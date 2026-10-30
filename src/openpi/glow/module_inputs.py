"""Data transform for GLOW's planning module read by learned encoders (GlowModulePI0Pytorch).

`GlowModuleInputs` runs before the stock tokenizer, like `GlowPlanAsPrompt`, and drops the task
string the same way (the prompt becomes empty). It emits the WHOLE plan of the sample's task as
fixed-shape arrays, split between the two encoders so that each carries what the other cannot:

    Spatial   what every stage does and where: per stage its atom, subgoal_operation, object and
              target nouns, predicate and anchors (the grounding), in the order `stage_order`
              gives: "sorted" (a set, saying nothing about execution), "execution", or
              "remaining" (the active stage first, then the later ones in execution order; the
              done stages are not shown)
    Temporal  when: the runtime_key of every stage in execution order, the stage count K, the
              active stage k (the pointer) and the frames since stage k began (the stage clock)

- `glow_text` / `glow_text_mask` [TEXT_LEN]: token ids (PaliGemma tokenizer, no BOS/EOS), one
  zero-padded segment per (field, stage j) in execution order, laid out by TEXT_SEGMENTS:
    atom{j} subgoal{j} object{j} target{j}   (Spatial)     rk{j}   (Temporal)
- `glow_ints` [INTS_LEN]: active stage k, stage count K, the predicate id of every stage
  (PREDICATES), the Spatial slot order (INT_SPATIAL_SLOT[i] = the stage, by execution index,
  shown at the i-th Spatial group), how many groups are shown (INT_SHOWN), the stage clock
  (INT_STAGE_FRAMES, -1 when unknown), and per stage one token id per HEAD_FIELDS entry
  (INT_HEAD, `head_index`; 0 = no word): the head words the encoders' "head" anchors embed.
- `glow_floats` [FLOATS_LEN]: per stage, object xyz, object valid, target xyz, target valid - the
  pack's normalized anchors, zero where a stage has none. With `episode_grounding`, a training
  sample's own stages take its episode's anchors (the pack's per-episode grasp / release
  positions) where the pack has them; inference always reads the per-task anchors.
Stages past K are zero and masked.

The active stage comes from the labels at training (`episode_index` / `frame_index`) and from
`oracle_plan_index` at inference (the client's tracker, or a forced stage); an optional
`plan_task` steers (that task's plan, index clamped to its stages) and an optional `plan_chain`
(a list of task strings) runs their stages as one plan in that order, the index counting over
the chain. `prev_plan_index`, sent by every GLOW client, is accepted and ignored. The stage clock
comes from `frame_index` and the episode's stage starts at training and from `plan_stage_frames`
(the client's count since the stage it sends last changed) at inference; with `stage_clock` an
inference sample without it raises. A field longer than its segment raises rather than being cut.

Chain augmentation (`chain_p` > 0, training samples only): a sample of a one-stage task in
`chain_tasks` becomes, with probability chain_p, a two-stage plan with another one-stage task
of `chain_tasks` (a partner that moves a different object), the two in random order and the
sample's own stage the active one. The partner's scene is the sample's own, so the images say
nothing about which stage is active: only the Temporal pointer does.

With a `teacher_tokenizer`, training samples also carry the task string as the stock pi05 prompt
(`glow_teacher_prompt` / `glow_teacher_prompt_mask`), for the distillation teacher pass of
GlowModulePI0Pytorch. Inference samples never do.
"""

import dataclasses
import re

import numpy as np

from openpi.glow import plan_pack as _plan_pack
from openpi.models import tokenizer as _tokenizer
from openpi.transforms import DataDict
from openpi.transforms import DataTransformFn

K_MAX = 4
PREDICATES = ("on", "in", "open", "close", "turnon", "turnoff")  # ids 1..; 0 = unknown
ATOM_LEN = 24
SUBGOAL_LEN = 24
NOUN_LEN = 8
RUNTIME_KEY_LEN = 24
STAGE_FIELDS = {"atom": ATOM_LEN, "subgoal": SUBGOAL_LEN, "object": NOUN_LEN, "target": NOUN_LEN, "rk": RUNTIME_KEY_LEN}
SPATIAL_FIELDS = ("atom", "subgoal", "object", "target")
TEMPORAL_FIELDS = ("rk",)
STAGE_TEXT_LEN = sum(STAGE_FIELDS.values())


def _segments() -> dict[str, tuple[int, int]]:
    out = {}
    for j in range(K_MAX):
        offset = j * STAGE_TEXT_LEN
        for field, length in STAGE_FIELDS.items():
            out[f"{field}{j}"] = (offset, offset + length)
            offset += length
    return out


TEXT_SEGMENTS = _segments()
TEXT_LEN = K_MAX * STAGE_TEXT_LEN
INT_STAGE, INT_NUM_STAGES = 0, 1
INT_PREDICATE = slice(2, 2 + K_MAX)
INT_SPATIAL_SLOT = slice(2 + K_MAX, 2 + 2 * K_MAX)
INT_SHOWN = 2 + 2 * K_MAX
INT_STAGE_FRAMES = 3 + 2 * K_MAX
# Per stage: the subgoal's verb, the object and target head nouns, the predicate as a word, and
# the stage's ordinal and count words ("second", "two" for stage 1): the Temporal head anchors
# gather the active stage's ordinal and the last stage's count.
HEAD_FIELDS = ("op", "object", "target", "atom", "ordinal", "count")
INT_HEAD = slice(4 + 2 * K_MAX, 4 + 2 * K_MAX + K_MAX * len(HEAD_FIELDS))
INTS_LEN = INT_HEAD.stop
STAGE_FLOATS = 8
FLOATS_LEN = K_MAX * STAGE_FLOATS
STAGE_ORDERS = ("sorted", "execution", "remaining")
PREDICATE_WORDS = {"on": "on", "in": "in", "open": "open", "close": "close", "turnon": "on", "turnoff": "off"}
COUNT_WORDS = ("one", "two", "three", "four")
ORDINAL_WORDS = ("first", "second", "third", "fourth")


def float_slices(j: int) -> tuple[slice, int, slice, int]:
    """Stage j's (object xyz, object valid, target xyz, target valid) in glow_floats."""
    base = j * STAGE_FLOATS
    return slice(base, base + 3), base + 3, slice(base + 4, base + 7), base + 7


def head_index(j: int, field: str) -> int:
    """Position in glow_ints of stage j's head-word id for `field` (HEAD_FIELDS)."""
    return INT_HEAD.start + j * len(HEAD_FIELDS) + HEAD_FIELDS.index(field)


def head_noun(noun: str) -> str | None:
    """The last word of a noun phrase, parentheticals dropped: "bowl (black)" -> "bowl"."""
    words = re.sub(r"\(.*?\)", " ", noun).replace("_", " ").split()
    return words[-1].lower() if words else None


def head_words(stage: _plan_pack.Stage, j: int) -> dict[str, str | None]:
    """Stage j's head word per HEAD_FIELDS entry (None: the stage has none)."""
    verb = stage.subgoal_operation.split()
    return {
        "op": verb[0].lower() if verb else None,
        "object": head_noun(stage.object_noun),
        "target": head_noun(stage.target_noun),
        "atom": PREDICATE_WORDS.get(stage.predicate),
        "ordinal": ORDINAL_WORDS[j],
        "count": COUNT_WORDS[j],
    }


def predicate_id(stage: _plan_pack.Stage) -> int:
    return PREDICATES.index(stage.predicate) + 1 if stage.predicate in PREDICATES else 0


def moved_object(stage: _plan_pack.Stage) -> str | None:
    """The object instance a stage moves ("in(butter_1, basket_1_contain_region)" -> "butter_1"),
    None for a stage that moves none (open / close / turnon / turnoff)."""
    if stage.predicate not in ("on", "in"):
        return None
    inside = stage.runtime_key.split("(", 1)[1]
    return inside.split(",", 1)[0].strip().rstrip(")")


def chain_partners(pack: _plan_pack.PlanPack, task_indices) -> dict[int, tuple[_plan_pack.TaskPlan, ...]]:
    """For each one-stage task in `task_indices`, the other one-stage tasks of `task_indices`
    whose stage moves a different object (or none), in task-index order."""
    by_index = {t.task_index: t for t in pack.tasks.values()}
    members = []
    for i in sorted({int(i) for i in task_indices}):
        if i not in by_index:
            raise KeyError(f"chain task {i} is not in the plan pack")
        if by_index[i].num_plans != 1:
            raise ValueError(
                f"chain task {i} ({by_index[i].task!r}) has {by_index[i].num_plans} stages; chains take one-stage tasks"
            )
        members.append(by_index[i])
    out = {}
    for own in members:
        mine = moved_object(own.stages[0])
        out[own.task_index] = tuple(
            t for t in members if t is not own and (mine is None or moved_object(t.stages[0]) != mine)
        )
    return out


def _scalar(x):
    return x if isinstance(x, str | int) else np.asarray(x).item()


@dataclasses.dataclass(frozen=True)
class GlowModuleInputs(DataTransformFn):
    pack: _plan_pack.PlanPack
    tokenizer: _tokenizer.PaligemmaTokenizer
    teacher_tokenizer: _tokenizer.PaligemmaTokenizer | None = None
    # The order of the Spatial groups: "sorted" (by runtime key, so the Spatial encoder sees a
    # set and the execution order lives in the Temporal arrays alone), "execution", or
    # "remaining" (the active stage first, then the later stages; done stages not shown).
    stage_order: str = "sorted"
    # Inference samples must carry plan_stage_frames (the model reads the stage clock).
    stage_clock: bool = False
    # Training samples' own stages take their episode's anchors where the pack has them.
    episode_grounding: bool = False
    # Chain augmentation of training samples (see the module docstring); chain_tasks are the
    # one-stage tasks (task indices) that may form chains.
    chain_p: float = 0.0
    chain_tasks: tuple[int, ...] = ()
    # Random draws for the augmentation; None uses numpy's global generator, which the DataLoader
    # seeds per worker.
    rng: np.random.Generator | None = None

    def __post_init__(self):
        if self.stage_order not in STAGE_ORDERS:
            raise ValueError(f"stage_order must be one of {STAGE_ORDERS}, got {self.stage_order!r}")
        if not 0.0 <= self.chain_p <= 1.0:
            raise ValueError(f"chain_p must be in [0, 1], got {self.chain_p}")
        if self.chain_p > 0 and not self.chain_tasks:
            raise ValueError("chain_p > 0 needs chain_tasks")
        object.__setattr__(self, "partners", chain_partners(self.pack, self.chain_tasks))
        object.__setattr__(self, "_ids", {})

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.get("prompt")) is None:
            raise ValueError("GlowModuleInputs: prompt is required")
        prompt = _scalar(prompt)
        plan_task = data.pop("plan_task", None)
        plan_task = None if plan_task is None or not _scalar(plan_task) else _scalar(plan_task)
        plan_chain = data.pop("plan_chain", None)
        plan_chain = None if plan_chain is None or len(plan_chain) == 0 else [_scalar(t) for t in plan_chain]
        data.pop("prev_plan_index", None)
        stage_frames = data.pop("plan_stage_frames", None)
        grounding = None

        if "episode_index" in data and "frame_index" in data:
            if plan_task is not None or plan_chain is not None:
                raise ValueError("GlowModuleInputs: plan_task and plan_chain are inference-time steering inputs")
            task = self.pack.task(prompt)
            episode_index = int(_scalar(data.pop("episode_index")))
            frame_index = int(_scalar(data.pop("frame_index")))
            episode = self.pack.episodes[episode_index]
            if episode.task_index != task.task_index:
                raise ValueError(
                    f"episode {episode_index} belongs to task {episode.task_index}, "
                    f"but its prompt {prompt!r} is task {task.task_index}"
                )
            own_index = episode.plan_index(frame_index)
            stage_frames = frame_index - episode.stage_starts[own_index]
            own = episode.anchors if self.episode_grounding else None
            stages, plan_index = self.chain(task, own_index)
            if own is not None:
                # The own stages are the plan, or (a chain) its stage plan_index; a partner keeps
                # the per-task anchors.
                if stages is task.stages:
                    grounding = list(own)
                else:
                    grounding = [None] * len(stages)
                    grounding[plan_index] = own[own_index]
            if self.teacher_tokenizer is not None:
                tokens, mask = self.teacher_tokenizer.tokenize(prompt)
                data["glow_teacher_prompt"], data["glow_teacher_prompt_mask"] = tokens, mask
        elif (oracle := data.pop("oracle_plan_index", None)) is not None:
            if stage_frames is None and self.stage_clock:
                raise ValueError(
                    "GlowModuleInputs: this model reads the stage clock; the client must send plan_stage_frames"
                )
            stage_frames = -1 if stage_frames is None else int(_scalar(stage_frames))
            plan_index = int(_scalar(oracle))
            if plan_chain is not None:
                if plan_task is not None:
                    raise ValueError("GlowModuleInputs: plan_task and plan_chain exclude each other")
                stages = tuple(s for t in plan_chain for s in self.pack.task(t).stages)
            else:
                stages = self.pack.task(prompt if plan_task is None else plan_task).stages
                if plan_task is not None:
                    plan_index = min(plan_index, len(stages) - 1)
        else:
            raise ValueError(
                "GlowModuleInputs: the module encoders read a given stage; serve with the LIBERO client's "
                "--args.plan-mode oracle (the stage tracker's) or forced, which send oracle_plan_index"
            )

        data.update(self.encode_stages(stages, plan_index, grounding=grounding, stage_frames=stage_frames))
        data["prompt"] = ""
        return data

    def chain(self, task: _plan_pack.TaskPlan, plan_index: int) -> tuple[tuple[_plan_pack.Stage, ...], int]:
        """A training sample's plan: the task's own stages, or (chain augmentation) its one stage
        and a partner's in random order, the own stage active."""
        partners = self.partners.get(task.task_index, ())
        if self.chain_p == 0 or not partners:
            return task.stages, plan_index
        rng = self.rng if self.rng is not None else np.random
        if rng.random() >= self.chain_p:
            return task.stages, plan_index
        partner = partners[int(rng.choice(len(partners)))]
        own_first = rng.random() < 0.5
        stages = (task.stages[0], partner.stages[0]) if own_first else (partner.stages[0], task.stages[0])
        return stages, 0 if own_first else 1

    def encode(self, task_string: str, plan_index: int, *, stage_frames: int = -1) -> dict[str, np.ndarray]:
        """The module arrays for stage `plan_index` of `task_string`'s task (its per-task anchors)."""
        task = self.pack.task(task_string)
        if not 0 <= plan_index < task.num_plans:
            raise IndexError(f"plan_index {plan_index} outside 0..{task.num_plans - 1} for {task_string!r}")
        return self.encode_stages(task.stages, plan_index, stage_frames=stage_frames)

    def _first_id(self, word: str | None) -> int:
        """The first SentencePiece id of `word` (0 for None), cached."""
        if word is None:
            return 0
        if word not in self._ids:
            self._ids[word] = int(self.tokenizer.encode_text(word)[0])
        return self._ids[word]

    def encode_stages(self, stages, plan_index: int, *, grounding=None, stage_frames: int = -1) -> dict[str, np.ndarray]:
        """The module arrays for a plan of `stages` (execution order) with stage `plan_index` active.
        `grounding` optionally replaces anchors per stage: None, or (object xyz, target xyz) where a
        None keeps that stage's own anchor. `stage_frames` is the stage clock (-1: unknown)."""
        stages = tuple(stages)
        num = len(stages)
        if not 0 <= plan_index < num:
            raise IndexError(f"plan_index {plan_index} outside 0..{num - 1}")
        if num > K_MAX:
            raise ValueError(f"a plan of {num} stages; the encoders read {K_MAX}")
        if grounding is not None and len(grounding) != num:
            raise ValueError(f"grounding for {len(grounding)} stages; the plan has {num}")

        text = np.zeros(TEXT_LEN, dtype=np.int32)
        mask = np.zeros(TEXT_LEN, dtype=bool)
        for j, stage in enumerate(stages):
            pieces = {
                "atom": stage.atom,
                "subgoal": stage.subgoal_operation,
                "object": stage.object_noun,
                "target": stage.target_noun,
                "rk": stage.runtime_key,
            }
            for field, piece in pieces.items():
                start, stop = TEXT_SEGMENTS[f"{field}{j}"]
                ids = self.tokenizer.encode_text(piece)
                if len(ids) > stop - start:
                    raise ValueError(f"{field} {piece!r} is {len(ids)} tokens; its segment holds {stop - start}")
                text[start : start + len(ids)] = ids
                mask[start : start + len(ids)] = True

        ints = np.zeros(INTS_LEN, dtype=np.int32)
        ints[INT_STAGE] = plan_index
        ints[INT_NUM_STAGES] = num
        ints[INT_PREDICATE] = [predicate_id(s) for s in stages] + [0] * (K_MAX - num)
        if self.stage_order == "sorted":
            order = sorted(range(num), key=lambda j: stages[j].runtime_key)
        elif self.stage_order == "remaining":
            order = list(range(plan_index, num)) + list(range(plan_index))
        else:
            order = list(range(num))
        ints[INT_SPATIAL_SLOT] = order + list(range(num, K_MAX))
        ints[INT_SHOWN] = num - plan_index if self.stage_order == "remaining" else num
        ints[INT_STAGE_FRAMES] = stage_frames
        for j in range(K_MAX):
            words = head_words(stages[j], j) if j < num else {"ordinal": ORDINAL_WORDS[j], "count": COUNT_WORDS[j]}
            for field in HEAD_FIELDS:
                ints[head_index(j, field)] = self._first_id(words.get(field))

        floats = np.zeros(FLOATS_LEN, dtype=np.float32)
        for j, stage in enumerate(stages):
            obj, obj_valid, tgt, tgt_valid = float_slices(j)
            own = grounding[j] if grounding is not None and grounding[j] is not None else (None, None)
            obj_xyz = own[0] if own[0] is not None else stage.object_xyz
            tgt_xyz = own[1] if own[1] is not None else stage.target_xyz
            if obj_xyz is not None:
                floats[obj] = obj_xyz
                floats[obj_valid] = 1.0
            if tgt_xyz is not None:
                floats[tgt] = tgt_xyz
                floats[tgt_valid] = 1.0
        return {"glow_text": text, "glow_text_mask": mask, "glow_ints": ints, "glow_floats": floats}
