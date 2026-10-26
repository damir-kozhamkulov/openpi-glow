"""Data transform for GLOW's planning module read by learned encoders (GlowModulePI0Pytorch).

`GlowModuleInputs` runs before the stock tokenizer, like `GlowPlanAsPrompt`, and drops the task
string the same way (the prompt becomes empty). Instead of rendering the active stage as text it
emits it as fixed-shape arrays that the model's Spatial and Temporal encoders read:

- `glow_text` / `glow_text_mask` [TEXT_LEN]: token ids (PaliGemma tokenizer, no BOS/EOS), one
  zero-padded segment per field, laid out by TEXT_SEGMENTS:
    subgoal   the active stage's subgoal_operation          (Spatial)
    object    its object noun                               (Spatial)
    target    its target noun                               (Spatial)
    rk0..rk3  the runtime_key of every stage of the task    (Temporal, K <= K_MAX)
- `glow_ints` [INTS_LEN]: predicate id (PREDICATES), active stage k, stage count K.
- `glow_floats` [FLOATS_LEN]: object xyz, object valid, target xyz, target valid - the pack's
  normalized anchors (the grounding), zero where a stage has none.

The active stage comes from the labels at training (`episode_index` / `frame_index`) and from
`oracle_plan_index` at inference (the client's tracker, or a forced stage); an optional
`plan_task` steers (that task's module, index clamped to its stages). `prev_plan_index`, sent by
every GLOW client, is accepted and ignored. A field longer than its segment raises rather than
being cut.

With a `teacher_tokenizer`, training samples also carry the task string as the stock pi05 prompt
(`glow_teacher_prompt` / `glow_teacher_prompt_mask`), for the distillation teacher pass of
GlowModulePI0Pytorch. Inference samples never do.
"""

import dataclasses

import numpy as np

from openpi.glow import plan_pack as _plan_pack
from openpi.models import tokenizer as _tokenizer
from openpi.transforms import DataDict
from openpi.transforms import DataTransformFn

K_MAX = 4
PREDICATES = ("on", "in", "open", "close", "turnon", "turnoff")  # ids 1..; 0 = unknown
SUBGOAL_LEN = 24
NOUN_LEN = 8
RUNTIME_KEY_LEN = 24

TEXT_SEGMENTS = {
    "subgoal": (0, SUBGOAL_LEN),
    "object": (SUBGOAL_LEN, SUBGOAL_LEN + NOUN_LEN),
    "target": (SUBGOAL_LEN + NOUN_LEN, SUBGOAL_LEN + 2 * NOUN_LEN),
    **{
        f"rk{j}": (
            SUBGOAL_LEN + 2 * NOUN_LEN + j * RUNTIME_KEY_LEN,
            SUBGOAL_LEN + 2 * NOUN_LEN + (j + 1) * RUNTIME_KEY_LEN,
        )
        for j in range(K_MAX)
    },
}
TEXT_LEN = SUBGOAL_LEN + 2 * NOUN_LEN + K_MAX * RUNTIME_KEY_LEN
INT_PREDICATE, INT_STAGE, INT_NUM_STAGES = 0, 1, 2
INTS_LEN = 3
FLOAT_OBJECT, FLOAT_OBJECT_VALID, FLOAT_TARGET, FLOAT_TARGET_VALID = slice(0, 3), 3, slice(4, 7), 7
FLOATS_LEN = 8


def predicate_id(stage: _plan_pack.Stage) -> int:
    return PREDICATES.index(stage.predicate) + 1 if stage.predicate in PREDICATES else 0


def _scalar(x):
    return x if isinstance(x, str | int) else np.asarray(x).item()


@dataclasses.dataclass(frozen=True)
class GlowModuleInputs(DataTransformFn):
    pack: _plan_pack.PlanPack
    tokenizer: _tokenizer.PaligemmaTokenizer
    teacher_tokenizer: _tokenizer.PaligemmaTokenizer | None = None

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.get("prompt")) is None:
            raise ValueError("GlowModuleInputs: prompt is required")
        prompt = _scalar(prompt)
        plan_task = data.pop("plan_task", None)
        plan_prompt = prompt if plan_task is None or not _scalar(plan_task) else _scalar(plan_task)
        data.pop("prev_plan_index", None)

        if "episode_index" in data and "frame_index" in data:
            if plan_prompt != prompt:
                raise ValueError("GlowModuleInputs: plan_task is an inference-time steering input")
            task = self.pack.task(prompt)
            episode_index = int(_scalar(data.pop("episode_index")))
            frame_index = int(_scalar(data.pop("frame_index")))
            if self.pack.episodes[episode_index].task_index != task.task_index:
                raise ValueError(
                    f"episode {episode_index} belongs to task {self.pack.episodes[episode_index].task_index}, "
                    f"but its prompt {prompt!r} is task {task.task_index}"
                )
            plan_index = self.pack.plan_index(episode_index, frame_index)
            if self.teacher_tokenizer is not None:
                tokens, mask = self.teacher_tokenizer.tokenize(prompt)
                data["glow_teacher_prompt"], data["glow_teacher_prompt_mask"] = tokens, mask
        elif (oracle := data.pop("oracle_plan_index", None)) is not None:
            plan_index = int(_scalar(oracle))
            if plan_prompt != prompt:
                plan_index = min(plan_index, self.pack.task(plan_prompt).num_plans - 1)
        else:
            raise ValueError(
                "GlowModuleInputs: the module encoders read a given stage; serve with the LIBERO client's "
                "--args.plan-mode oracle (the stage tracker's) or forced, which send oracle_plan_index"
            )

        data.update(self.encode(plan_prompt, plan_index))
        data["prompt"] = ""
        return data

    def encode(self, task_string: str, plan_index: int) -> dict[str, np.ndarray]:
        """The module arrays for stage `plan_index` of `task_string`'s task."""
        task = self.pack.task(task_string)
        if not 0 <= plan_index < task.num_plans:
            raise IndexError(f"plan_index {plan_index} outside 0..{task.num_plans - 1} for {task_string!r}")
        if task.num_plans > K_MAX:
            raise ValueError(f"task {task_string!r} has {task.num_plans} stages; the temporal encoder reads {K_MAX}")
        stage = task.stages[plan_index]

        text = np.zeros(TEXT_LEN, dtype=np.int32)
        mask = np.zeros(TEXT_LEN, dtype=bool)
        pieces = {"subgoal": stage.subgoal_operation, "object": stage.object_noun, "target": stage.target_noun}
        pieces |= {f"rk{j}": s.runtime_key for j, s in enumerate(task.stages)}
        for name, piece in pieces.items():
            start, stop = TEXT_SEGMENTS[name]
            ids = self.tokenizer.encode_text(piece)
            if len(ids) > stop - start:
                raise ValueError(f"{name} {piece!r} is {len(ids)} tokens; its segment holds {stop - start}")
            text[start : start + len(ids)] = ids
            mask[start : start + len(ids)] = True

        ints = np.zeros(INTS_LEN, dtype=np.int32)
        ints[INT_PREDICATE] = predicate_id(stage)
        ints[INT_STAGE] = plan_index
        ints[INT_NUM_STAGES] = task.num_plans

        floats = np.zeros(FLOATS_LEN, dtype=np.float32)
        if stage.object_xyz is not None:
            floats[FLOAT_OBJECT] = stage.object_xyz
            floats[FLOAT_OBJECT_VALID] = 1.0
        if stage.target_xyz is not None:
            floats[FLOAT_TARGET] = stage.target_xyz
            floats[FLOAT_TARGET_VALID] = 1.0
        return {"glow_text": text, "glow_text_mask": mask, "glow_ints": ints, "glow_floats": floats}
