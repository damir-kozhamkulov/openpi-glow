"""Does a GLOW module model's velocity follow the plan's ACTIVE stage? Forward only, no simulator.

Every plan here has two LIBERO-Goal tasks (A, B), and the distillation teacher (the same pretrained
model reading one task's instruction as its prompt) gives the velocity each task calls for on the
same images, state, noisy actions and time. The score of a velocity v for active task A against
the other task B, pooled over samples:

    margin = (sum |v - T_B|^2 - sum |v - T_A|^2) / sum |T_A - T_B|^2

+1 is exactly the active task's teacher, -1 exactly the other's, 0 halfway. rel_err = sum |v - T_A|^2
/ sum |T_A - T_B|^2 separates a blend (margin ~0, rel_err ~0.25) from a velocity near neither.

  rows    A trained checkpoint on its own prefix (`name=config:weights_dir`), ROW_VARIANTS:
            order_ab         plan [A, B], pointer 0: active A, as a chain-01 eval starts
            order_ba         plan [B, A], pointer 0: active B, as a chain-10 eval starts
            swap_ab          plan [A, B], pointer 1: active B, as a chain-swap eval
            second_ba        plan [B, A], pointer 1: active A
            one_stage_a      plan [A]: active A with no second stage listed
            temporal_off_ab  order_ab with the Temporal tokens masked out of the prefix
            temporal_off_ba  order_ba with the Temporal tokens masked out of the prefix
            gate_off_ab      order_ab with every stage's gate at 1 (a gated row only)
            gate_off_ba      order_ba with every stage's gate at 1 (a gated row only)
          temporal_off also sets a gated row's gates to 1, so it removes Temporal altogether. Also
          how far the Spatial tokens move between the two orders (0 for a "sorted" row: its order
          can only come through Temporal) and, for a gated row, the smaller group's mean gate.
  grid    LAYOUTS, at the fine-tuning init (`--grid-weights`): real word embeddings at a word
          token's scale, no learned residual - what the frozen backbone can read before any
          training, over order_ab and order_ba. Spatial is the two stages' words (op, object, target,
          predicate) plus a summary, in sorted order; the Temporal words vary:
            set               no active word: nothing says which stage is active
            set_active        the active object noun after the set (v2t's layout)
            active_set        the same before the set
            set_list          the active and the next object noun after the set
            list_set          the same before the set
            set_active_group  the active stage's four words again after the set
            active_group_set  the same before the set
            set_gate25        set_active, the other stage's group at a quarter of its size
            v2s               the active stage's group first, then the other's (v2s's layout)
            active_only       the active stage's group alone

Samples are held-out demos of the Goal tasks: "start" frames (below --start-window), where the scene
is valid for any Goal task, with a pair drawn from EVAL_PAIRS; "mid" frames (up to --mid-window),
with a pair holding the demo's own task as A, so only variants with A active mean anything there.
No images are augmented. The batches are built once by a background thread and reused by every
row, so the GPU is not left waiting on the PVC or on image decoding; the log reports the seconds
spent waiting. Episodes are read straight from their parquet files: only the probe's few episodes
are touched, no dataset-wide Arrow cache is built.

  python scripts/glow_temporal_probe.py --rows v2t=pi05_libero_glow_v2t:<dir> \
      --grid-weights <dir with the fine-tuning init> --norm-stats-dir <assets dir> \
      --dataset-root <lerobot repo dir> --exclude-episodes 379:427 --out temporal_probe.json
"""

import dataclasses
import json
import logging
import pathlib
import sys
import threading
import time
import types

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import glow_flow_probe as _flow_probe  # noqa: E402
import glow_module_diag as _diag  # noqa: E402

# The chain evals' (own, partner) pairs, as the client's chain lines print them (PLAN_CHAIN_OFFSET=1).
EVAL_PAIRS = (
    ("open the middle drawer of the cabinet", "put the bowl on the stove"),
    ("put the bowl on the stove", "put the wine bottle on top of the cabinet"),
    ("put the wine bottle on top of the cabinet", "open the top drawer and put the bowl inside"),
    ("open the top drawer and put the bowl inside", "push the plate to the front of the stove"),
    ("put the bowl on top of the cabinet", "push the plate to the front of the stove"),
    ("push the plate to the front of the stove", "put the cream cheese in the bowl"),
    ("put the cream cheese in the bowl", "turn on the stove"),
    ("turn on the stove", "put the bowl on the plate"),
    ("put the bowl on the plate", "put the wine bottle on the rack"),
    ("put the wine bottle on the rack", "open the middle drawer of the cabinet"),
)
# name -> (plan as "A"/"B" in execution order, pointer, what of Temporal is switched off).
ROW_VARIANTS = {
    "order_ab": ("AB", 0, None),
    "order_ba": ("BA", 0, None),
    "swap_ab": ("AB", 1, None),
    "second_ba": ("BA", 1, None),
    "one_stage_a": ("A", 0, None),
    "temporal_off_ab": ("AB", 0, "temporal"),
    "temporal_off_ba": ("BA", 0, "temporal"),
    "gate_off_ab": ("AB", 0, "gate"),
    "gate_off_ba": ("BA", 0, "gate"),
}
LAYOUTS = (
    "set",
    "set_active",
    "active_set",
    "set_list",
    "list_set",
    "set_active_group",
    "active_group_set",
    "set_gate25",
    "v2s",
    "active_only",
)
GATE = 0.25
GOAL_FIRST_EPISODE, GOAL_END_EPISODE = 379, 807
BOOTSTRAP = 200


@dataclasses.dataclass
class Args:
    # Assets directory whose <repo_id>/norm_stats.json normalizes state and actions.
    norm_stats_dir: str
    # LeRobot dataset root: the directory holding meta/ and data/.
    dataset_root: str
    # Trained checkpoints, name=config:weights_dir (the directory holding model.safetensors).
    rows: tuple[str, ...] = ()
    # The fine-tuning init (no encoders) for the layout grid; None skips the grid.
    grid_weights: str | None = None
    # The config whose data pipeline renders the samples, and whose model the grid uses.
    config: str = "pi05_libero_glow_v2t"
    repo_id: str = "physical-intelligence/libero"
    # Episodes the rows trained on, as OPENPI_TRAIN_EPISODES: never sampled.
    exclude_episodes: str = ""
    episodes_per_task: int = 4
    start_samples: int = 320
    mid_samples: int = 192
    start_window: int = 40
    mid_window: int = 120
    batch_size: int = 8
    seed: int = 0
    precision: str = "float32"
    device: str = "cuda"
    out: str | None = None


def parse_row(row: str) -> tuple[str, str, str]:
    """name=config:weights_dir -> (name, config, weights_dir)."""
    name, _, rest = row.partition("=")
    config, _, weights = rest.partition(":")
    if not (name and config and weights):
        raise ValueError(f"a row is name=config:weights_dir, got {row!r}")
    return name, config, weights


def heldout_episodes(pack, exclude, per_task: int, rng) -> dict[int, list[int]]:
    """Per Goal task index, `per_task` of its demos that are not in `exclude`."""
    by_task: dict[int, list[int]] = {}
    excluded = set(exclude)
    for ep in range(GOAL_FIRST_EPISODE, GOAL_END_EPISODE):
        if ep not in excluded:
            by_task.setdefault(pack.episodes[ep].task_index, []).append(ep)
    out = {}
    for task_index, episodes in sorted(by_task.items()):
        if len(episodes) < per_task:
            raise ValueError(f"task {task_index} has {len(episodes)} held-out demos, the probe wants {per_task}")
        out[task_index] = sorted(int(e) for e in rng.choice(episodes, size=per_task, replace=False))
    return out


def plan_frames(pack, heldout, *, start_samples, mid_samples, start_window, mid_window, batch_size, rng) -> list[dict]:
    """The probe's samples in batch order (the start frames first): episode, frame, bucket, pair
    (A, B) as task strings. Each bucket's size is rounded down to whole batches."""
    start_samples -= start_samples % batch_size
    mid_samples -= mid_samples % batch_size
    tasks_by_index = {t.task_index: t.task for t in pack.tasks.values()}
    episodes = [e for _, eps in sorted(heldout.items()) for e in eps]

    def own_pairs(task: str) -> list[tuple[str, str]]:
        return [(a, b) if a == task else (b, a) for a, b in EVAL_PAIRS if task in (a, b)]

    def draw(lo: int, hi: int, n: int) -> list[tuple[int, int]]:
        pool = [(e, f) for e in episodes for f in range(lo, min(hi, pack.episodes[e].n_frames))]
        return [pool[i] for i in rng.choice(len(pool), size=n, replace=False)]

    samples = []
    for ep, frame in draw(0, start_window, start_samples):
        a, b = EVAL_PAIRS[int(rng.integers(len(EVAL_PAIRS)))]
        samples.append({"episode": ep, "frame": frame, "bucket": "start", "pair": (a, b)})
    for ep, frame in draw(start_window, mid_window, mid_samples):
        pairs = own_pairs(tasks_by_index[pack.episodes[ep].task_index])
        samples.append({"episode": ep, "frame": frame, "bucket": "mid", "pair": pairs[int(rng.integers(len(pairs)))]})
    return samples


def read_task_strings(root: pathlib.Path) -> dict[int, str]:
    tasks = {}
    with open(root / "meta/tasks.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            tasks[int(row["task_index"])] = row["task"]
    return tasks


def read_frames(root: pathlib.Path, episode: int, frames, action_horizon: int, task_strings) -> dict[int, dict]:
    """The frames of one episode as the training loader hands them over: images, state, the action
    chunk (the last action repeated past the episode's end, as LeRobot pads), indices and the prompt."""
    import io

    from PIL import Image
    import pyarrow.parquet as pq

    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    path = root / info["data_path"].format(episode_chunk=episode // info["chunks_size"], episode_index=episode)
    table = pq.read_table(path, columns=["image", "wrist_image", "state", "actions", "task_index"])
    actions = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
    out = {}
    for frame in frames:
        chunk = actions[np.minimum(np.arange(frame, frame + action_horizon), len(actions) - 1)]
        task_index = table["task_index"][frame].as_py()

        def decode(column: str, frame: int = frame) -> np.ndarray:
            data = table[column][frame].as_py()["bytes"]
            return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)

        out[frame] = {
            "image": decode("image"),
            "wrist_image": decode("wrist_image"),
            "state": np.asarray(table["state"][frame].as_py(), dtype=np.float32),
            "actions": chunk,
            "episode_index": episode,
            "frame_index": frame,
            "task_index": task_index,
            "prompt": task_strings[task_index],
        }
    return out


class BatchStore:
    """Builds the probe's batches on a background thread, once, in order; `get` waits for one and
    counts the wait, so the log shows whether the data ever held the GPU back."""

    def __init__(self, plan, build_batch, batch_size: int):
        self.num_batches = len(plan) // batch_size
        self._plan, self._build, self._size = plan, build_batch, batch_size
        self._batches: list = []
        self._error: BaseException | None = None
        self._ready = threading.Condition()
        self.waited = 0.0
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        try:
            for b in range(self.num_batches):
                batch = self._build(self._plan[b * self._size : (b + 1) * self._size])
                with self._ready:
                    self._batches.append(batch)
                    self._ready.notify_all()
        except BaseException as e:  # noqa: BLE001 - re-raised on the main thread by get
            with self._ready:
                self._error = e
                self._ready.notify_all()

    def get(self, b: int):
        t0 = time.monotonic()
        with self._ready:
            while len(self._batches) <= b and self._error is None:
                self._ready.wait(timeout=5.0)
            if self._error is not None:
                raise self._error
            batch = self._batches[b]
        self.waited += time.monotonic() - t0
        return batch


def make_build_batch(module_pipeline, root: pathlib.Path, action_horizon: int, task_strings):
    def build(samples: list[dict]) -> dict:
        frames_by_episode: dict[int, list[int]] = {}
        for s in samples:
            frames_by_episode.setdefault(s["episode"], []).append(s["frame"])
        raw = {
            ep: read_frames(root, ep, sorted(set(frames)), action_horizon, task_strings)
            for ep, frames in frames_by_episode.items()
        }
        rendered = [module_pipeline(dict(raw[s["episode"]][s["frame"]])) for s in samples]
        return {"samples": samples, "batch": _flow_probe._collate(rendered)}  # noqa: SLF001

    return build


def plan_stages(pack, pair: tuple[str, str], plan: str) -> tuple:
    """The stages of `plan` ("AB", "BA" or "A") for the (A, B) task strings."""
    by_name = {"A": pack.task(pair[0]).stages[0], "B": pack.task(pair[1]).stages[0]}
    return tuple(by_name[c] for c in plan)


def module_arrays(module, samples, plan: str, pointer: int, device):
    """The module arrays of one plan for every sample, stacked, as a namespace `module_tokens` reads."""
    import torch

    arrays = [
        module.encode_stages(plan_stages(module.pack, s["pair"], plan), pointer, stage_frames=s["frame"])
        for s in samples
    ]
    return types.SimpleNamespace(
        **{name: torch.as_tensor(np.stack([a[name] for a in arrays])).to(device) for name in arrays[0]}
    )


def teacher_velocities(model, module, samples, inputs, x_t, flow_time, device) -> dict:
    """The teacher's velocity for task A and for task B of each sample's pair."""
    import torch

    from openpi.glow import module_inputs as _mi

    images, img_masks, state = inputs
    out = {}
    for which, index in (("A", 0), ("B", 1)):
        tokens = []
        for s in samples:
            task = module.pack.task(s["pair"][index])
            tokens.append(module.teacher_tokenizer.tokenize(_mi.teacher_prompt(task.stages[0], task.task)))
        ids = torch.as_tensor(np.stack([t[0] for t in tokens])).to(device)
        mask = torch.as_tensor(np.stack([t[1] for t in tokens])).to(device)
        prefix = model.embed_prefix(images, img_masks, ids, mask)
        out[which] = model.velocity_from_prefix(*prefix, state, x_t, flow_time)
    return out


def squared(a, b) -> np.ndarray:
    return (a - b).float().pow(2).sum(dim=(1, 2)).cpu().numpy()


def margin_stats(da: np.ndarray, db: np.ndarray, dab: np.ndarray, rng=None) -> dict:
    """Pooled margin and rel_err of per-sample squared distances (active, other, between teachers),
    with a bootstrap standard error of the margin over samples."""

    def margin(i):
        return float((db[i].sum() - da[i].sum()) / max(dab[i].sum(), 1e-12))

    everything = np.arange(len(da))
    out = {
        "samples": len(da),
        "margin": margin(everything),
        "rel_err": float(da.sum() / max(dab.sum(), 1e-12)),
    }
    if rng is not None and len(da) > 1:
        out["margin_se"] = float(np.std([margin(rng.integers(len(da), size=len(da))) for _ in range(BOOTSTRAP)]))
    return out


def rescale(tokens, rms):
    """Each token to the target RMS `rms` [B, 1, 1], as GlowModulePI0Pytorch rescales its tokens; a
    zero token stays zero."""
    import torch

    return tokens * torch.rsqrt(tokens.pow(2).mean(-1, keepdim=True) + 1e-6) * rms


def layout_tokens(name: str, model, arrays):
    """Layout `name`'s tokens [B, N, D] at a word token's scale, for two-stage plans with `arrays`
    (GlowModuleInputs "sorted" arrays). Reproduces the fresh encoders' anchors: set_active is what
    pi05_libero_glow_v2t emits at init and v2s what pi05_libero_glow_v2s emits."""
    import torch

    from openpi.glow import module_encoders as _enc
    from openpi.glow import module_inputs as _mi

    ints = arrays.glow_ints
    embed = model.paligemma_with_expert.embed_language_tokens
    head = _enc.head_embeddings(embed, ints)
    rows = torch.arange(ints.shape[0], device=ints.device)
    stage = ints[..., _mi.INT_STAGE].long()
    if int(ints[..., _mi.INT_NUM_STAGES].min()) != 2 or int(ints[..., _mi.INT_NUM_STAGES].max()) != 2:
        raise ValueError("layout_tokens takes two-stage plans")
    fields = [_mi.HEAD_FIELDS.index(_enc.SLOT_HEAD[slot]) for slot in _enc.SPATIAL_SLOTS]
    groups = head[:, :2][:, :, fields]  # [B, 2, 4, D]
    order = ints[..., _mi.INT_SPATIAL_SLOT].long()[:, :2]
    first, second = groups[rows, order[:, 0]], groups[rows, order[:, 1]]
    active, other = groups[rows, stage], groups[rows, 1 - stage]
    atom = head[:, :2, _mi.HEAD_FIELDS.index("atom")]
    ctx_mean, ctx_active = atom.mean(dim=1, keepdim=True), atom[rows, stage][:, None]
    count = head[rows, ints[..., _mi.INT_NUM_STAGES].long() - 1, _mi.HEAD_FIELDS.index("count")][:, None]
    ordinal = head[rows, stage, _mi.HEAD_FIELDS.index("ordinal")][:, None]
    active_noun = _enc.active_noun(head, ints)[:, None]
    other_ints = ints.clone()
    other_ints[..., _mi.INT_STAGE] = 1 - stage
    next_noun = _enc.active_noun(head, other_ints)[:, None]
    spatial_set = [first, second, ctx_mean]

    rms = model.module_text_rms(arrays)
    if name == "set_gate25":
        gate = torch.where(order == stage[:, None], 1.0, GATE).to(first.dtype)  # [B, 2]
        return torch.cat(
            [
                rescale(first, rms) * gate[:, 0, None, None],
                rescale(second, rms) * gate[:, 1, None, None],
                rescale(torch.cat([ctx_mean, count, active_noun], dim=1), rms),
            ],
            dim=1,
        )
    parts = {
        "set": [*spatial_set, count],
        "set_active": [*spatial_set, count, active_noun],
        "active_set": [count, active_noun, *spatial_set],
        "set_list": [*spatial_set, count, active_noun, next_noun],
        "list_set": [count, active_noun, next_noun, *spatial_set],
        "set_active_group": [*spatial_set, count, active],
        "active_group_set": [count, active, *spatial_set],
        "v2s": [active, other, ctx_active, count, ordinal],
        "active_only": [active, ctx_active, count, ordinal],
    }
    if name not in parts:
        raise ValueError(f"unknown layout {name!r}; choose from {LAYOUTS}")
    return rescale(torch.cat(parts[name], dim=1), rms)


def layout_prefix(model, tokens, images, img_masks, lang_tokens, lang_masks):
    """The stock prefix with `tokens` between the images and the language, as the model places its own."""
    import torch

    embs, pad_masks, att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    at = embs.shape[1] - lang_tokens.shape[1]
    b, n = tokens.shape[:2]
    return (
        torch.cat([embs[:, :at], tokens.to(embs.dtype), embs[:, at:]], dim=1),
        torch.cat([pad_masks[:, :at], pad_masks.new_ones(b, n), pad_masks[:, at:]], dim=1),
        torch.cat([att_masks[:, :at], att_masks.new_zeros(b, n), att_masks[:, at:]], dim=1),
    )


class Accumulator:
    """Per-sample squared distances per variant, with each sample's bucket and pair."""

    def __init__(self):
        self.rows: dict[str, list] = {}

    def add(self, name, samples, da, db, dab, active_is_a: bool) -> None:
        rows = self.rows.setdefault(name, [])
        for s, x, y, z in zip(samples, da, db, dab, strict=True):
            rows.append((s["bucket"], s["pair"], s["pair"][0 if active_is_a else 1], float(x), float(y), float(z)))

    def stats(self, names, rng, *, bucket: str = "start", key=None) -> dict:
        """The pooled stats of `names` on `bucket`, overall or split by `key` (a row -> group)."""
        rows = [r for n in names for r in self.rows.get(n, []) if r[0] == bucket]
        if not rows:
            return {}

        def pooled(sel, rng=None):
            return margin_stats(*(np.array([r[i] for r in sel]) for i in (3, 4, 5)), rng)

        if key is None:
            return pooled(rows, rng)
        groups: dict = {}
        for r in rows:
            groups.setdefault(key(r), []).append(r)
        return {" | ".join(k) if isinstance(k, tuple) else k: pooled(v) for k, v in sorted(groups.items())}


def run_row(name, model, module, store, args, device, *, grid: bool) -> dict:
    """One checkpoint over every batch (`grid`: the LAYOUTS over the start batches)."""
    import jax
    import torch

    from openpi.glow import module_encoders as _enc
    from openpi.models import model as _model

    def to_device(x):
        t = torch.as_tensor(x)
        # Normalize can promote state to float64 (numpy 2 promotion); the model computes in float32.
        return t.to(device, torch.float32) if t.dtype == torch.float64 else t.to(device)

    acc = Accumulator()
    moved, smaller_gate = [], []
    temporal = getattr(model, "glow_temporal_encoder", None)
    gated = temporal is not None and temporal.gate is not None
    t0, forwards = time.monotonic(), 0
    for b in range(store.num_batches):
        item = store.get(b)
        samples = item["samples"]
        if grid and any(s["bucket"] != "start" for s in samples):
            continue
        batch = jax.tree.map(to_device, item["batch"])
        obs = _model.Observation.from_dict(batch)
        with torch.no_grad():
            images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(obs, train=False)  # noqa: SLF001
            torch.manual_seed(args.seed * 100_003 + b)
            actions = torch.as_tensor(batch["actions"]).to(device, torch.float32)
            noise = model.sample_noise(actions.shape, device)
            flow_time = model.sample_time(actions.shape[0], device)
            expanded = flow_time[:, None, None]
            x_t = expanded * noise + (1 - expanded) * actions
            teachers = teacher_velocities(model, module, samples, (images, img_masks, state), x_t, flow_time, device)
            dab = squared(teachers["A"], teachers["B"])
            forwards += 2

            def score(variant, v, active_is_a):
                t_active, t_other = (teachers["A"], teachers["B"]) if active_is_a else (teachers["B"], teachers["A"])
                acc.add(variant, samples, squared(v, t_active), squared(v, t_other), dab, active_is_a)

            if grid:
                for plan, active_is_a in (("AB", True), ("BA", False)):
                    arrays = module_arrays(module, samples, plan, 0, device)
                    for layout in LAYOUTS:
                        tokens = layout_tokens(layout, model, arrays)
                        prefix = layout_prefix(model, tokens, images, img_masks, lang_tokens, lang_masks)
                        score(layout, model.velocity_from_prefix(*prefix, state, x_t, flow_time), active_is_a)
                        forwards += 1
            else:
                embed = model.paligemma_with_expert.embed_language_tokens
                spatial = []
                for variant, (plan, pointer, off) in ROW_VARIANTS.items():
                    if off == "gate" and not gated:
                        continue
                    arrays = module_arrays(module, samples, plan, pointer, device)
                    if variant in ("order_ab", "order_ba"):
                        spatial.append(
                            model.glow_spatial_encoder(
                                embed, arrays.glow_text, arrays.glow_text_mask, arrays.glow_ints, arrays.glow_floats.float()
                            )
                        )
                    if off and gated:
                        temporal.stage_gates = lambda _, ints: _enc.stage_valid(ints).to(torch.float32)
                    if off == "temporal":
                        own_mask = model.module_token_mask
                        keep = torch.ones_like(own_mask(arrays))
                        keep[:, -_enc.TEMPORAL_TOKENS :] = False
                        model.module_token_mask = lambda o, own_mask=own_mask, keep=keep: own_mask(o) & keep
                    prefix = model.embed_observation_prefix(arrays, images, img_masks, lang_tokens, lang_masks)
                    if off == "temporal":
                        del model.module_token_mask
                    if off and gated:
                        del temporal.stage_gates
                    if gated and variant == "order_ab":
                        gates = model.module_tokens_and_gates(arrays)[1]
                        smaller_gate.append(float(gates[:, [0, len(_enc.SPATIAL_SLOTS)]].min(dim=1).values.mean()))
                    active = plan[pointer]
                    score(variant, model.velocity_from_prefix(*prefix, state, x_t, flow_time), active == "A")
                    forwards += 1
                moved.append(float((spatial[0] - spatial[1]).abs().max()))
        if b % 8 == 0:
            elapsed = time.monotonic() - t0
            logging.info(
                f"{name}: batch {b + 1}/{store.num_batches}, {forwards / max(elapsed, 1e-9):.2f} forwards/s, "
                f"waited {store.waited:.0f} s on data so far"
            )
    logging.info(f"{name}: {forwards} forwards in {time.monotonic() - t0:.0f} s, {store.waited:.0f} s of data wait in all")

    rng = np.random.default_rng(args.seed)
    result: dict = {}
    if grid:
        result["layouts"] = {layout: acc.stats([layout], rng) for layout in LAYOUTS}
        result["layouts_by_active_task"] = {
            layout: acc.stats([layout], rng, key=lambda r: r[2]) for layout in LAYOUTS
        }
        for layout in LAYOUTS:
            s = result["layouts"][layout]
            logging.info(
                f"LAYOUT {layout:17s} margin {s['margin']:+.3f} +- {s['margin_se']:.3f}  rel_err {s['rel_err']:.3f}  "
                f"({s['samples']} scores)"
            )
        return result

    plan_first = ("order_ab", "order_ba")
    result["spatial_tokens_moved_between_orders_max"] = max(moved)
    result["plan_first"] = acc.stats(plan_first, rng)
    result["plan_first_by_pair"] = acc.stats(plan_first, rng, key=lambda r: tuple(r[1]))
    result["plan_first_by_active_task"] = acc.stats(plan_first, rng, key=lambda r: r[2])
    result["variants"] = {
        bucket: {v: acc.stats([v], rng, bucket=bucket) for v in ROW_VARIANTS} for bucket in ("start", "mid")
    }
    result["temporal_off"] = acc.stats(("temporal_off_ab", "temporal_off_ba"), rng)
    if gated:
        result["gate_off"] = acc.stats(("gate_off_ab", "gate_off_ba"), rng)
        result["smaller_group_gate_mean"] = float(np.mean(smaller_gate))
        logging.info(
            f"GATE {name}: gates at 1 margin {result['gate_off']['margin']:+.3f} +- "
            f"{result['gate_off']['margin_se']:.3f}; smaller group's mean gate {result['smaller_group_gate_mean']:.3f}"
        )
    logging.info(f"ORDER {name}: Spatial tokens move at most {max(moved):.3e} between the two orders")
    logging.info(
        f"ORDER {name}: plan_first margin {result['plan_first']['margin']:+.3f} +- "
        f"{result['plan_first']['margin_se']:.3f}, rel_err {result['plan_first']['rel_err']:.3f}; Temporal off "
        f"{result['temporal_off']['margin']:+.3f} +- {result['temporal_off']['margin_se']:.3f}"
    )
    for bucket, variants in result["variants"].items():
        for v, s in variants.items():
            if s:
                logging.info(
                    f"VARIANT {name} {bucket:5s} {v:16s} margin {s['margin']:+.3f} +- {s['margin_se']:.3f}  "
                    f"rel_err {s['rel_err']:.3f}"
                )
    for task, s in result["plan_first_by_active_task"].items():
        logging.info(f"TASK {name} {task!r}: margin {s['margin']:+.3f} rel_err {s['rel_err']:.3f} ({s['samples']})")
    for pair, s in result["plan_first_by_pair"].items():
        logging.info(f"PAIR {name} {pair!r}: margin {s['margin']:+.3f} rel_err {s['rel_err']:.3f} ({s['samples']})")
    return result


def release(model, device) -> None:
    """Frees a finished row's GPU memory. Dropping the name is not enough: the model's compiled
    sample_actions keeps it referenced, so its tensors are moved off the device instead."""
    import torch

    model.to("meta")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        logging.info(f"released a model: {torch.cuda.memory_allocated(device) / 2**30:.2f} GiB still allocated")


def main(args: Args) -> dict:
    import torch

    from openpi import transforms as _transforms
    from openpi.glow import module_inputs as _mi
    from openpi.training import config as _config
    from openpi.training import data_loader as _data_loader

    if not args.rows and args.grid_weights is None:
        raise ValueError("nothing to do: give --rows and/or --grid-weights")
    rows = [parse_row(r) for r in args.rows]
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    assets = _config.AssetsConfig(assets_dir=args.norm_stats_dir, asset_id=args.repo_id)

    def data_config(config_name):
        cfg = _config.get_config(config_name)
        dc = dataclasses.replace(cfg.data, assets=assets).create(cfg.assets_dirs, cfg.model)
        if dc.norm_stats is None:
            raise ValueError(f"no norm_stats under {args.norm_stats_dir}/{args.repo_id}")
        inputs = list(dc.model_transforms.inputs)
        at = [i for i, t in enumerate(inputs) if isinstance(t, _mi.GlowModuleInputs)]
        if len(at) != 1:
            raise ValueError(f"expected one GlowModuleInputs in {config_name}'s model transforms, got {len(at)}")
        # The probe builds its plans itself: no chain or counterfactual plan in the rendering.
        module = dataclasses.replace(inputs[at[0]], chain_p=0.0, chain_tasks=(), counterfactual_p=0.0)
        inputs[at[0]] = module
        return cfg, dc, module, inputs

    cfg, dc, module, inputs = data_config(args.config)
    pipeline = _transforms.compose(
        [
            *dc.repack_transforms.inputs,
            *dc.data_transforms.inputs,
            _transforms.Normalize(dc.norm_stats, use_quantiles=dc.use_quantile_norm),
            *inputs,
        ]
    )
    rng = np.random.default_rng(args.seed)
    exclude = _data_loader.resolve_train_episodes(args.exclude_episodes) if args.exclude_episodes else []
    heldout = heldout_episodes(module.pack, exclude, args.episodes_per_task, rng)
    plan = plan_frames(
        module.pack,
        heldout,
        start_samples=args.start_samples,
        mid_samples=args.mid_samples,
        start_window=args.start_window,
        mid_window=args.mid_window,
        batch_size=args.batch_size,
        rng=rng,
    )
    root = pathlib.Path(args.dataset_root)
    build = make_build_batch(pipeline, root, cfg.model.action_horizon, read_task_strings(root))
    # Started before any model loads, so the first batches wait for the GPU, not the other way round.
    store = BatchStore(plan, build, args.batch_size)
    logging.info(f"{len(plan)} samples from {sum(map(len, heldout.values()))} held-out episodes: {heldout}")

    result: dict = {"args": dataclasses.asdict(args), "heldout": heldout, "rows": {}}

    def write():
        if args.out:
            out = pathlib.Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=1), encoding="utf-8")

    def load(config_name, weights, init):
        ns = types.SimpleNamespace(
            weights_dir=weights, precision=args.precision, init=init, init_seed=args.seed, config=config_name
        )
        # Built on the device: on the host, creating a model peaks above 14 GiB of RSS against ~3 GiB
        # while the probe runs.
        with device:
            return _diag.load_model(_config.get_config(config_name), ns, device)

    for name, config_name, weights in rows:
        model = load(config_name, weights, False)
        _, _, row_module, _ = data_config(config_name)
        logging.info(f"{name}: {config_name} from {weights}, Spatial order {row_module.stage_order}")
        result["rows"][name] = run_row(name, model, row_module, store, args, device, grid=False)
        write()
        release(model, device)
        del model
    if args.grid_weights is not None:
        model = load(args.config, args.grid_weights, True)
        if module.stage_order != "sorted":
            raise ValueError(f"the grid reads sorted arrays; {args.config} has order {module.stage_order!r}")
        result["grid"] = run_row("grid", model, module, store, args, device, grid=True)
        write()
    if args.out:
        logging.info(f"wrote {args.out}")
    return result


if __name__ == "__main__":
    import tyro

    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
