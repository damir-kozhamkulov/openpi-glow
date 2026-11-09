"""Offline test of the progress monitor: can frozen camera tokens plus the plan's runtime keys tell
when each stage of a plan is done? No simulator and no action model; the result decides whether
the monitor can drive the stage gates.

1. Frames: every --stride-th frame of each multi-stage episode, every --single-stride-th of each
   one-stage episode, rendered by --config's data pipeline (GlowModuleInputs on the labels path, the
   episode's own plan, no chain or counterfactual plans). No normalization: only the images and
   the module arrays are read.
2. Camera tokens: the frozen VLM's image embedding (vision tower + projector, the tokens
   embed_prefix puts in the prefix), average-pooled over --pool x --pool patches, kept on the
   device. Only those weights and the token-embedding table are read from the checkpoint.
3. Labels, per stage key in execution order: done from the frame the next stage starts
   (stage_starts[j + 1]). The last stage's key is never reached inside a demo (demos end at
   success): not done, and its last --tail frames are left unlabeled.
4. Variants trained on the training split, same steps and seed:
     monitor         the key text in the queries; with probability --swap-p a multi-stage plan's
                     stages are permuted (labels follow their keys)
     monitor_noswap  the same with no permuted plans
     nokey           no key text: position and stage count only
5. Held-out metrics per family (libero10, goal, chain):
     own      AUROC of done over labeled (frame, stage) entries; false done in the first
              --start-window frames
     switch   per stage boundary, the latched switch (first sampled frame whose done probability is
              above --threshold on two consecutive samples) against the label's frame
     swapped  the same plans with their stages cyclically shifted: AUROC, and false done after the
              first boundary (where a model that only sees "something is finished" fires)
   plus a clock baseline (done once the task's median training start of the next stage has
   passed; position-based, so it ignores the keys too).

Held out: chain episodes whose initial state is in --heldout-inits (default the largest one in the
pack's provenance); every --holdout-every-th episode of each other task.

  python scripts/glow_progress_monitor.py --weights <dir with model.safetensors> \
      --dataset-root <lerobot repo dir> --plan-pack <pack.json> --out progress_monitor.json
"""

import dataclasses
import json
import logging
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import glow_flow_probe as _flow_probe  # noqa: E402
import glow_temporal_probe as _probe  # noqa: E402

GOAL_FIRST_EPISODE, GOAL_END_EPISODE = 379, 807
FAMILIES = ("libero10", "goal", "chain")
VARIANTS = {"monitor": (True, None), "monitor_noswap": (True, 0.0), "nokey": (False, 0.0)}
PASS_AUROC, PASS_SWITCH_FRAMES, PASS_START_FALSE = 0.98, 10.0, 0.02


@dataclasses.dataclass
class Args:
    # Directory holding model.safetensors: its vision tower, projector and token embedding are read.
    weights: str
    # LeRobot dataset root: the directory holding meta/ and data/.
    dataset_root: str
    # Plan pack with every sampled episode (and provenance["chains"] for chain episodes).
    plan_pack: str
    config: str = "pi05_libero_glow_v2t_fixed"
    # Model size override for the config ("dummy" for a dummy-size checkpoint).
    paligemma_variant: str | None = None
    action_expert_variant: str | None = None
    # Episodes to use, as OPENPI_TRAIN_EPISODES ("" = every pack episode the dataset has).
    episodes: str = ""
    # At most this many episodes per task (0 = all), the first ones by index.
    max_per_task: int = 0
    stride: int = 5
    single_stride: int = 10
    tail: int = 15
    pool: int = 2
    holdout_every: int = 5
    heldout_inits: tuple[int, ...] = ()
    # Monitor and training.
    width: int = 512
    depth: int = 2
    heads: int = 8
    steps: int = 6000
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup: int = 200
    swap_p: float = 0.3
    # Evaluation.
    threshold: float = 0.7
    start_window: int = 40
    render_batch: int = 32
    seed: int = 0
    precision: str = "float32"
    device: str = "cuda"
    out: str | None = None
    # Each variant's trained monitor goes to <save_dir>/<variant>/model.safetensors (parameter names
    # under MONITOR_PREFIX, as GlowModulePI0Pytorch.load_progress_monitor reads them) with monitor.json.
    save_dir: str | None = None


# ---------------------------------------------------------------------------------------------
# Episodes, frames, labels


def task_by_index(pack) -> dict:
    return {t.task_index: t for t in pack.tasks.values()}


def chain_inits(pack) -> dict[int, int]:
    """Chain episode -> its initial state, from the pack's provenance["chains"]["episodes"]
    (build_chain_dataset.py's records; empty without chains)."""
    records = pack.provenance.get("chains", {}).get("episodes", [])
    return {int(c["episode_index"]): int(c["init"]) for c in records}


def family(episode: int, chains: dict[int, int]) -> str:
    if episode in chains:
        return "chain"
    if episode >= GOAL_END_EPISODE:
        raise ValueError(f"episode {episode} is past LIBERO-Goal and not in the pack's provenance['chains']")
    return "libero10" if episode < GOAL_FIRST_EPISODE else "goal"


def select_episodes(pack, available: int, spec: list[int] | None, max_per_task: int) -> list[int]:
    """Pack episodes below `available` (and in `spec` if given), at most `max_per_task` per task."""
    wanted = set(spec) if spec is not None else None
    by_task: dict[int, list[int]] = {}
    for ep in sorted(pack.episodes):
        if ep < available and (wanted is None or ep in wanted):
            by_task.setdefault(pack.episodes[ep].task_index, []).append(ep)
    out = []
    for eps in by_task.values():
        out.extend(eps[:max_per_task] if max_per_task > 0 else eps)
    return sorted(out)


def split(pack, episodes, chains: dict[int, int], holdout_every: int, heldout_inits) -> dict[int, bool]:
    """Episode -> held out. Chains by initial state (default the largest present); every other task
    by its every `holdout_every`-th episode in index order."""
    inits = set(heldout_inits)
    present = {chains[e] for e in episodes if e in chains}
    if not inits and present:
        inits = {max(present)}
    out, rank = {}, {}
    for ep in episodes:
        if ep in chains:
            out[ep] = chains[ep] in inits
        else:
            task = pack.episodes[ep].task_index
            out[ep] = rank.get(task, 0) % holdout_every == 0
            rank[task] = rank.get(task, 0) + 1
    return out


def labels(episode_plan, frame: int, tail: int) -> tuple[np.ndarray, np.ndarray]:
    """(done, labeled) [K] in execution order for one frame: module_inputs.done_labels, the labels
    a monitor row trains on."""
    from openpi.glow import module_inputs as _mi

    return _mi.done_labels(episode_plan, frame, tail)


def frame_plan(pack, episodes, chains, heldout, stride: int, single_stride: int, tail: int) -> list[dict]:
    """Every sampled frame with its labels."""
    out = []
    for ep in episodes:
        plan = pack.episodes[ep]
        step = stride if len(plan.stage_starts) > 1 else single_stride
        for frame in range(0, plan.n_frames, step):
            done, labeled = labels(plan, frame, tail)
            out.append(
                {
                    "episode": ep,
                    "frame": frame,
                    "task_index": plan.task_index,
                    "family": family(ep, chains),
                    "heldout": heldout[ep],
                    "done": done,
                    "labeled": labeled,
                }
            )
    return out


def padded(x: np.ndarray, k_max: int, fill) -> np.ndarray:
    out = np.full((k_max, *x.shape[1:]), fill, dtype=x.dtype)
    out[: len(x)] = x
    return out


# ---------------------------------------------------------------------------------------------
# Permuted plans


def cyclic_perm(valid):
    """module_encoders.cyclic_stage_perm."""
    from openpi.glow import module_encoders as _enc

    return _enc.cyclic_stage_perm(valid)


def random_perm(valid, generator=None):
    """module_encoders.random_stage_perm."""
    from openpi.glow import module_encoders as _enc

    return _enc.random_stage_perm(valid, generator)


def permute(perm, *xs):
    """module_encoders.permute_stages."""
    from openpi.glow import module_encoders as _enc

    return _enc.permute_stages(perm, *xs)


# ---------------------------------------------------------------------------------------------
# Metrics


def auroc(scores: np.ndarray, targets: np.ndarray) -> float | None:
    """Area under the ROC curve (ties count half); None without both classes."""
    scores, targets = np.asarray(scores, dtype=np.float64), np.asarray(targets).astype(bool)
    pos, neg = int(targets.sum()), int((~targets).sum())
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[targets].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def latched_switch(frames, probs, threshold: float) -> int | None:
    """The first sampled frame whose probability and the previous sample's are above `threshold`."""
    for i in range(1, len(frames)):
        if probs[i] > threshold and probs[i - 1] > threshold:
            return int(frames[i])
    return None


def switch_stats(errors: list[int | None], tolerance: float) -> dict:
    hit = [e for e in errors if e is not None]
    return {
        "boundaries": len(errors),
        "missed": len(errors) - len(hit),
        "median_abs": float(np.median(np.abs(hit))) if hit else None,
        "mean": float(np.mean(hit)) if hit else None,
        "within": float(np.mean([abs(e) <= tolerance for e in hit])) if hit else None,
        "early": int(sum(e < -tolerance for e in hit)),
        "late": int(sum(e > tolerance for e in hit)),
    }


def evaluate(samples, idx, probs, pack, threshold: float, start_window: int, *, swapped=None) -> dict:
    """Metrics of done probabilities `probs` [N, K_MAX] for the samples `idx`; `swapped` = (probs,
    labels, labeled) of the cyclically shifted plans for the same samples."""
    out: dict = {}
    for fam in FAMILIES:
        rows = [i for i in idx if samples[i]["family"] == fam]
        if not rows:
            continue
        s, t, frames = [], [], []
        for n, i in enumerate(idx):
            if samples[i]["family"] != fam:
                continue
            k = len(samples[i]["done"])
            for j in range(k):
                if samples[i]["labeled"][j]:
                    s.append(probs[n, j])
                    t.append(samples[i]["done"][j])
                    frames.append(samples[i]["frame"])
        s, t, frames = np.array(s), np.array(t) > 0.5, np.array(frames)
        start = (frames < start_window) & ~t
        result = {
            "frames": len(rows),
            "entries": len(s),
            "positives": int(t.sum()),
            "auroc": auroc(s, t),
            "accuracy": float(np.mean((s > 0.5) == t)) if len(s) else None,
            "start_false_done": float(np.mean(s[start] > 0.5)) if start.any() else None,
        }
        result["switch"] = switch_stats(switch_errors(samples, idx, probs, pack, fam, threshold), PASS_SWITCH_FRAMES)
        result["per_task"] = per_task_auroc(samples, idx, probs, fam)
        if swapped is not None:
            result["swapped"] = swapped_stats(samples, idx, *swapped, pack, fam)
        out[fam] = result
    return out


def switch_errors(samples, idx, probs, pack, fam: str, threshold: float) -> list[int | None]:
    series: dict[int, list[tuple[int, int]]] = {}
    for n, i in enumerate(idx):
        if samples[i]["family"] == fam:
            series.setdefault(samples[i]["episode"], []).append((samples[i]["frame"], n))
    errors = []
    for ep, rows in sorted(series.items()):
        rows.sort()
        starts = pack.episodes[ep].stage_starts
        for j in range(len(starts) - 1):
            frames = [f for f, _ in rows]
            switch = latched_switch(frames, [probs[n, j] for _, n in rows], threshold)
            errors.append(None if switch is None else switch - starts[j + 1])
    return errors


def per_task_auroc(samples, idx, probs, fam: str) -> dict:
    by_task: dict[int, tuple[list, list]] = {}
    for n, i in enumerate(idx):
        if samples[i]["family"] != fam:
            continue
        s, t = by_task.setdefault(samples[i]["task_index"], ([], []))
        for j in range(len(samples[i]["done"])):
            if samples[i]["labeled"][j]:
                s.append(probs[n, j])
                t.append(samples[i]["done"][j] > 0.5)
    return {str(k): auroc(np.array(s), np.array(t)) for k, (s, t) in sorted(by_task.items())}


def swapped_stats(samples, idx, probs, done, labeled, pack, fam: str) -> dict:
    """Shifted plans: AUROC, and false done after the first boundary (multi-stage samples only)."""
    s, t, after = [], [], []
    for n, i in enumerate(idx):
        sample = samples[i]
        k = len(sample["done"])
        if sample["family"] != fam or k < 2:
            continue
        boundary = pack.episodes[sample["episode"]].stage_starts[1]
        for j in range(k):
            if labeled[n, j]:
                s.append(probs[n, j])
                t.append(done[n, j] > 0.5)
                after.append(sample["frame"] >= boundary)
    if not s:
        return {}
    s, t, after = np.array(s), np.array(t), np.array(after)
    neg_after = after & ~t
    return {
        "entries": len(s),
        "auroc": auroc(s, t),
        "false_done_after_boundary": float(np.mean(s[neg_after] > 0.5)) if neg_after.any() else None,
    }


def clock_scores(samples, train_idx, pack, k_max: int) -> np.ndarray:
    """[N, K_MAX] done scores of the clock baseline: frames past the task's median training start of
    stage j + 1 (the last stage never: -inf)."""
    starts: dict[tuple[int, int], list[int]] = {}
    for i in train_idx:
        s = pack.episodes[samples[i]["episode"]].stage_starts
        for j in range(len(s) - 1):
            starts.setdefault((samples[i]["task_index"], j), []).append(s[j + 1])
    medians = {key: float(np.median(v)) for key, v in starts.items()}
    out = np.full((len(samples), k_max), -np.inf)
    for i, sample in enumerate(samples):
        for j in range(len(sample["done"]) - 1):
            median = medians.get((sample["task_index"], j))
            if median is not None:
                out[i, j] = sample["frame"] - median
    return out


CLOCK_SCALE = 2.0


def clock_probs(scores: np.ndarray) -> np.ndarray:
    """Clock scores as probabilities, sigmoid(frames past the median / CLOCK_SCALE): monotone in the
    score (AUROC unchanged) and above --threshold 0.7 within ~2 frames of the median."""
    with np.errstate(over="ignore"):
        return (1.0 / (1.0 + np.exp(-scores / CLOCK_SCALE))).astype(np.float32)


def passes(result: dict) -> dict:
    """T1's bars from the design, for the monitor against the baselines, per family with positives."""
    out = {}
    variants = result["variants"]
    for fam, own in variants.get("monitor", {}).items():
        if not own.get("positives"):
            continue
        bars = {
            "auroc": own["auroc"] is not None and own["auroc"] >= PASS_AUROC,
            "switch_median": own["switch"]["median_abs"] is not None
            and own["switch"]["median_abs"] <= PASS_SWITCH_FRAMES,
            "start_false_done": own["start_false_done"] is not None and own["start_false_done"] <= PASS_START_FALSE,
        }
        swap = own.get("swapped", {})
        for base in ("nokey", "clock"):
            other = (variants.get(base) or result.get("baselines", {}).get(base) or {}).get(fam, {})
            other_swap = other.get("swapped", {})
            if swap.get("auroc") is not None and other_swap.get("auroc") is not None:
                bars[f"swapped_beats_{base}"] = swap["auroc"] > other_swap["auroc"]
        out[fam] = bars
    return out


# ---------------------------------------------------------------------------------------------
# Model pieces


def build_pipeline(args):
    """(train config, rendering pipeline, GlowModuleInputs) with the episode's own plans only."""
    from openpi import transforms as _transforms
    from openpi.glow import module_inputs as _mi
    from openpi.training import config as _config

    cfg = _config.get_config(args.config)
    model_cfg = cfg.model
    replace = {k: v for k, v in (("paligemma_variant", args.paligemma_variant),
                                 ("action_expert_variant", args.action_expert_variant)) if v}
    if replace:
        model_cfg = dataclasses.replace(model_cfg, **replace)
    # No norm stats are read: the pipeline below does not normalize.
    assets = _config.AssetsConfig(assets_dir="no-norm-stats", asset_id=None)
    data = dataclasses.replace(cfg.data, assets=assets, plan_pack_path=args.plan_pack)
    dc = data.create(cfg.assets_dirs, model_cfg)
    inputs = list(dc.model_transforms.inputs)
    at = [i for i, t in enumerate(inputs) if isinstance(t, _mi.GlowModuleInputs)]
    if len(at) != 1:
        raise ValueError(f"expected one GlowModuleInputs in {args.config}'s model transforms, got {len(at)}")
    module = dataclasses.replace(inputs[at[0]], chain_p=0.0, chain_tasks=(), counterfactual_p=0.0)
    inputs[at[0]] = module
    pipeline = _transforms.compose([*dc.repack_transforms.inputs, *dc.data_transforms.inputs, *inputs])
    return dataclasses.replace(cfg, model=model_cfg), pipeline, module


def load_vision(model_cfg, weights: str, precision: str, device):
    """The model built on the meta device with only the vision tower, the projector and the token
    embedding materialized from `weights` (a tied embedding is stored as the VLM's lm_head)."""
    from safetensors import safe_open
    import torch

    model_cfg = dataclasses.replace(model_cfg, dtype=precision)
    with torch.device("meta"):
        model = model_cfg.create_pytorch()
    vlm = model.paligemma_with_expert.paligemma
    parts = {
        "vision": vlm.model.vision_tower,
        "projector": vlm.model.multi_modal_projector,
        "embed": vlm.model.language_model.embed_tokens,
    }
    names = {id(m): n for n, m in model.named_modules()}
    tied = names[id(vlm)] + ".lm_head.weight"
    path = str(pathlib.Path(weights) / "model.safetensors")
    with safe_open(path, framework="pt", device=str(device)) as f:
        keys = set(f.keys())
        for part, module in parts.items():
            prefix = names[id(module)] + "."
            state = {k[len(prefix) :]: f.get_tensor(k) for k in keys if k.startswith(prefix)}
            if part == "embed" and not state and tied in keys:
                state = {"weight": f.get_tensor(tied)}
            if not state:
                raise KeyError(f"{path} has no tensors for the {part} ({prefix}*)")
            module.to_empty(device=device)
            module.load_state_dict(state, strict=True)
            persistent = set(module.state_dict())
            for name, buf in list(module.named_buffers()):
                if name in persistent:
                    continue
                owner_name, _, attr = name.rpartition(".")
                owner = module.get_submodule(owner_name) if owner_name else module
                if attr != "position_ids":
                    raise RuntimeError(f"{part}: non-persistent buffer {name} has no known initial value")
                owner.register_buffer(
                    attr, torch.arange(buf.shape[-1], device=device).expand(1, -1), persistent=False
                )
    model.eval()
    return model


def pooled_tokens(tokens, pool: int):
    """module_encoders.pool_image_tokens."""
    from openpi.glow import module_encoders as _enc

    return _enc.pool_image_tokens(tokens, pool)


class Store:
    """Camera tokens [N, C, T, D] (bfloat16, on the device) and per-task plan inputs."""

    def __init__(self):
        self.images = None
        self.image_mask = None
        self.task_rows: dict[int, int] = {}
        self.keys, self.key_heads, self.valid = [], [], []

    def add_tasks(self, task_indices, keys, key_heads, valid) -> list[int]:
        import torch

        rows = []
        for n, task in enumerate(task_indices):
            if task not in self.task_rows:
                self.task_rows[task] = len(self.keys)
                self.keys.append(keys[n])
                self.key_heads.append(key_heads[n])
                self.valid.append(valid[n])
            else:
                r = self.task_rows[task]
                if not (torch.equal(self.valid[r], valid[n]) and torch.allclose(self.keys[r], keys[n])):
                    raise ValueError(f"task {task}: plan inputs differ between its samples")
            rows.append(self.task_rows[task])
        return rows

    def stack(self):
        import torch

        self.keys = torch.stack(self.keys)
        self.key_heads = torch.stack(self.key_heads)
        self.valid = torch.stack(self.valid)


def make_build(pipeline, root: pathlib.Path, action_horizon: int, task_strings, cache: int = 4):
    """build(samples) -> {"samples", "batch"}: the frames as the training loader hands them over
    (as glow_temporal_probe.read_frames), rendered and collated. The last `cache` episodes' tables
    stay in memory, so batches that share an episode read its file once."""
    import collections
    import io

    from PIL import Image
    import pyarrow.parquet as pq

    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    tables: collections.OrderedDict = collections.OrderedDict()

    def table(ep: int):
        if ep in tables:
            tables.move_to_end(ep)
            return tables[ep]
        path = root / info["data_path"].format(episode_chunk=ep // info["chunks_size"], episode_index=ep)
        t = pq.read_table(path, columns=["image", "wrist_image", "state", "actions", "task_index"])
        actions = np.asarray(t["actions"].to_pylist(), dtype=np.float32)
        tables[ep] = (t, actions)
        if len(tables) > cache:
            tables.popitem(last=False)
        return tables[ep]

    def frame(ep: int, f: int) -> dict:
        t, actions = table(ep)

        def decode(column: str) -> np.ndarray:
            return np.asarray(Image.open(io.BytesIO(t[column][f].as_py()["bytes"])).convert("RGB"), dtype=np.uint8)

        task_index = t["task_index"][f].as_py()
        return {
            "image": decode("image"),
            "wrist_image": decode("wrist_image"),
            "state": np.asarray(t["state"][f].as_py(), dtype=np.float32),
            "actions": actions[np.minimum(np.arange(f, f + action_horizon), len(actions) - 1)],
            "episode_index": ep,
            "frame_index": f,
            "task_index": task_index,
            "prompt": task_strings[task_index],
        }

    def build(samples: list[dict]) -> dict:
        rendered = [pipeline(frame(s["episode"], s["frame"])) for s in samples]
        return {"samples": samples, "batch": _flow_probe._collate(rendered)}  # noqa: SLF001

    return build


class Prefetch:
    """Iterates build(chunk) over `chunks` in order, built on a background thread at most `depth`
    ahead; a consumed batch is dropped, so host memory holds depth + 1 batches whatever the run's
    size. `waited` counts the seconds the consumer spent waiting for data."""

    def __init__(self, chunks, build, depth: int = 4):
        import queue
        import threading

        self._queue: queue.Queue = queue.Queue(maxsize=depth)
        self.waited = 0.0
        threading.Thread(target=self._run, args=(list(chunks), build), daemon=True).start()

    def _run(self, chunks, build) -> None:
        try:
            for chunk in chunks:
                self._queue.put(("batch", build(chunk)))
            self._queue.put(("end", None))
        except BaseException as e:  # noqa: BLE001 - re-raised on the consumer's thread
            self._queue.put(("error", e))

    def __iter__(self):
        while True:
            t0 = time.monotonic()
            kind, item = self._queue.get()
            self.waited += time.monotonic() - t0
            if kind == "end":
                return
            if kind == "error":
                raise item
            yield item


def rss_gib() -> float | None:
    """This process's resident memory in GiB (None without psutil)."""
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process().memory_info().rss / 2**30


def to_torch(batch, device):
    import jax
    import torch

    def one(x):
        t = torch.as_tensor(np.asarray(x))
        return t.to(device, torch.float32) if t.dtype == torch.float64 else t.to(device)

    return jax.tree.map(one, batch)


def dump(samples, pipeline, model, args, device, root, action_horizon) -> tuple[Store, np.ndarray]:
    """Renders every sample, embeds its cameras and its plan; returns the store and each sample's
    task row."""
    import torch

    from openpi.glow import module_encoders as _enc
    from openpi.models import model as _model

    embed = model.paligemma_with_expert.embed_language_tokens
    build = make_build(pipeline, root, action_horizon, _probe.read_task_strings(root))
    plan = [{"episode": s["episode"], "frame": s["frame"]} for s in samples]
    batches = Prefetch((plan[i : i + args.render_batch] for i in range(0, len(plan), args.render_batch)), build)
    store = Store()
    task_row = np.zeros(len(samples), dtype=np.int64)
    cameras = None
    t0 = time.monotonic()

    def run(batch, rows):
        nonlocal cameras
        obs = _model.Observation.from_dict(to_torch(batch, device))
        with torch.no_grad():
            images, masks, *_ = model._preprocess_observation(obs, train=False)  # noqa: SLF001
            shown = [i for i, m in enumerate(masks) if bool(m.all())]
            if cameras is None:
                cameras = shown
                if any(bool(m.any()) for i, m in enumerate(masks) if i not in shown):
                    raise ValueError("a camera is present in some samples only")
                # The monitor's camera embedding is by position: the cameras kept must be the
                # prefix's first image slots, as the model passes them.
                if cameras != list(range(len(cameras))):
                    raise ValueError(f"cameras {cameras} are not the prefix's first image slots")
            elif shown != cameras:
                raise ValueError(f"cameras changed between batches: {cameras} vs {shown}")
            feats = torch.stack(
                [pooled_tokens(model.paligemma_with_expert.embed_image(images[i]), args.pool) for i in cameras], 1
            ).to(torch.bfloat16)
            if store.images is None:
                store.images = torch.empty((len(samples), *feats.shape[1:]), dtype=torch.bfloat16, device=device)
                logging.info(
                    f"camera tokens: {len(samples)} x {tuple(feats.shape[1:])} = "
                    f"{store.images.numel() * 2 / 2**30:.2f} GiB on {device}"
                )
            store.images[rows] = feats
            keys, key_heads, valid = _enc.monitor_key_inputs(
                embed, obs.glow_text, obs.glow_text_mask, obs.glow_ints
            )
        tasks = [samples[i]["task_index"] for i in rows]
        task_row[rows] = store.add_tasks(tasks, keys, key_heads, valid)

    for b, item in enumerate(batches):
        start = b * args.render_batch
        rows = list(range(start, start + len(item["samples"])))
        run(item["batch"], rows)
        if b % 50 == 0:
            rss = rss_gib()
            logging.info(
                f"rendered {rows[-1] + 1}/{len(samples)} frames, {time.monotonic() - t0:.0f} s"
                + (f", host RSS {rss:.2f} GiB" if rss is not None else "")
            )
    store.image_mask = torch.ones(len(cameras), dtype=torch.bool, device=device)
    store.stack()
    logging.info(
        f"dump done: {len(samples)} frames in {time.monotonic() - t0:.0f} s, data wait {batches.waited:.0f} s, "
        f"{len(store.task_rows)} tasks, cameras {cameras}"
    )
    return store, task_row


def batch_tensors(store, task_row, done, labeled, idx, device):
    import torch

    rows = torch.as_tensor(task_row[idx], device=device)
    images = store.images[torch.as_tensor(idx, device=device)].float()
    mask = store.image_mask[None].expand(len(idx), -1)
    return (
        store.keys[rows],
        store.key_heads[rows],
        store.valid[rows],
        images,
        mask,
        done[torch.as_tensor(idx, device=device)],
        labeled[torch.as_tensor(idx, device=device)],
    )


def train_variant(name, keys: bool, swap_p: float, store, task_row, done, labeled, train_idx, args, device):
    import torch

    from openpi.glow import module_encoders as _enc

    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed + 1)
    rng = np.random.default_rng(args.seed)
    monitor = _enc.ProgressMonitor(
        store.keys.shape[-1],
        store.images.shape[-1],
        args.width,
        args.depth,
        keys=keys,
        heads=args.heads,
    ).to(device)
    train_labeled = labeled[torch.as_tensor(train_idx, device=device)]
    train_done = done[torch.as_tensor(train_idx, device=device)]
    pos = float((train_done * train_labeled).sum())
    neg = float(((1 - train_done) * train_labeled).sum())
    pos_weight = torch.tensor(neg / max(pos, 1.0), device=device)
    opt = torch.optim.AdamW(monitor.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_at(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    t0, running = time.monotonic(), []
    monitor.train()
    for step in range(args.steps):
        idx = rng.choice(train_idx, size=min(args.batch_size, len(train_idx)), replace=False)
        k, h, v, im, m, d, lab = batch_tensors(store, task_row, done, labeled, idx, device)
        if swap_p > 0:
            perm = swap_perm(v, swap_p, gen)
            k, h, d, lab = permute(perm, k, h, d, lab)
        logits = monitor(k, h, v, im, m)
        logits = torch.where(lab, logits, torch.zeros_like(logits))
        per = torch.nn.functional.binary_cross_entropy_with_logits(logits, d, pos_weight=pos_weight, reduction="none")
        loss = (per * lab).sum() / lab.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(monitor.parameters(), 1.0)
        opt.step()
        sched.step()
        running.append(float(loss))
        if (step + 1) % 500 == 0 or step == 0:
            logging.info(
                f"{name} step {step + 1}/{args.steps} loss {np.mean(running):.4f} "
                f"lr {sched.get_last_lr()[0]:.2e} {time.monotonic() - t0:.0f} s"
            )
            running = []
    monitor.eval()
    return monitor


MONITOR_PREFIX = "glow_progress_monitor."


def save_monitor(monitor, out_dir: pathlib.Path, args, *, keys: bool) -> pathlib.Path:
    """The monitor's parameters under MONITOR_PREFIX, and its shape and input settings (monitor.json)."""
    import safetensors.torch

    out_dir.mkdir(parents=True, exist_ok=True)
    state = {MONITOR_PREFIX + k: v.detach().contiguous().cpu() for k, v in monitor.state_dict().items()}
    path = out_dir / "model.safetensors"
    safetensors.torch.save_file(state, str(path))
    meta = {
        "width": args.width,
        "depth": args.depth,
        "heads": args.heads,
        "pool": args.pool,
        "keys": keys,
        "config": args.config,
        "weights": args.weights,
        "plan_pack": args.plan_pack,
    }
    (out_dir / "monitor.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return path


def swap_perm(valid, swap_p: float, gen=None):
    """Per sample: a random other order of its stages with probability swap_p, else its own."""
    import torch

    j = torch.arange(valid.shape[-1], device=valid.device).expand_as(valid)
    swap = torch.rand(valid.shape[0], generator=gen, device=valid.device) < swap_p
    return torch.where(swap[:, None], random_perm(valid, gen), j)


def predict(monitor, store, task_row, done, labeled, idx, device, *, shift: bool, batch: int = 512):
    """Done probabilities [n, K_MAX] (and with `shift` the shifted labels) for samples `idx`."""
    import torch

    probs, dones, labs = [], [], []
    with torch.no_grad():
        for start in range(0, len(idx), batch):
            part = idx[start : start + batch]
            k, h, v, im, m, d, lab = batch_tensors(store, task_row, done, labeled, part, device)
            if shift:
                k, h, d, lab = permute(cyclic_perm(v), k, h, d, lab)
            probs.append(torch.sigmoid(monitor(k, h, v, im, m)).cpu().numpy())
            dones.append(d.cpu().numpy())
            labs.append(lab.cpu().numpy())
    return np.concatenate(probs), np.concatenate(dones), np.concatenate(labs)


def shifted_labels(done: np.ndarray, labeled: np.ndarray, valid: np.ndarray):
    """numpy twin of permute(cyclic_perm(valid), done, labeled), for the clock baseline."""
    import torch

    v = torch.as_tensor(valid)
    d, lab = permute(cyclic_perm(v), torch.as_tensor(done), torch.as_tensor(labeled))
    return d.numpy(), lab.numpy()


# ---------------------------------------------------------------------------------------------


def main(args: Args) -> dict:
    import torch

    from openpi.glow import module_inputs as _mi
    from openpi.training import data_loader as _data_loader

    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cfg, pipeline, module = build_pipeline(args)
    pack = module.pack
    root = pathlib.Path(args.dataset_root)
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    spec = _data_loader.resolve_train_episodes(args.episodes) if args.episodes else None
    episodes = select_episodes(pack, int(info["total_episodes"]), spec, args.max_per_task)
    chains = chain_inits(pack)
    heldout = split(pack, episodes, chains, args.holdout_every, args.heldout_inits)
    samples = frame_plan(pack, episodes, chains, heldout, args.stride, args.single_stride, args.tail)
    k_max = _mi.K_MAX
    done = np.stack([padded(s["done"], k_max, 0.0) for s in samples])
    labeled = np.stack([padded(s["labeled"], k_max, False) for s in samples])
    train_idx = np.array([i for i, s in enumerate(samples) if not s["heldout"]])
    test_idx = np.array([i for i, s in enumerate(samples) if s["heldout"]])
    counts = {
        fam: {
            "episodes": len({s["episode"] for s in samples if s["family"] == fam}),
            "train_frames": int(sum(1 for s in samples if s["family"] == fam and not s["heldout"])),
            "heldout_frames": int(sum(1 for s in samples if s["family"] == fam and s["heldout"])),
        }
        for fam in FAMILIES
    }
    logging.info(f"{len(episodes)} episodes, {len(samples)} frames: {counts}")
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("the split leaves no training or no held-out frames")

    result: dict = {"args": dataclasses.asdict(args), "counts": counts, "variants": {}, "baselines": {}}

    def write():
        if args.out:
            out = pathlib.Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=1), encoding="utf-8")

    model = load_vision(cfg.model, args.weights, args.precision, device)
    store, task_row = dump(samples, pipeline, model, args, device, root, cfg.model.action_horizon)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    done_t = torch.as_tensor(done, device=device)
    labeled_t = torch.as_tensor(labeled, device=device)

    for name, (keys, swap) in VARIANTS.items():
        swap_p = args.swap_p if swap is None else swap
        monitor = train_variant(name, keys, swap_p, store, task_row, done_t, labeled_t, train_idx, args, device)
        own, _, _ = predict(monitor, store, task_row, done_t, labeled_t, test_idx, device, shift=False)
        shifted = predict(monitor, store, task_row, done_t, labeled_t, test_idx, device, shift=True)
        result["variants"][name] = evaluate(
            samples, test_idx, own, pack, args.threshold, args.start_window, swapped=shifted
        )
        result.setdefault("variant_swap_p", {})[name] = swap_p
        if args.save_dir:
            path = save_monitor(monitor, pathlib.Path(args.save_dir) / name, args, keys=keys)
            logging.info(f"saved {name} to {path}")
        write()
        del monitor

    scores = clock_scores(samples, train_idx, pack, k_max)
    valid = np.stack([padded(np.ones(len(s["done"]), dtype=bool), k_max, False) for s in samples])
    clock_own = clock_probs(scores[test_idx])
    d_shift, lab_shift = shifted_labels(done[test_idx], labeled[test_idx], valid[test_idx])
    result["baselines"]["clock"] = evaluate(
        samples, test_idx, clock_own, pack, args.threshold, args.start_window,
        swapped=(clock_own, d_shift, lab_shift),
    )
    result["pass"] = passes(result)
    write()
    for name, fams in {**result["variants"], **result["baselines"]}.items():
        for fam, r in fams.items():
            sw, swp = r["switch"], r.get("swapped", {})
            logging.info(
                f"RESULT {name:15s} {fam:9s} auroc {r['auroc']} start_false {r['start_false_done']} "
                f"switch median {sw['median_abs']} within {sw['within']} missed {sw['missed']}/{sw['boundaries']} "
                f"swapped auroc {swp.get('auroc')} false_after {swp.get('false_done_after_boundary')}"
            )
    logging.info(f"PASS {result['pass']}")
    if args.out:
        logging.info(f"wrote {args.out}")
    return result


if __name__ == "__main__":
    import tyro

    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
