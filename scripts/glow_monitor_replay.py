"""Replays a monitor row's simulator queries offline: why does the progress monitor call a stage done
before it is? Reads the episodes a LIBERO client wrote with --args.dump-dir (examples/libero/
glow_query_dump.py) and scores every query again, four ways:

  B  served   the served policy's own input transforms, prefix and monitor
              (GlowModulePI0Pytorch.observation_prefix), done before and after the client's latch
              floor (glow_prev_done); the floored value must reproduce the recorded one
  C  fp32     the served camera tokens and plan inputs through a float32 copy of the monitor
  A  M1 path  the camera tokens as scripts/glow_progress_monitor.py made them for training (the
              vision tower of --weights in float32, pooled, rounded to bfloat16) through the float32
              monitor, with the plan inputs from --weights' token embedding
  D  demo     the served path with each early query's (frame < --start-window) camera images swapped
              for the same frame of a training demo of the same task (--dataset-root)

So B vs C is the monitor's precision, C vs A the camera tokens (vision weights and precision), and
B vs D the frames themselves. Also checked: the vision tower, projector and token embedding of
--policy-dir against --weights, tensor by tensor; and each chain stage's plan inputs against that
task's own one-stage plan.

Per run directory under --dump-dir: early false done (episodes whose first stage's done passes
--threshold before --start-window frames) for B, A and D; premature latches (the latch of the first
stage, --threshold on --consecutive queries, before its goal atom first held, or with the atom never
holding) for B and A; and the B / C / A gaps. Each premature episode's latch frame goes to
--png-dir (sim agent view | sim wrist | demo agent view | demo wrist).

  python scripts/glow_monitor_replay.py --dump-dir <dir> --policy-dir <ckpt> --monitor-from <dir> \
      --weights <pi05_libero dir> --dataset-root <lerobot root> --out replay.json
"""

import ast
import collections
import copy
import dataclasses
import json
import logging
import pathlib
import re
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "examples" / "libero"))
import glow_progress_monitor as _m1  # noqa: E402
import glow_query_dump as _dump  # noqa: E402

GOAL_END_EPISODE = _m1.GOAL_END_EPISODE
PARITY_PARTS = ("vision_tower.", "multi_modal_projector.", "embed_tokens.", "lm_head.")


@dataclasses.dataclass
class Args:
    # Directory holding the client's task*_ep*.npz files, one subdirectory per run (any depth).
    dump_dir: str
    # The served checkpoint and monitor (serve_policy --policy.dir / --policy.monitor-from).
    policy_dir: str
    monitor_from: str
    # The checkpoint scripts/glow_progress_monitor.py read the camera tokens from (its --weights).
    weights: str
    # LeRobot dataset root with the training demos (meta/ and data/).
    dataset_root: str
    config: str = "pi05_libero_glow_monitor"
    # Model size override for the config ("dummy" for a dummy-size checkpoint).
    paligemma_variant: str | None = None
    action_expert_variant: str | None = None
    threshold: float = 0.7
    consecutive: int = 2
    start_window: int = 40
    # Demos per task the demo control draws from (episode n uses demo n mod this).
    demo_episodes: int = 5
    # Local directory the demo files are copied to first (default: read in place).
    demo_cache: str | None = None
    batch_size: int = 32
    png_dir: str | None = None
    png_max: int = 60
    out: str | None = None
    device: str = "cuda"


# ---------------------------------------------------------------------------------------------
# Traces, latches and episode classes


def parse_trace(line: str) -> dict:
    """A client "plan trace:" line as a dict; list fields ([...]) parsed, the rest strings."""
    if "plan trace:" not in line:
        return {}
    out = {}
    for key, value in re.findall(r"(\w+)=(\[[^\]]*\]|\S+)", line.split("plan trace:", 1)[1]):
        out[key] = ast.literal_eval(value) if value.startswith("[") else value
    return out


def latched_frame(frames, probs, threshold: float, consecutive: int) -> int | None:
    """The frame at which a MonitorLatch(threshold, consecutive) latches `probs` (one stage's done
    per query): the query completing the first run of `consecutive` above the threshold."""
    run = 0
    for frame, p in zip(frames, probs, strict=True):
        run = run + 1 if p > threshold else 0
        if run >= consecutive:
            return int(frame)
    return None


def first_met(trace: dict) -> int | None:
    """The frame the plan's first stage's goal atom first held (None if it never did)."""
    met = trace.get("atoms_first_met") or [None]
    frames = int(trace.get("frames", 0))
    return None if met[0] is None or met[0] >= frames else int(met[0])


def premature(latch: int | None, met: int | None) -> bool:
    """A latch of the first stage before its goal atom held (or with the atom never holding)."""
    return latch is not None and (met is None or latch < met)


# ---------------------------------------------------------------------------------------------
# Requests


def element_from(ep: dict, i: int, image=None, wrist=None) -> dict:
    """Query `i` of a dumped episode as the client sent it (other camera images if given)."""
    out = {}
    for key, value in ep["request"][i].items():
        numeric = isinstance(value, list) and value and all(isinstance(v, (int, float)) for v in value)
        out[key] = np.asarray(value, dtype=np.float32) if numeric else value
    out[_dump.IMAGE_KEY] = ep["image"][i] if image is None else image
    out[_dump.WRIST_KEY] = ep["wrist_image"][i] if wrist is None else wrist
    out[_dump.STATE_KEY] = ep["state"][i]
    return out


def run_episodes(dump_dir: pathlib.Path) -> dict[str, list[pathlib.Path]]:
    """Run directory (relative to dump_dir) -> its episode files, in (task, episode) order."""
    runs = collections.defaultdict(list)
    for path in dump_dir.rglob("task*_ep*.npz"):
        runs[str(path.parent.relative_to(dump_dir)) or "."].append(path)

    def order(p):
        m = re.match(r"task(\d+)_ep(\d+)\.npz", p.name)
        return (int(m.group(1)), int(m.group(2))) if m else (1 << 30, 0)

    return {run: sorted(paths, key=order) for run, paths in sorted(runs.items())}


# ---------------------------------------------------------------------------------------------
# Weights


def parity(policy_file: str, weights_file: str, parts=PARITY_PARTS) -> dict:
    """Every tensor whose name contains one of `parts` in either file: present in both, same shape,
    and the largest absolute difference (in float32)."""
    from safetensors import safe_open

    with safe_open(policy_file, framework="pt") as a, safe_open(weights_file, framework="pt") as b:
        names_a = {k for k in a.keys() if any(p in k for p in parts)}  # noqa: SIM118
        names_b = {k for k in b.keys() if any(p in k for p in parts)}  # noqa: SIM118
        out = {
            "compared": 0,
            "identical": 0,
            "max_abs_diff": 0.0,
            "only_policy": sorted(names_a - names_b)[:20],
            "only_weights": sorted(names_b - names_a)[:20],
            "shape_mismatch": [],
            "differ": [],
        }
        for name in sorted(names_a & names_b):
            ta, tb = a.get_tensor(name).float(), b.get_tensor(name).float()
            if ta.shape != tb.shape:
                out["shape_mismatch"].append(name)
                continue
            diff = float((ta - tb).abs().max()) if ta.numel() else 0.0
            out["compared"] += 1
            out["identical"] += diff == 0.0
            out["max_abs_diff"] = max(out["max_abs_diff"], diff)
            if diff > 0 and len(out["differ"]) < 20:
                out["differ"].append([name, diff])
    return out


# ---------------------------------------------------------------------------------------------
# Demo frames


class Demos:
    """Training-demo camera frames by task string, as a client would send them (224 x 224 uint8):
    the one-task episodes below GOAL_END_EPISODE (LIBERO-10 and LIBERO-Goal), the first `per_task`
    of each task. With `cache`, the indexed files are copied there first, `parallel` at a time (a
    slow shared volume is then read once per file, concurrently)."""

    def __init__(
        self, root: pathlib.Path, per_task: int, size: int = 224, cache: pathlib.Path | None = None, parallel: int = 16
    ):
        self.root = root
        self.size = size
        self.info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
        self.by_task: dict[str, list[int]] = collections.defaultdict(list)
        for line in (root / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            ep, tasks = int(rec["episode_index"]), rec.get("tasks", [])
            if (
                ep < GOAL_END_EPISODE
                and len(tasks) == 1
                and len(self.by_task[tasks[0]]) < per_task
                and self.path(ep).exists()
            ):
                self.by_task[tasks[0]].append(ep)
        self._cache: collections.OrderedDict = collections.OrderedDict()
        if cache is not None:
            import concurrent.futures
            import shutil

            files = [self.path(ep) for eps in self.by_task.values() for ep in eps]
            targets = [cache / f.relative_to(root) for f in files]
            for t in targets:
                t.parent.mkdir(parents=True, exist_ok=True)
            with concurrent.futures.ThreadPoolExecutor(parallel) as pool:
                list(pool.map(shutil.copyfile, files, targets))
            self.root = cache

    def path(self, ep: int) -> pathlib.Path:
        rel = self.info["data_path"].format(episode_chunk=ep // self.info["chunks_size"], episode_index=ep)
        return self.root / rel

    def episode(self, task: str, n: int) -> int | None:
        eps = self.by_task.get(task)
        return eps[n % len(eps)] if eps else None

    def frames(self, ep: int) -> tuple[np.ndarray, np.ndarray]:
        """(image, wrist_image) [T, size, size, 3] of demo `ep`."""
        if ep in self._cache:
            self._cache.move_to_end(ep)
            return self._cache[ep]
        import io

        from openpi_client import image_tools
        from PIL import Image
        import pyarrow.parquet as pq

        t = pq.read_table(self.path(ep), columns=["image", "wrist_image"])

        def column(name):
            imgs = [np.asarray(Image.open(io.BytesIO(x["bytes"])).convert("RGB")) for x in t[name].to_pylist()]
            return image_tools.convert_to_uint8(image_tools.resize_with_pad(np.stack(imgs), self.size, self.size))

        self._cache[ep] = (column("image"), column("wrist_image"))
        if len(self._cache) > 8:
            self._cache.popitem(last=False)
        return self._cache[ep]


# ---------------------------------------------------------------------------------------------
# Scoring


def masked_probs(logits, valid):
    import torch

    return torch.sigmoid(logits.float()).masked_fill(~valid, 0.0)


class Scorer:
    """The four scorings of a batch of client elements (see the module docstring)."""

    def __init__(self, policy, m1_model, device):
        import torch

        self.policy = policy
        self.model = policy._model  # noqa: SLF001
        self.transform = policy._input_transform  # noqa: SLF001
        self.model.eval()
        self.m1 = m1_model
        self.device = device
        self.pool = self.model.config.module_monitor_pool
        self.monitor32 = copy.deepcopy(self.model.glow_progress_monitor).float().eval()
        self.served_dtype = next(self.model.glow_progress_monitor.parameters()).dtype
        self.torch = torch

    def prepare(self, elements) -> dict:
        """The served input transforms of `elements`, collated (host arrays; any thread)."""
        return _m1.collate([self.transform(e) for e in elements])

    def observation(self, batch):
        from openpi.models import model as _model

        return _model.Observation.from_dict(_m1.to_torch(batch, self.device))

    def score(self, batch, *, m1: bool = True) -> dict:
        """[B, K_MAX] numpy arrays for a prepared batch: served (floored), served_raw, fp32 and m1;
        plus the per-sample relative token gap (served vs M1 path) and the plan-input gap."""
        torch = self.torch
        from openpi.glow import module_encoders as _enc

        obs = self.observation(batch)
        with torch.no_grad():
            images, img_masks, lang_tokens, lang_masks, _ = self.model._preprocess_observation(obs, train=False)  # noqa: SLF001
            _, _, _, extras = self.model.observation_prefix(obs, images, img_masks, lang_tokens, lang_masks)
            keys, heads, valid, tokens, mask = extras["monitor_inputs"]
            out = {
                "served": extras["glow_done"].float(),
                "served_raw": masked_probs(extras["done_logits"], valid),
                "fp32": masked_probs(self.monitor32(keys.float(), heads.float(), valid, tokens.float(), mask), valid),
            }
            if m1 and self.m1 is not None:
                cams = [i for i, m in enumerate(img_masks) if bool(m.all())]
                vlm = self.m1.paligemma_with_expert
                tok = (
                    torch.stack([_m1.pooled_tokens(vlm.embed_image(images[i].float()), self.pool) for i in cams], 1)
                    .to(torch.bfloat16)
                    .float()
                )
                keys1, heads1, valid1 = _enc.monitor_key_inputs(
                    vlm.embed_language_tokens, obs.glow_text, obs.glow_text_mask, obs.glow_ints
                )
                mask1 = torch.ones(tok.shape[:2], dtype=torch.bool, device=tok.device)
                out["m1"] = masked_probs(self.monitor32(keys1, heads1, valid1, tok, mask1), valid1)
                served_tok = tokens[:, cams].float()
                out["token_gap"] = (served_tok - tok).norm(dim=-1).mean((1, 2)) / tok.norm(dim=-1).mean((1, 2))
                out["key_gap"] = (keys.float() - keys1).abs().amax((1, 2))
            out["valid"] = valid
        return {k: v.detach().cpu().numpy() for k, v in out.items()}

    def keys(self, batch):
        """A prepared batch's plan inputs as the monitor reads them: keys [B, K, D], valid [B, K]."""
        torch = self.torch
        from openpi.glow import module_encoders as _enc

        obs = self.observation(batch)
        with torch.no_grad():
            keys, _, valid = _enc.monitor_key_inputs(
                self.model.paligemma_with_expert.embed_language_tokens, obs.glow_text, obs.glow_text_mask, obs.glow_ints
            )
        return keys.float().cpu().numpy(), valid.cpu().numpy()


def chunks(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def scored(score, batches, **kw) -> dict:
    """score() over prepared batches, concatenated."""
    parts = [score(b, **kw) for b in batches]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


@dataclasses.dataclass
class Prepared:
    """One dumped episode with its queries' served inputs, ready for the GPU."""

    path: pathlib.Path
    ep: dict
    sim: list  # prepared batches of every query
    early: np.ndarray  # indices of the queries before the start window
    demo_episode: int | None
    demo: list  # prepared batches of the early queries with demo frames
    demo_frames: tuple | None  # the demo's (image, wrist_image), for the PNGs
    keys: dict | None = None  # a chain episode 0's chain_key_elements, prepared


def prepare_episode(scorer, demos, path: pathlib.Path, start_window: int, batch_size: int) -> Prepared:
    """Loads one episode and runs every CPU step of its scoring (background thread)."""
    ep = _dump.load(path)
    frames = ep["frame"]
    sim = [scorer.prepare(c) for c in chunks([element_from(ep, i) for i in range(len(frames))], batch_size)]
    early = np.flatnonzero(frames < start_window)
    demo_ep = demos.episode(ep["prompt"], ep["episode"])
    demo, demo_frames = [], None
    if demo_ep is not None and early.size:
        demo_frames = demos.frames(demo_ep)
        d_img, d_wrist = demo_frames
        last = len(d_img) - 1
        elements = [
            element_from(ep, i, d_img[min(int(frames[i]), last)], d_wrist[min(int(frames[i]), last)]) for i in early
        ]
        demo = [scorer.prepare(c) for c in chunks(elements, batch_size)]
    key_elements = chain_key_elements(ep) if ep["episode"] == 0 else None
    keys = scorer.prepare(key_elements) if key_elements else None
    return Prepared(path, ep, sim, early, demo_ep, demo, demo_frames, keys)


def chain_key_elements(ep: dict) -> list[dict] | None:
    """For a chain request (plan_chain): its first query, then each chain task's own one-stage
    request (that task as the prompt, no plan fields)."""
    chain = ep["request"][0].get("plan_chain")
    if not chain:
        return None
    first = element_from(ep, 0)
    singles = []
    for task in chain:
        single = {k: v for k, v in first.items() if not k.startswith("plan_")}
        single["prompt"] = task
        singles.append(single)
    return [first, *singles]


def chain_key_gaps(keys: np.ndarray, valid: np.ndarray) -> list[float | None]:
    """Row 0 is the chain, row 1 + j task j's own plan: stage j's key against that plan's stage 0."""
    return [
        float(np.abs(keys[0, j] - keys[1 + j, 0]).max()) if valid[0, j] and valid[1 + j, 0] else None
        for j in range(keys.shape[0] - 1)
    ]


def episode_row(prep: Prepared, s: dict, d: dict | None, args) -> dict:
    """One episode's scores against its trace: latches, early false done, the B / C / A gaps and
    the replay's fidelity to the recorded done."""
    ep, frames = prep.ep, prep.ep["frame"]
    q = len(frames)
    met = first_met(parse_trace(ep["trace"]))
    valid0 = bool(s["valid"][0, 0])
    recorded = ep["done"]
    row = {
        "file": prep.path.name,
        "task": ep["task"],
        "episode": ep["episode"],
        "prompt": ep["prompt"],
        "stages": int(s["valid"][0].sum()),
        "met": met,
        "latch_recorded": (
            latched_frame(frames, np.nan_to_num(recorded[:, 0]), args.threshold, args.consecutive)
            if recorded.shape[1]
            else None
        ),
        "latch_served": latched_frame(frames, s["served_raw"][:, 0], args.threshold, args.consecutive),
        "latch_m1": latched_frame(frames, s["m1"][:, 0], args.threshold, args.consecutive),
        "demo_episode": prep.demo_episode,
    }
    for key in ("recorded", "served", "m1"):
        row[f"premature_{key}"] = row["stages"] > 1 and premature(row[f"latch_{key}"], met)
    early = prep.early
    row["early_served"] = valid0 and bool((s["served_raw"][early, 0] > args.threshold).any())
    row["early_m1"] = valid0 and bool((s["m1"][early, 0] > args.threshold).any())
    row["early_served_max"] = float(s["served_raw"][early, 0].max()) if early.size else None
    if d is not None:
        row["early_demo"] = valid0 and bool((d["served_raw"][:, 0] > args.threshold).any())
        row["early_demo_max"] = float(d["served_raw"][:, 0].max())
    ok = ~np.isnan(recorded).any(1) if recorded.shape[1] else np.zeros(q, bool)
    k = min(recorded.shape[1], s["served"].shape[1])
    if ok.any() and k:
        row["fidelity"] = float(np.abs(s["served"][ok, :k] - recorded[ok, :k]).max())
    row["served_vs_fp32"] = float(np.abs(s["served_raw"] - s["fp32"]).max())
    row["fp32_vs_m1"] = float(np.abs(s["fp32"] - s["m1"]).max())
    row["token_gap"] = float(np.median(s["token_gap"]))
    row["key_gap"] = float(np.max(s["key_gap"]))
    return row


def png_name(row: dict, s: dict) -> str:
    return f"task{row['task']}_ep{row['episode']}_f{row['latch_served']}_met{row['met']}.png"


def latch_panels(prep: Prepared, row: dict) -> list[np.ndarray]:
    """The latch query's sim agent view and wrist, then the demo's at the same frame."""
    frames = prep.ep["frame"]
    i = int(np.flatnonzero(frames == row["latch_served"])[0])
    panels = [prep.ep["image"][i], prep.ep["wrist_image"][i]]
    if prep.demo_frames is not None:
        d_img, d_wrist = prep.demo_frames
        f = min(int(frames[i]), len(d_img) - 1)
        panels += [d_img[f], d_wrist[f]]
    return panels


def save_png(path: pathlib.Path, panels: list[np.ndarray]) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(panels, axis=1)).save(path)


def summarize(rows: list[dict], start_window: int) -> dict:
    """Per-run counts from the per-episode rows."""
    n = len(rows)

    def count(key):
        return int(sum(bool(r.get(key)) for r in rows))

    out = {
        "episodes": n,
        "early_false_served": count("early_served"),
        "early_false_m1": count("early_m1"),
        "early_false_demo": count("early_demo"),
        "premature_served": count("premature_served"),
        "premature_m1": count("premature_m1"),
        "premature_recorded": count("premature_recorded"),
        "first_met": int(sum(r["met"] is not None for r in rows)),
        "start_window": start_window,
    }
    for key in ("fidelity", "served_vs_fp32", "fp32_vs_m1", "token_gap", "key_gap"):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if vals:
            out[key] = {"median": float(np.median(vals)), "max": float(np.max(vals))}
    return out


def main(args: Args) -> dict:
    import torch

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    device = torch.device(args.device)
    cfg = _config.get_config(args.config)
    replace = {
        k: v
        for k, v in (
            ("paligemma_variant", args.paligemma_variant),
            ("action_expert_variant", args.action_expert_variant),
        )
        if v
    }
    if replace:
        cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, **replace))
    result: dict = {"args": dataclasses.asdict(args), "runs": {}}

    result["parity"] = parity(
        str(pathlib.Path(args.policy_dir) / "model.safetensors"), str(pathlib.Path(args.weights) / "model.safetensors")
    )
    logging.info(f"PARITY vision/projector/embed, {args.policy_dir} vs {args.weights}: {result['parity']}")
    meta_path = pathlib.Path(args.monitor_from) / "monitor.json"
    if meta_path.exists():
        result["monitor_meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
        if result["monitor_meta"].get("pool") != cfg.model.module_monitor_pool:
            raise ValueError(
                f"monitor.json pool {result['monitor_meta'].get('pool')} != the config's "
                f"{cfg.model.module_monitor_pool}"
            )

    policy = _policy_config.create_trained_policy(
        cfg, args.policy_dir, monitor_from=args.monitor_from, pytorch_device=str(device)
    )
    m1_model = _m1.load_vision(cfg.model, args.weights, "float32", device)
    scorer = Scorer(policy, m1_model, device)
    logging.info(f"served monitor dtype {scorer.served_dtype}, pool {scorer.pool}")
    t_demo = time.monotonic()
    cache = pathlib.Path(args.demo_cache) if args.demo_cache else None
    demos = Demos(pathlib.Path(args.dataset_root), args.demo_episodes, cache=cache)
    logging.info(
        f"demos: {sum(len(v) for v in demos.by_task.values())} episodes of {len(demos.by_task)} tasks"
        + (f", copied to {cache} in {time.monotonic() - t_demo:.0f} s" if cache else "")
    )
    png_dir = pathlib.Path(args.png_dir) if args.png_dir else None
    pngs = 0

    t0 = time.monotonic()
    for run, paths in run_episodes(pathlib.Path(args.dump_dir)).items():
        rows, key_checks = [], []
        stream = _m1.Prefetch(
            paths, lambda p: prepare_episode(scorer, demos, p, args.start_window, args.batch_size), depth=3
        )
        for prep in stream:
            s = scored(scorer.score, prep.sim)
            d = scored(scorer.score, prep.demo, m1=False) if prep.demo else None
            row = episode_row(prep, s, d, args)
            if png_dir is not None and row["premature_served"] and pngs < args.png_max:
                save_png(png_dir / run / png_name(row, s), latch_panels(prep, row))
                pngs += 1
            if prep.keys is not None:
                keys, valid = scorer.keys(prep.keys)
                chain = prep.ep["request"][0]["plan_chain"]
                key_checks.append(
                    {"task": prep.ep["task"], "chain": list(chain), "stage_key_gap": chain_key_gaps(keys, valid)}
                )
            rows.append(row)
        summary = summarize(rows, args.start_window)
        summary["chain_key_checks"] = key_checks
        result["runs"][run] = {"summary": summary, "episodes": rows}
        logging.info(f"REPLAY {run}: {json.dumps({k: v for k, v in summary.items() if k != 'chain_key_checks'})}")
        if key_checks:
            logging.info(f"REPLAY {run} chain keys: {json.dumps(key_checks)}")
        logging.info(
            f"{run}: {len(rows)} episodes, {time.monotonic() - t0:.0f} s, waited {stream.waited:.0f} s for data, "
            f"host RSS {_m1.rss_gib()}"
        )
        if args.out:
            pathlib.Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    if args.out:
        logging.info(f"wrote {args.out}")
    return result


if __name__ == "__main__":
    import tyro

    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
