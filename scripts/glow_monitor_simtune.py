"""Trains the progress monitor on closed-loop simulator frames as well as the training demos, and
measures how often it latches a stage before the stage is done.

scripts/glow_progress_monitor.py trains on demo frames only: one-stage plans on the one-task demos,
two-stage plans only on the chain demos. A chain evaluation runs a two-stage plan on a one-task
scene, which those demos never show. Here the episodes a monitor row's clients dumped
(--args.dump-dir, examples/libero/glow_query_dump.py) become training data as well:

1. Sim frames: every query of the dumped episodes (runs in --sim-exclude left out: a steered run's
   atoms score another plan), with the plan inputs the server built from the request
   (--sim-config's input transforms). Labels per stage from the simulator's goal atoms (the
   trace's atoms_first_met, execution order): not done before the atom first held, done from
   --sim-done-after frames after it, unlabeled in between; a stage whose atom never held is not
   done throughout. Held out: episodes --sim-heldout-episodes of every task (their own initial
   states), all queries; training episodes keep every --sim-train-every-th query.
2. Demo frames: M1's (--config's pipeline, its split and labels).
3. Camera tokens for both as M1 made them (--weights' vision tower in --precision, pooled, bfloat16
   on the device).
4. Variants: m1 (the --init-monitor as it is), finetune (from it, --finetune-steps at
   --finetune-lr), scratch (M1's recipe). Each training batch draws --sim-frac of its samples from
   the sim frames, and permutes multi-stage plans with probability --swap-p (labels follow).
5. Metrics: on the held-out sim episodes per family (libero10, chain, goal), each stage boundary's
   latch (--threshold on --consecutive queries) against the frame its atom first held: premature
   (before it, or with the atom never holding), missed, lag; first-stage false done before
   --start-window; AUROC. On the held-out demos, M1's metrics and T1 bars.

  python scripts/glow_monitor_simtune.py --weights <dir> --dataset-root <lerobot root> \
      --plan-pack <pack.json> --dump-dir <dumps> --init-monitor <M1 monitor dir> --out out.json
"""

import concurrent.futures
import dataclasses
import json
import logging
import pathlib
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "examples" / "libero"))
import glow_monitor_replay as _replay  # noqa: E402
import glow_progress_monitor as _m1  # noqa: E402
import glow_query_dump as _dump  # noqa: E402

SIM_FAMILIES = ("libero10", "chain", "goal")
VARIANTS = ("m1", "finetune", "scratch")
# Held-out sim chain episodes: premature latches at most this share of boundaries and at most this
# fraction of M1's; missed switches at most this share; the median lag at most this many frames.
PASS_PREMATURE, PASS_PREMATURE_VS_M1, PASS_MISSED, PASS_LAG = 0.05, 1 / 3, 0.05, 15.0


@dataclasses.dataclass
class Args:
    # M1's inputs (glow_progress_monitor.py): the checkpoint the camera tokens come from, the
    # demos and their plan pack.
    weights: str
    dataset_root: str
    plan_pack: str
    # The clients' --args.dump-dir root (one subdirectory per run) and the monitor to start from.
    dump_dir: str
    init_monitor: str
    config: str = "pi05_libero_glow_v2t_fixed"
    # The served config: its input transforms turn a dumped request into the monitor's plan inputs.
    sim_config: str = "pi05_libero_glow_monitor"
    # Plan pack for the dumped requests (default: --sim-config's own, the one the server read).
    sim_plan_pack: str | None = None
    paligemma_variant: str | None = None
    action_expert_variant: str | None = None
    # Demo frames, as M1.
    episodes: str = ""
    max_per_task: int = 0
    stride: int = 5
    single_stride: int = 10
    tail: int = 15
    pool: int = 2
    holdout_every: int = 5
    heldout_inits: tuple[int, ...] = ()
    # Sim frames.
    sim_exclude: tuple[str, ...] = ("goal-steer",)
    sim_heldout_episodes: tuple[int, ...] = (8, 9)
    sim_train_every: int = 2
    sim_done_after: int = 10
    # Local directory the dump files are copied to first (default: read in place).
    dump_cache: str | None = None
    # Monitor shape: read from --init-monitor's monitor.json.
    width: int = 512
    depth: int = 2
    heads: int = 8
    variants: tuple[str, ...] = VARIANTS
    steps: int = 6000
    lr: float = 3e-4
    finetune_steps: int = 3000
    finetune_lr: float = 1e-4
    batch_size: int = 256
    weight_decay: float = 0.01
    warmup: int = 200
    swap_p: float = 0.3
    sim_frac: float = 0.5
    threshold: float = 0.7
    consecutive: int = 2
    start_window: int = 40
    render_batch: int = 32
    seed: int = 0
    precision: str = "float32"
    device: str = "cuda"
    out: str | None = None
    save_dir: str | None = None


# ---------------------------------------------------------------------------------------------
# Sim episodes and labels


@dataclasses.dataclass
class SimEpisode:
    run: str
    path: pathlib.Path
    task: int
    episode: int
    family: str
    heldout: bool
    met: list  # frame each stage's goal atom first held (None: never), execution order
    queries: np.ndarray  # indices of the episode's queries kept
    frames: np.ndarray  # their frames
    rows: np.ndarray | None = None  # their rows in the sim store


def sim_family(run: str) -> str:
    if run == "10" or run.endswith("/10"):
        return "libero10"
    return "chain" if ("chain" in run or "return" in run) else "goal"


def sim_labels(frames: np.ndarray, met: list, stages: int, done_after: int, k_max: int):
    """(done, labeled) [Q, k_max]: per stage, not done before its atom first held, done from
    `done_after` frames after, unlabeled in between; never held = not done throughout."""
    done = np.zeros((len(frames), k_max), dtype=np.float32)
    labeled = np.zeros((len(frames), k_max), dtype=bool)
    for j in range(stages):
        m = met[j] if j < len(met) else None
        if m is None:
            labeled[:, j] = True
            continue
        done[:, j] = frames >= m + done_after
        labeled[:, j] = (frames < m) | (frames >= m + done_after)
    return done, labeled


def kept_queries(n: int, heldout: bool, every: int) -> np.ndarray:
    return np.arange(n) if heldout else np.arange(0, n, max(1, every))


def list_sim_episodes(dump_dir: pathlib.Path, exclude, heldout_episodes, every: int) -> list[SimEpisode]:
    """Every dumped episode outside the excluded runs, with its split, atoms and kept queries."""
    out = []
    for run, paths in _replay.run_episodes(dump_dir).items():
        if run in exclude or pathlib.Path(run).name in exclude:
            continue
        for path in paths:
            with np.load(str(path)) as z:
                frames = z["frame"]
                trace = _replay.parse_trace(str(z["trace"]))
                task, episode = int(z["task"]), int(z["episode"])
            met = [None if m is None else int(m) for m in trace.get("atoms_first_met", [])]
            heldout = episode in heldout_episodes
            q = kept_queries(len(frames), heldout, every)
            out.append(SimEpisode(run, path, task, episode, sim_family(run), heldout, met, q, frames[q]))
    return out


def cache_files(paths: list[pathlib.Path], src_root: pathlib.Path, dst_root: pathlib.Path, parallel: int = 16):
    """Copies `paths` (under src_root) to the same relative paths under dst_root, `parallel` at a time;
    returns the new paths."""
    targets = [dst_root / p.relative_to(src_root) for p in paths]
    for t in targets:
        t.parent.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(parallel) as pool:
        list(pool.map(shutil.copyfile, paths, targets))
    return targets


def plan_id(request: dict) -> str:
    """What the monitor's plan inputs depend on in a request."""
    return json.dumps({k: request.get(k) for k in ("prompt", "plan_chain", "plan_task")}, sort_keys=True)


# ---------------------------------------------------------------------------------------------
# Metrics


def boundary_events(frames, probs, met, stages: int, threshold: float, consecutive: int) -> list[dict]:
    """Per stage boundary (every stage but the last): the latch against the atom's first-met frame."""
    out = []
    for j in range(stages - 1):
        latch = _replay.latched_frame(frames, probs[:, j], threshold, consecutive)
        m = met[j] if j < len(met) else None
        out.append(
            {
                "stage": j,
                "met": m,
                "latch": latch,
                "premature": _replay.premature(latch, m),
                "missed": m is not None and latch is None,
                "lag": None if latch is None or m is None or latch < m else latch - m,
            }
        )
    return out


def sim_metrics(episodes: list[SimEpisode], probs: dict, labels: dict, args) -> dict:
    """Per family over held-out episodes: boundary latches, first-stage early false done, AUROC."""
    out = {}
    for fam in SIM_FAMILIES:
        eps = [e for e in episodes if e.heldout and e.family == fam]
        if not eps:
            continue
        events, early, scores, targets = [], 0, [], []
        for e in eps:
            p = probs[e.path]
            done, labeled = labels[e.path]
            stages = int(labeled.any(0).sum())
            events += boundary_events(e.frames, p, e.met, stages, args.threshold, args.consecutive)
            m0 = e.met[0] if e.met else None
            window = (e.frames < args.start_window) & ((e.frames < m0) if m0 is not None else True)
            early += bool((p[window, 0] > args.threshold).any())
            scores.append(p[labeled])
            targets.append(done[labeled] > 0.5)
        lags = [ev["lag"] for ev in events if ev["lag"] is not None]
        n = len(events)
        out[fam] = {
            "episodes": len(eps),
            "boundaries": n,
            "premature": int(sum(ev["premature"] for ev in events)),
            "premature_rate": float(np.mean([ev["premature"] for ev in events])) if n else None,
            "missed": int(sum(ev["missed"] for ev in events)),
            "missed_rate": float(np.mean([ev["missed"] for ev in events])) if n else None,
            "lag_median": float(np.median(lags)) if lags else None,
            "early_false_done": early,
            "auroc": _m1.auroc(np.concatenate(scores), np.concatenate(targets)) if scores else None,
        }
    return out


def sim_passes(variant: dict, m1: dict) -> dict:
    """The held-out chain bars against M1 (see PASS_*)."""
    own, base = variant.get("chain", {}), m1.get("chain", {})
    if not own.get("boundaries"):
        return {}
    rate, base_rate = own["premature_rate"], base.get("premature_rate")
    return {
        "premature": rate <= PASS_PREMATURE,
        "premature_vs_m1": base_rate is not None and rate <= PASS_PREMATURE_VS_M1 * base_rate,
        "missed": own["missed_rate"] <= PASS_MISSED,
        "lag": own["lag_median"] is not None and own["lag_median"] <= PASS_LAG,
    }


# ---------------------------------------------------------------------------------------------
# Tokens


def request_pipeline(args):
    """--sim-config's input transforms of a client request (the served ones, without normalization)."""
    from openpi import transforms as _transforms

    sim_args = dataclasses.replace(args, config=args.sim_config)
    cfg = _config_with_variants(sim_args)
    data = dataclasses.replace(cfg.data, assets=_m1_assets())
    if args.sim_plan_pack:
        data = dataclasses.replace(data, plan_pack_path=args.sim_plan_pack)
    dc = data.create(cfg.assets_dirs, cfg.model)
    return _transforms.compose([*dc.data_transforms.inputs, *dc.model_transforms.inputs])


def _m1_assets():
    from openpi.training import config as _config

    return _config.AssetsConfig(assets_dir="no-norm-stats", asset_id=None)


def _config_with_variants(args):
    from openpi.training import config as _config

    cfg = _config.get_config(args.config)
    replace = {
        k: v
        for k, v in (
            ("paligemma_variant", args.paligemma_variant),
            ("action_expert_variant", args.action_expert_variant),
        )
        if v
    }
    return dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, **replace)) if replace else cfg


def embed_sim(episodes: list[SimEpisode], pipeline, model, args, device):
    """Camera tokens and plan inputs of every kept sim query: a store (rows per episode set on each
    SimEpisode) and each row's plan row."""
    import torch

    from openpi.glow import module_encoders as _enc
    from openpi.models import model as _model

    total = sum(len(e.queries) for e in episodes)
    store = _m1.Store()
    plan_row = np.zeros(total, dtype=np.int64)
    ids: dict[str, int] = {}
    embed = model.paligemma_with_expert.embed_language_tokens

    def build(e: SimEpisode):
        ep = _dump.load(e.path)
        elements = [_replay.element_from(ep, int(i)) for i in e.queries]
        plans = [plan_id(ep["request"][int(i)]) for i in e.queries]
        batches = [
            (
                _m1.collate([pipeline(x) for x in elements[s : s + args.render_batch]]),
                plans[s : s + args.render_batch],
            )
            for s in range(0, len(elements), args.render_batch)
        ]
        return e, batches

    t0, row, cameras = time.monotonic(), 0, None
    stream = _m1.Prefetch(episodes, build, depth=4)
    for n, (e, batches) in enumerate(stream):
        start = row
        for batch, plans in batches:
            obs = _model.Observation.from_dict(_m1.to_torch(batch, device))
            with torch.no_grad():
                images, masks, *_ = model._preprocess_observation(obs, train=False)  # noqa: SLF001
                shown = [i for i, m in enumerate(masks) if bool(m.all())]
                if cameras is None:
                    cameras = shown
                elif shown != cameras:
                    raise ValueError(f"cameras changed between batches: {cameras} vs {shown}")
                feats = torch.stack(
                    [_m1.pooled_tokens(model.paligemma_with_expert.embed_image(images[i]), args.pool) for i in cameras],
                    1,
                ).to(torch.bfloat16)
                if store.images is None:
                    store.images = torch.empty((total, *feats.shape[1:]), dtype=torch.bfloat16, device=device)
                    logging.info(
                        f"sim camera tokens: {total} x {tuple(feats.shape[1:])} = "
                        f"{store.images.numel() * 2 / 2**30:.2f} GiB on {device}"
                    )
                rows = list(range(row, row + len(plans)))
                store.images[rows] = feats
                keys, heads, valid = _enc.monitor_key_inputs(embed, obs.glow_text, obs.glow_text_mask, obs.glow_ints)
            for p in plans:
                ids.setdefault(p, len(ids))
            plan_row[rows] = store.add_tasks([ids[p] for p in plans], keys, heads, valid)
            row += len(plans)
        e.rows = np.arange(start, row)
        if n % 50 == 0:
            logging.info(
                f"sim: {n + 1}/{len(episodes)} episodes, {row}/{total} queries, "
                f"{time.monotonic() - t0:.0f} s, host RSS {_m1.rss_gib()}"
            )
    store.image_mask = torch.ones(len(cameras), dtype=torch.bool, device=device)
    store.stack()
    logging.info(
        f"sim embedded: {row} queries, {len(ids)} plans, {time.monotonic() - t0:.0f} s, "
        f"data wait {stream.waited:.0f} s, cameras {cameras}"
    )
    return store, plan_row


# ---------------------------------------------------------------------------------------------
# Training


@dataclasses.dataclass
class Source:
    """One store of camera tokens with its labels and the rows a variant trains on."""

    store: object
    row: np.ndarray  # plan row per sample
    done: object  # [N, K_MAX] tensor
    labeled: object  # [N, K_MAX] tensor
    train_idx: np.ndarray


def load_init(path: str) -> tuple[dict, dict]:
    """(state dict without MONITOR_PREFIX, monitor.json) of a saved monitor."""
    from safetensors.torch import load_file

    root = pathlib.Path(path)
    state = load_file(str(root / "model.safetensors"))
    prefix = _m1.MONITOR_PREFIX
    meta = json.loads((root / "monitor.json").read_text(encoding="utf-8"))
    return {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}, meta


def train(name: str, sources: list[Source], fracs: list[float], init_state, steps: int, lr: float, args, device):
    """M1's training loop over several sources, each batch drawn from them in proportion `fracs`."""
    import torch

    from openpi.glow import module_encoders as _enc

    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed + 1)
    rng = np.random.default_rng(args.seed)
    first = sources[0].store
    monitor = _enc.ProgressMonitor(
        first.keys.shape[-1], first.images.shape[-1], args.width, args.depth, keys=True, heads=args.heads
    ).to(device)
    if init_state is not None:
        monitor.load_state_dict(init_state)
    if steps == 0:
        return monitor.eval()
    pos = neg = 0.0
    for s in sources:
        t = torch.as_tensor(s.train_idx, device=device)
        pos += float((s.done[t] * s.labeled[t]).sum())
        neg += float(((1 - s.done[t]) * s.labeled[t]).sum())
    pos_weight = torch.tensor(neg / max(pos, 1.0), device=device)
    opt = torch.optim.AdamW(monitor.parameters(), lr=lr, weight_decay=args.weight_decay)
    warmup = min(args.warmup, max(1, steps // 10))

    def lr_at(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + np.cos(np.pi * (step - warmup) / max(1, steps - warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    counts = [int(round(args.batch_size * f)) for f in fracs]
    counts[0] = args.batch_size - sum(counts[1:])
    t0, running = time.monotonic(), []
    monitor.train()
    for step in range(steps):
        parts = []
        for s, c in zip(sources, counts, strict=True):
            if c > 0:
                idx = rng.choice(s.train_idx, size=min(c, len(s.train_idx)), replace=False)
                parts.append(_m1.batch_tensors(s.store, s.row, s.done, s.labeled, idx, device))
        k, h, v, im, m, d, lab = (torch.cat(xs) for xs in zip(*parts, strict=True))
        if args.swap_p > 0:
            k, h, d, lab = _m1.permute(_m1.swap_perm(v, args.swap_p, gen), k, h, d, lab)
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
                f"{name} step {step + 1}/{steps} loss {np.mean(running):.4f} "
                f"lr {sched.get_last_lr()[0]:.2e} {time.monotonic() - t0:.0f} s"
            )
            running = []
    return monitor.eval()


def check_stage_counts(episodes: list[SimEpisode], store, plan_row: np.ndarray) -> None:
    """Every episode's plan has as many stages as its trace has goal atoms (the labels' stages)."""
    bad = []
    for e in episodes:
        if len(e.rows):
            stages = int(store.valid[int(plan_row[e.rows[0]])].sum())
            if stages != len(e.met):
                bad.append(f"{e.run}/{e.path.name}: plan {stages} stages, trace {len(e.met)} atoms")
    if bad:
        raise ValueError(f"{len(bad)} episodes' plans and atoms disagree, e.g. {bad[:5]}")


def sim_probs(monitor, source: Source, episodes: list[SimEpisode], device) -> dict:
    """Held-out episodes' done probabilities [Q, K_MAX] by path."""
    idx = np.concatenate([e.rows for e in episodes]) if episodes else np.zeros(0, np.int64)
    if not len(idx):
        return {}
    probs, _, _ = _m1.predict(monitor, source.store, source.row, source.done, source.labeled, idx, device, shift=False)
    out, at = {}, 0
    for e in episodes:
        out[e.path] = probs[at : at + len(e.rows)]
        at += len(e.rows)
    return out


# ---------------------------------------------------------------------------------------------


def main(args: Args) -> dict:
    import torch

    from openpi.glow import module_inputs as _mi
    from openpi.training import data_loader as _data_loader

    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    init_state, meta = load_init(args.init_monitor)
    args = dataclasses.replace(args, width=meta["width"], depth=meta["depth"], heads=meta["heads"])
    if meta.get("pool") != args.pool:
        raise ValueError(f"--init-monitor pool {meta.get('pool')} != --pool {args.pool}")
    result: dict = {"args": dataclasses.asdict(args), "init_meta": meta, "sim": {}, "demo": {}, "pass": {}}

    def write():
        if args.out:
            out = pathlib.Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=1), encoding="utf-8")

    # Sim episodes (copied locally first if asked).
    dump_dir = pathlib.Path(args.dump_dir)
    if args.dump_cache:
        t0 = time.monotonic()
        files = [p for paths in _replay.run_episodes(dump_dir).values() for p in paths]
        cache_files(files, dump_dir, pathlib.Path(args.dump_cache))
        logging.info(f"copied {len(files)} dump files to {args.dump_cache} in {time.monotonic() - t0:.0f} s")
        dump_dir = pathlib.Path(args.dump_cache)
    sim_eps = list_sim_episodes(dump_dir, set(args.sim_exclude), set(args.sim_heldout_episodes), args.sim_train_every)
    if not any(not e.heldout for e in sim_eps) or not any(e.heldout for e in sim_eps):
        raise ValueError("the sim split leaves no training or no held-out episodes")
    k_max = _mi.K_MAX
    labels = {}
    for e in sim_eps:
        stages = len(e.met)
        labels[e.path] = sim_labels(e.frames, e.met, stages, args.sim_done_after, k_max)
    counts = {
        fam: {
            "train_episodes": sum(1 for e in sim_eps if e.family == fam and not e.heldout),
            "heldout_episodes": sum(1 for e in sim_eps if e.family == fam and e.heldout),
            "queries": int(sum(len(e.queries) for e in sim_eps if e.family == fam)),
        }
        for fam in SIM_FAMILIES
    }
    result["sim_counts"] = counts
    logging.info(f"sim episodes: {counts}")

    # Demo frames, as M1.
    cfg, pipeline, module = _m1.build_pipeline(args)
    pack = module.pack
    root = pathlib.Path(args.dataset_root)
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    spec = _data_loader.resolve_train_episodes(args.episodes) if args.episodes else None
    episodes = _m1.select_episodes(pack, int(info["total_episodes"]), spec, args.max_per_task)
    chains = _m1.chain_inits(pack)
    heldout = _m1.split(pack, episodes, chains, args.holdout_every, args.heldout_inits)
    samples = _m1.frame_plan(pack, episodes, chains, heldout, args.stride, args.single_stride, args.tail)
    done = np.stack([_m1.padded(s["done"], k_max, 0.0) for s in samples])
    labeled = np.stack([_m1.padded(s["labeled"], k_max, False) for s in samples])
    demo_train = np.array([i for i, s in enumerate(samples) if not s["heldout"]])
    demo_test = np.array([i for i, s in enumerate(samples) if s["heldout"]])
    logging.info(f"demo frames: {len(samples)} ({len(demo_train)} train, {len(demo_test)} held out)")

    model = _m1.load_vision(cfg.model, args.weights, args.precision, device)
    demo_store, demo_row = _m1.dump(samples, pipeline, model, args, device, root, cfg.model.action_horizon)
    sim_store, sim_row = embed_sim(sim_eps, request_pipeline(args), model, args, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    check_stage_counts(sim_eps, sim_store, sim_row)
    sim_done = np.zeros((len(sim_row), k_max), dtype=np.float32)
    sim_labeled = np.zeros((len(sim_row), k_max), dtype=bool)
    for e in sim_eps:
        sim_done[e.rows], sim_labeled[e.rows] = labels[e.path]
    demo = Source(
        demo_store, demo_row, torch.as_tensor(done, device=device), torch.as_tensor(labeled, device=device), demo_train
    )
    sim_train = np.concatenate([e.rows for e in sim_eps if not e.heldout])
    sim = Source(
        sim_store,
        sim_row,
        torch.as_tensor(sim_done, device=device),
        torch.as_tensor(sim_labeled, device=device),
        sim_train,
    )
    sim_test = [e for e in sim_eps if e.heldout]

    plans = {
        "m1": (init_state, 0, args.lr),
        "finetune": (init_state, args.finetune_steps, args.finetune_lr),
        "scratch": (None, args.steps, args.lr),
    }
    for name in args.variants:
        init, steps, lr = plans[name]
        monitor = train(name, [demo, sim], [1 - args.sim_frac, args.sim_frac], init, steps, lr, args, device)
        result["sim"][name] = sim_metrics(sim_test, sim_probs(monitor, sim, sim_test, device), labels, args)
        own, _, _ = _m1.predict(monitor, demo.store, demo.row, demo.done, demo.labeled, demo_test, device, shift=False)
        shifted = _m1.predict(monitor, demo.store, demo.row, demo.done, demo.labeled, demo_test, device, shift=True)
        result["demo"][name] = _m1.evaluate(
            samples, demo_test, own, pack, args.threshold, args.start_window, swapped=shifted
        )
        result["pass"][name] = {
            "demo": _m1.passes({"variants": {"monitor": result["demo"][name]}}),
            "sim": sim_passes(result["sim"][name], result["sim"].get("m1", {})),
        }
        if args.save_dir and name != "m1":
            path = _m1.save_monitor(monitor, pathlib.Path(args.save_dir) / name, args, keys=True)
            logging.info(f"saved {name} to {path}")
        for fam, r in result["sim"][name].items():
            logging.info(f"SIM {name:9s} {fam:9s} {json.dumps(r)}")
        for fam, r in result["demo"][name].items():
            sw = r["switch"]
            logging.info(
                f"DEMO {name:9s} {fam:9s} auroc {r['auroc']} start_false {r['start_false_done']} "
                f"switch median {sw['median_abs']} within {sw['within']} missed {sw['missed']}/{sw['boundaries']} "
                f"swapped auroc {r.get('swapped', {}).get('auroc')}"
            )
        logging.info(f"PASS {name} {json.dumps(result['pass'][name])}")
        write()
        del monitor
    if args.out:
        logging.info(f"wrote {args.out}")
    return result


if __name__ == "__main__":
    import tyro

    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
