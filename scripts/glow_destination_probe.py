"""Does a GLOW module model's velocity follow the plan's DESTINATION? Forward only, no simulator.

Each pair here moves the same object to two destinations (DESTINATION_PAIRS), so the plans differ
only in where the object goes. The samples are held-out demos of those tasks, A being the demo's
own task and B the other destination, in three buckets cut at the demo's gripper:

    reach   before the gripper closes, when the motion does not depend on the destination
    early   the first half of the closed run, from `settle` frames after it closes
    late    the rest of the closed run

The plan has one stage, as a goal-oracle eval's, and the score is glow_temporal_probe's margin against
the distillation teachers of A and B: +1 is the teacher of the task scored, -1 the other's, 0 halfway.
VARIANTS sets which task each encoder reads:

    plan_own                    both encoders read A, scored for A
    plan_other                  both encoders read B, scored for B
    temporal_off_own            Spatial reads A, the Temporal tokens masked out, scored for A
    temporal_off_other          Spatial reads B, the Temporal tokens masked out, scored for B
    spatial_own_temporal_other  Spatial reads A, Temporal reads B, scored for Spatial's task
    spatial_other_temporal_own  Spatial reads B, Temporal reads A, scored for Spatial's task

The demo's images favour A whatever the plan says, so a variant is read pooled with its mirror: a
model that ignores the destination in the plan pools to 0. follows_plan pools the first two,
spatial_alone the next two, and spatial_vs_temporal the last two: +1 the velocity follows Spatial,
-1 Temporal. teacher_gap is the mean squared distance between the two teachers: a small gap means
the destination does not change the motion there.

`--spatial-from name=weights_dir` gives row `name` the Spatial encoder saved in another checkpoint, as
in glow_temporal_probe. With its Temporal tokens masked, such a row is that checkpoint's Spatial encoder
alone, so its spatial_alone reproduces that checkpoint's own row on the same samples. `--fixed-gate
name=value` scales its Spatial groups by a constant, also as in glow_temporal_probe.

`--two-stage` makes every plan [the destination's task, SECOND_STAGE], the first stage active, so a turned
down partner can matter: a one-stage plan has no other group for a gate to scale. A masked variant then
keeps the row's gates, so spatial_alone is the Spatial tokens at their gated sizes.

  python scripts/glow_destination_probe.py --rows key=pi05_libero_glow_v2t_key:<dir> \
      --norm-stats-dir <assets dir> --dataset-root <lerobot repo dir> \
      --exclude-episodes 379:427 --out destination_probe.json
"""

import dataclasses
import json
import logging
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import glow_temporal_probe as _gtp  # noqa: E402

DESTINATION_PAIRS = (
    ("put the bowl on the stove", "put the bowl on the plate"),
    ("put the bowl on the stove", "put the bowl on top of the cabinet"),
    ("put the bowl on the plate", "put the bowl on top of the cabinet"),
    ("put the wine bottle on the rack", "put the wine bottle on top of the cabinet"),
)
# The plans' second stage with --two-stage: another object, the same for both destinations.
SECOND_STAGE = "open the middle drawer of the cabinet"
BUCKETS = ("reach", "early", "late")
# name -> (the task Spatial reads, the task Temporal reads or None for masked), "A" or "B". A variant
# is scored for the task Spatial reads.
VARIANTS = {
    "plan_own": ("A", "A"),
    "plan_other": ("B", "B"),
    "temporal_off_own": ("A", None),
    "temporal_off_other": ("B", None),
    "spatial_own_temporal_other": ("A", "B"),
    "spatial_other_temporal_own": ("B", "A"),
}
POOLED = {
    "follows_plan": ("plan_own", "plan_other"),
    "spatial_alone": ("temporal_off_own", "temporal_off_other"),
    "spatial_vs_temporal": ("spatial_own_temporal_other", "spatial_other_temporal_own"),
}
# The gripper command at or above which a frame's gripper is closing or closed.
GRIPPER_CLOSED = 0.0


@dataclasses.dataclass
class Args:
    # Assets directory whose <repo_id>/norm_stats.json normalizes state and actions.
    norm_stats_dir: str
    # LeRobot dataset root: the directory holding meta/ and data/.
    dataset_root: str
    # Trained checkpoints, name=config:weights_dir (the directory holding model.safetensors).
    rows: tuple[str, ...] = ()
    # Rows that take their Spatial encoder from another checkpoint, name=weights_dir.
    spatial_from: tuple[str, ...] = ()
    # Rows whose inactive stages' Spatial groups enter at a constant, name=value.
    fixed_gate: tuple[str, ...] = ()
    # Plans of two stages, SECOND_STAGE after the destination's task.
    two_stage: bool = False
    # The config whose data pipeline renders the samples.
    config: str = "pi05_libero_glow_v2t"
    repo_id: str = "physical-intelligence/libero"
    # Episodes the rows trained on, as OPENPI_TRAIN_EPISODES: never sampled.
    exclude_episodes: str = ""
    episodes_per_task: int = 8
    # Samples per task in each bucket.
    reach_samples: int = 24
    early_samples: int = 48
    late_samples: int = 24
    # Frames after the gripper closes that belong to no bucket: the grasp, before the lift.
    settle: int = 10
    batch_size: int = 8
    seed: int = 0
    precision: str = "float32"
    device: str = "cuda"
    out: str | None = None


def closed_run(gripper: np.ndarray) -> tuple[int, int]:
    """The longest run of closed-gripper frames, as [start, stop)."""
    closed = np.concatenate([[False], np.asarray(gripper) >= GRIPPER_CLOSED, [False]])
    edges = np.flatnonzero(closed[1:] != closed[:-1])
    if len(edges) == 0:
        raise ValueError("the gripper never closes in this episode")
    starts, stops = edges[::2], edges[1::2]
    longest = int(np.argmax(stops - starts))
    return int(starts[longest]), int(stops[longest])


def bucket_frames(start: int, stop: int, settle: int) -> dict[str, range]:
    """Each bucket's frames for a closed run [start, stop)."""
    middle = (start + settle + stop) // 2
    return {"reach": range(start), "early": range(start + settle, middle), "late": range(middle, stop)}


def read_gripper(root: pathlib.Path, episode: int) -> np.ndarray:
    """One episode's gripper commands, the last action dimension."""
    import pyarrow.parquet as pq

    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    path = root / info["data_path"].format(episode_chunk=episode // info["chunks_size"], episode_index=episode)
    return np.asarray(pq.read_table(path, columns=["actions"])["actions"].to_pylist(), dtype=np.float32)[:, -1]


def partners(task: str) -> list[str]:
    return [b if a == task else a for a, b in DESTINATION_PAIRS if task in (a, b)]


def plan_samples(tasks_by_index, heldout, runs, *, per_bucket, settle, rng) -> list[dict]:
    """The samples, shuffled: episode, frame, bucket and pair (the demo's task, another destination).
    `heldout` maps a task index to its demos, `runs` a demo to its closed run, `per_bucket` a bucket
    to its samples per task."""
    samples = []
    for task_index, episodes in sorted(heldout.items()):
        task = tasks_by_index[task_index]
        others = partners(task)
        by_bucket = {bucket: [] for bucket in BUCKETS}
        for episode in episodes:
            for bucket, frames in bucket_frames(*runs[episode], settle).items():
                by_bucket[bucket].extend((episode, f) for f in frames)
        for bucket, pool in by_bucket.items():
            if len(pool) < per_bucket[bucket]:
                raise ValueError(f"{task!r} has {len(pool)} {bucket} frames, the probe wants {per_bucket[bucket]}")
            for i in rng.choice(len(pool), size=per_bucket[bucket], replace=False):
                other = others[int(rng.integers(len(others)))]
                samples.append({"episode": pool[i][0], "frame": pool[i][1], "bucket": bucket, "pair": (task, other)})
    return [samples[i] for i in rng.permutation(len(samples))]


def variant_prefix(model, spatial, temporal, off: bool, images, img_masks, lang_tokens, lang_masks):
    """The model's prefix with its Spatial encoder reading `spatial`, and its Temporal encoder reading
    `temporal` or (`off`) masked out."""
    import torch

    from openpi.glow import module_encoders as _enc

    encoder = model.glow_temporal_encoder
    own_forward = encoder.forward
    encoder.forward = lambda embed, _text, _mask, _ints: own_forward(
        embed, temporal.glow_text, temporal.glow_text_mask, temporal.glow_ints
    )
    if off:
        own_mask = model.module_token_mask
        keep = torch.ones_like(own_mask(spatial))
        keep[:, -_enc.TEMPORAL_TOKENS :] = False
        model.module_token_mask = lambda o: own_mask(o) & keep
    try:
        return model.embed_observation_prefix(spatial, images, img_masks, lang_tokens, lang_masks)
    finally:
        del encoder.forward
        if off:
            del model.module_token_mask


def run_row(name, model, module, store, args, device) -> dict:
    """One checkpoint over every batch."""
    import jax
    import torch

    from openpi.models import model as _model

    def to_device(x):
        t = torch.as_tensor(x)
        # Normalize can promote state to float64 (numpy 2 promotion); the model computes in float32.
        return t.to(device, torch.float32) if t.dtype == torch.float64 else t.to(device)

    acc = _gtp.Accumulator()
    t0, forwards = time.monotonic(), 0
    for b in range(store.num_batches):
        item = store.get(b)
        samples = item["samples"]
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
            teachers = _gtp.teacher_velocities(model, module, samples, (images, img_masks, state), x_t, flow_time, device)
            dab = _gtp.squared(teachers["A"], teachers["B"])
            forwards += 2
            then = SECOND_STAGE if args.two_stage else None
            arrays = {which: _gtp.module_arrays(module, samples, which, 0, device, then) for which in ("A", "B")}
            for variant, (spatial, temporal) in VARIANTS.items():
                prefix = variant_prefix(
                    model,
                    arrays[spatial],
                    arrays[temporal or spatial],
                    temporal is None,
                    images,
                    img_masks,
                    lang_tokens,
                    lang_masks,
                )
                v = model.velocity_from_prefix(*prefix, state, x_t, flow_time)
                scored, other = (teachers["A"], teachers["B"]) if spatial == "A" else (teachers["B"], teachers["A"])
                acc.add(variant, samples, _gtp.squared(v, scored), _gtp.squared(v, other), dab, spatial == "A")
                forwards += 1
        if b % 8 == 0:
            elapsed = time.monotonic() - t0
            logging.info(
                f"{name}: batch {b + 1}/{store.num_batches}, {forwards / max(elapsed, 1e-9):.2f} forwards/s, "
                f"waited {store.waited:.0f} s on data so far"
            )
    logging.info(f"{name}: {forwards} forwards in {time.monotonic() - t0:.0f} s, {store.waited:.0f} s of data wait in all")

    rng = np.random.default_rng(args.seed)
    result: dict = {"buckets": {}}
    for bucket in BUCKETS:
        rows = acc.rows["plan_own"]
        gap = float(np.mean([r[5] for r in rows if r[0] == bucket]))
        entry: dict = {"teacher_gap": gap}
        for pooled, names in POOLED.items():
            entry[pooled] = acc.stats(names, rng, bucket=bucket)
            # A variant's two mirrors score the same pair from both ends, so the pair is unordered.
            entry[f"{pooled}_by_pair"] = acc.stats(names, rng, bucket=bucket, key=lambda r: tuple(sorted(r[1])))
        entry["variants"] = {v: acc.stats([v], rng, bucket=bucket) for v in VARIANTS}
        result["buckets"][bucket] = entry
        logging.info(f"DEST {name} {bucket:5s} teacher_gap {gap:.4g}")
        for pooled in POOLED:
            s = entry[pooled]
            logging.info(
                f"DEST {name} {bucket:5s} {pooled:20s} margin {s['margin']:+.3f} +- {s['margin_se']:.3f}  "
                f"rel_err {s['rel_err']:.3f}"
            )
        for v, s in entry["variants"].items():
            logging.info(
                f"VARIANT {name} {bucket:5s} {v:27s} margin {s['margin']:+.3f} +- {s['margin_se']:.3f}  "
                f"rel_err {s['rel_err']:.3f}"
            )
        for pooled in POOLED:
            for pair, s in entry[f"{pooled}_by_pair"].items():
                logging.info(
                    f"PAIR {name} {bucket:5s} {pooled:20s} {pair!r}: margin {s['margin']:+.3f} "
                    f"rel_err {s['rel_err']:.3f} ({s['samples']})"
                )
    return result


def main(args: Args) -> dict:
    import torch

    from openpi.training import data_loader as _data_loader

    if not args.rows:
        raise ValueError("nothing to do: give --rows")
    rows = [_gtp.parse_row(r) for r in args.rows]
    spatial_from = _gtp.parse_spatial_from(args.spatial_from, rows)
    fixed_gate = _gtp.parse_fixed_gate(args.fixed_gate, rows)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg, dc, module, inputs = _gtp.data_config(args.config, args.norm_stats_dir, args.repo_id)
    pipeline = _gtp.render_pipeline(dc, inputs)
    rng = np.random.default_rng(args.seed)
    exclude = _data_loader.resolve_train_episodes(args.exclude_episodes) if args.exclude_episodes else []
    tasks_by_index = {t.task_index: t.task for t in module.pack.tasks.values()}
    wanted = {t for pair in DESTINATION_PAIRS for t in pair}
    heldout = {
        i: episodes
        for i, episodes in _gtp.heldout_episodes(module.pack, exclude, args.episodes_per_task, rng).items()
        if tasks_by_index[i] in wanted
    }
    root = pathlib.Path(args.dataset_root)
    runs = {e: closed_run(read_gripper(root, e)) for episodes in heldout.values() for e in episodes}
    per_bucket = {"reach": args.reach_samples, "early": args.early_samples, "late": args.late_samples}
    plan = plan_samples(tasks_by_index, heldout, runs, per_bucket=per_bucket, settle=args.settle, rng=rng)
    build = _gtp.make_build_batch(pipeline, root, cfg.model.action_horizon, _gtp.read_task_strings(root))
    # Started before any model loads, so the first batches wait for the GPU, not the other way round.
    store = _gtp.BatchStore(plan, build, args.batch_size)
    logging.info(f"{len(plan)} samples from {len(runs)} held-out episodes: {heldout}")

    result: dict = {"args": dataclasses.asdict(args), "heldout": heldout, "closed_runs": runs, "rows": {}}

    def write():
        if args.out:
            out = pathlib.Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=1), encoding="utf-8")

    for name, config_name, weights in rows:
        model = _gtp.load_model(
            config_name,
            weights,
            args,
            device,
            init=False,
            spatial_weights=spatial_from.get(name),
            fixed_gate=fixed_gate.get(name),
        )
        _, _, row_module, _ = _gtp.data_config(config_name, args.norm_stats_dir, args.repo_id)
        logging.info(f"{name}: {config_name} from {weights}")
        result["rows"][name] = run_row(name, model, row_module, store, args, device)
        write()
        _gtp.release(model, device)
        del model
    if args.out:
        logging.info(f"wrote {args.out}")
    return result


if __name__ == "__main__":
    import tyro

    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
