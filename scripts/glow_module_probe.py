"""Flow loss of a GLOW module model (pi05_libero_glow_*) under counterfactual module inputs.

Forward only, no gradients. Every variant sees the same frames, images, state, actions, flow
noise, flow time and image augmentation; only the planning-module arrays differ, so the paired
per-sample differences from `correct` measure what the actions read from each encoder input:

  correct              the labeled stage of the sample's own task
  correct_again        correct repeated: the run-to-run floor (nondeterministic kernels)
  other_task           every module input from another task of the same suite (the next one in
                       task-index order; same stage, or that task's last) - the goal is wrong
  other_task_spatial   only the Spatial inputs (subgoal, nouns, predicate, anchors) from other_task
  other_task_temporal  only the Temporal inputs (runtime keys, stage pointer, stage count) from
                       other_task
  wrong_stage          the other stage of the same task (a one-stage task keeps its stage) - the
                       Temporal pointer and the Spatial stage are wrong
  wrong_stage_spatial  only the Spatial inputs from wrong_stage (the pointer stays right)
  wrong_stage_pointer  only the stage pointer from wrong_stage (the Spatial stage stays right)
  grounding_other      only the anchors (glow_floats) from other_task - the grounding is wrong
  spatial_blank        subgoal, nouns, predicate and anchors removed
  temporal_blank       the runtime keys removed (stage pointer kept)

A model whose actions read an input has a clearly positive difference when it is wrong; one that
ignores it has ~0. The `_spatial` / `_temporal` / `_pointer` splits say which encoder carries a
task's or a stage's effect. Everything is also summarized on multi-stage samples only, and on
those by the sample's labeled stage (a wrong stage ahead of the scene vs behind it).

  python scripts/glow_module_probe.py --weights-dir <dir with model.safetensors> \
      --norm-stats-dir <assets dir holding physical-intelligence/libero/norm_stats.json> \
      --dataset-root <lerobot repo dir> --episodes 379:807 --out module_probe.json
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

VARIANTS = (
    "correct",
    "correct_again",
    "other_task",
    "other_task_spatial",
    "other_task_temporal",
    "wrong_stage",
    "wrong_stage_spatial",
    "wrong_stage_pointer",
    "grounding_other",
    "spatial_blank",
    "temporal_blank",
)
# LIBERO-10 is task indices 0-9 and LIBERO-Goal 10-19 in physical-intelligence/libero.
SUITE_SIZE = 10


@dataclasses.dataclass
class Args:
    weights_dir: str
    # Assets directory whose <repo_id>/norm_stats.json normalizes state and actions.
    norm_stats_dir: str
    # LeRobot dataset root: the directory holding meta/ and data/.
    dataset_root: str
    config: str = "pi05_libero_glow_st"
    repo_id: str = "physical-intelligence/libero"
    # Episodes, as OPENPI_TRAIN_EPISODES (LIBERO-Goal = 379:807).
    episodes: str = "0:807"
    num_samples: int = 512
    batch_size: int = 8
    seed: int = 0
    precision: str = "float32"
    device: str = "cuda"
    out: str | None = None


def other_task(pack, task_string: str):
    """The next task of the same suite in task-index order."""
    task = pack.task(task_string)
    suite = task.task_index // SUITE_SIZE
    same = sorted((t for t in pack.tasks.values() if t.task_index // SUITE_SIZE == suite), key=lambda t: t.task_index)
    return same[(same.index(task) + 1) % len(same)]


def splice(base: dict, donor: dict, part: str) -> dict:
    """`base`'s module arrays with one encoder's inputs taken from `donor`: "spatial" (subgoal,
    object and target text, predicate, anchors) or "temporal" (runtime keys, stage pointer, stage
    count)."""
    from openpi.glow import module_inputs as _mi

    if part == "spatial":
        segments = ("subgoal", "object", "target")
        ints = (_mi.INT_PREDICATE,)
    elif part == "temporal":
        segments = tuple(f"rk{j}" for j in range(_mi.K_MAX))
        ints = (_mi.INT_STAGE, _mi.INT_NUM_STAGES)
    else:
        raise ValueError(f"unknown part {part!r}")
    out = {k: v.copy() for k, v in base.items()}
    for name in segments:
        start, stop = _mi.TEXT_SEGMENTS[name]
        out["glow_text"][start:stop] = donor["glow_text"][start:stop]
        out["glow_text_mask"][start:stop] = donor["glow_text_mask"][start:stop]
    for i in ints:
        out["glow_ints"][i] = donor["glow_ints"][i]
    if part == "spatial":
        out["glow_floats"][:] = donor["glow_floats"]
    return out


def module_variants(encode, pack, task_string: str, plan_index: int, correct: dict) -> dict[str, dict]:
    """The module arrays of every variant for one sample (`correct` = its own arrays)."""
    from openpi.glow import module_inputs as _mi

    task = pack.task(task_string)
    other = other_task(pack, task_string)
    cross = encode(other.task, min(plan_index, other.num_plans - 1))
    swapped = encode(task_string, (plan_index + 1) % task.num_plans)

    pointer = {k: v.copy() for k, v in correct.items()}
    pointer["glow_ints"][_mi.INT_STAGE] = swapped["glow_ints"][_mi.INT_STAGE]

    spatial_blank = {k: v.copy() for k, v in correct.items()}
    for name in ("subgoal", "object", "target"):
        start, stop = _mi.TEXT_SEGMENTS[name]
        spatial_blank["glow_text"][start:stop] = 0
        spatial_blank["glow_text_mask"][start:stop] = False
    spatial_blank["glow_ints"][_mi.INT_PREDICATE] = 0
    spatial_blank["glow_floats"][:] = 0.0

    temporal_blank = {k: v.copy() for k, v in correct.items()}
    for j in range(_mi.K_MAX):
        start, stop = _mi.TEXT_SEGMENTS[f"rk{j}"]
        temporal_blank["glow_text"][start:stop] = 0
        temporal_blank["glow_text_mask"][start:stop] = False

    return {
        "correct": correct,
        "correct_again": correct,
        "other_task": cross,
        "other_task_spatial": splice(correct, cross, "spatial"),
        "other_task_temporal": splice(correct, cross, "temporal"),
        "wrong_stage": swapped,
        "wrong_stage_spatial": splice(correct, swapped, "spatial"),
        "wrong_stage_pointer": pointer,
        "grounding_other": {**correct, "glow_floats": cross["glow_floats"]},
        "spatial_blank": spatial_blank,
        "temporal_blank": temporal_blank,
    }


def main(args: Args) -> dict:
    import jax
    from lerobot.common.datasets import lerobot_dataset
    import safetensors.torch
    import torch

    from openpi import transforms as _transforms
    from openpi.glow import module_inputs as _mi
    from openpi.models import model as _model
    from openpi.training import config as _config
    from openpi.training import data_loader as _data_loader

    cfg = _config.get_config(args.config)
    assets = _config.AssetsConfig(assets_dir=args.norm_stats_dir, asset_id=args.repo_id)
    dc = dataclasses.replace(cfg.data, assets=assets).create(cfg.assets_dirs, cfg.model)
    if dc.norm_stats is None:
        raise ValueError(f"no norm_stats under {args.norm_stats_dir}/{args.repo_id}")
    model_inputs = list(dc.model_transforms.inputs)
    at_module = [i for i, t in enumerate(model_inputs) if isinstance(t, _mi.GlowModuleInputs)]
    if len(at_module) != 1:
        raise ValueError(f"expected one GlowModuleInputs in {args.config}'s model transforms, got {len(at_module)}")
    module = model_inputs[at_module[0]]

    episodes = _data_loader.resolve_train_episodes(args.episodes)
    meta = lerobot_dataset.LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    # The smallest 0..n-1 prefix holding the episodes (what the pinned LeRobot can load), as the
    # training loader does.
    raw = lerobot_dataset.LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        episodes=list(range(episodes[-1] + 1)),
        delta_timestamps={
            key: [t / meta.fps for t in range(cfg.model.action_horizon)] for key in dc.action_sequence_keys
        },
    )
    rows = _data_loader.episode_frame_indices(raw.episode_data_index, episodes)
    raw = _data_loader.TransformedDataset(raw, [_transforms.PromptFromLeRobotTask(meta.tasks)])
    pipeline = _transforms.compose(
        [
            *dc.repack_transforms.inputs,
            *dc.data_transforms.inputs,
            _transforms.Normalize(dc.norm_stats, use_quantiles=dc.use_quantile_norm),
            *dc.model_transforms.inputs,
        ]
    )

    rng = np.random.default_rng(args.seed)
    n = args.num_samples - args.num_samples % args.batch_size
    indices = rng.choice(rows, size=n, replace=False)

    device = torch.device(args.device)

    def _to_device(x):
        t = torch.as_tensor(x)
        # Normalize can promote state to float64 (numpy 2 promotion); the model computes in float32.
        return t.to(device, torch.float32) if t.dtype == torch.float64 else t.to(device)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model_cfg = cfg.model
    object.__setattr__(model_cfg, "dtype", args.precision)
    model = model_cfg.create_pytorch().to(device)
    safetensors.torch.load_model(model, str(pathlib.Path(args.weights_dir) / "model.safetensors"), device=str(device))
    model.eval()

    flows = {v: [] for v in VARIANTS}
    tasks, multistage, stages = [], [], []
    t0 = time.monotonic()
    for b in range(n // args.batch_size):
        idx = indices[b * args.batch_size : (b + 1) * args.batch_size]
        frames = [raw[int(i)] for i in idx]
        samples = []
        for f in frames:
            task_string = f["prompt"]
            ep, frame = int(np.asarray(f["episode_index"])), int(np.asarray(f["frame_index"]))
            k = module.pack.plan_index(ep, frame)
            sample = pipeline(dict(f))
            correct = {key: sample[key] for key in ("glow_text", "glow_text_mask", "glow_ints", "glow_floats")}
            samples.append((sample, module_variants(module.encode, module.pack, task_string, k, correct)))
            tasks.append(int(np.asarray(f["task_index"])))
            multistage.append(module.pack.task(task_string).num_plans > 1)
            stages.append(k)

        torch.manual_seed(args.seed * 100_003 + b)
        base = _flow_probe._collate([s for s, _ in samples])  # noqa: SLF001
        actions = torch.as_tensor(base["actions"]).to(device, torch.float32)
        noise = model.sample_noise(actions.shape, device)
        flow_time = model.sample_time(actions.shape[0], device)
        for name in VARIANTS:
            batch = _flow_probe._collate([{**s, **v[name]} for s, v in samples])  # noqa: SLF001
            obs = _model.Observation.from_dict(jax.tree.map(_to_device, batch))
            torch.manual_seed(args.seed * 100_003 + b)  # identical image augmentation
            with torch.no_grad():
                per_element = model.forward(obs, actions, noise=noise, time=flow_time)
            flows[name].append(per_element.float().mean(dim=(1, 2)).cpu().numpy())
        if b % 8 == 0:
            done = (b + 1) * args.batch_size
            means = {k: float(np.concatenate(v).mean()) for k, v in flows.items()}
            logging.info(f"{done}/{n} samples ({done / (time.monotonic() - t0):.1f}/s) running means {means}")

    flows = {k: np.concatenate(v) for k, v in flows.items()}
    tasks = np.asarray(tasks)
    multistage = np.asarray(multistage, dtype=bool)
    stages = np.asarray(stages)

    def subset(sel):
        return (
            _flow_probe.paired_summary({k: v[sel] for k, v in flows.items()}, tasks[sel], "correct")
            if sel.sum() > 1
            else None
        )

    summary = {
        "vs_correct": _flow_probe.paired_summary(flows, tasks, "correct"),
        "vs_correct_multistage": subset(multistage),
        "vs_correct_multistage_by_stage": {int(k): subset(multistage & (stages == k)) for k in np.unique(stages)},
        "args": dataclasses.asdict(args),
    }
    overall = summary["vs_correct"]["overall"]
    by_stage = {k: s for k, s in summary["vs_correct_multistage_by_stage"].items() if s is not None}
    counts = ", ".join(f"{k}: {s['samples']}" for k, s in by_stage.items())
    logging.info(f"{len(tasks)} samples, {int(multistage.sum())} from multi-stage tasks (by labeled stage: {counts})")
    for name in flows:
        o = overall[name]
        line = f"FLOW {name:19s} {o['flow']:.5f}  x{o['ratio']:.3f}  diff {o['diff']:+.5f} +- {o['diff_se']:.5f}"
        if summary["vs_correct_multistage"] is not None:
            m = summary["vs_correct_multistage"]["overall"][name]
            line += f" | multi-stage x{m['ratio']:.3f} diff {m['diff']:+.5f} +- {m['diff_se']:.5f}"
        for k, s in by_stage.items():
            line += f" | stage {k} x{s['overall'][name]['ratio']:.3f}"
        logging.info(line)
    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **summary,
            "per_sample": {k: v.tolist() for k, v in flows.items()},
            "indices": indices.tolist(),
            "stages": stages.tolist(),
        }
        out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        logging.info(f"wrote {out}")
    return summary


if __name__ == "__main__":
    import tyro

    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
