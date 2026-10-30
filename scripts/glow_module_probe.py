"""Flow loss of a GLOW module model (pi05_libero_glow_*) under counterfactual module inputs.

Forward only, no gradients. Every variant sees the same frames, images, state, actions, flow
noise, flow time and image augmentation; only the planning-module arrays differ, so the paired
per-sample differences from `correct` measure what the actions read from each encoder input:

  correct              the sample's own plan with its labeled stage active
  correct_again        correct repeated: the run-to-run floor (nondeterministic kernels)
  other_task           every module input from another task of the same suite (the next one in
                       task-index order; same stage, or that task's last) - the goal is wrong
  other_task_spatial   only the Spatial inputs (atoms, subgoals, nouns, predicates, anchors, slot
                       order) from other_task
  other_task_temporal  only the Temporal inputs (runtime keys, stage pointer, stage count) from
                       other_task
  wrong_stage          the other stage of the same task active (a one-stage task keeps its stage)
                       - the Temporal pointer is wrong; the Spatial set is the same plan
  wrong_stage_pointer  only the pointer of wrong_stage (the same as wrong_stage when the Spatial
                       groups do not depend on the pointer; with the "remaining" order they do)
  grounding_other      only the anchors (glow_floats) from other_task - the grounding is wrong
  grounding_mean       the per-task anchors instead of the sample's own (its episode's, with
                       episode_grounding; otherwise identical to correct)
  spatial_blank        atoms, subgoals, nouns, predicates, head words and anchors removed
  temporal_blank       the runtime keys removed (stage pointer kept)
  clock_unknown        the stage clock unknown (-1)
Every variant keeps the sample's stage clock, and wrong_stage its anchors (the same task's).

Chain variants, for samples of a one-stage task that has chain partners (module_inputs
.chain_partners over the suite's one-stage tasks; the partner is the first in task-index order
after the sample's task); other samples repeat `correct` and are left out of their summary:

  chain_correct        the sample's stage first, the partner's second, the pointer at 0
  chain_second         the partner's stage first ("done"), the sample's second, the pointer at 1 -
                       the same active stage: a model that reads the pointer stays near chain_correct
  chain_wrong_pointer  chain_correct with the pointer at 1 (the partner's stage active): a model
                       that reads the pointer acts on the partner's goal - clearly positive
  chain_temporal_blank chain_correct with the runtime keys removed
  chain_partner_blank  chain_correct with the partner's Spatial text, predicate and anchors removed

A model whose actions read an input has a clearly positive difference when it is wrong; one that
ignores it has ~0. The `_spatial` / `_temporal` splits say which encoder carries a task's
effect. Everything is also summarized on multi-stage samples only, and on those by the sample's
labeled stage (a wrong stage ahead of the scene vs behind it); the chain variants against
chain_correct on the chainable samples.

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
    "wrong_stage_pointer",
    "grounding_other",
    "grounding_mean",
    "spatial_blank",
    "temporal_blank",
    "clock_unknown",
    "chain_correct",
    "chain_second",
    "chain_wrong_pointer",
    "chain_temporal_blank",
    "chain_partner_blank",
)
CHAIN_VARIANTS = tuple(v for v in VARIANTS if v.startswith("chain_"))
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


def _segments(fields, stages=None):
    """The text segment names of `fields` over `stages` (default: every stage position)."""
    from openpi.glow import module_inputs as _mi

    return [f"{f}{j}" for j in (range(_mi.K_MAX) if stages is None else stages) for f in fields]


def _copy(arrays: dict) -> dict:
    return {k: v.copy() for k, v in arrays.items()}


def _blank_text(arrays: dict, names) -> None:
    from openpi.glow import module_inputs as _mi

    for name in names:
        start, stop = _mi.TEXT_SEGMENTS[name]
        arrays["glow_text"][start:stop] = 0
        arrays["glow_text_mask"][start:stop] = False


def splice(base: dict, donor: dict, part: str) -> dict:
    """`base`'s module arrays with one encoder's inputs taken from `donor`: "spatial" (atoms,
    subgoals, object and target text, predicates, anchors, slot order, shown groups, Spatial head
    words) or "temporal" (runtime keys, stage pointer, stage count, stage clock)."""
    from openpi.glow import module_inputs as _mi

    if part == "spatial":
        segments = _segments(_mi.SPATIAL_FIELDS)
        ints = (_mi.INT_PREDICATE, _mi.INT_SPATIAL_SLOT, _mi.INT_SHOWN, *_spatial_heads(range(_mi.K_MAX)))
    elif part == "temporal":
        segments = _segments(_mi.TEMPORAL_FIELDS)
        ints = (_mi.INT_STAGE, _mi.INT_NUM_STAGES, _mi.INT_STAGE_FRAMES)
    else:
        raise ValueError(f"unknown part {part!r}")
    out = _copy(base)
    for name in segments:
        start, stop = _mi.TEXT_SEGMENTS[name]
        out["glow_text"][start:stop] = donor["glow_text"][start:stop]
        out["glow_text_mask"][start:stop] = donor["glow_text_mask"][start:stop]
    for i in ints:
        out["glow_ints"][i] = donor["glow_ints"][i]
    if part == "spatial":
        out["glow_floats"][:] = donor["glow_floats"]
    return out


def _spatial_heads(stages) -> list[int]:
    """glow_ints positions of the Spatial head words (verb, nouns, predicate word) of `stages`."""
    from openpi.glow import module_inputs as _mi

    return [_mi.head_index(j, f) for j in stages for f in ("op", "object", "target", "atom")]


def spatial_blank(arrays: dict, stages=None) -> dict:
    """`arrays` with the Spatial text, predicates and anchors of `stages` (default: all) removed."""
    from openpi.glow import module_inputs as _mi

    out = _copy(arrays)
    stages = list(range(_mi.K_MAX)) if stages is None else list(stages)
    _blank_text(out, _segments(_mi.SPATIAL_FIELDS, stages))
    for j in stages:
        out["glow_ints"][_mi.INT_PREDICATE.start + j] = 0
        out["glow_floats"][j * _mi.STAGE_FLOATS : (j + 1) * _mi.STAGE_FLOATS] = 0.0
    out["glow_ints"][_spatial_heads(stages)] = 0
    return out


def temporal_blank(arrays: dict) -> dict:
    """`arrays` with the runtime keys removed (the stage pointer and count kept)."""
    from openpi.glow import module_inputs as _mi

    out = _copy(arrays)
    _blank_text(out, _segments(_mi.TEMPORAL_FIELDS))
    return out


def with_pointer(arrays: dict, stage: int) -> dict:
    from openpi.glow import module_inputs as _mi

    out = _copy(arrays)
    out["glow_ints"][_mi.INT_STAGE] = stage
    return out


def chain_partner(pack, task_string: str):
    """The sample's chain partner: the first of its partners (module_inputs.chain_partners over
    the suite's one-stage tasks) after it in task-index order, cyclically; None without one."""
    from openpi.glow import module_inputs as _mi

    task = pack.task(task_string)
    suite = task.task_index // SUITE_SIZE
    ones = [t.task_index for t in pack.tasks.values() if t.task_index // SUITE_SIZE == suite and t.num_plans == 1]
    partners = _mi.chain_partners(pack, ones).get(task.task_index, ())
    if not partners:
        return None
    later = [p for p in partners if p.task_index > task.task_index]
    return (later or list(partners))[0]


def module_variants(module, task_string: str, plan_index: int, correct: dict) -> dict[str, dict]:
    """The module arrays of every variant for one sample (`correct` = its own arrays); `module`
    is the GlowModuleInputs transform (its encode / encode_stages and pack)."""
    from openpi.glow import module_inputs as _mi

    pack = module.pack
    task = pack.task(task_string)
    other = other_task(pack, task_string)
    cross = module.encode(other.task, min(plan_index, other.num_plans - 1))
    swapped = module.encode(task_string, (plan_index + 1) % task.num_plans)
    swapped["glow_floats"] = correct["glow_floats"].copy()
    clock = int(correct["glow_ints"][_mi.INT_STAGE_FRAMES])
    unknown = _copy(correct)
    unknown["glow_ints"][_mi.INT_STAGE_FRAMES] = -1

    out = {
        "correct": correct,
        "correct_again": correct,
        "other_task": cross,
        "other_task_spatial": splice(correct, cross, "spatial"),
        "other_task_temporal": splice(correct, cross, "temporal"),
        "wrong_stage": swapped,
        "wrong_stage_pointer": with_pointer(correct, int(swapped["glow_ints"][_mi.INT_STAGE])),
        "grounding_other": {**correct, "glow_floats": cross["glow_floats"]},
        "grounding_mean": {**correct, "glow_floats": module.encode(task_string, plan_index)["glow_floats"]},
        "spatial_blank": spatial_blank(correct),
        "temporal_blank": temporal_blank(correct),
        "clock_unknown": unknown,
    }
    partner = chain_partner(pack, task_string) if task.num_plans == 1 else None
    if partner is None:
        out.update(dict.fromkeys(CHAIN_VARIANTS, correct))
    else:
        first = module.encode_stages((task.stages[0], partner.stages[0]), 0)
        out |= {
            "chain_correct": first,
            "chain_second": module.encode_stages((partner.stages[0], task.stages[0]), 1),
            "chain_wrong_pointer": with_pointer(first, 1),
            "chain_temporal_blank": temporal_blank(first),
            "chain_partner_blank": spatial_blank(first, stages=[1]),
        }
    for name, arrays in out.items():
        if name != "clock_unknown" and arrays is not correct and arrays["glow_ints"][_mi.INT_STAGE_FRAMES] != clock:
            out[name] = {**arrays, "glow_ints": arrays["glow_ints"].copy()}
            out[name]["glow_ints"][_mi.INT_STAGE_FRAMES] = clock
    return out


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

    # The sample's own plan, never a chain: the probe builds its chains itself.
    module = dataclasses.replace(module, chain_p=0.0, chain_tasks=())
    flows = {v: [] for v in VARIANTS}
    tasks, multistage, stages, chainable = [], [], [], []
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
            variants = module_variants(module, task_string, k, correct)
            samples.append((sample, variants))
            tasks.append(int(np.asarray(f["task_index"])))
            multistage.append(module.pack.task(task_string).num_plans > 1)
            stages.append(k)
            chainable.append(variants["chain_correct"] is not correct)

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
    chainable = np.asarray(chainable, dtype=bool)

    def subset(sel, baseline="correct", names=None):
        picked = {k: v[sel] for k, v in flows.items() if names is None or k in names}
        return _flow_probe.paired_summary(picked, tasks[sel], baseline) if sel.sum() > 1 else None

    chain_names = ("correct", *CHAIN_VARIANTS)
    summary = {
        "vs_correct": _flow_probe.paired_summary(flows, tasks, "correct"),
        "vs_correct_multistage": subset(multistage),
        "vs_correct_multistage_by_stage": {int(k): subset(multistage & (stages == k)) for k in np.unique(stages)},
        "vs_chain_correct_chainable": subset(chainable, "chain_correct", chain_names),
        "args": dataclasses.asdict(args),
    }
    overall = summary["vs_correct"]["overall"]
    by_stage = {k: s for k, s in summary["vs_correct_multistage_by_stage"].items() if s is not None}
    counts = ", ".join(f"{k}: {s['samples']}" for k, s in by_stage.items())
    logging.info(
        f"{len(tasks)} samples, {int(multistage.sum())} from multi-stage tasks (by labeled stage: {counts}), "
        f"{int(chainable.sum())} chainable"
    )
    for name in flows:
        o = overall[name]
        line = f"FLOW {name:20s} {o['flow']:.5f}  x{o['ratio']:.3f}  diff {o['diff']:+.5f} +- {o['diff_se']:.5f}"
        if summary["vs_correct_multistage"] is not None:
            m = summary["vs_correct_multistage"]["overall"][name]
            line += f" | multi-stage x{m['ratio']:.3f} diff {m['diff']:+.5f} +- {m['diff_se']:.5f}"
        for k, s in by_stage.items():
            line += f" | stage {k} x{s['overall'][name]['ratio']:.3f}"
        logging.info(line)
    if summary["vs_chain_correct_chainable"] is not None:
        for name in chain_names:
            c = summary["vs_chain_correct_chainable"]["overall"][name]
            logging.info(
                f"CHAIN {name:20s} {c['flow']:.5f}  x{c['ratio']:.3f} vs chain_correct  diff {c['diff']:+.5f} "
                f"+- {c['diff_se']:.5f}  (chainable samples)"
            )
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
