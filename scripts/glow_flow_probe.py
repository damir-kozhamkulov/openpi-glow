"""Flow loss of one set of weights under prompt-layout variants, on identical training samples.

Forward only, no gradients. Every variant sees the same frames, images, state, actions, flow
noise, flow time and image augmentation; only the language tokens and their masks differ, so the
per-sample differences isolate what the prompt layout does to the action expert.

  stock            pi05_libero's own prompt: "{task}" + "\\n"
  stock_again      stock repeated: the run-to-run floor (nondeterministic kernels)
  was              "{task}. Was: {plan t-delta}" + "\\n", no plan block
  unified          + causal "Now: {plan t}" eos block (run3a's layout, input uncorrupted)
  unified_bidir    the same tokens with the block bidirectional (ar mask 0)
  unified_noeos    the block without its eos token
  unified_corrupt  input plan always replaced by another stage of the task (corrupt_p 1)

Counterfactual plans (the plan-sensitivity of trained weights; "unified" is the correct plan):

  now_swapped      the "Now:" block names the other stage of the task (a one-stage task keeps its
                   stage), "Was:" as in unified
  both_swapped     "Was:" and "Now:" both name the other stage: what forced mode sends when it is wrong
  now_cross_task   the "Now:" block is a different task's plan (the next task in plan-pack order, same
                   stage or its last), task string and "Was:" unchanged

Reports each variant's mean flow loss and its paired difference, with a standard error, from
`stock` and from `unified`, overall and per task, and from `unified` on multi-stage samples only
(where now_swapped / both_swapped differ from it). A model whose actions read the plan has a
clearly positive difference for the counterfactual plans; one that ignores the plan has ~0.

  python scripts/glow_flow_probe.py --weights-dir <dir with model.safetensors> \
      --norm-stats-dir <assets dir holding physical-intelligence/libero/norm_stats.json> \
      --dataset-root <lerobot repo dir> --out flow_probe.json
"""

import dataclasses
import json
import logging
import pathlib
import time

import numpy as np

VARIANTS = (
    "stock",
    "stock_again",
    "was",
    "unified",
    "unified_bidir",
    "unified_noeos",
    "unified_corrupt",
    "now_swapped",
    "both_swapped",
    "now_cross_task",
)
COUNTERFACTUAL_MODES = ("now_swapped", "both_swapped", "now_cross_task")


@dataclasses.dataclass
class Args:
    weights_dir: str
    # Assets directory whose <repo_id>/norm_stats.json normalizes state and actions.
    norm_stats_dir: str
    # LeRobot dataset root: the directory holding meta/ and data/.
    dataset_root: str
    # Further checkpoint directories whose weights are averaged uniformly with weights_dir's in
    # memory (a checkpoint average, as SWA/EMA do over time). Empty: weights_dir alone.
    average_with: tuple[str, ...] = ()
    config: str = "pi05_libero_glow_unified"
    stock_config: str = "pi05_libero"
    repo_id: str = "physical-intelligence/libero"
    # Episode range "start:stop" (LIBERO-10 = 0:379).
    episodes: str = "0:379"
    num_samples: int = 512
    batch_size: int = 8
    seed: int = 0
    # Variants to run (default: all); stock and unified always run, the differences are taken from them.
    variants: tuple[str, ...] = VARIANTS
    precision: str = "float32"
    device: str = "cuda"
    out: str | None = None


def block_variants(batch: dict) -> dict[str, dict]:
    """The token-level variants of one unified batch (numpy arrays, leading batch axis)."""
    loss = batch["token_loss_mask"].astype(bool)
    was = {**batch, "tokenized_prompt_mask": batch["tokenized_prompt_mask"] & ~loss}
    was["token_ar_mask"] = np.zeros_like(batch["token_ar_mask"])
    was["token_loss_mask"] = np.zeros_like(loss)
    bidir = {**batch, "token_ar_mask": np.zeros_like(batch["token_ar_mask"])}
    last = loss.shape[1] - 1 - np.argmax(loss[:, ::-1], axis=1)
    eos = np.zeros_like(loss)
    eos[np.arange(len(last)), last] = loss.any(axis=1)
    noeos = {
        **batch,
        "tokenized_prompt_mask": batch["tokenized_prompt_mask"] & ~eos,
        "token_ar_mask": np.where(eos, 0, batch["token_ar_mask"]),
        "token_loss_mask": loss & ~eos,
    }
    return {"was": was, "unified": batch, "unified_bidir": bidir, "unified_noeos": noeos}


@dataclasses.dataclass(frozen=True)
class ProbePlanState:
    """GlowPlanState for training samples with a counterfactual plan (see COUNTERFACTUAL_MODES).

    The input plan is the label `delta` frames earlier, never corrupted; the mode then replaces
    the "Now:" plan, or both plans, as the module docstring describes.
    """

    pack: object
    delta: int
    mode: str
    nouns: bool = True
    anchors: bool = True

    def __call__(self, data: dict) -> dict:
        from openpi.glow import plan_prompt as _plan_prompt

        if self.mode not in COUNTERFACTUAL_MODES:
            raise ValueError(f"unknown probe mode {self.mode!r}")
        prompt = data["prompt"] if isinstance(data["prompt"], str) else np.asarray(data["prompt"]).item()
        episode = int(np.asarray(data.pop("episode_index")).item())
        frame = int(np.asarray(data.pop("frame_index")).item())
        task = self.pack.task(prompt)
        now = self.pack.plan_index(episode, frame)
        prev = self.pack.plan_index(episode, max(0, frame - self.delta))
        other = (now + 1) % task.num_plans

        def render(p, k):
            return self.pack.render_plan(p, k, nouns=self.nouns, anchors=self.anchors)

        if self.mode == "now_swapped":
            target = render(prompt, other)
        elif self.mode == "both_swapped":
            prev = other
            target = render(prompt, other)
        else:
            ordered = sorted(self.pack.tasks.values(), key=lambda t: t.task_index)
            cross = ordered[(ordered.index(task) + 1) % len(ordered)]
            target = render(cross.task, min(now, cross.num_plans - 1))
        data["glow_plan_target"] = _plan_prompt.TARGET_PREFIX + target
        data["prompt"] = f"{prompt}. {_plan_prompt.INPUT_PREFIX}{render(prompt, prev)}"
        return data


def paired_summary(flows: dict[str, np.ndarray], tasks: np.ndarray, reference: str = "stock") -> dict:
    """Mean flow per variant and its paired difference from `reference` (mean, standard error)."""

    def one(sel):
        base = flows[reference][sel]
        out = {}
        for name, f in flows.items():
            d = f[sel] - base
            out[name] = {
                "flow": float(f[sel].mean()),
                "ratio": float(f[sel].mean() / base.mean()),
                "diff": float(d.mean()),
                "diff_se": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan"),
            }
        return out

    every = np.ones(len(tasks), dtype=bool)
    return {
        "reference": reference,
        "samples": len(tasks),
        "overall": one(every),
        "per_task": {int(t): {"samples": int((tasks == t).sum()), **one(tasks == t)} for t in np.unique(tasks)},
    }


def _collate(items):
    import jax

    # As data_loader._collate_fn.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def main(args: Args) -> dict:
    import jax
    from lerobot.common.datasets import lerobot_dataset
    import safetensors
    import safetensors.torch
    import torch

    from openpi.glow import plan_prompt as _plan_prompt
    from openpi.models import model as _model
    from openpi.training import config as _config
    from openpi.training import data_loader as _data_loader
    from openpi import transforms as _transforms

    cfg = _config.get_config(args.config)
    stock_cfg = _config.get_config(args.stock_config)
    if cfg.model.action_horizon != stock_cfg.model.action_horizon:
        raise ValueError("the two configs differ in action_horizon")
    assets = _config.AssetsConfig(assets_dir=args.norm_stats_dir, asset_id=args.repo_id)

    def data_config(c, **kw):
        factory = dataclasses.replace(c.data, assets=assets, **kw)
        return factory.create(c.assets_dirs, c.model)

    unknown = set(args.variants) - set(VARIANTS)
    if unknown:
        raise ValueError(f"unknown variants {sorted(unknown)}; choose from {VARIANTS}")
    variants_run = [v for v in VARIANTS if v in {"stock", "unified", *args.variants}]

    pipelines = {
        "stock": data_config(stock_cfg),
        "unified": data_config(cfg, plan_corrupt_p=0.0),
    }
    if "unified_corrupt" in variants_run:
        pipelines["unified_corrupt"] = data_config(cfg, plan_corrupt_p=1.0)
    # The counterfactual pipelines: the unified one with its plan-state transform swapped out.
    unified_dc = pipelines["unified"]
    model_inputs = list(unified_dc.model_transforms.inputs)
    at_plan = [i for i, t in enumerate(model_inputs) if isinstance(t, _plan_prompt.GlowPlanState)]
    if len(at_plan) != 1:
        raise ValueError(f"expected one GlowPlanState in {args.config}'s model transforms, got {len(at_plan)}")
    plan_state = model_inputs[at_plan[0]]
    for mode in COUNTERFACTUAL_MODES:
        if mode not in variants_run:
            continue
        swapped = ProbePlanState(
            plan_state.pack, plan_state.delta, mode, nouns=plan_state.nouns, anchors=plan_state.anchors
        )
        inputs = [*model_inputs[: at_plan[0]], swapped, *model_inputs[at_plan[0] + 1 :]]
        pipelines[mode] = dataclasses.replace(
            unified_dc, model_transforms=dataclasses.replace(unified_dc.model_transforms, inputs=inputs)
        )
    for name, dc in pipelines.items():
        if dc.norm_stats is None:
            raise ValueError(f"{name}: no norm_stats under {args.norm_stats_dir}/{args.repo_id}")

    start, stop = (int(x) for x in args.episodes.split(":"))
    meta = lerobot_dataset.LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    raw = lerobot_dataset.LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        episodes=list(range(start, stop)),
        delta_timestamps={
            key: [t / meta.fps for t in range(cfg.model.action_horizon)]
            for key in pipelines["unified"].action_sequence_keys
        },
    )
    raw = _data_loader.TransformedDataset(raw, [_transforms.PromptFromLeRobotTask(meta.tasks)])
    # The training sample transforms (data_loader.transform_dataset's list), applied per frame so
    # each frame is decoded once and fed to every pipeline.
    pipeline_fns = {
        name: _transforms.compose(
            [
                *dc.repack_transforms.inputs,
                *dc.data_transforms.inputs,
                _transforms.Normalize(dc.norm_stats, use_quantiles=dc.use_quantile_norm),
                *dc.model_transforms.inputs,
            ]
        )
        for name, dc in pipelines.items()
    }

    rng = np.random.default_rng(args.seed)
    n = args.num_samples - args.num_samples % args.batch_size
    indices = rng.choice(len(raw), size=n, replace=False)

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
    if args.average_with:
        # Every name of every parameter, tied aliases included: save_model keeps one name per
        # shared tensor, so a file key may be any alias. Each tensor is summed once.
        by_name = dict(model.named_parameters(remove_duplicate=False))
        unique = {p.data_ptr(): p for p in by_name.values()}
        with torch.no_grad():
            for other_dir in args.average_with:
                summed = set()
                # One tensor at a time, from CPU: a second full copy never sits on the GPU.
                with safetensors.safe_open(str(pathlib.Path(other_dir) / "model.safetensors"), "pt") as f:
                    for key in f.keys():  # noqa: SIM118
                        p = by_name.get(key)
                        if p is None or p.data_ptr() in summed:
                            continue
                        p.add_(f.get_tensor(key).to(device=p.device, dtype=p.dtype))
                        summed.add(p.data_ptr())
                if summed != set(unique):
                    raise ValueError(f"{other_dir}: {len(set(unique) - summed)} parameters not found in its file")
            for p in unique.values():
                p.div_(1 + len(args.average_with))
        logging.info(f"averaged weights of {1 + len(args.average_with)} checkpoints")
    model.eval()
    # The per-element flow tensor: lambda 0 returns it whatever the plan block.
    model.subtask_ce_weight = 0.0

    flows = {v: [] for v in variants_run}
    tasks = []
    multistage = []
    t0 = time.monotonic()
    for b in range(n // args.batch_size):
        idx = indices[b * args.batch_size : (b + 1) * args.batch_size]
        frames = [raw[int(i)] for i in idx]
        tasks.extend(int(np.asarray(f["task_index"])) for f in frames)
        multistage.extend(plan_state.pack.task(f["prompt"]).num_plans > 1 for f in frames)
        batches = {name: _collate([fn(dict(f)) for f in frames]) for name, fn in pipeline_fns.items()}
        # Same frames through every pipeline: everything but the language tokens must agree.
        for name in batches:
            for key in ("state", "actions"):
                np.testing.assert_array_equal(batches[name][key], batches["stock"][key])
            for cam, img in batches["stock"]["image"].items():
                np.testing.assert_array_equal(batches[name]["image"][cam], img)
        variants = {
            "stock_again": batches["stock"],
            **block_variants(batches["unified"]),
            **batches,
        }

        # The model's own noise and time distributions, drawn once per batch.
        torch.manual_seed(args.seed * 100_003 + b)
        actions_shape = batches["stock"]["actions"].shape
        noise = model.sample_noise(actions_shape, device)
        flow_time = model.sample_time(actions_shape[0], device)
        for name in variants_run:
            obs = _model.Observation.from_dict(jax.tree.map(_to_device, variants[name]))
            actions = torch.as_tensor(variants[name]["actions"]).to(device, torch.float32)
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
    summary = {
        "vs_stock": paired_summary(flows, tasks, "stock"),
        "vs_unified": paired_summary(flows, tasks, "unified"),
        "vs_unified_multistage": paired_summary(
            {k: v[multistage] for k, v in flows.items()}, tasks[multistage], "unified"
        ),
        "args": dataclasses.asdict(args),
    }
    vs_stock, vs_unified = summary["vs_stock"]["overall"], summary["vs_unified"]["overall"]
    vs_multi = summary["vs_unified_multistage"]["overall"]
    logging.info(f"{len(tasks)} samples, {int(multistage.sum())} from multi-stage tasks")
    for name in flows:
        s, u, m = vs_stock[name], vs_unified[name], vs_multi[name]
        logging.info(
            f"FLOW {name:16s} {s['flow']:.5f}  x{s['ratio']:.2f} stock  diff {s['diff']:+.5f} +- {s['diff_se']:.5f}"
            f" | vs unified {u['diff']:+.5f} +- {u['diff_se']:.5f}"
            f" | multi-stage vs unified {m['diff']:+.5f} +- {m['diff_se']:.5f}"
        )
    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {**summary, "per_sample": {k: v.tolist() for k, v in flows.items()}, "indices": indices.tolist()}
        out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        logging.info(f"wrote {out}")
    return summary


if __name__ == "__main__":
    import tyro

    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
