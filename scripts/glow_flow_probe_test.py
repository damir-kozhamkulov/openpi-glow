"""glow_flow_probe: the token variants, the paired summary, and main() end to end.

The end-to-end test needs the local LeRobot copy (HF_LEROBOT_HOME with repo `libero_staging`),
the plan pack and pi05_libero's norm_stats; skipped otherwise. It runs a dummy-size GLOW model
with random weights, so it checks the wiring, not the numbers.
"""

import os
import pathlib

import numpy as np
import pytest

from . import glow_flow_probe as gfp

REPO = pathlib.Path(__file__).resolve().parents[2]  # glow_repo (openpi is its submodule)
PACK = REPO / "derived/plan_pack.json"
DATASET = pathlib.Path(os.environ.get("HF_LEROBOT_HOME", "")) / "libero_staging"
ASSETS = REPO / "data/assets/pi05_libero"
NORM_STATS = ASSETS / "physical-intelligence/libero/norm_stats.json"


def _batch():
    # Two rows: 3 prompt tokens + a 3-token block (last = eos), then padding.
    tokens = np.array([[2, 5, 6, 7, 8, 1, 0, 0], [2, 5, 7, 8, 1, 0, 0, 0]])
    mask = tokens != 0
    loss = np.zeros_like(mask)
    loss[0, 3:6] = True
    loss[1, 2:5] = True
    return {
        "tokenized_prompt": tokens,
        "tokenized_prompt_mask": mask,
        "token_ar_mask": loss.astype(np.int32),
        "token_loss_mask": loss,
        "state": np.zeros((2, 8)),
    }


def test_block_variants():
    batch = _batch()
    v = gfp.block_variants(batch)
    assert v["unified"] is batch
    # was: the block is gone from the attention (pad mask), leaving the prompt alone.
    assert v["was"]["tokenized_prompt_mask"].tolist() == [[1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0]]
    assert not v["was"]["token_ar_mask"].any() and not v["was"]["token_loss_mask"].any()
    # bidir: same tokens visible, no causal segment.
    assert np.array_equal(v["unified_bidir"]["tokenized_prompt_mask"], batch["tokenized_prompt_mask"])
    assert not v["unified_bidir"]["token_ar_mask"].any()
    # noeos: exactly the last block token of each row is dropped.
    dropped = batch["tokenized_prompt_mask"] & ~v["unified_noeos"]["tokenized_prompt_mask"]
    assert np.argwhere(dropped).tolist() == [[0, 5], [1, 4]]
    assert v["unified_noeos"]["token_loss_mask"].sum() == batch["token_loss_mask"].sum() - 2
    # The input batch is untouched.
    assert batch["tokenized_prompt_mask"].sum() == 11 and batch["token_ar_mask"].sum() == 6


def test_paired_summary():
    flows = {"stock": np.array([1.0, 2.0, 3.0, 4.0]), "unified": np.array([2.0, 4.0, 6.0, 8.0])}
    s = gfp.paired_summary(flows, np.array([0, 0, 1, 1]))
    assert s["samples"] == 4 and s["reference"] == "stock"
    assert s["overall"]["stock"]["diff"] == 0.0 and s["overall"]["unified"]["ratio"] == 2.0
    assert s["overall"]["unified"]["diff"] == 2.5
    assert s["per_task"][1]["samples"] == 2 and s["per_task"][1]["unified"]["flow"] == 7.0
    u = gfp.paired_summary(flows, np.array([0, 0, 1, 1]), "unified")
    assert u["overall"]["unified"]["diff"] == 0.0 and u["overall"]["stock"]["diff"] == -2.5


@pytest.mark.skipif(not PACK.exists(), reason="plan pack not present")
def test_probe_plan_state():
    from openpi.glow import plan_pack as _plan_pack

    pack = _plan_pack.PlanPack.load(PACK)
    ep = next(e for e, p in sorted(pack.episodes.items()) if len(p.stage_starts) == 2)
    episode = pack.episodes[ep]
    task = next(t for t in pack.tasks.values() if t.task_index == episode.task_index)
    frame = episode.stage_starts[1] + 20  # well inside stage 2, and 20 > delta: input is stage 2 too

    def run(mode):
        data = {"prompt": task.task, "episode_index": np.asarray(ep), "frame_index": np.asarray(frame)}
        return gfp.ProbePlanState(pack, 5, mode)(data)

    stage1, stage2 = (pack.render_plan(task.task, k) for k in (0, 1))
    out = run("now_swapped")
    assert out["glow_plan_target"] == "Now: " + stage1 and out["prompt"] == f"{task.task}. Was: {stage2}"
    assert "episode_index" not in out and "frame_index" not in out
    out = run("both_swapped")
    assert out["glow_plan_target"] == "Now: " + stage1 and out["prompt"] == f"{task.task}. Was: {stage1}"
    out = run("now_cross_task")
    ordered = sorted(pack.tasks.values(), key=lambda t: t.task_index)
    cross = ordered[(ordered.index(task) + 1) % len(ordered)]
    assert out["glow_plan_target"] == "Now: " + pack.render_plan(cross.task, min(1, cross.num_plans - 1))
    assert out["prompt"] == f"{task.task}. Was: {stage2}"
    with pytest.raises(ValueError):
        run("unified")


@pytest.mark.skipif(
    not ((DATASET / "meta/info.json").exists() and PACK.exists() and NORM_STATS.exists()),
    reason="local LeRobot copy, plan pack or norm_stats not present",
)
def test_main_end_to_end(monkeypatch, tmp_path):
    import dataclasses
    import json

    import safetensors.torch
    import torch
    from transformers.models.auto import CONFIG_MAPPING

    import openpi.models_pytorch.gemma_pytorch as gemma_pytorch
    import openpi.shared.download as download
    import openpi.training.config as _config

    orig = CONFIG_MAPPING["paligemma"]

    def tiny():
        cfg = orig()
        cfg.vision_config.hidden_size = 32
        cfg.vision_config.num_hidden_layers = 2
        cfg.vision_config.num_attention_heads = 4
        cfg.vision_config.patch_size = 56
        return cfg

    monkeypatch.setattr(gemma_pytorch, "CONFIG_MAPPING", {**CONFIG_MAPPING, "paligemma": tiny})
    monkeypatch.setenv("OPENPI_PLAN_PACK", str(PACK))
    # Windows paths read as URL schemes ("w:"); the cluster never takes this path.
    real_download = download.maybe_download
    monkeypatch.setattr(
        download, "maybe_download", lambda p, **k: real_download(p, **k) if "://" in str(p) else pathlib.Path(p)
    )
    real_get = _config.get_config

    def get_config(name):
        c = real_get(name)
        return dataclasses.replace(
            c,
            model=dataclasses.replace(
                c.model, paligemma_variant="dummy", action_expert_variant="dummy", pytorch_compile_mode=None
            ),
        )

    monkeypatch.setattr(_config, "get_config", get_config)
    weights = tmp_path / "weights"
    weights.mkdir()
    torch.manual_seed(0)
    safetensors.torch.save_model(get_config("pi05_libero_glow_unified").model.create_pytorch(), str(weights / "model.safetensors"))

    out = tmp_path / "probe.json"
    s = gfp.main(
        gfp.Args(
            weights_dir=str(weights),
            norm_stats_dir=str(ASSETS),
            dataset_root=str(DATASET),
            episodes="0:20",
            num_samples=12,
            batch_size=4,
            device="cpu",
            out=str(out),
        )
    )
    o = s["vs_stock"]["overall"]
    assert s["vs_stock"]["samples"] == 12 and set(o) == set(gfp.VARIANTS)
    assert o["stock"]["diff"] == 0.0 and all(np.isfinite(v["flow"]) for v in o.values())
    # Fixed noise, time and augmentation: on CPU a repeat is exact.
    assert o["stock_again"]["diff"] == 0.0
    # Random weights still see the tokens: changing the prompt or the plan changes the flow loss.
    assert o["was"]["diff"] != 0.0 and o["unified"]["diff"] != 0.0
    u = s["vs_unified"]["overall"]
    assert u["unified"]["diff"] == 0.0 and u["now_cross_task"]["diff"] != 0.0
    # Episodes 0-19 hold multi-stage tasks, where a swapped stage is a different plan.
    assert s["vs_unified_multistage"]["samples"] > 0
    assert s["vs_unified_multistage"]["overall"]["now_swapped"]["diff"] != 0.0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload["per_sample"]["unified"]) == 12 and len(payload["indices"]) == 12

    # A subset runs only the named variants plus the two references.
    s = gfp.main(
        gfp.Args(
            weights_dir=str(weights),
            norm_stats_dir=str(ASSETS),
            dataset_root=str(DATASET),
            episodes="0:20",
            num_samples=4,
            batch_size=4,
            device="cpu",
            variants=("now_swapped",),
        )
    )
    assert set(s["vs_stock"]["overall"]) == {"stock", "unified", "now_swapped"}

    # Averaging in memory equals probing a checkpoint that holds the averaged weights.
    weights2, averaged = tmp_path / "weights2", tmp_path / "averaged"
    weights2.mkdir()
    averaged.mkdir()
    torch.manual_seed(1)
    other = get_config("pi05_libero_glow_unified").model.create_pytorch()
    safetensors.torch.save_model(other, str(weights2 / "model.safetensors"))
    torch.manual_seed(0)
    first = get_config("pi05_libero_glow_unified").model.create_pytorch()
    with torch.no_grad():
        for p, q in zip(first.parameters(), other.parameters(), strict=True):
            p.add_(q).div_(2)
    safetensors.torch.save_model(first, str(averaged / "model.safetensors"))

    def probe(**kw):
        a = gfp.Args(
            norm_stats_dir=str(ASSETS),
            dataset_root=str(DATASET),
            episodes="0:20",
            num_samples=4,
            batch_size=4,
            device="cpu",
            variants=("stock",),
            **kw,
        )
        return gfp.main(a)["vs_stock"]["overall"]["unified"]["flow"]

    in_memory = probe(weights_dir=str(weights), average_with=(str(weights2),))
    assert np.isclose(in_memory, probe(weights_dir=str(averaged)), rtol=1e-6)
    assert not np.isclose(in_memory, probe(weights_dir=str(weights)), rtol=1e-6)
