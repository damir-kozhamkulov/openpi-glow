"""glow_module_probe: the counterfactual module arrays, and main() end to end.

Needs the LIBERO-10 + Goal plan pack; the end-to-end test also needs the local LeRobot copy
(HF_LEROBOT_HOME with repo `libero_staging`) and pi05_libero's norm_stats. It runs a dummy-size
module model with random weights, so it checks the wiring, not the numbers.
"""

import dataclasses
import os
import pathlib

import numpy as np
import pytest

from . import glow_module_probe as gmp

REPO = pathlib.Path(__file__).resolve().parents[2]  # glow_repo (openpi is its submodule)
GOAL_PACK = REPO / "derived/plan_pack_libero10_goal.json"
DATASET = pathlib.Path(os.environ.get("HF_LEROBOT_HOME", "")) / "libero_staging"
ASSETS = REPO / "data/assets/pi05_libero"
NORM_STATS = ASSETS / "physical-intelligence/libero/norm_stats.json"

pytestmark = pytest.mark.skipif(not GOAL_PACK.exists(), reason="Goal plan pack not present")


@pytest.fixture(scope="module")
def module():
    from openpi.glow import module_inputs as _mi
    from openpi.glow import plan_pack as _plan_pack
    from openpi.models import tokenizer as _tokenizer

    return _mi.GlowModuleInputs(_plan_pack.PlanPack.load(GOAL_PACK), _tokenizer.PaligemmaTokenizer(8))


def test_other_task_stays_in_the_suite(module):
    for t in module.pack.tasks.values():
        other = gmp.other_task(module.pack, t.task)
        assert other.task_index != t.task_index
        assert other.task_index // gmp.SUITE_SIZE == t.task_index // gmp.SUITE_SIZE


def test_module_variants(module):
    from openpi.glow import module_inputs as _mi

    two = next(t for t in module.pack.tasks.values() if t.num_plans == 2)
    correct = module.encode(two.task, 0)
    v = gmp.module_variants(module.encode, module.pack, two.task, 0, correct)
    assert set(v) == set(gmp.VARIANTS)
    other = gmp.other_task(module.pack, two.task)
    np.testing.assert_array_equal(v["other_task"]["glow_text"], module.encode(other.task, 0)["glow_text"])
    np.testing.assert_array_equal(v["wrong_stage"]["glow_text"], module.encode(two.task, 1)["glow_text"])
    np.testing.assert_array_equal(v["grounding_other"]["glow_text"], correct["glow_text"])
    np.testing.assert_array_equal(v["grounding_other"]["glow_floats"], v["other_task"]["glow_floats"])
    # Blanks remove exactly their encoder's text, and leave `correct` untouched.
    spatial = [_mi.TEXT_SEGMENTS[n] for n in ("subgoal", "object", "target")]
    temporal = [_mi.TEXT_SEGMENTS[f"rk{j}"] for j in range(_mi.K_MAX)]
    for blank, removed, kept in (("spatial_blank", spatial, temporal), ("temporal_blank", temporal, spatial)):
        for start, stop in removed:
            assert not v[blank]["glow_text_mask"][start:stop].any()
        for start, stop in kept:
            np.testing.assert_array_equal(v[blank]["glow_text_mask"][start:stop], correct["glow_text_mask"][start:stop])
    assert v["spatial_blank"]["glow_ints"][_mi.INT_PREDICATE] == 0 and not v["spatial_blank"]["glow_floats"].any()
    assert correct["glow_ints"][_mi.INT_PREDICATE] != 0 and correct["glow_text_mask"].sum() > 0


def _segments(arrays, names):
    from openpi.glow import module_inputs as _mi

    return np.concatenate([arrays["glow_text"][slice(*_mi.TEXT_SEGMENTS[n])] for n in names])


def test_splits_take_one_encoders_inputs(module):
    from openpi.glow import module_inputs as _mi

    spatial = ("subgoal", "object", "target")
    temporal = tuple(f"rk{j}" for j in range(_mi.K_MAX))
    # Task 0 (two stages) against task 1: different subgoals, anchors and runtime keys.
    two = next(t for t in module.pack.tasks.values() if t.task_index == 0)
    correct = module.encode(two.task, 0)
    v = gmp.module_variants(module.encode, module.pack, two.task, 0, correct)
    for split, donor in (("other_task", v["other_task"]), ("wrong_stage", v["wrong_stage"])):
        s = v[f"{split}_spatial"]
        np.testing.assert_array_equal(_segments(s, spatial), _segments(donor, spatial))
        np.testing.assert_array_equal(_segments(s, temporal), _segments(correct, temporal))
        np.testing.assert_array_equal(s["glow_floats"], donor["glow_floats"])
        assert s["glow_ints"][_mi.INT_PREDICATE] == donor["glow_ints"][_mi.INT_PREDICATE]
        assert s["glow_ints"][_mi.INT_STAGE] == 0
    t = v["other_task_temporal"]
    np.testing.assert_array_equal(_segments(t, spatial), _segments(correct, spatial))
    np.testing.assert_array_equal(_segments(t, temporal), _segments(v["other_task"], temporal))
    np.testing.assert_array_equal(t["glow_floats"], correct["glow_floats"])
    np.testing.assert_array_equal(t["glow_ints"][1:], v["other_task"]["glow_ints"][1:])
    assert not np.array_equal(_segments(t, temporal), _segments(correct, temporal))
    # The pointer split moves the stage index and nothing else.
    p = v["wrong_stage_pointer"]
    assert p["glow_ints"][_mi.INT_STAGE] == 1 and correct["glow_ints"][_mi.INT_STAGE] == 0
    for key in ("glow_text", "glow_text_mask", "glow_floats"):
        np.testing.assert_array_equal(p[key], correct[key])
    # Variants never write into `correct`.
    np.testing.assert_array_equal(correct["glow_text"], module.encode(two.task, 0)["glow_text"])
    np.testing.assert_array_equal(correct["glow_ints"], module.encode(two.task, 0)["glow_ints"])


@pytest.mark.skipif(
    not ((DATASET / "meta/info.json").exists() and NORM_STATS.exists()),
    reason="local LeRobot copy or norm_stats not present",
)
def test_main_end_to_end(monkeypatch, tmp_path):
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
                c.model,
                paligemma_variant="dummy",
                action_expert_variant="dummy",
                pytorch_compile_mode=None,
                module_width=32,
            ),
            data=dataclasses.replace(c.data, plan_pack_path=str(GOAL_PACK)),
        )

    monkeypatch.setattr(_config, "get_config", get_config)
    weights = tmp_path / "weights"
    weights.mkdir()
    torch.manual_seed(0)
    safetensors.torch.save_model(get_config("pi05_libero_glow_st").model.create_pytorch(), str(weights / "model.safetensors"))

    out = tmp_path / "probe.json"
    # LIBERO-10 episodes (two-stage tasks) and Goal episodes, not a 0..n-1 prefix.
    s = gmp.main(
        gmp.Args(
            weights_dir=str(weights),
            norm_stats_dir=str(ASSETS),
            dataset_root=str(DATASET),
            episodes="0:6,379:385",
            num_samples=16,
            batch_size=4,
            device="cpu",
            out=str(out),
        )
    )
    o = s["vs_correct"]["overall"]
    assert s["vs_correct"]["samples"] == 16 and set(o) == set(gmp.VARIANTS)
    assert o["correct"]["diff"] == 0.0 and o["correct_again"]["diff"] == 0.0  # exact repeat on CPU
    assert all(np.isfinite(v["flow"]) for v in o.values())
    # Random weights still read the module: every counterfactual moves the flow loss.
    for name in ("other_task", "grounding_other", "spatial_blank", "temporal_blank"):
        assert o[name]["diff"] != 0.0, name
    assert s["vs_correct_multistage"]["overall"]["wrong_stage"]["diff"] != 0.0
    by_stage = [b for b in s["vs_correct_multistage_by_stage"].values() if b is not None]
    assert by_stage and sum(b["samples"] for b in by_stage) <= s["vs_correct_multistage"]["samples"]
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload["per_sample"]["correct"]) == 16
