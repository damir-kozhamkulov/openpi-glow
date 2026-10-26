"""glow_module_diag: collapse statistics, attention grouping, and main() end to end.

The end-to-end tests need the LIBERO-10 + Goal plan pack, the local LeRobot copy (HF_LEROBOT_HOME
with repo `libero_staging`) and pi05_libero's norm_stats. They run dummy-size models with random
weights, so they check the wiring, not the numbers.
"""

import dataclasses
import json
import os
import pathlib

import numpy as np
import pytest

from . import glow_module_diag as gmd

REPO = pathlib.Path(__file__).resolve().parents[2]  # glow_repo (openpi is its submodule)
GOAL_PACK = REPO / "derived/plan_pack_libero10_goal.json"
DATASET = pathlib.Path(os.environ.get("HF_LEROBOT_HOME", "")) / "libero_staging"
ASSETS = REPO / "data/assets/pi05_libero"
NORM_STATS = ASSETS / "physical-intelligence/libero/norm_stats.json"

needs_data = pytest.mark.skipif(
    not (GOAL_PACK.exists() and (DATASET / "meta/info.json").exists() and NORM_STATS.exists()),
    reason="plan pack, local LeRobot copy or norm_stats not present",
)


def test_collapse_stats():
    same = gmd.collapse_stats(np.tile(np.arange(1.0, 5.0), (3, 1)))
    assert same["mean_cos"] == pytest.approx(1.0) and same["spread"] == pytest.approx(0.0, abs=1e-7)
    assert same["min_1_minus_cos"] == pytest.approx(0.0, abs=1e-7)
    orth = gmd.collapse_stats(np.eye(4) * 5.0)  # scale does not matter
    assert orth["mean_cos"] == pytest.approx(0.0) and orth["min_1_minus_cos"] == pytest.approx(1.0)
    assert orth["spread"] > 0.8


def test_attention_recorder_groups_and_masses():
    import torch

    # One sample: 4 image tokens (one padded), 2 module, 2 language, then 3 action tokens.
    prefix = torch.tensor([[0, 0, 0, -1, 1, 1, 2, 2]])
    t = prefix.shape[1] + 3
    rec = gmd.AttentionRecorder(num_layers=2)
    rec.prefix = prefix
    # Uniform attention over the 10 valid keys for every query; padding gets none.
    w = torch.zeros(1, 8, t, t)
    valid = torch.ones(t, dtype=torch.bool)
    valid[3] = False
    w[..., valid] = 1.0 / valid.sum()
    value = torch.ones(1, 1, t, 4)
    for _ in range(2):
        rec.record(w, value)
    s = rec.summary()
    assert s["keys_per_sample"] == {"images": 3, "module": 2, "language": 2, "actions": 3}
    a = s["by_query"]["actions"]
    assert a["mass_mean"]["module"] == pytest.approx(0.2) and a["mass_mean"]["images"] == pytest.approx(0.3)
    assert sum(a["mass_mean"].values()) == pytest.approx(1.0)
    # Uniform attention: every prefix group gets an average prefix token's share.
    for v in a["per_token_vs_prefix_mean"].values():
        assert v == pytest.approx(1.0)
    assert s["actions_value_weighted_share_mean"]["module"] == pytest.approx(0.2)
    # A second sample starts a new sequence (the next call is layer 0 again).
    rec.prefix = prefix
    rec.groups = None
    rec.record(w, value)
    assert rec.samples == 2 and rec.layer == 1


def _dummy_configs(monkeypatch):
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
        small = {"paligemma_variant": "dummy", "action_expert_variant": "dummy", "pytorch_compile_mode": None}
        if hasattr(c.model, "module_width"):
            small["module_width"] = 32
        return dataclasses.replace(
            c,
            model=dataclasses.replace(c.model, **small),
            data=dataclasses.replace(c.data, plan_pack_path=str(GOAL_PACK)),
        )

    monkeypatch.setattr(_config, "get_config", get_config)
    return get_config


def _args(weights, tmp_path, **kw):
    return gmd.Args(
        weights_dir=str(weights),
        norm_stats_dir=str(ASSETS),
        dataset_root=str(DATASET),
        # LIBERO-10 (two-stage tasks) and Goal episodes, not a 0..n-1 prefix.
        episodes="0:6,379:385",
        num_samples=8,
        batch_size=4,
        device="cpu",
        out=str(tmp_path / "diag.json"),
        **kw,
    )


@needs_data
def test_main_module_model(monkeypatch, tmp_path):
    import safetensors.torch
    import torch

    get_config = _dummy_configs(monkeypatch)
    weights = tmp_path / "weights"
    weights.mkdir()
    torch.manual_seed(0)
    safetensors.torch.save_model(
        get_config("pi05_libero_glow_st").model.create_pytorch(), str(weights / "model.safetensors")
    )

    s = gmd.main(_args(weights, tmp_path))

    slots = s["collapse"]["slots"]
    assert list(slots) == [*gmd.SPATIAL_SLOTS, *gmd.TEMPORAL_SLOTS]
    pack_modules = len(s["collapse"]["modules"])
    assert slots["OP"]["all"]["modules"] == pack_modules > 20
    assert slots["OP"]["libero_goal"]["modules"] == 10  # every Goal task has one stage
    # A random encoder separates modules; its CTX slot is a learned constant query, but the
    # transformer mixes the other slots into it.
    assert slots["OBJ"]["all"]["spread"] > 0 and slots["OBJ"]["all"]["mean_cos"] < 1.0

    o = s["flow"]["overall"]
    assert set(o) == set(gmd.FLOW_VARIANTS) and s["flow"]["samples"] == 8
    assert all(np.isfinite(v["flow"]) for v in o.values())
    assert o["no_module"]["diff"] != 0.0 and o["task_prompt"]["diff"] != 0.0

    att = s["attention"]
    assert att["keys_per_sample"]["module"] == 6
    assert att["keys_per_sample"]["language"] == 2  # BOS and the newline answer token
    for q in ("actions", "images", "module"):
        assert sum(att["by_query"][q]["mass_mean"].values()) == pytest.approx(1.0, abs=1e-4)
    # Prefix queries never attend to the action tokens (stock prefix-LM mask).
    assert att["by_query"]["images"]["mass_mean"]["actions"] == pytest.approx(0.0, abs=1e-6)
    layers = get_config("pi05_libero_glow_st").model.create_pytorch().paligemma_with_expert.paligemma.config
    assert len(att["by_query"]["actions"]["mass_per_layer"]["module"]) == layers.text_config.num_hidden_layers
    assert sum(att["actions_value_weighted_share_mean"].values()) == pytest.approx(1.0, abs=1e-4)
    assert set(s["embedding_rms"]) == {"images", "module", "language"}
    # The encoder tokens are rescaled to the language tokens' RMS (pooled over the tokens; the
    # diagnostic reports the mean per-token RMS, so the two agree only closely).
    assert s["embedding_rms"]["module"] == pytest.approx(s["embedding_rms"]["language"], rel=1e-2)

    payload = json.loads((tmp_path / "diag.json").read_text(encoding="utf-8"))
    assert len(payload["per_sample_flow"]["correct"]) == 8


@needs_data
def test_main_init_loads_without_encoders(monkeypatch, tmp_path):
    import safetensors.torch
    import torch

    from openpi.models import pi0_config

    _dummy_configs(monkeypatch)
    weights = tmp_path / "init"
    weights.mkdir()
    stock = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        max_token_len=8,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        pytorch_compile_mode=None,
    )
    torch.manual_seed(1)
    safetensors.torch.save_model(stock.create_pytorch(), str(weights / "model.safetensors"))

    with pytest.raises(Exception):  # noqa: B017 - strict load must refuse the missing encoders
        gmd.main(_args(weights, tmp_path))
    a = gmd.main(_args(weights, tmp_path, init=True))
    b = gmd.main(_args(weights, tmp_path, init=True))
    assert a["flow"]["overall"]["correct"]["flow"] == b["flow"]["overall"]["correct"]["flow"]  # seeded encoders
    # Without module tokens the init model is the stock one: both prompts' losses do not depend
    # on the encoders' seed.
    c = gmd.main(_args(weights, tmp_path, init=True, init_seed=5))
    assert c["flow"]["overall"]["no_module"]["flow"] == a["flow"]["overall"]["no_module"]["flow"]
    assert c["flow"]["overall"]["task_prompt"]["flow"] == a["flow"]["overall"]["task_prompt"]["flow"]
    assert c["flow"]["overall"]["correct"]["flow"] != a["flow"]["overall"]["correct"]["flow"]


@needs_data
def test_main_plan_prompt_reference(monkeypatch, tmp_path):
    import safetensors.torch
    import torch

    get_config = _dummy_configs(monkeypatch)
    weights = tmp_path / "weights"
    weights.mkdir()
    torch.manual_seed(0)
    safetensors.torch.save_model(
        get_config("pi05_libero_plan_prompt").model.create_pytorch(), str(weights / "model.safetensors")
    )

    s = gmd.main(_args(weights, tmp_path, config="pi05_libero_plan_prompt"))
    assert "collapse" not in s
    assert set(s["flow"]["overall"]) == {"correct"}
    att = s["attention"]
    assert att["keys_per_sample"]["module"] == 0 and att["keys_per_sample"]["language"] > 2  # the plan text
    assert "module" not in att["by_query"]["actions"]["per_token_vs_prefix_mean"]
    assert sum(att["by_query"]["actions"]["mass_mean"].values()) == pytest.approx(1.0, abs=1e-4)


@needs_data
def test_main_anchored_init_is_the_text(monkeypatch, tmp_path):
    """A fresh text-anchored encoder emits its anchors whatever its seed: the --init flow loss and
    the tokens' collapse statistics do not depend on the encoders' initialization."""
    import safetensors.torch
    import torch

    from openpi.models import pi0_config

    _dummy_configs(monkeypatch)
    weights = tmp_path / "init"
    weights.mkdir()
    stock = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        max_token_len=8,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        pytorch_compile_mode=None,
    )
    torch.manual_seed(1)
    safetensors.torch.save_model(stock.create_pytorch(), str(weights / "model.safetensors"))

    runs = [
        gmd.main(_args(weights, tmp_path, config="pi05_libero_glow_st_anchor", init=True, init_seed=seed))
        for seed in (0, 5)
    ]
    a, b = (r["flow"]["overall"]["correct"]["flow"] for r in runs)
    assert a == pytest.approx(b, rel=1e-6)
    for slot in gmd.SPATIAL_SLOTS + gmd.TEMPORAL_SLOTS:
        sa, sb = (r["collapse"]["slots"][slot]["all"] for r in runs)
        assert sa["mean_cos"] == pytest.approx(sb["mean_cos"], abs=1e-6), slot
