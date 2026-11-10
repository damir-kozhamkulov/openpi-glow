"""Shared pieces of the GLOW script tests: the local data they need and dummy-size configs."""

import dataclasses
import os
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]  # glow_repo (openpi is its submodule)
GOAL_PACK = REPO / "derived/plan_pack_libero10_goal.json"
DATASET = pathlib.Path(os.environ.get("HF_LEROBOT_HOME", "")) / "libero_staging"


def dummy_configs(monkeypatch):
    """Patches openpi.training.config.get_config to return dummy-size models (with a small SigLIP)
    that read GOAL_PACK; returns the patched get_config."""
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
