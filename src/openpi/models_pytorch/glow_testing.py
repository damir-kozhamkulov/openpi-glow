"""Shared pieces of the GLOW tests: dummy-size model configs and random observations (CPU)."""

import pytest
import torch
from transformers.models.auto import CONFIG_MAPPING

import openpi.models.model as _model
import openpi.models_pytorch.gemma_pytorch as gemma_pytorch

L = 32
B = 4
N_PROMPT = (9, 12, 7, 15)


@pytest.fixture(scope="module", autouse=True)
def tiny_vision_tower():
    """Dummy Gemma width is 64; give it a 2-layer SigLIP so the test runs in seconds on CPU.
    A test module turns it on by importing it."""
    orig = CONFIG_MAPPING["paligemma"]

    def tiny():
        cfg = orig()
        cfg.vision_config.hidden_size = 32
        cfg.vision_config.num_hidden_layers = 2
        cfg.vision_config.num_attention_heads = 4
        cfg.vision_config.image_size = 224
        cfg.vision_config.patch_size = 56
        return cfg

    saved = gemma_pytorch.CONFIG_MAPPING
    gemma_pytorch.CONFIG_MAPPING = {**CONFIG_MAPPING, "paligemma": tiny}
    yield
    gemma_pytorch.CONFIG_MAPPING = saved


def dummy_config(cls, **kw):
    """`cls` (Pi0Config or a subclass) as pi05_libero at dummy size, float32, uncompiled."""
    return cls(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        dtype="float32",
        pytorch_compile_mode=None,
        max_token_len=L,
        **kw,
    )


def observation_arrays(seed, bsize=B):
    """Images (two cameras present, the third masked), their masks, state, and one prompt row per
    sample (N_PROMPT tokens each)."""
    g = torch.Generator().manual_seed(seed)
    imgs = {
        k: torch.randint(0, 256, (bsize, 224, 224, 3), generator=g, dtype=torch.uint8)
        for k in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    }
    masks = {k: torch.ones(bsize, dtype=torch.bool) for k in imgs}
    masks["right_wrist_0_rgb"] = torch.zeros(bsize, dtype=torch.bool)
    tokens = torch.zeros(bsize, L, dtype=torch.int64)
    tmask = torch.zeros(bsize, L, dtype=torch.bool)
    for b in range(bsize):
        n = N_PROMPT[b]
        tokens[b, :n] = torch.randint(3, 1000, (n,), generator=g)
        tmask[b, :n] = True
    state = torch.rand(bsize, 32, generator=g) * 2 - 1
    return imgs, masks, state, tokens, tmask


def make_observation(seed, bsize=B):
    imgs, masks, state, tokens, tmask = observation_arrays(seed, bsize)
    return _model.Observation.from_dict(
        {"image": imgs, "image_mask": masks, "state": state, "tokenized_prompt": tokens, "tokenized_prompt_mask": tmask}
    )
