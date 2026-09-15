"""GLOW model gates at dummy scale (CPU, float32, eager attention, gradient checkpointing on).

T1 (lambda = 0 identity): GlowPI0Pytorch on a batch that carries the subtask target block is
bitwise the stock PI0Pytorch on the same batch without it - per-element flow loss and every
parameter gradient - while the stock model on the block-carrying batch differs (the block is
visible to it, so the masking is what removes it). Plus the CE head's basic behaviour.

Runs in-container with `uv pip install --python /venv-server/bin/python pytest pynvml`.
"""

import math

import pytest
import torch
from transformers.models.auto import CONFIG_MAPPING

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
from openpi.glow.model_config import GlowPi0Config
import openpi.models_pytorch.gemma_pytorch as gemma_pytorch

L = 32
B = 4
VOCAB = 257152


@pytest.fixture(scope="module", autouse=True)
def tiny_vision_tower():
    """Dummy Gemma width is 64; give it a 2-layer SigLIP so the test runs in seconds on CPU."""
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


def _cfg(cls, **kw):
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


def _make_obs(seed, with_block, block_rows=(0, 2)):
    g = torch.Generator().manual_seed(seed)
    gb = torch.Generator().manual_seed(seed + 1000)  # block tokens on their own stream, so prompts match
    imgs = {
        k: torch.randint(0, 256, (B, 224, 224, 3), generator=g, dtype=torch.uint8)
        for k in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    }
    masks = {k: torch.ones(B, dtype=torch.bool) for k in imgs}
    masks["right_wrist_0_rgb"] = torch.zeros(B, dtype=torch.bool)
    n_prompt = [9, 12, 7, 15]
    tokens = torch.zeros(B, L, dtype=torch.int64)
    tmask = torch.zeros(B, L, dtype=torch.bool)
    ar = torch.zeros(B, L, dtype=torch.int32)
    loss = torch.zeros(B, L, dtype=torch.bool)
    for b in range(B):
        tokens[b, : n_prompt[b]] = torch.randint(3, 1000, (n_prompt[b],), generator=g)
        tmask[b, : n_prompt[b]] = True
        if with_block and b in block_rows:
            m = 6
            tokens[b, n_prompt[b] : n_prompt[b] + m] = torch.randint(3, 1000, (m,), generator=gb)
            tokens[b, n_prompt[b] + m - 1] = 1  # eos
            tmask[b, n_prompt[b] : n_prompt[b] + m] = True
            ar[b, n_prompt[b] : n_prompt[b] + m] = 1
            loss[b, n_prompt[b] : n_prompt[b] + m] = True
    state = torch.rand(B, 32, generator=g) * 2 - 1
    d = {"image": imgs, "image_mask": masks, "state": state, "tokenized_prompt": tokens, "tokenized_prompt_mask": tmask}
    if with_block:
        d["token_ar_mask"] = ar
        d["token_loss_mask"] = loss
    return _model.Observation.from_dict(d)


def _run(model, obs, actions, noise, time):
    torch.manual_seed(0)  # image-augmentation draws
    model.train()
    model.zero_grad(set_to_none=True)
    out = model(obs, actions, noise=noise, time=time)
    loss = out["loss"] if isinstance(out, dict) else out.mean()
    loss.backward()
    grads = {n: (p.grad.detach().clone() if p.grad is not None else None) for n, p in model.named_parameters()}
    return out, grads


@pytest.fixture(scope="module")
def models():
    from openpi.models_pytorch.glow_pi0_pytorch import GlowPI0Pytorch
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    torch.manual_seed(1234)
    stock = PI0Pytorch(_cfg(pi0_config.Pi0Config))
    glow0 = GlowPI0Pytorch(_cfg(GlowPi0Config, subtask_ce_weight=0.0))
    glow1 = GlowPI0Pytorch(_cfg(GlowPi0Config, subtask_ce_weight=0.1))
    sd = stock.state_dict()
    glow0.load_state_dict(sd)
    glow1.load_state_dict(sd)
    for m in (stock, glow0, glow1):
        m.gradient_checkpointing_enable()
    return stock, glow0, glow1


@pytest.fixture(scope="module")
def batch():
    g = torch.Generator().manual_seed(7)
    actions = torch.rand(B, 10, 32, generator=g) * 2 - 1
    noise = torch.randn(B, 10, 32, generator=g)
    time = torch.rand(B, generator=g) * 0.998 + 0.001
    return actions, noise, time


def _same_grads(a, b):
    return all(
        (x is None and y is None) or (x is not None and y is not None and torch.equal(x, y))
        for (_, x), (_, y) in zip(sorted(a.items()), sorted(b.items()), strict=True)
    )


def test_lambda_zero_is_bitwise_stock(models, batch):
    stock, glow0, _ = models
    actions, noise, time = batch
    out_s, g_s = _run(stock, _make_obs(11, with_block=False), actions, noise, time)
    out_0, g_0 = _run(glow0, _make_obs(11, with_block=True), actions, noise, time)
    assert isinstance(out_0, torch.Tensor) and out_0.shape == out_s.shape
    assert torch.equal(out_0, out_s)
    assert _same_grads(g_s, g_0)


def test_block_is_visible_to_the_stock_model(models, batch):
    stock, _, _ = models
    actions, noise, time = batch
    out_s, _ = _run(stock, _make_obs(11, with_block=False), actions, noise, time)
    out_sb, _ = _run(stock, _make_obs(11, with_block=True), actions, noise, time)
    assert not torch.equal(out_sb, out_s)


def test_ce_term(models, batch):
    stock, _, glow1 = models
    actions, noise, time = batch
    out_s, g_s = _run(stock, _make_obs(11, with_block=False), actions, noise, time)
    out_1, g_1 = _run(glow1, _make_obs(11, with_block=True), actions, noise, time)
    assert {"loss", "flow", "subtask_ce", "ce_samples"} <= set(out_1)
    ce = float(out_1["subtask_ce"])
    assert math.isfinite(ce) and abs(ce - math.log(VOCAB)) < 3.0
    assert torch.equal(out_1["flow"], out_s.mean().detach())
    assert float(out_1["ce_samples"]) == 2.0
    assert torch.allclose(out_1["loss"].detach(), out_1["flow"] + 0.1 * out_1["subtask_ce"])
    expert = [n for n in g_1 if n.startswith("paligemma_with_expert.gemma_expert") or n.startswith("action_")]
    assert all(torch.equal(g_1[n], g_s[n]) for n in expert if g_s[n] is not None)
    lm = [n for n in g_1 if "language_model" in n]
    assert any(not torch.equal(g_1[n], g_s[n]) for n in lm if g_s[n] is not None and g_1[n] is not None)
    assert all(g_1[n] is not None for n in g_1 if "embed_tokens" in n)


def test_ce_pass_skipped_without_targets(models, batch):
    _, _, glow1 = models
    actions, noise, time = batch
    out, _ = _run(glow1, _make_obs(11, with_block=True, block_rows=()), actions, noise, time)
    assert "subtask_ce" not in out
    assert torch.equal(out["loss"].detach(), out["flow"])
    assert float(out["ce_samples"]) == 0.0


def test_ce_over_subset_equals_ce_over_rows(models):
    from openpi.models_pytorch.glow_pi0_pytorch import _select_rows

    _, _, glow1 = models
    obs = _make_obs(11, with_block=True)
    sel = obs.token_loss_mask.any(dim=1)
    # The CE pass redraws augmentation, whose factors are per-batch scalars, so both calls
    # need the same global seed to be comparable.
    torch.manual_seed(5)
    ce_all = glow1.subtask_ce(obs, sel)
    obs_02 = _select_rows(obs, torch.tensor([0, 2]))
    torch.manual_seed(5)
    ce_02 = glow1.subtask_ce(obs_02, torch.tensor([True, True]))
    assert torch.allclose(ce_all, ce_02, atol=1e-6)


def test_inference_is_stock(models):
    stock, _, glow1 = models
    stock.eval()
    glow1.eval()
    obs = _make_obs(11, with_block=False)
    noise = torch.randn(B, 10, 32, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        a_s = stock.sample_actions(torch.device("cpu"), obs, noise=noise, num_steps=3)
        a_g = glow1.sample_actions(torch.device("cpu"), obs, noise=noise, num_steps=3)
    assert torch.equal(a_s, a_g)
