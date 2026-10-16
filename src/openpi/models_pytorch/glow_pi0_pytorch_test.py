"""GLOW unified model at dummy scale (CPU, float32, eager attention, gradient checkpointing on).

Training pass: without a plan block it is bitwise the stock model (loss and every gradient); the
block is causal and invisible to everything before it; the action expert sees it; the plan CE
trains the VLM and leaves the action expert's gradients untouched. Inference: candidate scores
equal each candidate's own teacher-forced log-likelihood, the training pass's CE agrees with
them, and acting from the winner's sliced KV cache equals acting on the winner given outright.

Runs in-container with `uv pip install --python /venv-server/bin/python pytest pynvml`.
"""

import math

import pytest
import torch
from transformers.models.auto import CONFIG_MAPPING

from openpi.glow.model_config import GlowPi0Config
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models_pytorch.gemma_pytorch as gemma_pytorch

L = 32
B = 4
M = 6  # plan block length, eos included
N_PROMPT = (9, 12, 7, 15)
VOCAB = 257152
CPU = torch.device("cpu")


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


def _arrays(seed, block_seed, bsize=B, with_block=True):
    """Images, state and one token row per sample; the block's tokens come from `block_seed`."""
    g = torch.Generator().manual_seed(seed)
    gb = torch.Generator().manual_seed(block_seed)
    imgs = {
        k: torch.randint(0, 256, (bsize, 224, 224, 3), generator=g, dtype=torch.uint8)
        for k in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    }
    masks = {k: torch.ones(bsize, dtype=torch.bool) for k in imgs}
    masks["right_wrist_0_rgb"] = torch.zeros(bsize, dtype=torch.bool)
    tokens = torch.zeros(bsize, L, dtype=torch.int64)
    tmask = torch.zeros(bsize, L, dtype=torch.bool)
    ar = torch.zeros(bsize, L, dtype=torch.int32)
    loss = torch.zeros(bsize, L, dtype=torch.bool)
    for b in range(bsize):
        n = N_PROMPT[b]
        tokens[b, :n] = torch.randint(3, 1000, (n,), generator=g)
        tmask[b, :n] = True
        if with_block:
            tokens[b, n : n + M] = torch.randint(3, 1000, (M,), generator=gb)
            tokens[b, n + M - 1] = 1  # eos
            tmask[b, n : n + M] = True
            ar[b, n : n + M] = 1
            loss[b, n : n + M] = True
    state = torch.rand(bsize, 32, generator=g) * 2 - 1
    return imgs, masks, state, tokens, tmask, ar, loss


def _obs(imgs, masks, state, tokens, tmask, ar=None, loss=None):
    d = {"image": imgs, "image_mask": masks, "state": state, "tokenized_prompt": tokens, "tokenized_prompt_mask": tmask}
    if ar is not None:
        d["token_ar_mask"] = ar
        d["token_loss_mask"] = loss
    return _model.Observation.from_dict(d)


def _make_obs(seed, block_seed=1000, with_block=True, bsize=B):
    imgs, masks, state, tokens, tmask, ar, loss = _arrays(seed, block_seed, bsize, with_block)
    return _obs(imgs, masks, state, tokens, tmask, ar, loss)


def _candidates_obs(seed, block_seeds, bsize=2):
    """Candidate layout: K rows of L tokens side by side, [bsize, K * L]; same prompt and images
    per sample, block k from block_seeds[k]."""
    rows = [_arrays(seed, s, bsize) for s in block_seeds]
    imgs, masks, state = rows[0][:3]
    joined = [torch.cat([r[i] for r in rows], dim=1) for i in range(3, 7)]
    return _obs(imgs, masks, state, *joined)


def _given_obs(seed, block_seed, bsize=2):
    return _make_obs(seed, block_seed, bsize=bsize)


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


def _same_grads(a, b, names=None):
    names = sorted(a) if names is None else names
    return all(
        (a[n] is None and b[n] is None) or (a[n] is not None and b[n] is not None and torch.equal(a[n], b[n]))
        for n in names
    )


def _capture_prefix(model, obs, actions, noise, time):
    """The training pass's prefix output (after the final norm). The model calls `.forward`
    directly, which module hooks do not see, so the method itself is wrapped."""
    seen = {}
    inner = model.paligemma_with_expert.forward

    def capture(*args, **kwargs):
        out = inner(*args, **kwargs)
        seen.setdefault("prefix", out[0][0])
        return out

    model.paligemma_with_expert.forward = capture
    try:
        _run(model, obs, actions, noise, time)
    finally:
        del model.paligemma_with_expert.forward
    return seen["prefix"].detach()


def test_embed_prefix_matches_stock(models):
    stock, glow0, _ = models
    stock.eval()
    glow0.eval()
    imgs, masks, state, tokens, tmask, _, _ = _arrays(3, 1000, with_block=False)
    pre = stock._preprocess_observation(_obs(imgs, masks, state, tokens, tmask), train=False)  # noqa: SLF001
    with torch.no_grad():
        ref = stock.embed_prefix(*pre[:4])
        got = glow0.embed_prefix(*pre[:4])
        assert all(torch.equal(x, y) for x, y in zip(ref, got, strict=True))
        # image_repeats = 2 equals the stock prefix with every sample's row duplicated, sample-major.
        # (Compared against duplicated rows, not a stock pass over a doubled batch: SigLIP's output
        # moves by ~1e-7 with the batch size.)
        rep = glow0.embed_prefix(
            pre[0], pre[1], pre[2].repeat_interleave(2, 0), pre[3].repeat_interleave(2, 0), image_repeats=2
        )
        assert all(torch.equal(x, y.repeat_interleave(2, 0)) for x, y in zip(rep, ref, strict=True))


def test_no_block_is_bitwise_stock(models, batch):
    stock, glow0, glow1 = models
    actions, noise, time = batch
    obs_nb = _make_obs(11, with_block=False)  # all-zero ar / loss masks
    out_s, g_s = _run(stock, obs_nb, actions, noise, time)
    out_0, g_0 = _run(glow0, obs_nb, actions, noise, time)
    assert isinstance(out_0, torch.Tensor)
    assert torch.equal(out_0, out_s)
    assert _same_grads(g_s, g_0)
    # lambda > 0 without any block: no CE term, the loss is the flow loss.
    out_1, _ = _run(glow1, obs_nb, actions, noise, time)
    assert "plan_ce" not in out_1
    assert torch.equal(out_1["loss"].detach(), out_s.mean().detach())
    assert float(out_1["ce_samples"]) == 0.0


def test_block_is_causal_and_drives_the_actions(models, batch):
    _, glow0, _ = models
    actions, noise, time = batch
    obs_a, obs_b = _make_obs(11, block_seed=1000), _make_obs(11, block_seed=2000)
    pre_a = _capture_prefix(glow0, obs_a, actions, noise, time)
    pre_b = _capture_prefix(glow0, obs_b, actions, noise, time)
    num_img = pre_a.shape[1] - L
    per_image = num_img // 3
    for b, n in enumerate(N_PROMPT):
        # Valid images and prompt - including the prompt's last token, whose state predicts the
        # block's first token - cannot see the block: bitwise unchanged when it changes. The
        # masked right-wrist image is excluded: a padded query's attention row is fully masked,
        # so its (never read) state averages over every key, the block included, as in stock pi05.
        valid = [*range(2 * per_image), *range(num_img, num_img + n)]
        assert torch.equal(pre_a[b, valid], pre_b[b, valid])
        assert not torch.equal(pre_a[b, num_img + n + 1 : num_img + n + M], pre_b[b, num_img + n + 1 : num_img + n + M])
    out_a, _ = _run(glow0, obs_a, actions, noise, time)
    out_b, _ = _run(glow0, obs_b, actions, noise, time)
    assert not torch.equal(out_a, out_b)  # the action expert attends to the block


def test_plan_ce(models, batch):
    _, glow0, glow1 = models
    actions, noise, time = batch
    obs = _make_obs(11)
    out_0, g_0 = _run(glow0, obs, actions, noise, time)
    out_1, g_1 = _run(glow1, obs, actions, noise, time)
    assert {"loss", "flow", "plan_ce", "plan_token_acc", "plan_exact", "ce_samples"} <= set(out_1)
    ce = float(out_1["plan_ce"])
    assert math.isfinite(ce) and abs(ce - math.log(VOCAB)) < 3.0
    assert torch.equal(out_1["flow"], out_0.mean().detach())
    assert float(out_1["ce_samples"]) == float(B)
    assert 0.0 <= float(out_1["plan_token_acc"]) <= 1.0 and 0.0 <= float(out_1["plan_exact"]) <= 1.0
    assert torch.allclose(out_1["loss"].detach(), out_1["flow"] + 0.1 * out_1["plan_ce"])
    # The CE reaches the action expert through nothing: its gradients are the flow loss's alone.
    expert = [n for n in g_1 if n.startswith(("paligemma_with_expert.gemma_expert", "action_", "time_mlp"))]
    assert expert and _same_grads(g_0, g_1, expert)
    lm = [n for n in g_1 if "language_model" in n and g_0[n] is not None and g_1[n] is not None]
    assert any(not torch.equal(g_0[n], g_1[n]) for n in lm)
    assert all(g_1[n] is not None for n in g_1 if "embed_tokens" in n)


def _sample(model, obs, noise):
    model.eval()
    with torch.no_grad():
        return model.sample_actions(CPU, obs, noise=noise, num_steps=3)


def test_scoring_and_cached_winner(models):
    _, _, glow1 = models
    seeds = (1000, 2000, 3000)
    noise = torch.randn(2, 10, 32, generator=torch.Generator().manual_seed(3))
    scored = _sample(glow1, _candidates_obs(21, seeds), noise)
    assert scored["plan_logp"].shape == (2, 3) and scored["actions"].shape == (2, 10, 32)
    given = [_sample(glow1, _given_obs(21, s), noise) for s in seeds]
    for k, out in enumerate(given):
        # Each candidate's score is its own teacher-forced block log-likelihood.
        assert torch.allclose(scored["plan_logp"][:, k], out["plan_logp"][:, 0], atol=1e-4)
        assert (out["plan_choice"] == 0).all()
    assert torch.equal(scored["plan_choice"], scored["plan_logp"].argmax(dim=1))
    for b in range(2):
        winner = int(scored["plan_choice"][b])
        # Acting from the winner's sliced KV cache = acting on the winner given outright.
        assert torch.allclose(scored["actions"][b], given[winner]["actions"][b], atol=1e-5)
    # The given block changes the actions.
    assert not torch.allclose(given[0]["actions"], given[1]["actions"])


def test_token_width_checks(models, batch):
    _, _, glow1 = models
    obs = _candidates_obs(21, (1000, 2000))
    with pytest.raises(ValueError):  # candidate rows are inference-only
        _run(glow1, obs, *(x[:2] for x in batch))
    cut = obs.replace(
        **{
            k: getattr(obs, k)[:, :-1]
            for k in ("tokenized_prompt", "tokenized_prompt_mask", "token_ar_mask", "token_loss_mask")
        }
    )
    with pytest.raises(ValueError):  # not a whole number of rows
        _sample(glow1, cut, None)


def test_training_ce_agrees_with_inference_scores(models, monkeypatch):
    """The joint training pass and the prefix-only inference pass decode the same block likelihood."""
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    _, _, glow1 = models
    monkeypatch.setattr(
        glow1,
        "_preprocess_observation",
        lambda obs, train=True: PI0Pytorch._preprocess_observation(glow1, obs, train=False),
    )
    obs = _given_obs(21, 1000)
    g = torch.Generator().manual_seed(9)
    out, _ = _run(
        glow1, obs, torch.rand(2, 10, 32, generator=g), torch.randn(2, 10, 32, generator=g), torch.full((2,), 0.5)
    )
    monkeypatch.undo()
    logp = _sample(glow1, obs, torch.randn(2, 10, 32))["plan_logp"][:, 0]
    # Every block token is a target, the first one predicted from the prompt's last token.
    n_tokens = int(obs.token_loss_mask.sum())
    assert n_tokens == 2 * M
    assert math.isclose(float(out["plan_ce"]), float(-logp.sum()) / n_tokens, rel_tol=1e-4)


def test_sample_actions_requires_a_block(models):
    _, _, glow1 = models
    with pytest.raises(ValueError):
        _sample(glow1, _make_obs(5, with_block=False, bsize=2).replace(token_ar_mask=None, token_loss_mask=None), None)
