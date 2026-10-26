"""GLOW module model at dummy scale (CPU, float32): encoder tokens in the prefix.

Prefix layout and scale; the module inputs reach the actions (training flow and inference
actions move when they change, and stay put when they do not); every encoder input is read
(subgoal, nouns, anchors, predicate, runtime keys, stage); the ablation switches remove what
they name; the encoders get gradients; the stock model is unchanged by the prefix hook.

Runs in-container with `uv pip install --python /venv-server/bin/python pytest pynvml`.
"""

import numpy as np
import pytest
import torch

from openpi.glow import module_encoders as _enc
from openpi.glow import module_inputs as _mi
from openpi.glow.model_config import GlowModulePi0Config
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
from openpi.models_pytorch.glow_module_pi0_pytorch import GlowModulePI0Pytorch
from openpi.models_pytorch.glow_pi0_pytorch_test import _arrays
from openpi.models_pytorch.glow_pi0_pytorch_test import _cfg
from openpi.models_pytorch.glow_pi0_pytorch_test import tiny_vision_tower  # noqa: F401

B = 4
WIDTH = 64  # dummy Gemma width


def _module_arrays(seed, bsize=B):
    """Random but well-formed module arrays: every text segment partly filled, two-stage tasks."""
    g = np.random.default_rng(seed)
    text = np.zeros((bsize, _mi.TEXT_LEN), dtype=np.int64)
    mask = np.zeros((bsize, _mi.TEXT_LEN), dtype=bool)
    for name, (start, stop) in _mi.TEXT_SEGMENTS.items():
        n = 0 if name in ("rk2", "rk3") else int(g.integers(1, stop - start))
        text[:, start : start + n] = g.integers(3, 1000, (bsize, n))
        mask[:, start : start + n] = True
    ints = np.stack(
        [g.integers(1, len(_mi.PREDICATES) + 1, bsize), g.integers(0, 2, bsize), np.full(bsize, 2)], axis=1
    )
    floats = np.concatenate(
        [g.uniform(-1, 1, (bsize, 3)), np.ones((bsize, 1)), g.uniform(-1, 1, (bsize, 3)), np.ones((bsize, 1))], axis=1
    )
    return {"glow_text": text, "glow_text_mask": mask, "glow_ints": ints, "glow_floats": floats.astype(np.float32)}


def _obs(seed, module, bsize=B):
    imgs, masks, state, tokens, tmask, _, _ = _arrays(seed, 0, bsize, with_block=False)
    d = {"image": imgs, "image_mask": masks, "state": state, "tokenized_prompt": tokens, "tokenized_prompt_mask": tmask}
    d |= {k: torch.as_tensor(v) for k, v in module.items()}
    return _model.Observation.from_dict(d)


def _model_with(**kw):
    torch.manual_seed(1234)
    return GlowModulePI0Pytorch(_cfg(GlowModulePi0Config, module_width=32, **kw))


@pytest.fixture(scope="module")
def model():
    return _model_with()


def _flow(model, obs, seed=7):
    g = torch.Generator().manual_seed(seed)
    actions = torch.randn(B, 10, 32, generator=g)
    noise = torch.randn(B, 10, 32, generator=g)
    time = torch.rand(B, generator=g)
    torch.manual_seed(0)
    model.train()
    return model(obs, actions, noise=noise, time=time)


def _act(model, obs, seed=11):
    model.eval()
    noise = torch.randn(B, 10, 32, generator=torch.Generator().manual_seed(seed))
    return model.sample_actions(torch.device("cpu"), obs, noise=noise, num_steps=3)


def test_prefix_layout_and_scale(model):
    obs = _obs(0, _module_arrays(0))
    images, img_masks, lang_tokens, lang_masks, _ = model._preprocess_observation(obs, train=False)  # noqa: SLF001
    stock = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    embs, pad, att = model.embed_observation_prefix(obs, images, img_masks, lang_tokens, lang_masks)
    n = _enc.SPATIAL_TOKENS + _enc.TEMPORAL_TOKENS
    num_img = stock[0].shape[1] - lang_tokens.shape[1]
    assert embs.shape == (B, stock[0].shape[1] + n, WIDTH)
    # Images and language are the stock ones; the module tokens sit between them, valid and bidirectional.
    assert torch.equal(embs[:, :num_img], stock[0][:, :num_img])
    assert torch.equal(embs[:, num_img + n :], stock[0][:, num_img:])
    assert pad[:, num_img : num_img + n].all() and not att[:, num_img : num_img + n].any()
    assert torch.equal(pad[:, num_img + n :], stock[1][:, num_img:])
    # Module tokens have the RMS of the sample's language tokens.
    module_rms = embs[:, num_img : num_img + n].pow(2).mean(-1).sqrt()
    lang = stock[0][:, num_img:]
    lang_rms = (lang.pow(2).mean(-1) * lang_masks).sum(1) / lang_masks.sum(1)
    torch.testing.assert_close(module_rms, lang_rms.sqrt()[:, None].expand_as(module_rms), rtol=1e-4, atol=1e-4)


def test_module_inputs_reach_the_actions(model):
    base = _module_arrays(0)
    obs = _obs(0, base)
    flow = _flow(model, obs)
    assert torch.equal(flow, _flow(model, _obs(0, base)))
    acts = _act(model, obs)
    assert torch.equal(acts, _act(model, _obs(0, base)))
    other = _obs(0, _module_arrays(1))
    assert not torch.allclose(flow, _flow(model, other))
    assert not torch.allclose(acts, _act(model, other))


def _changed(arrays, edit):
    out = {k: v.copy() for k, v in arrays.items()}
    edit(out)
    return out


def _segment_edit(name):
    start, stop = _mi.TEXT_SEGMENTS[name]

    def edit(a):
        a["glow_text"][:, start] = (a["glow_text"][:, start] + 17) % 1000 + 3
        a["glow_text_mask"][:, start] = True

    return edit


EDITS = {
    "subgoal": _segment_edit("subgoal"),
    "object noun": _segment_edit("object"),
    "target noun": _segment_edit("target"),
    "runtime key": _segment_edit("rk1"),
    "predicate": lambda a: a["glow_ints"].__setitem__((slice(None), _mi.INT_PREDICATE), 0),
    "stage": lambda a: a["glow_ints"].__setitem__((slice(None), _mi.INT_STAGE), 1 - a["glow_ints"][:, _mi.INT_STAGE]),
    "object anchor": lambda a: a["glow_floats"].__setitem__((slice(None), _mi.FLOAT_OBJECT), 0.5),
    "target anchor": lambda a: a["glow_floats"].__setitem__((slice(None), _mi.FLOAT_TARGET), -0.5),
}


@pytest.mark.parametrize("field", list(EDITS))
def test_every_input_is_read(model, field):
    base = _module_arrays(0)
    edited = _changed(base, EDITS[field])
    torch.manual_seed(0)
    a = model.module_tokens(_obs(0, base))
    b = model.module_tokens(_obs(0, edited))
    assert not torch.allclose(a, b), field


@pytest.mark.parametrize(
    ("switch", "ignored", "tokens"),
    [
        ({"module_temporal": False}, ("runtime key",), _enc.SPATIAL_TOKENS),
        ({"module_spatial": False}, ("subgoal", "object noun", "target noun", "predicate", "object anchor"), 2),
        ({"module_grounding": False}, ("object anchor", "target anchor"), 6),
        ({"module_subgoal_text": False}, ("subgoal",), 6),
    ],
)
def test_ablation_switches(switch, ignored, tokens):
    m = _model_with(**switch)
    base = _module_arrays(0)
    out = m.module_tokens(_obs(0, base))
    assert out.shape == (B, tokens, WIDTH)
    for field in ignored:
        assert torch.equal(out, m.module_tokens(_obs(0, _changed(base, EDITS[field])))), field
    with pytest.raises(ValueError, match="spatial or the temporal"):
        _model_with(module_spatial=False, module_temporal=False)


def test_encoders_get_gradients(model):
    model.zero_grad(set_to_none=True)
    _flow(model, _obs(0, _module_arrays(0))).mean().backward()
    for prefix in GlowModulePI0Pytorch.NEW_PARAM_PREFIXES:
        named = [(n, p) for n, p in model.named_parameters() if n.startswith(prefix)]
        assert named, prefix
        # Unused rows of embedding tables get none; every module gets some.
        modules = {n.split(".")[1] for n, p in named if p.grad is not None and p.grad.abs().sum() > 0}
        assert modules >= {"text_proj", "transformer", "out"}, (prefix, modules)
    model.zero_grad(set_to_none=True)


def test_missing_module_arrays_raise(model):
    imgs, masks, state, tokens, tmask, _, _ = _arrays(0, 0, B, with_block=False)
    obs = _model.Observation.from_dict(
        {"image": imgs, "image_mask": masks, "state": state, "tokenized_prompt": tokens, "tokenized_prompt_mask": tmask}
    )
    with pytest.raises(ValueError, match="module_inputs"):
        _act(model, obs)


def test_stock_prefix_hook_is_the_stock_prefix():
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    torch.manual_seed(1234)
    stock = PI0Pytorch(_cfg(pi0_config.Pi0Config))
    obs = _obs(0, _module_arrays(0))
    images, img_masks, lang_tokens, lang_masks, _ = stock._preprocess_observation(obs, train=False)  # noqa: SLF001
    a = stock.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    b = stock.embed_observation_prefix(obs, images, img_masks, lang_tokens, lang_masks)
    for x, y in zip(a, b, strict=True):
        assert torch.equal(x, y)


def _pooled(model, arrays, names):
    """Mean backbone embedding of the valid tokens of the named segments, per sample [B, D]."""
    embed = model.paligemma_with_expert.embed_language_tokens
    ids = np.concatenate([arrays["glow_text"][:, slice(*_mi.TEXT_SEGMENTS[n])] for n in names], axis=1)
    valid = np.concatenate([arrays["glow_text_mask"][:, slice(*_mi.TEXT_SEGMENTS[n])] for n in names], axis=1)
    emb = embed(torch.as_tensor(ids)).float() * torch.as_tensor(valid)[..., None]
    return emb.sum(1) / torch.as_tensor(valid).sum(1, keepdim=True).clamp(min=1)


def test_text_anchor_init_emits_the_anchors():
    m = _model_with(module_text_anchor=True)
    arrays = _module_arrays(0)
    tokens = m.module_tokens(_obs(0, arrays))
    stage = torch.as_tensor(arrays["glow_ints"][:, _mi.INT_STAGE])
    every_rk = [f"rk{j}" for j in range(_mi.K_MAX)]
    expected = [
        _pooled(m, arrays, ["subgoal"]),
        _pooled(m, arrays, ["object"]),
        _pooled(m, arrays, ["target"]),
        _pooled(m, arrays, ["subgoal", "object", "target"]),
        _pooled(m, arrays, every_rk),
        torch.stack([_pooled(m, arrays, [f"rk{j}"]) for j in range(_mi.K_MAX)], dim=1)[torch.arange(B), stage],
    ]
    torch.testing.assert_close(tokens, torch.stack(expected, dim=1), rtol=1e-5, atol=1e-6)
    # The prefix carries them, rescaled to the language tokens' RMS as unanchored tokens are.
    obs = _obs(0, arrays)
    images, img_masks, lang_tokens, lang_masks, _ = m._preprocess_observation(obs, train=False)  # noqa: SLF001
    embs, _, _ = m.embed_observation_prefix(obs, images, img_masks, lang_tokens, lang_masks)
    num_img = embs.shape[1] - lang_tokens.shape[1] - tokens.shape[1]
    cos = torch.nn.functional.cosine_similarity(embs[:, num_img : num_img + tokens.shape[1]], tokens, dim=-1)
    torch.testing.assert_close(cos, torch.ones_like(cos), rtol=0, atol=1e-5)


def test_text_anchor_without_subgoal_text():
    m = _model_with(module_text_anchor=True, module_subgoal_text=False)
    arrays = _module_arrays(0)
    tokens = m.module_tokens(_obs(0, arrays))
    assert torch.equal(tokens[:, 0], torch.zeros_like(tokens[:, 0]))  # OP anchors to zero
    torch.testing.assert_close(tokens[:, 3], _pooled(m, arrays, ["object", "target"]), rtol=1e-5, atol=1e-6)


def test_text_anchor_residual_trains():
    m = _model_with(module_text_anchor=True)
    _flow(m, _obs(0, _module_arrays(0))).mean().backward()
    for enc in (m.glow_spatial_encoder, m.glow_temporal_encoder):
        # The zero-initialized output layer moves first; everything before it waits for it.
        assert enc.out[1].weight.grad.abs().sum() > 0 and enc.out[1].bias.grad.abs().sum() > 0
        assert all(p.grad is None or p.grad.abs().sum() == 0 for p in enc.transformer.parameters())
    m.zero_grad(set_to_none=True)


def _teacher(seed=0, bsize=B):
    """Task-string tokens for the teacher pass: the stock prompt rows of `_arrays`."""
    _, _, _, tokens, tmask, _, _ = _arrays(seed + 50, 0, bsize, with_block=False)
    return {"glow_teacher_prompt": tokens, "glow_teacher_prompt_mask": tmask}


def _distill_obs(seed=0):
    return _obs(seed, {**_module_arrays(seed), **_teacher(seed)})


def _flow_inputs(seed=7):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, 10, 32, generator=g), torch.randn(B, 10, 32, generator=g), torch.rand(B, generator=g)


def _by_hand(m, obs, actions, noise, time):
    """Teacher and student velocities as the distillation forward computes them (seed 0)."""
    torch.manual_seed(0)
    images, img_masks, lang_tokens, lang_masks, state = m._preprocess_observation(obs, train=True)  # noqa: SLF001
    x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
    with torch.no_grad():
        teacher = m.embed_prefix(images, img_masks, obs.glow_teacher_prompt, obs.glow_teacher_prompt_mask)
        v_teacher = m.velocity_from_prefix(*teacher, state, x_t, time)
    student = m.embed_observation_prefix(obs, images, img_masks, lang_tokens, lang_masks)
    return m.velocity_from_prefix(*student, state, x_t, time), v_teacher


def test_distillation_forward():
    m = _model_with(module_text_anchor=True, module_distill_weight=0.5, module_flow_weight=2.0)
    obs = _distill_obs()
    actions, noise, time = _flow_inputs()
    m.train()
    torch.manual_seed(0)
    out = m(obs, actions, noise=noise, time=time)
    assert set(out) == {"loss", "flow", "distill", "teacher_flow"}

    with torch.no_grad():
        v, v_teacher = _by_hand(m, obs, actions, noise, time)
    u_t = noise - actions
    flow, distill = (u_t - v).pow(2).mean(), (v - v_teacher).pow(2).mean()
    torch.testing.assert_close(out["flow"], flow)
    torch.testing.assert_close(out["distill"], distill)
    torch.testing.assert_close(out["teacher_flow"], (u_t - v_teacher).pow(2).mean())
    torch.testing.assert_close(out["loss"], 2.0 * flow + 0.5 * distill)
    assert out["distill"] > 0 and out["loss"].requires_grad and not out["distill"].requires_grad

    # Evaluation mode is the plain per-element flow loss.
    m.eval()
    plain = m(obs, actions, noise=noise, time=time)
    assert isinstance(plain, torch.Tensor) and plain.shape == actions.shape


def test_distillation_teacher_carries_no_gradient():
    """The gradient is that of flow + distillation toward a constant target."""
    m = _model_with(module_text_anchor=True, module_distill_weight=1.0)
    obs = _distill_obs()
    actions, noise, time = _flow_inputs()
    m.train()
    torch.manual_seed(0)
    m(obs, actions, noise=noise, time=time)["loss"].backward()
    got = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    m.zero_grad(set_to_none=True)

    v, v_teacher = _by_hand(m, obs, actions, noise, time)
    ((noise - actions - v).pow(2).mean() + (v - v_teacher).pow(2).mean()).backward()
    want = {n: p.grad for n, p in m.named_parameters() if p.grad is not None}
    assert got.keys() == want.keys()
    for n in want:
        torch.testing.assert_close(got[n], want[n], rtol=1e-4, atol=1e-7, msg=n)
    m.zero_grad(set_to_none=True)


def test_distillation_needs_the_teacher_prompt():
    m = _model_with(module_distill_weight=1.0)
    m.train()
    with pytest.raises(ValueError, match="glow_teacher_prompt"):
        m(_obs(0, _module_arrays(0)), torch.zeros(B, 10, 32))


def test_stock_forward_is_its_velocity_helper():
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    torch.manual_seed(1234)
    stock = PI0Pytorch(_cfg(pi0_config.Pi0Config))
    obs = _obs(0, _module_arrays(0))
    actions, noise, time = _flow_inputs()
    stock.eval()
    torch.manual_seed(0)
    loss = stock(obs, actions, noise=noise, time=time)
    torch.manual_seed(0)
    images, img_masks, lang_tokens, lang_masks, state = stock._preprocess_observation(obs, train=True)  # noqa: SLF001
    x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
    prefix = stock.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    v = stock.velocity_from_prefix(*prefix, state, x_t, time)
    torch.testing.assert_close(loss, (noise - actions - v).pow(2))
