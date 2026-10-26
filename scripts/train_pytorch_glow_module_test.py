"""Trainer support for GLOW's module encoders: the partial init load and the encoder LR group."""

import pytest
import safetensors.torch
import torch

from openpi.glow.model_config import GlowModulePi0Config
from openpi.models_pytorch.glow_module_pi0_pytorch import GlowModulePI0Pytorch
from openpi.models_pytorch.glow_pi0_pytorch_test import _cfg
from openpi.models_pytorch.glow_pi0_pytorch_test import tiny_vision_tower  # noqa: F401
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
import openpi.models.pi0_config as pi0_config

from . import train_pytorch


@pytest.fixture
def init_file(tmp_path):
    """A stock pi05 checkpoint: what pi05_libero_ref_pytorch is to the module model."""
    torch.manual_seed(0)
    stock = PI0Pytorch(_cfg(pi0_config.Pi0Config))
    path = tmp_path / "model.safetensors"
    safetensors.torch.save_model(stock, str(path))
    return stock, path


def _module_model():
    torch.manual_seed(1)
    return GlowModulePI0Pytorch(_cfg(GlowModulePi0Config, module_width=32))


def test_init_load_leaves_only_the_encoders_fresh(init_file):
    stock, path = init_file
    model = _module_model()
    fresh = {n: p.detach().clone() for n, p in model.named_parameters() if n.startswith(model.NEW_PARAM_PREFIXES)}
    train_pytorch.load_init_weights(model, str(path), "cpu")
    stock_params = dict(stock.named_parameters())
    for name, p in model.named_parameters():
        if name in fresh:
            assert torch.equal(p, fresh[name]), name
        else:
            assert torch.equal(p, stock_params[name]), name


def test_init_load_still_strict_outside_the_encoders(init_file, tmp_path):
    _, path = init_file
    tensors = safetensors.torch.load_file(str(path))
    dropped = next(k for k in tensors if k.startswith("action_out_proj"))
    del tensors[dropped]
    partial = tmp_path / "partial.safetensors"
    safetensors.torch.save_file(tensors, str(partial))
    with pytest.raises(RuntimeError, match="action_out_proj"):
        train_pytorch.load_init_weights(_module_model(), str(partial), "cpu")
    # A model without new parameters loads strictly, as before.
    with pytest.raises(RuntimeError):
        train_pytorch.load_init_weights(PI0Pytorch(_cfg(pi0_config.Pi0Config)), str(partial), "cpu")


def test_param_groups():
    model = _module_model()
    groups = train_pytorch.optimizer_param_groups(model, 16.0)
    assert [g["name"] for g in groups] == ["pretrained", "new"]
    assert [g["lr_scale"] for g in groups] == [1.0, 16.0]
    new = {id(p) for p in groups[1]["params"]}
    expected = {id(p) for n, p in model.named_parameters() if n.startswith(model.NEW_PARAM_PREFIXES)}
    assert new == expected and new
    assert len(groups[0]["params"]) + len(groups[1]["params"]) == len(list(model.parameters()))
    # The stock model has one group.
    stock_groups = train_pytorch.optimizer_param_groups(PI0Pytorch(_cfg(pi0_config.Pi0Config)), 16.0)
    assert len(stock_groups) == 1 and stock_groups[0]["lr_scale"] == 1.0


def test_grad_norm_of():
    a, b = torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(3))
    assert train_pytorch.grad_norm_of([a, b]) == 0.0
    a.grad = torch.tensor([3.0, 0.0])
    b.grad = torch.tensor([0.0, 4.0, 0.0])
    assert train_pytorch.grad_norm_of([a, b]) == pytest.approx(5.0)


def _align_model():
    torch.manual_seed(1)
    return GlowModulePI0Pytorch(
        _cfg(
            GlowModulePi0Config,
            module_width=32,
            module_text_anchor=True,
            freeze_pretrained=True,
            module_distill_weight=1.0,
        )
    )


def test_freeze_pretrained():
    model = _align_model()
    frozen = train_pytorch.freeze_pretrained(model)
    new = {n for n, _ in model.named_parameters() if n.startswith(model.NEW_PARAM_PREFIXES)}
    for name, p in model.named_parameters():
        assert p.requires_grad == (name in new), name
    assert frozen == sum(p.numel() for n, p in model.named_parameters() if n not in new) > 0
    # Frozen parameters stay out of the optimizer, and so does the group they empty.
    groups = train_pytorch.optimizer_param_groups(model, 16.0)
    assert [g["name"] for g in groups] == ["new"] and groups[0]["lr_scale"] == 16.0
    assert {id(p) for p in groups[0]["params"]} == {id(p) for n, p in model.named_parameters() if n in new}
    with pytest.raises(ValueError, match="nothing would train"):
        train_pytorch.freeze_pretrained(PI0Pytorch(_cfg(pi0_config.Pi0Config)))


def test_frozen_backbone_step_moves_only_the_encoders():
    from openpi.models_pytorch.glow_module_pi0_pytorch_test import _distill_obs
    from openpi.models_pytorch.glow_module_pi0_pytorch_test import _flow_inputs

    model = _align_model()
    train_pytorch.freeze_pretrained(model)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optim = torch.optim.AdamW(train_pytorch.optimizer_param_groups(model, 16.0), lr=1e-3)
    model.train()
    actions, noise, time = _flow_inputs()
    out = model(_distill_obs(), actions, noise=noise, time=time)
    out["loss"].backward()
    optim.step()
    for name, p in model.named_parameters():
        if name.startswith(model.NEW_PARAM_PREFIXES):
            assert p.grad is not None, name
        else:
            assert p.grad is None and torch.equal(p, before[name]), name
    # The zero-initialized output layers move on the first step.
    for enc in ("glow_spatial_encoder", "glow_temporal_encoder"):
        name = f"{enc}.out.1.weight"
        assert not torch.equal(dict(model.named_parameters())[name], before[name]), name
    assert all(id(p) in {id(q) for q in optim.state} for p in model.parameters() if p.requires_grad)
    assert len(optim.state) == sum(1 for p in model.parameters() if p.requires_grad)
