"""FSDP (OPENPI_PARALLEL=fsdp) on the GLOW module model's encoder-only stage, against a single
process: the backbone frozen, the loss flow + distillation (a teacher pass without gradients, then
the student pass, on the same sharded parameters).

Two CPU ranks over gloo each take half of every batch; the reference takes the whole batch.
Checked: per-step loss, the encoders' first-step gradients, the frozen parameters untouched on
both sides, and the encoders after AdamW steps. Image augmentation is off (see train_pytorch_fsdp_test); flow noise and time are
passed in.
"""

import dataclasses
import datetime
import os
import pathlib

import pytest
import safetensors.torch
import torch
import torch.distributed as dist
from torch.distributed.fsdp._fully_shard import _fsdp_collectives
import torch.multiprocessing as mp

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.glow.model_config import GlowModulePi0Config
from openpi.models_pytorch.glow_module_pi0_pytorch import GlowModulePI0Pytorch
from openpi.models_pytorch.glow_module_pi0_pytorch_test import B
from openpi.models_pytorch.glow_module_pi0_pytorch_test import _distill_obs
from openpi.models_pytorch.glow_pi0_pytorch_test import _cfg

from . import train_pytorch
from .train_pytorch_fsdp_test import _test_environment

_WORLD_SIZE = 2
_STEPS = 2
_LR = 1e-3
_MAX_NORM = 0.05
_ENCODER_LR_SCALE = 16.0
# Encoder parameter tolerance after the AdamW steps: 5% of the most a step can move one. The
# zero-initialized output layers make every gradient before them exactly zero on the first step
# and tiny on the second, where Adam's division by the running RMS turns float32 summation-order
# differences into visible ones (~2% here); a missing or unreduced gradient moves them by up to
# the whole step.
_PARAM_ATOL = 5e-2 * _LR * _ENCODER_LR_SCALE * _STEPS


def _make_model() -> torch.nn.Module:
    torch.manual_seed(1234)
    model = GlowModulePI0Pytorch(
        _cfg(
            GlowModulePi0Config,
            module_width=32,
            module_text_anchor=True,
            freeze_pretrained=True,
            module_distill_weight=1.0,
        )
    )
    train_pytorch.freeze_pretrained(model)
    model.gradient_checkpointing_enable()
    model.train()
    return model


def _select_rows(observation, selected):
    """The observation (every field, module and teacher arrays included) restricted to rows."""
    out = {}
    for field in dataclasses.fields(observation):
        value = getattr(observation, field.name)
        if isinstance(value, dict):
            value = {k: v[selected] for k, v in value.items()}
        elif value is not None:
            value = value[selected]
        out[field.name] = value
    return dataclasses.replace(observation, **out)


def _loss(model, step: int, rows: slice) -> torch.Tensor:
    obs = _distill_obs(100 + step)
    g = torch.Generator().manual_seed(200 + step)
    actions = torch.randn(B, 10, 32, generator=g)
    noise = torch.randn(B, 10, 32, generator=g)
    time = torch.rand(B, generator=g) * 0.98 + 0.01
    selected = torch.zeros(B, dtype=torch.bool)
    selected[rows] = True
    return model(_select_rows(obs, selected), actions[rows], noise=noise[rows], time=time[rows])["loss"]


def _optimizer(model) -> torch.optim.Optimizer:
    groups = train_pytorch.optimizer_param_groups(model, _ENCODER_LR_SCALE)
    return torch.optim.AdamW([{**g, "lr": _LR * g["lr_scale"]} for g in groups], lr=_LR, weight_decay=1e-4)


def _reference(tmp_path: pathlib.Path) -> dict:
    with _test_environment():
        model = _make_model()
        safetensors.torch.save_model(model, tmp_path / "init.safetensors")
        init = {n: p.detach().clone() for n, p in model.named_parameters()}
        optim = _optimizer(model)
        losses, grads = [], None
        for step in range(_STEPS):
            loss = _loss(model, step, slice(0, B))
            loss.backward()
            if step == 0:
                grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
            torch.nn.utils.clip_grad_norm_(model.parameters(), _MAX_NORM)
            losses.append(float(loss))
            optim.step()
            train_pytorch.zero_gradients(model, optim, keep_allocated=False)
        params = {n: p.detach().clone() for n, p in model.named_parameters()}
        return {"loss": losses, "grads": grads, "init": init, "params": params}


def _fsdp_worker(rank: int, tmp: str) -> None:
    tmp_path = pathlib.Path(tmp)
    os.environ["OPENPI_PARALLEL"] = "fsdp"
    dist.init_process_group(
        "gloo",
        init_method=(tmp_path / "store").as_uri(),
        rank=rank,
        world_size=_WORLD_SIZE,
        timeout=datetime.timedelta(seconds=300),
    )
    is_main = rank == 0
    device = torch.device("cpu")
    # gloo treats ReduceOp.AVG as a sum; see train_pytorch_fsdp_test.
    _fsdp_collectives._get_gradient_divide_factors = lambda *_, **__: (1.0, float(_WORLD_SIZE))  # noqa: SLF001
    try:
        with _test_environment():
            model = _make_model()
            safetensors.torch.load_model(model, tmp_path / "init.safetensors")
            model = train_pytorch._fsdp.shard_model(model, train_pytorch._fsdp.build_mesh(device))  # noqa: SLF001
            optim = _optimizer(model)
            rows = slice(2 * rank, 2 * rank + 2)
            losses, grads = [], None
            for step in range(_STEPS):
                loss = _loss(model, step, rows)
                loss.backward()
                train_pytorch._fsdp.reshard(model)  # noqa: SLF001
                if step == 0:
                    grads = {
                        n: train_pytorch._fsdp._gather(p.grad, is_main=is_main)  # noqa: SLF001
                        for n, p in model.named_parameters()
                        if p.grad is not None
                    }
                train_pytorch._fsdp.clip_grad_norm_(model.parameters(), _MAX_NORM)  # noqa: SLF001
                local = loss.detach().clone()
                dist.all_reduce(local)
                losses.append(float(local) / _WORLD_SIZE)
                optim.step()
                train_pytorch.zero_gradients(model, optim, keep_allocated=False)
            state = train_pytorch._fsdp.full_model_state_dict(model, is_main=is_main)  # noqa: SLF001
            if is_main:
                params = {n: state[n] for n, _ in model.named_parameters()}
                torch.save({"loss": losses, "grads": grads, "params": params}, tmp_path / "fsdp.pt")
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("fsdp_module")
    reference = _reference(tmp_path)
    mp.spawn(_fsdp_worker, args=(str(tmp_path),), nprocs=_WORLD_SIZE, join=True)
    return reference, torch.load(tmp_path / "fsdp.pt", weights_only=False)


def test_losses_match(runs):
    reference, fsdp = runs
    assert fsdp["loss"] == pytest.approx(reference["loss"], rel=1e-5)


def test_first_step_gradients_match(runs):
    reference, fsdp = runs
    # Gradients reach exactly the encoders, the same ones on both sides.
    assert fsdp["grads"].keys() == reference["grads"].keys()
    assert all(n.startswith(GlowModulePI0Pytorch.NEW_PARAM_PREFIXES) for n in reference["grads"])
    for name, want in reference["grads"].items():
        torch.testing.assert_close(fsdp["grads"][name], want, rtol=1e-4, atol=1e-7, msg=name)


def test_only_the_encoders_move(runs):
    reference, fsdp = runs
    prefixes = GlowModulePI0Pytorch.NEW_PARAM_PREFIXES
    assert fsdp["params"].keys() == reference["params"].keys()
    moved, off = 0, {}
    for name, want in reference["params"].items():
        got = fsdp["params"][name]
        if name.startswith(prefixes):
            diff = float((got - want).abs().max())
            if diff > _PARAM_ATOL:
                off[name] = diff
            moved += not torch.equal(want, reference["init"][name])
        else:
            assert torch.equal(want, reference["init"][name]), name
            assert torch.equal(got, reference["init"][name]), name
    assert not off, f"encoder parameters off by more than {_PARAM_ATOL}: {off}"
    assert moved > 0
