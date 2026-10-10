"""FSDP (OPENPI_PARALLEL=fsdp) against a single-process reference, on the GLOW model at dummy scale.

Two CPU ranks over gloo each take half of every batch; the reference takes the whole batch.
Checked: per-step loss, the full gradients, the clipped norm, the parameters after AdamW steps,
that an FSDP checkpoint is the reference's checkpoint (same model.safetensors keys and
duplicate-name metadata, same optimizer.pt layout), and that a run resumes under FSDP from a
checkpoint written without it - the DDP-to-FSDP path a resumed run takes - and gradient
accumulation over micro-batches.

Image augmentation is switched off (preprocessing in eval mode) so both sides see the same
pixels; flow noise and time are passed in.
"""

import contextlib
import dataclasses
import datetime
import os
import pathlib
import types

import pytest
import safetensors
import safetensors.torch
import torch
import torch.distributed as dist
from torch.distributed.fsdp._fully_shard import _fsdp_collectives
import torch.multiprocessing as mp
from transformers.models.auto import CONFIG_MAPPING

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.glow.model_config import GlowPi0Config
import openpi.models_pytorch.gemma_pytorch as gemma_pytorch
from openpi.models_pytorch.glow_pi0_pytorch import GlowPI0Pytorch
from openpi.models_pytorch.glow_pi0_pytorch import _select_rows
from openpi.models_pytorch.glow_pi0_pytorch_test import B
from openpi.models_pytorch.glow_pi0_pytorch_test import _cfg
from openpi.models_pytorch.glow_pi0_pytorch_test import _make_obs
import openpi.models_pytorch.preprocessing_pytorch as preprocessing

from . import train_pytorch

_WORLD_SIZE = 2
_STEPS = 2
_LR = 1e-3
# Small enough that clipping is active on every step.
_MAX_NORM = 0.05
# Parameter tolerance after the AdamW steps: 1% of the most a step can move a parameter (lr).
# Adam divides each gradient component by its own running RMS, so components near eps (rows of
# the tied embedding the CE head barely touches) turn float32 summation-order differences into
# visible parameter differences, ~7e-6 here; a missing or unreduced gradient moves parameters by
# up to lr.
_PARAM_ATOL = 1e-2 * _LR * _STEPS


@dataclasses.dataclass
class _Config:
    save_interval: int
    num_train_steps: int
    checkpoint_dir: pathlib.Path
    wandb_enabled: bool = False


_DATA_CONFIG = types.SimpleNamespace(norm_stats=None, asset_id=None)


@contextlib.contextmanager
def _test_environment():
    """Tiny SigLIP (the dummy Gemma is 64 wide) and augmentation-free preprocessing."""
    orig_paligemma = CONFIG_MAPPING["paligemma"]

    def tiny():
        cfg = orig_paligemma()
        cfg.vision_config.hidden_size = 32
        cfg.vision_config.num_hidden_layers = 2
        cfg.vision_config.num_attention_heads = 4
        cfg.vision_config.image_size = 224
        cfg.vision_config.patch_size = 56
        return cfg

    orig_preprocess = preprocessing.preprocess_observation_pytorch

    def no_augmentation(observation, *, train=True, **kwargs):
        return orig_preprocess(observation, train=False, **kwargs)

    saved_mapping = gemma_pytorch.CONFIG_MAPPING
    gemma_pytorch.CONFIG_MAPPING = {**CONFIG_MAPPING, "paligemma": tiny}
    preprocessing.preprocess_observation_pytorch = no_augmentation
    try:
        yield
    finally:
        gemma_pytorch.CONFIG_MAPPING = saved_mapping
        preprocessing.preprocess_observation_pytorch = orig_preprocess


def _make_model(ce_weight: float) -> torch.nn.Module:
    torch.manual_seed(1234)
    model = GlowPI0Pytorch(_cfg(GlowPi0Config, subtask_ce_weight=ce_weight))
    model.gradient_checkpointing_enable()
    model.train()
    return model


def _batch(step: int, rows: slice):
    """Step `step`'s batch of 4 restricted to `rows`. Rows 0 and 2 carry a 6-token subtask
    block, so each half holds the same number of CE targets and the per-rank CE means average
    to the full-batch mean."""
    obs = _make_obs(100 + step, with_block=True, block_rows=(0, 2))
    g = torch.Generator().manual_seed(200 + step)
    actions = torch.randn(B, 10, 32, generator=g)
    noise = torch.randn(B, 10, 32, generator=g)
    time = torch.rand(B, generator=g) * 0.98 + 0.01
    selected = torch.zeros(B, dtype=torch.bool)
    selected[rows] = True
    return _select_rows(obs, selected), actions[rows], noise[rows], time[rows]


def _loss(model, step: int, rows: slice) -> torch.Tensor:
    obs, actions, noise, time = _batch(step, rows)
    out = model(obs, actions, noise=noise, time=time)
    return out["loss"] if isinstance(out, dict) else out.mean()


def _optimizer(model) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=_LR, weight_decay=1e-4)


def _reference(tmp_path: pathlib.Path, ce_weight: float) -> dict:
    """Single process, whole batches. Writes the init weights and checkpoints at steps 1 and
    _STEPS in the single-process format."""
    with _test_environment():
        model = _make_model(ce_weight)
        safetensors.torch.save_model(model, tmp_path / "init.safetensors")
        optim = _optimizer(model)
        record = {"loss": [], "grad_norm": [], "grads": None}
        for step in range(_STEPS):
            loss = _loss(model, step, slice(0, 4))
            loss.backward()
            if step == 0:
                record["grads"] = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
            record["grad_norm"].append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), _MAX_NORM)))
            record["loss"].append(float(loss))
            optim.step()
            train_pytorch.zero_gradients(model, optim, keep_allocated=False)
            # After the first step: the checkpoint the FSDP run resumes from.
            if step == 0:
                config = _Config(save_interval=1, num_train_steps=100, checkpoint_dir=tmp_path / "ckpt_resume_from")
                train_pytorch.save_checkpoint(model, optim, 1, config, is_main=True, data_config=_DATA_CONFIG)
        config = _Config(save_interval=_STEPS, num_train_steps=100, checkpoint_dir=tmp_path / "ckpt_ref")
        train_pytorch.save_checkpoint(model, optim, _STEPS, config, is_main=True, data_config=_DATA_CONFIG)
        record["params"] = {n: p.detach().clone() for n, p in model.named_parameters()}
        return record


def _fsdp_worker(rank: int, tmp: str, ce_weight: float, mode: str) -> None:
    """mode: "fresh" (from the init weights), "resume" (from the reference's step-1
    checkpoint) or "accum" (fresh, each rank's half batch as two micro-batches of one)."""
    tmp_path = pathlib.Path(tmp)
    resume = mode == "resume"
    accum_steps = 2 if mode == "accum" else 1
    os.environ["OPENPI_PARALLEL"] = "fsdp"
    dist.init_process_group(
        "gloo",
        init_method=(tmp_path / f"store_{mode}").as_uri(),
        rank=rank,
        world_size=_WORLD_SIZE,
        timeout=datetime.timedelta(seconds=300),
    )
    is_main = rank == 0
    device = torch.device("cpu")
    # FSDP2 reduce-scatters float32 gradients with ReduceOp.AVG, which NCCL implements and gloo
    # silently treats as a sum. Route it through FSDP2's own sum-then-divide path (the one it
    # uses for float16) so the CPU ranks average too.
    _fsdp_collectives._get_gradient_divide_factors = lambda *_, **__: (1.0, float(_WORLD_SIZE))  # noqa: SLF001
    try:
        with _test_environment():
            model = _make_model(ce_weight)
            if resume:
                ckpt_root = tmp_path / "ckpt_resume_from"
                train_pytorch.load_model_weights(model, ckpt_root / "1", device)
            else:
                safetensors.torch.load_model(model, tmp_path / "init.safetensors")
            model = train_pytorch._fsdp.shard_model(model, train_pytorch._fsdp.build_mesh(device))  # noqa: SLF001
            optim = _optimizer(model)
            first_step = 0
            if resume:
                first_step = train_pytorch.load_checkpoint(model, optim, ckpt_root, device, fsdp=True)
            rows = slice(2 * rank, 2 * rank + 2)
            record = {"loss": [], "grad_norm": [], "grads": None}
            for step in range(first_step, _STEPS):
                micro = 2 // accum_steps
                loss = 0.0
                for start in range(rows.start, rows.stop, micro):
                    micro_loss = _loss(model, step, slice(start, start + micro)) / accum_steps
                    micro_loss.backward()
                    train_pytorch._fsdp.reshard(model)  # noqa: SLF001
                    loss += micro_loss.detach()
                if step == 0:
                    grads = {
                        n: train_pytorch._fsdp._gather(p.grad, is_main=is_main)  # noqa: SLF001
                        for n, p in model.named_parameters()
                        if p.grad is not None
                    }
                    record["grads"] = grads
                norm = train_pytorch._fsdp.clip_grad_norm_(model.parameters(), _MAX_NORM)  # noqa: SLF001
                local_loss = loss.detach().clone()
                dist.all_reduce(local_loss)
                record["loss"].append(float(local_loss) / _WORLD_SIZE)
                record["grad_norm"].append(float(norm))
                optim.step()
                train_pytorch.zero_gradients(model, optim, keep_allocated=accum_steps > 1)
            config = _Config(save_interval=_STEPS, num_train_steps=100, checkpoint_dir=tmp_path / f"ckpt_fsdp_{mode}")
            train_pytorch.save_checkpoint(model, optim, _STEPS, config, is_main, _DATA_CONFIG, fsdp=True)
            state = train_pytorch._fsdp.full_model_state_dict(model, is_main=is_main)  # noqa: SLF001
            record["params"] = {n: state[n] for n, _ in model.named_parameters()} if is_main else None
            record["param_types"] = sorted({type(p).__name__ for p in model.parameters()})
            if is_main:
                torch.save(record, tmp_path / f"fsdp_{mode}.pt")
    finally:
        dist.destroy_process_group()


def _run_fsdp(tmp_path: pathlib.Path, ce_weight: float, mode: str) -> dict:
    mp.spawn(_fsdp_worker, args=(str(tmp_path), ce_weight, mode), nprocs=_WORLD_SIZE, join=True)
    return torch.load(tmp_path / f"fsdp_{mode}.pt", weights_only=False)


def _assert_params_close(got: dict, want: dict) -> None:
    assert got.keys() == want.keys()
    for name, value in want.items():
        torch.testing.assert_close(got[name], value, rtol=0, atol=_PARAM_ATOL, msg=name)


@pytest.fixture(scope="module", params=[0.1, 0.0], ids=["glow_ce", "stock_flow"])
def runs(request, tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp(f"fsdp_{request.param}")
    reference = _reference(tmp_path, request.param)
    fsdp = _run_fsdp(tmp_path, request.param, "fresh")
    resumed = _run_fsdp(tmp_path, request.param, "resume")
    return tmp_path, reference, fsdp, resumed


def test_parameters_are_sharded(runs):
    _, _, fsdp, _ = runs
    assert fsdp["param_types"] == ["DTensor"]


def test_losses_and_clipped_norms_match(runs):
    _, reference, fsdp, _ = runs
    torch.testing.assert_close(fsdp["loss"], reference["loss"], rtol=1e-5, atol=0)
    torch.testing.assert_close(fsdp["grad_norm"], reference["grad_norm"], rtol=1e-4, atol=0)
    assert all(norm > _MAX_NORM for norm in reference["grad_norm"])


def test_gradients_match(runs):
    _, reference, fsdp, _ = runs
    # Same set of parameters with gradients: the unused ones (the expert's lm_head, and under the
    # flow loss alone the last PaliGemma layer's MLP) have none on either side.
    assert fsdp["grads"].keys() == reference["grads"].keys()
    for name, grad in reference["grads"].items():
        torch.testing.assert_close(fsdp["grads"][name], grad, rtol=1e-4, atol=1e-7, msg=name)


def test_parameters_after_adamw_steps_match(runs):
    _, reference, fsdp, _ = runs
    _assert_params_close(fsdp["params"], reference["params"])


def test_resume_from_single_process_checkpoint_matches(runs):
    _, reference, _, resumed = runs
    _assert_params_close(resumed["params"], reference["params"])


def test_checkpoint_files_match_single_process_format(runs):
    tmp_path, _, _, _ = runs
    ref_dir = tmp_path / "ckpt_ref" / str(_STEPS)
    fsdp_dir = tmp_path / "ckpt_fsdp_fresh" / str(_STEPS)

    with (
        safetensors.safe_open(ref_dir / "model.safetensors", "pt") as ref,
        safetensors.safe_open(fsdp_dir / "model.safetensors", "pt") as got,
    ):
        assert got.metadata() == ref.metadata()
        assert ref.metadata()  # the tied embedding / lm_head pair is recorded
        assert sorted(got.keys()) == sorted(ref.keys())
        for key in ref.keys():  # noqa: SIM118
            torch.testing.assert_close(got.get_tensor(key), ref.get_tensor(key), rtol=0, atol=_PARAM_ATOL, msg=key)

    ref_opt = torch.load(ref_dir / "optimizer.pt", weights_only=False)
    got_opt = torch.load(fsdp_dir / "optimizer.pt", weights_only=False)
    assert got_opt["param_groups"] == ref_opt["param_groups"]
    assert got_opt["state"].keys() == ref_opt["state"].keys()
    for index, entry in ref_opt["state"].items():
        assert got_opt["state"][index].keys() == entry.keys()
        for key, value in entry.items():
            got = got_opt["state"][index][key]
            assert type(got) is torch.Tensor
            assert got.shape == value.shape
            assert got.device == value.device
            if key == "step":
                torch.testing.assert_close(got, value, rtol=0, atol=0)
            else:
                torch.testing.assert_close(got, value, rtol=1e-4, atol=1e-9)


def test_accumulation_matches_full_batch(tmp_path):
    # Flow loss only: with the CE term, micro-batches that hold no subtask target weight the CE
    # differently from the full batch whatever the parallelism.
    reference = _reference(tmp_path, 0.0)
    accumulated = _run_fsdp(tmp_path, 0.0, "accum")
    torch.testing.assert_close(accumulated["grad_norm"], reference["grad_norm"], rtol=1e-4, atol=0)
    _assert_params_close(accumulated["params"], reference["params"])


def test_parallel_mode_is_validated(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENPI_PARALLEL", "fsdp")
    assert train_pytorch._fsdp.resolve_parallelism() == "fsdp"  # noqa: SLF001
    monkeypatch.setenv("OPENPI_PARALLEL", "zero3")
    with pytest.raises(ValueError, match="OPENPI_PARALLEL"):
        train_pytorch._fsdp.resolve_parallelism()  # noqa: SLF001
