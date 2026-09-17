import contextlib
import datetime
import os
import pathlib

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

os.environ["JAX_PLATFORMS"] = "cpu"

from . import train_pytorch

_WORLD_SIZE = 2
_ACCUM_STEPS = 2
_OPTIMIZER_STEPS = 3
_MICRO_BATCH = 4


def test_ddp_timeout_defaults_to_an_hour(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENPI_DDP_TIMEOUT_MIN", raising=False)
    assert train_pytorch.resolve_ddp_timeout() == datetime.timedelta(minutes=60)


def test_ddp_timeout_reads_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENPI_DDP_TIMEOUT_MIN", "90")
    assert train_pytorch.resolve_ddp_timeout() == datetime.timedelta(minutes=90)


def test_ddp_timeout_rejects_non_positive(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENPI_DDP_TIMEOUT_MIN", "0")
    with pytest.raises(ValueError, match="OPENPI_DDP_TIMEOUT_MIN"):
        train_pytorch.resolve_ddp_timeout()


def _make_model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(6, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1)).double()


def _micro_batch(step: int, rank: int, micro: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(10_000 * step + 100 * rank + micro)
    x = torch.randn(_MICRO_BATCH, 6, generator=gen, dtype=torch.float64)
    y = torch.randn(_MICRO_BATCH, 1, generator=gen, dtype=torch.float64)
    return x, y


def _loss(model: torch.nn.Module, step: int, rank: int, micro: int) -> torch.Tensor:
    x, y = _micro_batch(step, rank, micro)
    return torch.nn.functional.mse_loss(model(x), y)


def _ddp_worker(rank: int, store: str, out_dir: str, keep_allocated: bool, sync_until_allocated: bool) -> None:  # noqa: FBT001
    """Runs train_pytorch.py's accumulation cycle on a small model under DDP (gloo, CPU).

    `sync_until_allocated=False` runs every non-final micro-batch under `no_sync()` regardless
    of whether the gradients are allocated. Records the final parameters and how many times the
    gradient storage moved between consecutive backward passes.
    """
    dist.init_process_group(
        "gloo",
        init_method=pathlib.Path(store).as_uri(),
        rank=rank,
        world_size=_WORLD_SIZE,
        timeout=datetime.timedelta(seconds=120),
    )
    try:
        # The trainer's DDP flags.
        model = torch.nn.parallel.DistributedDataParallel(
            _make_model(), find_unused_parameters=True, gradient_as_bucket_view=True
        )
        optim = torch.optim.AdamW(model.parameters(), lr=1e-2)
        grads_allocated = False
        pointers = None
        storage_moves = 0
        for step in range(_OPTIMIZER_STEPS):
            for micro in range(_ACCUM_STEPS):
                skip = train_pytorch.skip_grad_sync(
                    use_ddp=True,
                    is_last_micro_batch=micro + 1 == _ACCUM_STEPS,
                    grads_allocated=grads_allocated or not sync_until_allocated,
                )
                with model.no_sync() if skip else contextlib.nullcontext():
                    (_loss(model, step, rank, micro) / _ACCUM_STEPS).backward()
                current = [p.grad.data_ptr() for p in model.parameters()]
                if pointers is not None and current != pointers:
                    storage_moves += 1
                pointers = current
            optim.step()
            train_pytorch.zero_gradients(model, optim, keep_allocated=keep_allocated)
            grads_allocated = keep_allocated
        torch.save(
            {"params": [p.detach().clone() for p in model.module.parameters()], "storage_moves": storage_moves},
            os.path.join(out_dir, f"rank{rank}.pt"),
        )
    finally:
        dist.destroy_process_group()


def _reference_params() -> list[torch.Tensor]:
    """The same optimizer steps on one process, each over the full global batch."""
    model = _make_model()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for step in range(_OPTIMIZER_STEPS):
        for rank in range(_WORLD_SIZE):
            for micro in range(_ACCUM_STEPS):
                (_loss(model, step, rank, micro) / (_ACCUM_STEPS * _WORLD_SIZE)).backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
    return [p.detach() for p in model.parameters()]


def _run_ddp(tmp_path: pathlib.Path, *, keep_allocated: bool, sync_until_allocated: bool) -> list[dict]:
    mp.spawn(
        _ddp_worker,
        args=(str(tmp_path / "store"), str(tmp_path), keep_allocated, sync_until_allocated),
        nprocs=_WORLD_SIZE,
        join=True,
    )
    return [torch.load(tmp_path / f"rank{rank}.pt") for rank in range(_WORLD_SIZE)]


@pytest.mark.parametrize(("keep_allocated", "sync_until_allocated"), [(True, True), (False, False)])
def test_ddp_accumulation_matches_full_batch(
    tmp_path: pathlib.Path, *, keep_allocated: bool, sync_until_allocated: bool
):
    reference = _reference_params()
    results = _run_ddp(tmp_path, keep_allocated=keep_allocated, sync_until_allocated=sync_until_allocated)
    for result in results:
        for got, want in zip(result["params"], reference, strict=True):
            torch.testing.assert_close(got, want, rtol=0, atol=1e-12)


def test_trainer_accumulation_keeps_gradients_in_ddp_buckets(tmp_path: pathlib.Path):
    # The trainer's settings under accumulation: every backward writes into the storage the
    # first one allocated, so no second set of gradients ever exists.
    for result in _run_ddp(tmp_path, keep_allocated=True, sync_until_allocated=True):
        assert result["storage_moves"] == 0


def test_unsynced_backward_into_freed_gradients_allocates_outside_buckets(tmp_path: pathlib.Path):
    # Control for the storage check. The first micro-batch of every cycle allocates gradients
    # outside the buckets and the synced last micro-batch moves them in: two moves per cycle,
    # minus the first allocation, which is not a move.
    for result in _run_ddp(tmp_path, keep_allocated=False, sync_until_allocated=False):
        assert result["storage_moves"] == 2 * _OPTIMIZER_STEPS - 1
