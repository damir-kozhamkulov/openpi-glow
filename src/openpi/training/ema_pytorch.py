"""Exponential moving average (EMA) of model parameters for the PyTorch trainer.

Follows the JAX trainer (`scripts/train.py`, `TrainState.ema_params`): the average is
``decay * ema + (1 - decay) * param`` after every optimizer step, seeded with the initial
weights, with no bias correction. Checkpoints hold the averaged weights where evaluation
looks for them — the same substitution the JAX trainer makes in
`openpi/training/checkpoints.py:_split_params`.

The accumulator is float32 even when the model trains in bfloat16. With ``decay=0.999``
one update moves the accumulator by 0.1% of ``(param - accumulator)``; bfloat16's 8-bit
mantissa spaces representable values ~0.4% apart, so that increment rounds away entirely
and the average never leaves its starting point. See
`ema_pytorch_test.py::test_bfloat16_accumulator_would_stall`.
"""

from collections.abc import Iterator
import contextlib
import logging

import torch

# Element count per slice when comparing the accumulator against live parameters. Keeps the
# float32 temporaries bounded (~16 MiB) instead of allocating one the size of the largest
# parameter - the token embedding alone would be 2 GiB, at the point in the step where memory
# is already at its peak.
_COMPARE_CHUNK_ELEMENTS = 4_000_000


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module, whether or not `model` is wrapped in DDP."""
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


class ParameterEma:
    """A float32 shadow copy of a model's trainable parameters, updated in place each step.

    Only parameters with ``requires_grad`` are tracked. Frozen parameters never change, so
    averaging them is a no-op that the JAX trainer performs and we skip; the live model
    already holds the correct values and `swapped_into` leaves them untouched.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.dtype = dtype
        self._shadow: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, param in self._tracked(model):
                self._shadow[name] = param.detach().to(
                    device=device if device is not None else param.device, dtype=dtype, copy=True
                )

    @staticmethod
    def _tracked(model: torch.nn.Module) -> Iterator[tuple[str, torch.nn.Parameter]]:
        for name, param in unwrap_model(model).named_parameters():
            if param.requires_grad and param.is_floating_point():
                yield name, param

    @property
    def device(self) -> torch.device:
        return next(iter(self._shadow.values())).device

    @property
    def num_parameters(self) -> int:
        return sum(t.numel() for t in self._shadow.values())

    @property
    def num_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self._shadow.values())

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """Apply ``ema = decay * ema + (1 - decay) * param``. Call once per optimizer step."""
        for name, param in self._tracked(model):
            shadow = self._shadow.get(name)
            if shadow is None:
                continue
            source = param.detach()
            if source.device != shadow.device:
                source = source.to(shadow.device, non_blocking=True)
            # mul_ + add_ rather than lerp_: add_ promotes a bfloat16 `source` inside the
            # kernel, so no float32 temporary the size of the parameter is ever materialized.
            shadow.mul_(self.decay).add_(source, alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_from(self, model: torch.nn.Module) -> None:
        """Reset the accumulator to the model's current parameters (used when resuming)."""
        for name, param in self._tracked(model):
            shadow = self._shadow.get(name)
            if shadow is not None:
                shadow.copy_(param.detach())

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        """Overwrite the model's parameters with the averaged ones. Not reversible."""
        for name, param in self._tracked(model):
            shadow = self._shadow.get(name)
            if shadow is not None:
                param.copy_(shadow)

    @contextlib.contextmanager
    def swapped_into(
        self, model: torch.nn.Module, *, backup_device: str | torch.device = "cpu"
    ) -> Iterator[torch.nn.Module]:
        """Temporarily install the averaged parameters, e.g. to serialize them.

        The displaced parameters are held on `backup_device` (host memory by default, so the
        swap costs no GPU memory at a moment when a checkpoint write is already in flight) and
        restored on the way out, including when the body raises.
        """
        module = unwrap_model(model)
        backup: dict[str, torch.Tensor] = {}
        try:
            with torch.no_grad():
                for name, param in self._tracked(module):
                    shadow = self._shadow.get(name)
                    if shadow is None:
                        continue
                    backup[name] = param.detach().to(backup_device, copy=True)
                    param.copy_(shadow)
            yield module
        finally:
            with torch.no_grad():
                for name, param in self._tracked(module):
                    saved = backup.get(name)
                    if saved is not None:
                        param.copy_(saved)

    @torch.no_grad()
    def relative_distance(self, model: torch.nn.Module) -> float:
        """||ema - params|| / ||params||, over all tracked parameters.

        Check: nonzero whenever averaging is actually moving the accumulator.
        """
        diff_sq: torch.Tensor | None = None
        ref_sq: torch.Tensor | None = None
        for name, param in self._tracked(model):
            shadow = self._shadow.get(name)
            if shadow is None:
                continue
            source = param.detach()
            if source.device != shadow.device:
                source = source.to(shadow.device)
            flat_shadow = shadow.reshape(-1)
            flat_source = source.reshape(-1)
            for shadow_chunk, source_chunk in zip(
                flat_shadow.split(_COMPARE_CHUNK_ELEMENTS), flat_source.split(_COMPARE_CHUNK_ELEMENTS), strict=True
            ):
                source_chunk = source_chunk.to(self.dtype)  # noqa: PLW2901
                delta = (shadow_chunk - source_chunk).pow_(2).sum()
                norm = source_chunk.pow(2).sum()
                diff_sq = delta if diff_sq is None else diff_sq + delta
                ref_sq = norm if ref_sq is None else ref_sq + norm
        if diff_sq is None or ref_sq is None or float(ref_sq) == 0.0:
            return 0.0
        return float(torch.sqrt(diff_sq / ref_sq))


def create_ema(
    model: torch.nn.Module,
    decay: float,
    *,
    device_preference: str = "auto",
) -> ParameterEma:
    """Build a `ParameterEma`, falling back to host memory if the device copy does not fit.

    The accumulator holds the trainable parameters in float32 (~13 GiB for pi05). If that
    allocation OOMs on the training device, a host-resident accumulator behaves identically
    at the cost of one device-to-host parameter copy per step.
    """
    if device_preference == "cpu":
        return ParameterEma(model, decay, device=torch.device("cpu"))
    device = None if device_preference == "auto" else torch.device(device_preference)
    try:
        return ParameterEma(model, decay, device=device)
    except torch.cuda.OutOfMemoryError:
        logging.warning(
            "EMA accumulator did not fit on the training device; falling back to host memory. "
            "This adds a parameter-sized host copy per step. Set OPENPI_PYTORCH_EMA_DEVICE=cpu "
            "to choose this up front, or OPENPI_PYTORCH_EMA=0 to train without EMA."
        )
    # Outside the handler: the partially built accumulator is reachable from the exception's
    # traceback until the handler exits, and its device memory is not released before then.
    torch.cuda.empty_cache()
    return ParameterEma(model, decay, device=torch.device("cpu"))
