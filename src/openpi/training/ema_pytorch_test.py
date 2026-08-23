import copy

import pytest
import torch

from openpi.training import ema_pytorch


class _TinyModel(torch.nn.Module):
    """Two layers with a frozen one in between, to cover parameter selection."""

    def __init__(self, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.trainable = torch.nn.Linear(4, 4, dtype=dtype)
        self.frozen = torch.nn.Linear(4, 4, dtype=dtype)
        self.frozen.weight.requires_grad = False
        self.frozen.bias.requires_grad = False


def _step(model: torch.nn.Module, delta: float) -> None:
    """Stand-in for an optimizer step: move every parameter by a fixed amount."""
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                param.add_(torch.full_like(param, delta))


def test_matches_the_jax_recursion():
    torch.manual_seed(0)
    model = _TinyModel()
    decay = 0.9
    ema = ema_pytorch.ParameterEma(model, decay)

    # Reference: the exact recursion from scripts/train.py, in float64, with no bias
    # correction and seeded at the initial parameters.
    reference = {name: p.detach().double().clone() for name, p in model.named_parameters() if p.requires_grad}
    for _ in range(20):
        _step(model, 0.25)
        ema.update(model)
        for name, param in model.named_parameters():
            if param.requires_grad:
                reference[name] = decay * reference[name] + (1 - decay) * param.detach().double()

    with ema.swapped_into(model) as swapped:
        for name, param in swapped.named_parameters():
            if param.requires_grad:
                torch.testing.assert_close(param.double(), reference[name], rtol=0, atol=1e-6)


def test_tracks_only_trainable_parameters():
    model = _TinyModel()
    ema = ema_pytorch.ParameterEma(model, 0.9)

    assert set(ema._shadow) == {"trainable.weight", "trainable.bias"}  # noqa: SLF001
    assert ema.num_parameters == 20  # 4x4 weight + 4 bias
    assert ema.num_bytes == 80  # float32

    frozen_before = model.frozen.weight.detach().clone()
    _step(model, 1.0)
    ema.update(model)
    with ema.swapped_into(model):
        # The frozen parameter is left alone rather than restored from a shadow copy.
        torch.testing.assert_close(model.frozen.weight, frozen_before)


def test_bfloat16_accumulator_would_stall():
    """The reason the accumulator is float32 while the model trains in bfloat16.

    With decay=0.999 each update moves the accumulator by 0.1% of the gap, which is below
    half a ULP in bfloat16: the increment rounds away and the average never leaves its
    starting point. A checkpoint written from such an accumulator would be the pretrained
    weights, not the trained ones.
    """
    model = _TinyModel(dtype=torch.bfloat16)
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(1.0)
    start = model.trainable.weight.detach().float().clone()

    in_bfloat16 = ema_pytorch.ParameterEma(model, 0.999, dtype=torch.bfloat16)
    in_float32 = ema_pytorch.ParameterEma(model, 0.999, dtype=torch.float32)
    for _ in range(50):
        _step(model, 0.01)
        in_bfloat16.update(model)
        in_float32.update(model)

    stalled = in_bfloat16._shadow["trainable.weight"].float()  # noqa: SLF001
    moved = in_float32._shadow["trainable.weight"].float()  # noqa: SLF001
    torch.testing.assert_close(stalled, start, rtol=0, atol=0)
    assert (moved - start).abs().max() > 0


def test_swap_restores_parameters_even_on_error():
    torch.manual_seed(0)
    model = _TinyModel()
    ema = ema_pytorch.ParameterEma(model, 0.5)
    _step(model, 1.0)
    ema.update(model)
    before = copy.deepcopy(model.state_dict())

    def fail_while_swapped():
        with ema.swapped_into(model):
            assert not torch.equal(model.trainable.weight, before["trainable.weight"])
            raise RuntimeError("checkpoint write failed")

    with pytest.raises(RuntimeError):
        fail_while_swapped()

    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)


def test_copy_from_reseeds_the_accumulator():
    model = _TinyModel()
    ema = ema_pytorch.ParameterEma(model, 0.9)
    _step(model, 1.0)
    ema.update(model)
    assert ema.relative_distance(model) > 0

    # What resuming does: the accumulator is rebuilt from the averaged weights on disk.
    ema.copy_from(model)
    assert ema.relative_distance(model) == pytest.approx(0.0, abs=1e-7)


def test_relative_distance_is_scale_free():
    model = _TinyModel()
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(2.0)
    ema = ema_pytorch.ParameterEma(model, 0.5)
    _step(model, 2.0)  # parameters 2 -> 4, accumulator 2 -> 3 at decay 0.5
    ema.update(model)
    assert ema.relative_distance(model) == pytest.approx(0.25, rel=1e-5)


def test_rejects_invalid_decay():
    model = _TinyModel()
    for decay in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="decay"):
            ema_pytorch.ParameterEma(model, decay)


def test_unwrap_model_passes_through_plain_modules():
    model = _TinyModel()
    assert ema_pytorch.unwrap_model(model) is model
