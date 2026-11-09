import math

import pytest
import torch

from openpi.glow import module_encoders as _enc
from openpi.glow import module_inputs as _mi


def _ints(num_stages, stage=0):
    ints = torch.zeros(len(num_stages), _mi.INT_HEAD.stop, dtype=torch.long)
    ints[:, _mi.INT_NUM_STAGES] = torch.tensor(num_stages)
    ints[:, _mi.INT_STAGE] = stage
    return ints


def test_soft_active_is_the_first_stage_not_done_at_saturation():
    valid = _enc.stage_valid(_ints([2, 2, 2, 3]))
    done = torch.tensor([[1.0, 0.0, 0, 0], [0.0, 0.0, 0, 0], [1.0, 1.0, 0, 0], [1.0, 0.0, 1.0, 0]])
    expected = torch.tensor([[0.0, 1, 0, 0], [1.0, 0, 0, 0], [0.0, 1, 0, 0], [0.0, 1, 0, 0]])
    torch.testing.assert_close(_enc.soft_active(done, valid), expected)


def test_soft_active_is_a_distribution_over_the_plan():
    torch.manual_seed(0)
    ints = _ints([1, 2, 3, 4])
    valid = _enc.stage_valid(ints)
    # Past the plan the probabilities are garbage on purpose: they must not count.
    active = _enc.soft_active(torch.rand(4, _mi.K_MAX), valid)
    torch.testing.assert_close(active.sum(-1), torch.ones(4))
    assert bool((active[~valid] == 0).all())
    assert bool((active >= 0).all())


def test_soft_stage_gates_at_a_one_hot_are_the_fixed_gates():
    for stage in (0, 1):
        ints = _ints([2, 3], stage=stage)
        valid = _enc.stage_valid(ints)
        one_hot = torch.nn.functional.one_hot(torch.tensor([stage, stage]), _mi.K_MAX).float()
        torch.testing.assert_close(
            _enc.soft_stage_gates(one_hot, valid, 0.06), _enc.fixed_stage_gates(ints, 0.06).float()
        )


def test_monitor_key_inputs_pool_each_stage_key_in_execution_order():
    embed = torch.nn.Embedding(50, 6)
    text = torch.zeros(1, _mi.TEXT_LEN, dtype=torch.long)
    mask = torch.zeros(1, _mi.TEXT_LEN, dtype=torch.bool)
    for j, ids in enumerate(([3, 4], [7])):
        start, _ = _mi.TEXT_SEGMENTS[f"rk{j}"]
        text[0, start : start + len(ids)] = torch.tensor(ids)
        mask[0, start : start + len(ids)] = True
    ints = _ints([2])
    heads = ints[:, _mi.INT_HEAD].reshape(1, _mi.K_MAX, len(_mi.HEAD_FIELDS)).clone()
    heads[0, 1, _mi.HEAD_FIELDS.index("key_object")] = 9
    ints[:, _mi.INT_HEAD] = heads.reshape(1, -1)
    keys, key_heads, valid = _enc.monitor_key_inputs(embed, text, mask, ints)
    with torch.no_grad():
        torch.testing.assert_close(keys[0, 0], embed.weight[[3, 4]].mean(0))
        torch.testing.assert_close(keys[0, 1], embed.weight[7])
        torch.testing.assert_close(key_heads[0, 1, 1], embed.weight[9])
    assert bool((key_heads[0, 0] == 0).all())
    assert valid.tolist() == [[True, True, False, False]]


def _monitor_inputs(b=3, c=3, t=5, d_text=8, d_image=12):
    torch.manual_seed(1)
    valid = _enc.stage_valid(_ints([2, 1, 4][:b]))
    return (
        torch.randn(b, _mi.K_MAX, d_text),
        torch.randn(b, _mi.K_MAX, 2, d_text),
        valid,
        torch.randn(b, c, t, d_image),
        torch.tensor([[True, True, False]] * b),
    )


@pytest.mark.parametrize("keys", [True, False])
def test_progress_monitor_masks_padding_stages_and_missing_cameras(keys):
    monitor = _enc.ProgressMonitor(8, 12, width=16, depth=1, heads=2, keys=keys).eval()
    keys_in, heads_in, valid, images, image_mask = _monitor_inputs()
    logits = monitor(keys_in, heads_in, valid, images, image_mask)
    assert logits.shape == (3, _mi.K_MAX)
    assert bool(torch.isinf(logits[~valid]).all()) and bool(torch.isfinite(logits[valid]).all())
    other = images.clone()
    other[:, 2] = torch.randn_like(other[:, 2]) * 100
    torch.testing.assert_close(monitor(keys_in, heads_in, valid, other, image_mask), logits)


def test_progress_monitor_without_keys_ignores_the_key_text():
    monitor = _enc.ProgressMonitor(8, 12, width=16, depth=1, heads=2, keys=False).eval()
    keys_in, heads_in, valid, images, image_mask = _monitor_inputs()
    logits = monitor(keys_in, heads_in, valid, images, image_mask)
    torch.testing.assert_close(monitor(-keys_in, heads_in * 3, valid, images, image_mask), logits)
    with_keys = _enc.ProgressMonitor(8, 12, width=16, depth=1, heads=2, keys=True).eval()
    a = with_keys(keys_in, heads_in, valid, images, image_mask)
    assert not torch.allclose(with_keys(-keys_in, heads_in, valid, images, image_mask)[valid], a[valid])


def test_progress_monitor_reads_the_cameras():
    monitor = _enc.ProgressMonitor(8, 12, width=16, depth=1, heads=2).eval()
    keys_in, heads_in, valid, images, image_mask = _monitor_inputs()
    a = monitor(keys_in, heads_in, valid, images, image_mask)
    b = monitor(keys_in, heads_in, valid, images.flip(0), image_mask)
    assert not torch.allclose(a[0, :2], b[0, :2])
    assert math.isinf(float(a[1, 1]))
