"""glow_progress_monitor: labels, the split, permuted plans, the metrics, the partial weight load and
main() end to end.

The end-to-end test needs the LIBERO-10 + Goal plan pack and the local LeRobot copy (HF_LEROBOT_HOME
with repo `libero_staging`). Models are dummy-size with random weights, so it checks the wiring,
not the numbers.
"""

import dataclasses
import itertools
import json
import time
import types

import numpy as np
import pytest

from . import glow_module_diag_test as _diag_test
from . import glow_progress_monitor as gpm

GOAL_PACK, DATASET = _diag_test.GOAL_PACK, _diag_test.DATASET
needs_pack_and_data = pytest.mark.skipif(
    not (GOAL_PACK.exists() and (DATASET / "meta/info.json").exists()),
    reason="plan pack or local LeRobot copy not present",
)


def _episode(task_index, n_frames, starts):
    return types.SimpleNamespace(task_index=task_index, n_frames=n_frames, stage_starts=tuple(starts))


def _pack(episodes, chains=()):
    # build_chain_dataset.py's provenance layout.
    chains = {"first_chain_episode": 807, "episodes": list(chains)}
    return types.SimpleNamespace(episodes=episodes, provenance={"chains": chains}, tasks={})


def test_labels_follow_the_next_stage_start_and_leave_the_tail_unlabeled():
    ep = _episode(0, 100, [0, 50])
    assert [gpm.labels(ep, f, 15)[0].tolist() for f in (0, 49, 50, 99)] == [[0, 0], [0, 0], [1, 0], [1, 0]]
    assert gpm.labels(ep, 84, 15)[1].tolist() == [True, True]
    assert gpm.labels(ep, 85, 15)[1].tolist() == [True, False]
    one = _episode(1, 40, [0])
    done, labeled = gpm.labels(one, 10, 15)
    assert done.tolist() == [0] and labeled.tolist() == [True]


def test_split_holds_out_chain_inits_and_every_nth_episode_of_a_task():
    episodes = {e: _episode(e // 10, 100, [0, 50]) for e in range(20)}
    episodes.update({30: _episode(5, 100, [0, 40]), 31: _episode(5, 100, [0, 40]), 32: _episode(6, 100, [0, 40])})
    chains = gpm.chain_inits(_pack(episodes, [{"episode_index": 30, "init": 10}, {"episode_index": 31, "init": 11},
                                               {"episode_index": 32, "init": 11}]))
    pack = _pack(episodes)
    held = gpm.split(pack, sorted(episodes), chains, 5, ())
    assert [e for e in range(20) if held[e]] == [0, 5, 10, 15]
    assert (held[30], held[31], held[32]) == (False, True, True)
    assert gpm.split(pack, sorted(episodes), chains, 5, (10,))[30] is True
    assert gpm.family(30, chains) == "chain" and gpm.family(5, chains) == "libero10"
    assert gpm.family(400, chains) == "goal"
    with pytest.raises(ValueError, match="provenance"):
        gpm.family(900, chains)


def test_select_episodes_keeps_the_first_per_task_within_the_dataset():
    episodes = {e: _episode(e % 2, 10, [0]) for e in range(10)}
    pack = _pack(episodes)
    assert gpm.select_episodes(pack, 8, None, 2) == [0, 1, 2, 3]
    assert gpm.select_episodes(pack, 10, [1, 3, 5, 9], 0) == [1, 3, 5, 9]


def test_permutations_move_only_plan_stages_and_never_keep_the_order():
    import torch

    valid = torch.tensor([[True, True, False, False], [True, True, True, False], [True, False, False, False]])
    assert gpm.cyclic_perm(valid).tolist() == [[1, 0, 2, 3], [1, 2, 0, 3], [0, 1, 2, 3]]
    gen = torch.Generator().manual_seed(0)
    j = torch.arange(4).expand(3, 4)
    for _ in range(50):
        perm = gpm.random_perm(valid, gen)
        assert sorted(perm[1, :3].tolist()) == [0, 1, 2] and perm[1, 3] == 3
        assert not torch.equal(perm[:2], j[:2])
        assert torch.equal(perm[2], j[2])
    assert torch.equal(gpm.swap_perm(valid, 0.0, gen), j)
    x = torch.arange(12.0).reshape(3, 4)
    (moved,) = gpm.permute(gpm.cyclic_perm(valid), x)
    assert moved[0].tolist() == [1.0, 0.0, 2.0, 3.0]


def test_auroc_matches_the_pairwise_count():
    rng = np.random.default_rng(0)
    scores = np.round(rng.random(60), 1)  # ties on purpose
    targets = rng.random(60) < 0.4
    pos, neg = scores[targets], scores[~targets]
    pairwise = np.mean([1.0 if p > n else 0.5 if p == n else 0.0 for p, n in itertools.product(pos, neg)])
    assert gpm.auroc(scores, targets) == pytest.approx(pairwise)
    assert gpm.auroc(np.array([0.1, 0.9]), np.array([False, True])) == 1.0
    assert gpm.auroc(np.array([0.9, 0.1]), np.array([False, True])) == 0.0
    assert gpm.auroc(np.array([0.3, 0.4]), np.array([True, True])) is None


def test_latched_switch_needs_two_consecutive_samples():
    frames = [0, 5, 10, 15, 20]
    assert gpm.latched_switch(frames, [0, 0.9, 0.1, 0.8, 0.9], 0.7) == 20
    assert gpm.latched_switch(frames, [0, 0, 0.9, 0.9, 0.9], 0.7) == 15
    assert gpm.latched_switch(frames, [0.9, 0, 0, 0, 0], 0.7) is None


def _samples(pack, episodes, heldout, stride=5, tail=15):
    return gpm.frame_plan(pack, episodes, {}, heldout, stride, 10, tail)


def test_evaluate_a_perfect_predictor():
    episodes = {0: _episode(0, 100, [0, 50]), 1: _episode(0, 100, [0, 60]), 400: _episode(3, 60, [0])}
    pack = _pack(episodes)
    held = {0: True, 1: True, 400: True}
    samples = _samples(pack, sorted(episodes), held)
    idx = np.arange(len(samples))
    done = np.stack([gpm.padded(s["done"], 4, 0.0) for s in samples])
    labeled = np.stack([gpm.padded(s["labeled"], 4, False) for s in samples])
    valid = np.stack([gpm.padded(np.ones(len(s["done"]), dtype=bool), 4, False) for s in samples])
    d_shift, lab_shift = gpm.shifted_labels(done, labeled, valid)
    out = gpm.evaluate(samples, idx, done, pack, 0.7, 40, swapped=(d_shift, d_shift, lab_shift))
    l10 = out["libero10"]
    assert l10["auroc"] == 1.0 and l10["start_false_done"] == 0.0
    # Latched on the second sample past the boundary: one stride late.
    assert l10["switch"] == {"boundaries": 2, "missed": 0, "median_abs": 5.0, "mean": 5.0, "within": 1.0,
                             "early": 0, "late": 0}
    assert l10["swapped"]["auroc"] == 1.0 and l10["swapped"]["false_done_after_boundary"] == 0.0
    assert out["goal"]["auroc"] is None and out["goal"]["start_false_done"] == 0.0
    # The original-order probabilities read against the shifted labels: done in the wrong slot.
    wrong = gpm.evaluate(samples, idx, done, pack, 0.7, 40, swapped=(done, d_shift, lab_shift))
    assert wrong["libero10"]["swapped"]["false_done_after_boundary"] == 1.0


def test_clock_baseline_uses_training_medians():
    episodes = {0: _episode(0, 100, [0, 40]), 1: _episode(0, 100, [0, 60]), 2: _episode(0, 100, [0, 50])}
    pack = _pack(episodes)
    samples = _samples(pack, [0, 1, 2], {0: False, 1: False, 2: True})
    scores = gpm.clock_scores(samples, [i for i, s in enumerate(samples) if not s["heldout"]], pack, 4)
    first = next(i for i, s in enumerate(samples) if s["episode"] == 2 and s["frame"] == 50)
    assert scores[first, 0] == 0.0 and np.isneginf(scores[first, 1])
    assert gpm.clock_probs(scores[first : first + 1])[0, 0] == pytest.approx(0.5)
    assert gpm.clock_probs(np.array([[-np.inf]]))[0, 0] == 0.0


def test_passes_reads_the_monitor_against_both_baselines():
    fam = {"positives": 3, "auroc": 0.99, "start_false_done": 0.0, "switch": {"median_abs": 5.0},
           "swapped": {"auroc": 0.9}}
    result = {
        "variants": {"monitor": {"chain": fam}, "nokey": {"chain": {"swapped": {"auroc": 0.5}}}},
        "baselines": {"clock": {"chain": {"swapped": {"auroc": 0.95}}}},
    }
    assert gpm.passes(result) == {"chain": {"auroc": True, "switch_median": True, "start_false_done": True,
                                            "swapped_beats_nokey": True, "swapped_beats_clock": False}}


def test_prefetch_keeps_order_stays_bounded_and_raises_the_builders_error():
    import threading

    built, lock = [], threading.Lock()

    def build(chunk):
        with lock:
            built.append(chunk)
        return chunk

    batches = gpm.Prefetch(range(20), build, depth=2)
    first = next(iter(batches))
    deadline = time.monotonic() + 2.0
    while len(built) < 4 and time.monotonic() < deadline:
        time.sleep(0.01)
    # One consumed, two queued, one built and waiting for room: never the whole run ahead.
    assert first == 0 and len(built) <= 4
    assert [first, *batches] == list(range(20))

    def failing(chunk):
        if chunk == 3:
            raise RuntimeError("bad episode")
        return chunk

    with pytest.raises(RuntimeError, match="bad episode"):
        list(gpm.Prefetch(range(10), failing))


def test_pooled_tokens_average_square_cells():
    import torch

    tokens = torch.arange(16.0).reshape(1, 16, 1)
    out = gpm.pooled_tokens(tokens, 2)
    assert out[..., 0].tolist() == [[2.5, 4.5, 10.5, 12.5]]
    with pytest.raises(ValueError, match="square grid"):
        gpm.pooled_tokens(torch.zeros(1, 15, 1), 2)


def test_load_vision_reads_only_the_vision_tower_projector_and_embedding(monkeypatch, tmp_path):
    import safetensors.torch
    import torch

    from openpi.training import config as _config

    _diag_test._dummy_configs(monkeypatch)  # noqa: SLF001
    cfg = _config.get_config("pi05_libero_glow_v2t_fixed")
    torch.manual_seed(0)
    full = dataclasses.replace(cfg.model, dtype="float32").create_pytorch()
    safetensors.torch.save_model(full, str(tmp_path / "model.safetensors"))
    full.eval()
    part = gpm.load_vision(cfg.model, str(tmp_path), "float32", torch.device("cpu"))
    image = torch.rand(2, 3, 224, 224) * 2 - 1
    with torch.no_grad():
        torch.testing.assert_close(
            part.paligemma_with_expert.embed_image(image), full.paligemma_with_expert.embed_image(image)
        )
        ids = torch.tensor([[2, 7, 11]])
        torch.testing.assert_close(
            part.paligemma_with_expert.embed_language_tokens(ids), full.paligemma_with_expert.embed_language_tokens(ids)
        )
    assert part.paligemma_with_expert.gemma_expert.lm_head.weight.is_meta


@needs_pack_and_data
def test_main_end_to_end_on_dummy_weights(monkeypatch, tmp_path):
    import safetensors.torch
    import torch

    from openpi.training import config as _config

    _diag_test._dummy_configs(monkeypatch)  # noqa: SLF001
    cfg = _config.get_config("pi05_libero_glow_v2t_fixed")
    torch.manual_seed(0)
    safetensors.torch.save_model(cfg.model.create_pytorch(), str(tmp_path / "model.safetensors"))
    args = gpm.Args(
        weights=str(tmp_path),
        dataset_root=str(DATASET),
        plan_pack=str(GOAL_PACK),
        episodes="0:3,50:53,379:382",
        holdout_every=3,
        steps=3,
        batch_size=8,
        width=16,
        heads=2,
        depth=1,
        stride=20,
        single_stride=40,
        device="cuda" if torch.cuda.is_available() else "cpu",
        out=str(tmp_path / "out.json"),
        save_dir=str(tmp_path / "monitors"),
    )
    result = gpm.main(args)
    # The saved monitor is what a monitor row loads (load_progress_monitor), shapes included.
    monitor_cfg = _config.get_config("pi05_libero_glow_monitor")
    row = dataclasses.replace(
        monitor_cfg.model, module_monitor_width=16, module_monitor_depth=1, module_monitor_heads=2
    ).create_pytorch()
    saved = tmp_path / "monitors" / "monitor"
    assert row.load_progress_monitor(str(saved)) > 0
    assert json.loads((saved / "monitor.json").read_text())["pool"] == args.pool
    assert set(result["variants"]) == set(gpm.VARIANTS)
    assert result["counts"]["libero10"]["episodes"] == 6 and result["counts"]["goal"]["episodes"] == 3
    assert "swapped" in result["variants"]["monitor"]["libero10"]
    assert "clock" in result["baselines"] and "libero10" in result["pass"]
    assert (tmp_path / "out.json").exists()
