"""glow_temporal_probe: the margin, the sample plan, the parquet reader and main() end to end.

The tests that read data need the LIBERO-10 + Goal plan pack, the local LeRobot copy
(HF_LEROBOT_HOME with repo `libero_staging`) and pi05_libero's norm_stats. Models are dummy-size with
random weights, so they check the wiring, not the numbers.
"""

import json
import pathlib

import numpy as np
import pytest

from . import glow_module_diag_test as _diag_test
from . import glow_temporal_probe as gtp
from openpi.glow import module_inputs as _mi  # noqa: E402
from openpi.glow import plan_pack as _plan_pack  # noqa: E402

DATASET, ASSETS, GOAL_PACK = _diag_test.DATASET, _diag_test.ASSETS, _diag_test.GOAL_PACK
needs_data = _diag_test.needs_data


def test_margin_stats():
    dab = np.array([4.0, 16.0])
    follows_active = gtp.margin_stats(np.zeros(2), dab, dab, np.random.default_rng(0))
    assert follows_active["margin"] == pytest.approx(1.0) and follows_active["rel_err"] == pytest.approx(0.0)
    assert follows_active["margin_se"] == pytest.approx(0.0)
    follows_other = gtp.margin_stats(dab, np.zeros(2), dab)
    assert follows_other["margin"] == pytest.approx(-1.0) and follows_other["rel_err"] == pytest.approx(1.0)
    halfway = gtp.margin_stats(dab / 4, dab / 4, dab)
    assert halfway["margin"] == pytest.approx(0.0) and halfway["rel_err"] == pytest.approx(0.25)


def test_parse_row():
    assert gtp.parse_row("v2t=pi05_libero_glow_v2t:/data/x:y/999") == ("v2t", "pi05_libero_glow_v2t", "/data/x:y/999")
    with pytest.raises(ValueError, match="name=config"):
        gtp.parse_row("v2t=pi05_libero_glow_v2t")


@pytest.mark.skipif(not GOAL_PACK.exists(), reason="plan pack not present")
def test_sample_plan():
    pack = _plan_pack.PlanPack.load(str(GOAL_PACK))
    goal = {t.task for t in pack.tasks.values() if t.task_index // 10 == 1}
    assert {t for pair in gtp.EVAL_PAIRS for t in pair} == goal
    assert all(pack.task(t).num_plans == 1 for t in goal)

    rng = np.random.default_rng(0)
    exclude = list(range(379, 427))
    heldout = gtp.heldout_episodes(pack, exclude, 4, rng)
    assert sorted(heldout) == list(range(10, 20)) and all(len(v) == 4 for v in heldout.values())
    chosen = {e for v in heldout.values() for e in v}
    assert not chosen & set(exclude) and all(379 <= e < 807 and pack.episodes[e].task_index // 10 == 1 for e in chosen)

    plan = gtp.plan_frames(
        pack, heldout, start_samples=21, mid_samples=19, start_window=40, mid_window=120, batch_size=8, rng=rng
    )
    assert [s["bucket"] for s in plan] == ["start"] * 16 + ["mid"] * 16
    tasks = {t.task_index: t.task for t in pack.tasks.values()}
    for s in plan:
        assert s["episode"] in chosen and s["frame"] < pack.episodes[s["episode"]].n_frames
        assert s["pair"] in gtp.EVAL_PAIRS or s["pair"][::-1] in gtp.EVAL_PAIRS
        if s["bucket"] == "start":
            assert 0 <= s["frame"] < 40
        else:
            assert 40 <= s["frame"] < 120 and s["pair"][0] == tasks[pack.episodes[s["episode"]].task_index]


def test_release_moves_every_tensor_off_the_device(monkeypatch):
    import torch

    get_config = _diag_test._dummy_configs(monkeypatch)  # noqa: SLF001
    model = get_config("pi05_libero_glow_v2t").model.create_pytorch()
    gtp.release(model, torch.device("cpu"))
    assert {p.device.type for p in model.parameters()} == {"meta"}
    assert {b.device.type for b in model.buffers()} <= {"meta"}


def test_batch_store_order_and_error():
    store = gtp.BatchStore(list(range(8)), lambda part: sum(part), 4)
    assert [store.get(i) for i in range(store.num_batches)] == [6, 22]

    def fails(_):
        raise RuntimeError("bad frame")

    with pytest.raises(RuntimeError, match="bad frame"):
        gtp.BatchStore(list(range(4)), fails, 4).get(0)


@needs_data
def test_read_frames_matches_the_training_loader():
    from lerobot.common.datasets import lerobot_dataset

    episode, horizon = 2, 10
    meta = lerobot_dataset.LeRobotDatasetMetadata("physical-intelligence/libero", root=str(DATASET))
    dataset = lerobot_dataset.LeRobotDataset(
        "physical-intelligence/libero",
        root=str(DATASET),
        episodes=list(range(episode + 1)),
        delta_timestamps={"actions": [t / meta.fps for t in range(horizon)]},
    )
    start = int(dataset.episode_data_index["from"][episode])
    last = int(dataset.episode_data_index["to"][episode]) - start - 1
    frames = [0, last - 3]  # the second chunk runs past the episode's end
    tasks = gtp.read_task_strings(DATASET)
    got = gtp.read_frames(DATASET, episode, frames, horizon, tasks)
    for frame in frames:
        ref, mine = dataset[start + frame], got[frame]
        assert mine["episode_index"] == int(ref["episode_index"]) and mine["frame_index"] == int(ref["frame_index"])
        assert mine["prompt"] == meta.tasks[int(ref["task_index"])]
        np.testing.assert_allclose(mine["state"], ref["state"].numpy())
        np.testing.assert_allclose(mine["actions"], ref["actions"].numpy())
        for key in ("image", "wrist_image"):
            as_float = np.moveaxis(mine[key], -1, 0) / 255.0
            np.testing.assert_allclose(as_float, ref[key].numpy(), atol=1e-6)


def _layout_inputs(model, batch=2):
    import torch

    images = [torch.rand(batch, 3, 224, 224) * 2 - 1 for _ in range(3)]
    masks = [torch.ones(batch, dtype=torch.bool) for _ in range(3)]
    tokens = torch.zeros(batch, 8, dtype=torch.long)
    tokens[:, :2] = torch.tensor([1, 108])
    lang_masks = tokens != 0
    return images, masks, tokens, lang_masks


@pytest.mark.skipif(not GOAL_PACK.exists(), reason="plan pack not present")
@pytest.mark.parametrize(
    ("layout", "config", "order"),
    [
        ("set_active", "pi05_libero_glow_v2t", "sorted"),
        ("set_gate25", "pi05_libero_glow_v2t_gate", "sorted"),
        ("v2s", "pi05_libero_glow_v2s", "remaining"),
    ],
)
def test_layout_is_what_a_fresh_model_emits(monkeypatch, layout, config, order):
    import torch

    from openpi.models import tokenizer as _tokenizer

    get_config = _diag_test._dummy_configs(monkeypatch)  # noqa: SLF001
    torch.manual_seed(0)
    cfg = get_config(config)
    object.__setattr__(cfg.model, "dtype", "float32")  # the precision the probe runs at
    model = cfg.model.create_pytorch().eval()
    pack = _plan_pack.PlanPack.load(str(GOAL_PACK))
    tokenizer = _tokenizer.PaligemmaTokenizer(8)
    own = _mi.GlowModuleInputs(pack, tokenizer, stage_order=order, stage_clock=True)
    sorted_ = _mi.GlowModuleInputs(pack, tokenizer, stage_order="sorted", stage_clock=True)
    samples = [{"pair": gtp.EVAL_PAIRS[0], "frame": 3}, {"pair": gtp.EVAL_PAIRS[6], "frame": 30}]
    inputs = _layout_inputs(model)
    with torch.no_grad():
        for plan in ("AB", "BA"):
            embs, pad, att = model.embed_observation_prefix(gtp.module_arrays(own, samples, plan, 0, "cpu"), *inputs)
            tokens = gtp.layout_tokens(layout, model, gtp.module_arrays(sorted_, samples, plan, 0, "cpu"))
            built, built_pad, built_att = gtp.layout_prefix(model, tokens, *inputs)
            # The model's padded Spatial groups take no attention and no position: compare without them.
            for i in range(len(samples)):
                np.testing.assert_allclose(embs[i][pad[i]].numpy(), built[i][built_pad[i]].numpy(), atol=1e-5)
                assert att[i][pad[i]].equal(built_att[i][built_pad[i]])


@pytest.mark.skipif(not GOAL_PACK.exists(), reason="plan pack not present")
def test_layout_sizes_and_gate(monkeypatch):
    import torch

    from openpi.models import tokenizer as _tokenizer

    get_config = _diag_test._dummy_configs(monkeypatch)  # noqa: SLF001
    cfg = get_config("pi05_libero_glow_v2t")
    object.__setattr__(cfg.model, "dtype", "float32")  # the precision the probe runs at
    model = cfg.model.create_pytorch().eval()
    pack = _plan_pack.PlanPack.load(str(GOAL_PACK))
    module = _mi.GlowModuleInputs(pack, _tokenizer.PaligemmaTokenizer(8), stage_order="sorted", stage_clock=True)
    samples = [{"pair": gtp.EVAL_PAIRS[0], "frame": 3}]
    arrays = gtp.module_arrays(module, samples, "AB", 0, "cpu")
    with torch.no_grad():
        sizes = {name: gtp.layout_tokens(name, model, arrays) for name in gtp.LAYOUTS}
    assert sizes["set"].shape[1] == 10 and sizes["set_active"].shape[1] == 11 and sizes["set_list"].shape[1] == 12
    assert sizes["set_active_group"].shape[1] == 14 and sizes["active_only"].shape[1] == 7
    # The gate shrinks the other stage's group to a quarter and leaves the active stage's.
    gated = sizes["set_gate25"].pow(2).mean(-1).sqrt()[0, :8]
    plain = sizes["set_active"].pow(2).mean(-1).sqrt()[0, :8]
    ratios = sorted({round(float(g / p), 3) for g, p in zip(gated, plain, strict=True) if p > 0})
    assert ratios == [gtp.GATE, 1.0]
    with pytest.raises(ValueError, match="unknown layout"):
        gtp.layout_tokens("nope", model, arrays)
    one_stage = gtp.module_arrays(module, samples, "A", 0, "cpu")
    with pytest.raises(ValueError, match="two-stage"):
        gtp.layout_tokens("set", model, one_stage)


@needs_data
def test_main_rows_and_grid(monkeypatch, tmp_path):
    import safetensors.torch
    import torch

    get_config = _diag_test._dummy_configs(monkeypatch)  # noqa: SLF001
    torch.manual_seed(0)
    v2t = tmp_path / "v2t"
    v2s = tmp_path / "v2s"
    init = tmp_path / "init"
    for d in (v2t, v2s, init):
        d.mkdir()
    # Trained-like weights: the encoders' residuals are not zero, so the Temporal tokens do something.
    model = get_config("pi05_libero_glow_v2t").model.create_pytorch()
    for name, p in model.named_parameters():
        if name.endswith("out.1.weight"):
            torch.nn.init.normal_(p, std=0.02)
    safetensors.torch.save_model(model, str(v2t / "model.safetensors"))
    safetensors.torch.save_model(model, str(v2s / "model.safetensors"))
    model.glow_spatial_encoder = None
    model.glow_temporal_encoder = None
    safetensors.torch.save_model(model, str(init / "model.safetensors"))

    out = tmp_path / "probe.json"
    s = gtp.main(
        gtp.Args(
            norm_stats_dir=str(ASSETS),
            dataset_root=str(DATASET),
            rows=(f"v2t=pi05_libero_glow_v2t:{v2t}", f"v2s=pi05_libero_glow_v2s:{v2s}"),
            grid_weights=str(init),
            exclude_episodes="379:427",
            episodes_per_task=1,
            start_samples=8,
            mid_samples=4,
            batch_size=4,
            device="cpu",
            out=str(out),
        )
    )
    written = json.loads(pathlib.Path(out).read_text(encoding="utf-8"))
    assert set(written["rows"]) == {"v2t", "v2s"} and set(written["grid"]["layouts"]) == set(gtp.LAYOUTS)

    t, v = s["rows"]["v2t"], s["rows"]["v2s"]
    assert t["spatial_tokens_moved_between_orders_max"] == 0.0
    assert v["spatial_tokens_moved_between_orders_max"] > 1e-3
    assert t["plan_first"]["samples"] == 16 and set(t["variants"]) == {"start", "mid"}
    for variant, (_, _, off) in gtp.ROW_VARIANTS.items():
        assert t["variants"]["start"][variant].get("samples", 0) == (0 if off == "gate" else 8)
        assert t["variants"]["mid"][variant].get("samples", 0) == (0 if off == "gate" else 4)
    # With Temporal masked a sorted row's two orders are one prefix, so the margins are opposite.
    off = t["variants"]["start"]
    assert off["temporal_off_ab"]["margin"] == pytest.approx(-off["temporal_off_ba"]["margin"], abs=1e-5)
    assert t["temporal_off"]["margin"] == pytest.approx(0.0, abs=1e-5)
    assert all(np.isfinite(g["margin"]) for g in s["grid"]["layouts"].values())
    assert s["grid"]["layouts"]["set"]["samples"] == 16
    # Nothing in the "set" layout says which stage is active.
    assert s["grid"]["layouts"]["set"]["margin"] == pytest.approx(0.0, abs=1e-5)


@needs_data
def test_main_gated_row(monkeypatch, tmp_path):
    import safetensors.torch
    import torch

    get_config = _diag_test._dummy_configs(monkeypatch)  # noqa: SLF001
    torch.manual_seed(0)
    model = get_config("pi05_libero_glow_v2t_gate").model.create_pytorch()
    for name, p in model.named_parameters():
        if name.endswith("out.1.weight"):
            torch.nn.init.normal_(p, std=0.02)
    safetensors.torch.save_model(model, str(tmp_path / "model.safetensors"))
    s = gtp.main(
        gtp.Args(
            norm_stats_dir=str(ASSETS),
            dataset_root=str(DATASET),
            rows=(f"gate=pi05_libero_glow_v2t_gate:{tmp_path}",),
            exclude_episodes="379:427",
            episodes_per_task=1,
            start_samples=8,
            mid_samples=4,
            batch_size=4,
            device="cpu",
        )
    )["rows"]["gate"]
    assert s["spatial_tokens_moved_between_orders_max"] == 0.0
    # The gate head is still zero: exactly the starting gate.
    assert s["smaller_group_gate_mean"] == pytest.approx(0.25)
    assert s["variants"]["start"]["gate_off_ab"]["samples"] == 8
    # With Temporal off altogether the two orders are one prefix again.
    assert s["temporal_off"]["margin"] == pytest.approx(0.0, abs=1e-5)
    assert s["gate_off"]["samples"] == 16
    # The gate does something on its own.
    assert s["plan_first"]["margin"] != pytest.approx(s["gate_off"]["margin"], abs=1e-4)
