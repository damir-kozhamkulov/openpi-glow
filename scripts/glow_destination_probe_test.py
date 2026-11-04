"""glow_destination_probe: the gripper buckets, the sample plan, the crossed prefix and main() end to end.

The tests that read data need the LIBERO-10 + Goal plan pack, the local LeRobot copy
(HF_LEROBOT_HOME with repo `libero_staging`) and pi05_libero's norm_stats. Models are dummy-size with
random weights, so they check the wiring, not the numbers.
"""

import numpy as np
import pytest

from . import glow_destination_probe as gdp
from . import glow_module_diag_test as _diag_test
from . import glow_temporal_probe as gtp
from openpi.glow import plan_pack as _plan_pack  # noqa: E402

DATASET, ASSETS, GOAL_PACK = _diag_test.DATASET, _diag_test.ASSETS, _diag_test.GOAL_PACK
needs_data = _diag_test.needs_data


def test_closed_run_takes_the_longest():
    gripper = np.array([-1, 1, -1, -1, 1, 1, 1, -1, 1, 1], dtype=np.float32)
    assert gdp.closed_run(gripper) == (4, 7)
    assert gdp.closed_run(np.array([-1, 1, 1], dtype=np.float32)) == (1, 3)
    with pytest.raises(ValueError, match="never closes"):
        gdp.closed_run(-np.ones(4))


def test_bucket_frames():
    buckets = gdp.bucket_frames(40, 90, 10)
    assert buckets["reach"] == range(40)
    assert buckets["early"] == range(50, 70) and buckets["late"] == range(70, 90)


def test_partners():
    assert gdp.partners("put the bowl on the plate") == [
        "put the bowl on the stove",
        "put the bowl on top of the cabinet",
    ]
    assert gdp.partners("put the wine bottle on the rack") == ["put the wine bottle on top of the cabinet"]


def test_plan_samples():
    tasks = {10: "put the bowl on the plate", 11: "put the wine bottle on the rack"}
    heldout = {10: [1, 2], 11: [3, 4]}
    runs = {1: (30, 90), 2: (30, 90), 3: (40, 100), 4: (40, 100)}
    per_bucket = {"reach": 4, "early": 8, "late": 2}
    plan = gdp.plan_samples(tasks, heldout, runs, per_bucket=per_bucket, settle=10, rng=np.random.default_rng(0))
    assert len(plan) == 28 and len({(s["episode"], s["frame"], s["bucket"]) for s in plan}) == 28
    for s in plan:
        task_index = 10 if s["episode"] < 3 else 11
        assert s["pair"][0] == tasks[task_index] and s["pair"][1] in gdp.partners(tasks[task_index])
        assert s["frame"] in gdp.bucket_frames(*runs[s["episode"]], 10)[s["bucket"]]
    for task_index in tasks:
        own = [s for s in plan if s["pair"][0] == tasks[task_index]]
        assert sorted(s["bucket"] for s in own) == sorted(b for b, n in per_bucket.items() for _ in range(n))
    # Shuffled, so a dropped remainder and any one batch favour no task or bucket.
    assert [s["bucket"] for s in plan] != sorted(s["bucket"] for s in plan)
    with pytest.raises(ValueError, match="early frames"):
        gdp.plan_samples(
            tasks, heldout, runs, per_bucket={**per_bucket, "early": 10_000}, settle=10, rng=np.random.default_rng(0)
        )


@pytest.mark.skipif(not GOAL_PACK.exists(), reason="plan pack not present")
def test_pairs_are_one_stage_tasks_of_one_object():
    from openpi.glow import module_inputs as _mi

    pack = _plan_pack.PlanPack.load(str(GOAL_PACK))
    for a, b in gdp.DESTINATION_PAIRS:
        stage_a, stage_b = pack.task(a), pack.task(b)
        assert stage_a.num_plans == 1 and stage_b.num_plans == 1
        assert _mi.moved_object(stage_a.stages[0]) == _mi.moved_object(stage_b.stages[0])
        assert stage_a.stages[0].runtime_key != stage_b.stages[0].runtime_key


@pytest.mark.skipif(not GOAL_PACK.exists(), reason="plan pack not present")
@pytest.mark.parametrize(
    "config", ["pi05_libero_glow_v2t", "pi05_libero_glow_v2t_gate", "pi05_libero_glow_v2t_key"]
)
def test_variant_prefix_reads_each_task_where_it_should(monkeypatch, config):
    import torch

    from openpi.glow import module_encoders as _enc
    from openpi.glow import module_inputs as _mi
    from openpi.models import tokenizer as _tokenizer

    get_config = _diag_test._dummy_configs(monkeypatch)  # noqa: SLF001
    torch.manual_seed(0)
    cfg = get_config(config)
    object.__setattr__(cfg.model, "dtype", "float32")  # the precision the probe runs at
    model = cfg.model.create_pytorch().eval()
    for name, p in model.named_parameters():
        if name.endswith("out.1.weight"):
            torch.nn.init.normal_(p, std=0.02)
    pack = _plan_pack.PlanPack.load(str(GOAL_PACK))
    module = _mi.GlowModuleInputs(pack, _tokenizer.PaligemmaTokenizer(8), stage_order="sorted", stage_clock=True)
    samples = [{"pair": gdp.DESTINATION_PAIRS[0], "frame": 50}]
    a = gtp.module_arrays(module, samples, "A", 0, "cpu")
    b = gtp.module_arrays(module, samples, "B", 0, "cpu")
    images = [torch.rand(1, 3, 224, 224) * 2 - 1 for _ in range(3)]
    masks = [torch.ones(1, dtype=torch.bool) for _ in range(3)]
    tokens = torch.zeros(1, 8, dtype=torch.long)
    tokens[:, :2] = torch.tensor([1, 108])
    inputs = (images, masks, tokens, tokens != 0)
    temporal = model.glow_temporal_encoder

    with torch.no_grad():
        plain, pad, _ = model.embed_observation_prefix(a, *inputs)
        same, same_pad, _ = gdp.variant_prefix(model, a, a, False, *inputs)
        assert plain.equal(same) and pad.equal(same_pad)

        crossed, _, _ = gdp.variant_prefix(model, a, b, False, *inputs)
        other, _, _ = model.embed_observation_prefix(b, *inputs)
        at = plain.shape[1] - tokens.shape[1]
        spatial = slice(at - _enc.SPATIAL_TOKENS - _enc.TEMPORAL_TOKENS, at - _enc.TEMPORAL_TOKENS)
        time = slice(at - _enc.TEMPORAL_TOKENS, at)
        # The two tasks' tokens differ; the crossed prefix has A's Spatial and B's Temporal.
        assert not plain[:, spatial].equal(other[:, spatial]) and not plain[:, time].equal(other[:, time])
        assert crossed[:, spatial].equal(plain[:, spatial])
        # B's Temporal tokens up to their size: the rescale's target comes from A's words.
        unit = lambda x: (x / x.pow(2).mean(-1, keepdim=True).sqrt()).numpy()  # noqa: E731
        np.testing.assert_allclose(unit(crossed[:, time]), unit(other[:, time]), atol=1e-4)

        _, off_pad, _ = gdp.variant_prefix(model, a, a, True, *inputs)
        assert off_pad[:, time].sum() == 0 and off_pad[:, spatial].equal(pad[:, spatial])
    # The patches are gone: the model is as it was.
    assert "forward" not in vars(temporal) and "module_token_mask" not in vars(model)


@pytest.mark.skipif(not GOAL_PACK.exists(), reason="plan pack not present")
def test_spatial_from_replaces_the_spatial_encoder_alone(monkeypatch, tmp_path):
    import safetensors.torch
    import torch

    from openpi.glow import module_encoders as _enc
    from openpi.glow import module_inputs as _mi
    from openpi.models import tokenizer as _tokenizer

    get_config = _diag_test._dummy_configs(monkeypatch)  # noqa: SLF001

    def build(config, seed):
        torch.manual_seed(seed)
        cfg = get_config(config)
        object.__setattr__(cfg.model, "dtype", "float32")  # the precision the probe runs at
        model = cfg.model.create_pytorch().eval()
        for name, p in model.named_parameters():
            if name.endswith("out.1.weight"):
                torch.nn.init.normal_(p, std=0.02)
        return model

    key, gate = build("pi05_libero_glow_v2t_key", 0), build("pi05_libero_glow_v2t_gate", 1)
    # One backbone, so that only the encoders tell the two models apart.
    backbone = {k: v for k, v in key.state_dict().items() if not k.startswith(key.NEW_PARAM_PREFIXES)}
    gate.load_state_dict(backbone, strict=False)
    safetensors.torch.save_model(gate, str(tmp_path / "model.safetensors"))
    temporal_before = {k: v.clone() for k, v in key.glow_temporal_encoder.state_dict().items()}
    assert not key.glow_spatial_encoder.out[1].weight.equal(gate.glow_spatial_encoder.out[1].weight)

    gtp.load_spatial(key, str(tmp_path))

    for k, v in gate.glow_spatial_encoder.state_dict().items():
        assert key.glow_spatial_encoder.state_dict()[k].equal(v), k
    for k, v in key.glow_temporal_encoder.state_dict().items():
        assert v.equal(temporal_before[k]), k

    pack = _plan_pack.PlanPack.load(str(GOAL_PACK))
    module = _mi.GlowModuleInputs(pack, _tokenizer.PaligemmaTokenizer(8), stage_order="sorted", stage_clock=True)
    samples = [{"pair": gdp.DESTINATION_PAIRS[0], "frame": 50}]
    a = gtp.module_arrays(module, samples, "A", 0, "cpu")
    images = [torch.rand(1, 3, 224, 224) * 2 - 1 for _ in range(3)]
    masks = [torch.ones(1, dtype=torch.bool) for _ in range(3)]
    tokens = torch.zeros(1, 8, dtype=torch.long)
    tokens[:, :2] = torch.tensor([1, 108])
    inputs = (images, masks, tokens, tokens != 0)
    with torch.no_grad():
        # Temporal masked, the combined model's Spatial tokens are the other checkpoint's.
        combined, combined_pad, combined_att = gdp.variant_prefix(key, a, a, True, *inputs)
        donor, donor_pad, donor_att = gdp.variant_prefix(gate, a, a, True, *inputs)
        at = combined.shape[1] - tokens.shape[1]
        spatial = slice(at - _enc.SPATIAL_TOKENS - _enc.TEMPORAL_TOKENS, at - _enc.TEMPORAL_TOKENS)
        assert combined[:, spatial].equal(donor[:, spatial])
        assert combined_pad.equal(donor_pad) and combined_att.equal(donor_att)
    # A checkpoint without the encoder is refused rather than leaving the row's own.
    safetensors.torch.save_file({"other": torch.zeros(1)}, str(tmp_path / "model.safetensors"))
    with pytest.raises(RuntimeError, match="Missing key"):
        gtp.load_spatial(key, str(tmp_path))


@pytest.mark.skipif(not GOAL_PACK.exists(), reason="plan pack not present")
def test_fixed_gate_turns_down_the_other_stage_alone(monkeypatch):
    import torch

    from openpi.glow import module_encoders as _enc
    from openpi.glow import module_inputs as _mi
    from openpi.models import tokenizer as _tokenizer

    get_config = _diag_test._dummy_configs(monkeypatch)  # noqa: SLF001
    torch.manual_seed(0)
    cfg = get_config("pi05_libero_glow_v2t_key")
    object.__setattr__(cfg.model, "dtype", "float32")  # the precision the probe runs at
    model = cfg.model.create_pytorch().eval()
    for name, p in model.named_parameters():
        if name.endswith("out.1.weight"):
            torch.nn.init.normal_(p, std=0.02)
    pack = _plan_pack.PlanPack.load(str(GOAL_PACK))
    module = _mi.GlowModuleInputs(pack, _tokenizer.PaligemmaTokenizer(8), stage_order="sorted", stage_clock=False)
    samples = [{"pair": gdp.DESTINATION_PAIRS[0], "frame": 50}]
    two = gtp.module_arrays(module, samples, "A", 0, "cpu", gdp.SECOND_STAGE)
    one = gtp.module_arrays(module, samples, "A", 0, "cpu")
    assert int(two.glow_ints[0, _mi.INT_NUM_STAGES]) == 2 and int(two.glow_ints[0, _mi.INT_STAGE]) == 0
    images = [torch.rand(1, 3, 224, 224) * 2 - 1 for _ in range(3)]
    masks = [torch.ones(1, dtype=torch.bool) for _ in range(3)]
    tokens = torch.zeros(1, 8, dtype=torch.long)
    tokens[:, :2] = torch.tensor([1, 108])
    inputs = (images, masks, tokens, tokens != 0)
    width = len(_enc.SPATIAL_SLOTS)

    with torch.no_grad():
        plain_two = model.embed_observation_prefix(two, *inputs)[0]
        plain_one = model.embed_observation_prefix(one, *inputs)[0]
        gtp.set_fixed_gate(model, 0.25)
        gated_two = model.embed_observation_prefix(two, *inputs)[0]
        gated_one = model.embed_observation_prefix(one, *inputs)[0]
        with gtp.gates_at_one(model):
            full_two = model.embed_observation_prefix(two, *inputs)[0]
        assert model.config.module_stage_gate == 0.25

    at = plain_two.shape[1] - tokens.shape[1] - _enc.SPATIAL_TOKENS - _enc.TEMPORAL_TOKENS
    groups = two.glow_ints[0, _mi.INT_SPATIAL_SLOT].long().tolist()
    active_at, other_at = groups.index(0), groups.index(1)
    # The active stage's group, the summary and the Temporal tokens as they were; the other group scaled.
    other = slice(at + other_at * width, at + (other_at + 1) * width)
    active = slice(at + active_at * width, at + (active_at + 1) * width)
    np.testing.assert_allclose(gated_two[:, other].numpy(), 0.25 * plain_two[:, other].numpy(), atol=1e-5)
    assert gated_two[:, active].equal(plain_two[:, active])
    summary_on = at + _enc.SPATIAL_TOKENS - 1
    assert gated_two[:, summary_on:].equal(plain_two[:, summary_on:])
    # A one-stage plan has nothing to turn down; gates at 1 are the ungated model.
    assert gated_one.equal(plain_one) and full_two.equal(plain_two)

    gate_cfg = get_config("pi05_libero_glow_v2t_gate")
    with pytest.raises(ValueError, match="learned gate"):
        gtp.set_fixed_gate(gate_cfg.model.create_pytorch(), 0.25)


def test_parse_fixed_gate():
    rows = [("combined", "cfg", "/a")]
    assert gtp.parse_fixed_gate(("combined=0.06",), rows) == {"combined": 0.06}
    with pytest.raises(ValueError, match="names no row"):
        gtp.parse_fixed_gate(("other=0.06",), rows)
    with pytest.raises(ValueError, match="name=value"):
        gtp.parse_fixed_gate(("combined",), rows)
    with pytest.raises(ValueError, match=r"in \(0, 1\]"):
        gtp.parse_fixed_gate(("combined=0",), rows)


def test_parse_spatial_from():
    rows = [("combined", "cfg", "/a"), ("key", "cfg", "/b")]
    assert gtp.parse_spatial_from(("combined=/c",), rows) == {"combined": "/c"}
    assert gtp.parse_spatial_from((), rows) == {}
    with pytest.raises(ValueError, match="names no row"):
        gtp.parse_spatial_from(("other=/c",), rows)
    with pytest.raises(ValueError, match="name=weights_dir"):
        gtp.parse_spatial_from(("combined",), rows)


@needs_data
def test_main_end_to_end(monkeypatch, tmp_path):
    import safetensors.torch
    import torch

    get_config = _diag_test._dummy_configs(monkeypatch)  # noqa: SLF001
    torch.manual_seed(0)
    model = get_config("pi05_libero_glow_v2t_key").model.create_pytorch()
    for name, p in model.named_parameters():
        if name.endswith("out.1.weight"):
            torch.nn.init.normal_(p, std=0.02)
    safetensors.torch.save_model(model, str(tmp_path / "model.safetensors"))
    spatial_dir = tmp_path / "spatial"
    spatial_dir.mkdir()
    torch.manual_seed(1)
    spatial = get_config("pi05_libero_glow_v2t_gate").model.create_pytorch()
    safetensors.torch.save_model(spatial, str(spatial_dir / "model.safetensors"))
    s = gdp.main(
        gdp.Args(
            norm_stats_dir=str(ASSETS),
            dataset_root=str(DATASET),
            rows=(f"key=pi05_libero_glow_v2t_key:{tmp_path}",),
            spatial_from=(f"key={spatial_dir}",),
            fixed_gate=("key=0.06",),
            two_stage=True,
            exclude_episodes="379:427",
            episodes_per_task=1,
            reach_samples=2,
            early_samples=2,
            late_samples=2,
            batch_size=4,
            device="cpu",
            out=str(tmp_path / "out.json"),
        )
    )
    assert (tmp_path / "out.json").exists()
    assert sorted(s["heldout"]) == [10, 11, 14, 17, 18]
    # Each of the 5 tasks gives 2 samples per bucket; 28 of the 30 fill whole batches.
    row = s["rows"]["key"]["buckets"]
    assert sum(row[b]["variants"]["plan_own"]["samples"] for b in gdp.BUCKETS) == 28
    for bucket in gdp.BUCKETS:
        assert row[bucket]["teacher_gap"] > 0
        for pooled, names in gdp.POOLED.items():
            assert np.isfinite(row[bucket][pooled]["margin"])
            assert row[bucket][pooled]["samples"] == 2 * row[bucket]["variants"][names[0]]["samples"]
        for pooled in gdp.POOLED:
            assert sum(p["samples"] for p in row[bucket][f"{pooled}_by_pair"].values()) == row[bucket][pooled]["samples"]
