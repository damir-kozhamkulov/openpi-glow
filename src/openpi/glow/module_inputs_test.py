"""The planning-module arrays the data pipeline produces for the GLOW module encoders.

Needs the LIBERO-10 + Goal plan pack (glow_repo/derived/plan_pack_libero10_goal.json) and its
label chains (plan_chain.csv, goal_plan_chain.csv) - skipped otherwise. The end-to-end loader
check additionally needs the local LeRobot copy (HF_LEROBOT_HOME with repo `libero_staging`).
"""

import collections
import csv
import dataclasses
import json
import os
import pathlib

import numpy as np
import pytest

from openpi.glow import module_inputs as _mi
from openpi.glow import plan_pack as _plan_pack

REPO = pathlib.Path(__file__).resolve().parents[4]  # glow_repo (openpi is its submodule)
GOAL_PACK = REPO / "derived/plan_pack_libero10_goal.json"
CHAINS = (REPO / "derived/plan_chain.csv", REPO / "derived/goal_plan_chain.csv")
CONFIGS = (
    "pi05_libero_glow_st",
    "pi05_libero_glow_s_only",
    "pi05_libero_glow_nogrounding",
    "pi05_libero_glow_nosubgoal",
    "pi05_libero_glow_chain",
    "pi05_libero_glow_chain_exec",
)
GOAL_TASKS = tuple(range(10, 20))

pytestmark = pytest.mark.skipif(
    not (GOAL_PACK.exists() and all(c.exists() for c in CHAINS)), reason="Goal plan pack / label chains not present"
)


@pytest.fixture(scope="module")
def pack():
    return _plan_pack.PlanPack.load(GOAL_PACK)


@pytest.fixture(scope="module")
def raw():
    return json.loads(GOAL_PACK.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def chain():
    out = {}
    for path in CHAINS:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out.setdefault(int(r["episode_index"]), []).append(r)
    for rows in out.values():
        rows.sort(key=lambda r: int(r["plan_index"]))
    return out


@pytest.fixture(scope="module")
def tokenizer():
    from openpi.models import tokenizer as _tokenizer

    return _tokenizer.PaligemmaTokenizer(8)


@pytest.fixture(scope="module")
def transform(pack, tokenizer):
    return _mi.GlowModuleInputs(pack, tokenizer)


def _segment(arrays, name):
    start, stop = _mi.TEXT_SEGMENTS[name]
    return arrays["glow_text"][start:stop][arrays["glow_text_mask"][start:stop]].tolist()


def _by_index(pack, task_index):
    return next(t for t in pack.tasks.values() if t.task_index == task_index)


def test_layout():
    spans = sorted(_mi.TEXT_SEGMENTS.values())
    assert spans[0][0] == 0 and spans[-1][1] == _mi.TEXT_LEN
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:], strict=False))
    assert len(_mi.TEXT_SEGMENTS) == _mi.K_MAX * len(_mi.STAGE_FIELDS)
    assert set(_mi.SPATIAL_FIELDS) | set(_mi.TEMPORAL_FIELDS) == set(_mi.STAGE_FIELDS)
    assert _mi.INTS_LEN == 4 + 2 * _mi.K_MAX + _mi.K_MAX * len(_mi.HEAD_FIELDS)
    assert _mi.FLOATS_LEN == _mi.K_MAX * _mi.STAGE_FLOATS
    heads = [_mi.head_index(j, f) for j in range(_mi.K_MAX) for f in _mi.HEAD_FIELDS]
    assert heads == list(range(_mi.INT_HEAD.start, _mi.INT_HEAD.stop)) and _mi.INT_HEAD.stop == _mi.INTS_LEN
    assert _mi.INT_SPATIAL_SLOT.stop == _mi.INT_SHOWN < _mi.INT_STAGE_FRAMES < _mi.INT_HEAD.start
    for j in range(_mi.K_MAX):
        obj, obj_valid, tgt, tgt_valid = _mi.float_slices(j)
        assert (obj.start, obj.stop, obj_valid, tgt.start, tgt.stop, tgt_valid) == tuple(
            j * _mi.STAGE_FLOATS + o for o in (0, 3, 3, 4, 7, 7)
        )


def test_encode_every_stage(pack, raw, transform, tokenizer):
    """Every stage of every task, against the pack JSON read independently: every stage's text,
    predicate and anchors at its execution position, the pointer, and the sorted slot order."""
    for task_string, entry in raw["tasks"].items():
        stages = sorted(entry["stages"], key=lambda s: s["plan_index"])
        order = sorted(range(len(stages)), key=lambda j: stages[j]["runtime_key"])
        for k in range(len(stages)):
            a = transform.encode(task_string, k)
            assert a["glow_text"].shape == a["glow_text_mask"].shape == (_mi.TEXT_LEN,)
            assert a["glow_ints"].shape == (_mi.INTS_LEN,) and a["glow_floats"].shape == (_mi.FLOATS_LEN,)
            for j in range(_mi.K_MAX):
                st = stages[j] if j < len(stages) else None
                expect = (
                    {
                        "atom": st["atom"],
                        "subgoal": st["subgoal_operation"],
                        "object": st["object_noun"],
                        "target": st["target_noun"],
                        "rk": st["runtime_key"],
                    }
                    if st
                    else dict.fromkeys(_mi.STAGE_FIELDS, "")
                )
                for field, piece in expect.items():
                    assert _segment(a, f"{field}{j}") == (tokenizer.encode_text(piece) if piece else [])
            # Valid tokens fill each segment from its start.
            for start, stop in _mi.TEXT_SEGMENTS.values():
                m = a["glow_text_mask"][start:stop]
                assert m.tolist() == sorted(m.tolist(), reverse=True)
                assert not a["glow_text"][start:stop][~m].any()
            predicates = [_mi.PREDICATES.index(s["atom"].split("(")[0].lower()) + 1 for s in stages]
            ints = a["glow_ints"].tolist()
            assert ints[:2] == [k, entry["K"]]
            assert ints[_mi.INT_PREDICATE] == predicates + [0] * (_mi.K_MAX - len(stages))
            assert ints[_mi.INT_SPATIAL_SLOT] == order + list(range(len(stages), _mi.K_MAX))
            for j in range(_mi.K_MAX):
                obj, obj_valid, tgt, tgt_valid = _mi.float_slices(j)
                for xyz, valid, anchor in (
                    (obj, obj_valid, stages[j]["object_anchor"] if j < len(stages) else None),
                    (tgt, tgt_valid, stages[j]["target_anchor"] if j < len(stages) else None),
                ):
                    if anchor is None:
                        assert a["glow_floats"][valid] == 0 and not a["glow_floats"][xyz].any()
                    else:
                        assert a["glow_floats"][valid] == 1
                        np.testing.assert_allclose(a["glow_floats"][xyz], anchor["normalized"], rtol=1e-6)


def test_sorted_order_differs_from_execution_order(pack, tokenizer):
    """The moka-pot task runs moka_pot_2 before moka_pot_1: sorted by key, its groups swap."""
    moka = _by_index(pack, 6)
    assert [s.runtime_key.split("(")[1].split(",")[0] for s in moka.stages] == ["moka_pot_2", "moka_pot_1"]
    execution = _mi.GlowModuleInputs(pack, tokenizer, stage_order="execution")
    sorted_ = _mi.GlowModuleInputs(pack, tokenizer)
    assert sorted_.encode(moka.task, 0)["glow_ints"][_mi.INT_SPATIAL_SLOT].tolist() == [1, 0, 2, 3]
    assert execution.encode(moka.task, 0)["glow_ints"][_mi.INT_SPATIAL_SLOT].tolist() == [0, 1, 2, 3]
    # Only the slot order differs: the text arrays stay in execution order either way.
    for key in ("glow_text", "glow_text_mask", "glow_floats"):
        np.testing.assert_array_equal(sorted_.encode(moka.task, 0)[key], execution.encode(moka.task, 0)[key])
    with pytest.raises(ValueError, match="stage_order"):
        _mi.GlowModuleInputs(pack, tokenizer, stage_order="random")


def test_training_stage_at_boundaries(pack, chain, transform):
    ep = next(e for e, rows in sorted(chain.items()) if len(rows) == 2)
    rows = chain[ep]
    task = _by_index(pack, int(rows[0]["task_index"])).task
    boundary = int(rows[1]["start_frame"])
    for frame, k, clock in ((0, 0, 0), (boundary - 1, 0, boundary - 1), (boundary, 1, 0)):
        out = transform({"prompt": task, "episode_index": np.int64(ep), "frame_index": np.int64(frame), "state": 1})
        assert out["prompt"] == "" and out["state"] == 1
        assert "episode_index" not in out and "frame_index" not in out
        expected = transform.encode(task, k, stage_frames=clock)
        for key, value in expected.items():
            np.testing.assert_array_equal(out[key], value)


def test_inference_and_steering(pack, transform):
    two = next(t for t in pack.tasks.values() if t.num_plans == 2)
    one = next(t for t in pack.tasks.values() if t.num_plans == 1)
    out = transform({"prompt": two.task, "oracle_plan_index": np.int64(1), "prev_plan_index": np.int64(0)})
    assert out["prompt"] == "" and "prev_plan_index" not in out and "oracle_plan_index" not in out
    np.testing.assert_array_equal(out["glow_text"], transform.encode(two.task, 1)["glow_text"])
    # Steering: the steering task's plan, the stage clamped to its stages.
    out = transform({"prompt": two.task, "oracle_plan_index": 1, "plan_task": one.task})
    for key, value in transform.encode(one.task, 0).items():
        np.testing.assert_array_equal(out[key], value)
    with pytest.raises(ValueError, match="oracle"):
        transform({"prompt": two.task, "prev_plan_index": 0})
    ep = next(e for e, p in pack.episodes.items() if p.task_index == two.task_index)
    with pytest.raises(ValueError, match="steering"):
        transform({"prompt": two.task, "episode_index": ep, "frame_index": 0, "plan_task": one.task})
    with pytest.raises(ValueError, match="belongs to task"):
        transform({"prompt": one.task, "episode_index": ep, "frame_index": 0})


def test_plan_chain_at_inference(pack, transform):
    """plan_chain runs several tasks' stages as one plan, the index counting over the chain."""
    a, b = _by_index(pack, 17), _by_index(pack, 16)  # bowl on the stove; turn on the stove
    out = transform({"prompt": a.task, "oracle_plan_index": 1, "plan_chain": [a.task, b.task]})
    expected = transform.encode_stages((a.stages[0], b.stages[0]), 1)
    for key, value in expected.items():
        np.testing.assert_array_equal(out[key], value)
    assert out["glow_ints"][:2].tolist() == [1, 2]
    assert _segment(out, "rk0") != _segment(out, "rk1")
    # Sorted groups: "on(akita_black_bowl_1, ...)" sorts before "turnon(...)".
    assert out["glow_ints"][_mi.INT_SPATIAL_SLOT].tolist() == [0, 1, 2, 3]
    out = transform({"prompt": a.task, "oracle_plan_index": 0, "plan_chain": [b.task, a.task]})
    assert out["glow_ints"][_mi.INT_SPATIAL_SLOT].tolist() == [1, 0, 2, 3]
    # An empty chain is no chain; a chain excludes plan_task; a training sample takes none.
    np.testing.assert_array_equal(
        transform({"prompt": a.task, "oracle_plan_index": 0, "plan_chain": []})["glow_text"],
        transform.encode(a.task, 0)["glow_text"],
    )
    with pytest.raises(ValueError, match="exclude"):
        transform({"prompt": a.task, "oracle_plan_index": 0, "plan_chain": [b.task], "plan_task": b.task})
    ep = next(e for e, p in pack.episodes.items() if p.task_index == a.task_index)
    with pytest.raises(ValueError, match="steering"):
        transform({"prompt": a.task, "episode_index": ep, "frame_index": 0, "plan_chain": [a.task, b.task]})
    with pytest.raises(ValueError, match=f"{_mi.K_MAX}"):
        transform.encode_stages(tuple(a.stages) * (_mi.K_MAX + 1), 0)


def test_plan_chain_through_policy_inputs(monkeypatch, pack):
    """A chain request as examples/libero sends it, over msgpack, through the served config's
    input transforms (LiberoInputs first): every index of the chain reaches the module arrays."""
    import openpi.training.config as _config
    from openpi import transforms as _transforms
    from openpi.policies import libero_policy
    from openpi_client import msgpack_numpy

    monkeypatch.setattr(_config.DataConfigFactory, "_load_norm_stats", lambda self, d, a: None)
    cfg = _config.get_config("pi05_libero_glow_chain")
    dc = dataclasses.replace(cfg.data, plan_pack_path=str(GOAL_PACK)).create(pathlib.Path("."), cfg.model)
    assert isinstance(dc.data_transforms.inputs[0], libero_policy.LiberoInputs)
    served = _transforms.compose(
        [
            *dc.data_transforms.inputs,
            _transforms.Normalize(None, use_quantiles=dc.use_quantile_norm),
            *dc.model_transforms.inputs,
        ]
    )
    module = next(t for t in dc.model_transforms.inputs if isinstance(t, _mi.GlowModuleInputs))
    a, b = _by_index(pack, 17), _by_index(pack, 16)  # bowl on the stove; turn on the stove
    for chain, k, stages in (
        ([a.task, b.task], 0, (a.stages[0], b.stages[0])),
        ([a.task, b.task], 1, (a.stages[0], b.stages[0])),
        ([b.task, a.task], 1, (b.stages[0], a.stages[0])),
    ):
        element = {
            "observation/image": np.zeros((224, 224, 3), np.uint8),
            "observation/wrist_image": np.zeros((224, 224, 3), np.uint8),
            "observation/state": np.zeros(8),
            "prompt": a.task,
            "oracle_plan_index": k,
            "plan_chain": list(chain),
        }
        out = served(msgpack_numpy.unpackb(msgpack_numpy.packb(element)))
        assert out["glow_ints"][:2].tolist() == [k, 2]
        for key, value in module.encode_stages(stages, k).items():
            np.testing.assert_array_equal(out[key], value)
        assert "plan_chain" not in out and out["tokenized_prompt_mask"].sum() == 2


def test_chain_partners(pack):
    partners = _mi.chain_partners(pack, GOAL_TASKS)
    assert set(partners) == set(GOAL_TASKS)
    moved = {i: _mi.moved_object(_by_index(pack, i).stages[0]) for i in GOAL_TASKS}
    assert moved[17] == "akita_black_bowl_1" and moved[16] is None and moved[19] is None
    for own, mates in partners.items():
        assert own not in {m.task_index for m in mates}
        for m in mates:
            assert moved[own] is None or _mi.moved_object(m.stages[0]) != moved[own]
        assert [m.task_index for m in mates] == sorted(m.task_index for m in mates)
    # The four bowl tasks never pair with each other; the stove tasks pair with everything.
    bowl = {i for i in GOAL_TASKS if moved[i] == "akita_black_bowl_1"}
    assert len(bowl) == 4
    for i in bowl:
        assert not bowl & {m.task_index for m in partners[i]}
    assert len(partners[16]) == len(GOAL_TASKS) - 1
    assert sum(len(v) for v in partners.values()) == 76
    with pytest.raises(ValueError, match="stages"):
        _mi.chain_partners(pack, (0, 10))
    with pytest.raises(KeyError):
        _mi.chain_partners(pack, (99,))


def test_chain_augmentation(pack, tokenizer):
    own = _by_index(pack, 17)
    ep = next(e for e, p in pack.episodes.items() if p.task_index == own.task_index)
    sample = {"prompt": own.task, "episode_index": np.int64(ep), "frame_index": np.int64(0)}
    always = _mi.GlowModuleInputs(pack, tokenizer, chain_p=1.0, chain_tasks=GOAL_TASKS, rng=np.random.default_rng(0))
    partners = {m.task_index for m in always.partners[own.task_index]}
    seen = collections.Counter()
    for _ in range(120):
        out = always(dict(sample))
        k, num = out["glow_ints"][:2].tolist()
        assert num == 2 and out["prompt"] == ""
        other = 1 - k
        # The own stage sits at the pointer, a partner's at the other position.
        assert _segment(out, f"rk{k}") == tokenizer.encode_text(own.stages[0].runtime_key)
        matches = [
            m
            for m in always.partners[own.task_index]
            if _segment(out, f"rk{other}") == tokenizer.encode_text(m.stages[0].runtime_key)
        ]
        assert len(matches) == 1
        pair = (own.stages[0], matches[0].stages[0]) if k == 0 else (matches[0].stages[0], own.stages[0])
        expected = always.encode_stages(pair, k, stage_frames=0)  # frame 0: the clock at 0
        for key, value in expected.items():
            np.testing.assert_array_equal(out[key], value)
        seen[(k, matches[0].task_index)] += 1
    assert {i for _, i in seen} == partners and {k for k, _ in seen} == {0, 1}
    # chain_p = 0 (the default) never chains; a task outside chain_tasks never chains.
    never = _mi.GlowModuleInputs(pack, tokenizer, chain_tasks=GOAL_TASKS)
    for key, value in never.encode(own.task, 0, stage_frames=0).items():
        np.testing.assert_array_equal(never(dict(sample))[key], value)
    l10 = _by_index(pack, 9)
    l10_ep = next(e for e, p in pack.episodes.items() if p.task_index == l10.task_index)
    out = always({"prompt": l10.task, "episode_index": np.int64(l10_ep), "frame_index": np.int64(0)})
    for key, value in always.encode(l10.task, 0, stage_frames=0).items():
        np.testing.assert_array_equal(out[key], value)
    # Half the time at chain_p 0.5, with the global generator too.
    half = _mi.GlowModuleInputs(pack, tokenizer, chain_p=0.5, chain_tasks=GOAL_TASKS)
    np.random.seed(0)
    chained = sum(half(dict(sample))["glow_ints"][_mi.INT_NUM_STAGES] == 2 for _ in range(400))
    assert 150 < chained < 250
    # Inference samples are never chained.
    for key, value in always.encode(own.task, 0).items():
        np.testing.assert_array_equal(always({"prompt": own.task, "oracle_plan_index": 0})[key], value)
    with pytest.raises(ValueError, match="chain_tasks"):
        _mi.GlowModuleInputs(pack, tokenizer, chain_p=0.5)
    with pytest.raises(ValueError, match="chain_p"):
        _mi.GlowModuleInputs(pack, tokenizer, chain_p=1.5, chain_tasks=GOAL_TASKS)


def test_teacher_prompt(pack, chain, transform, tokenizer):
    from openpi.models import tokenizer as _tokenizer

    teacher_tokenizer = _tokenizer.PaligemmaTokenizer(48)
    with_teacher = _mi.GlowModuleInputs(pack, tokenizer, teacher_tokenizer)
    ep = next(e for e, rows in sorted(chain.items()) if len(rows) == 2)
    task = _by_index(pack, int(chain[ep][0]["task_index"])).task
    sample = {"prompt": task, "episode_index": np.int64(ep), "frame_index": np.int64(0)}
    out = with_teacher(dict(sample))
    # The task string as the stock pi05 prompt (BOS, task, newline), the student's prompt emptied.
    tokens, mask = teacher_tokenizer.tokenize(task)
    np.testing.assert_array_equal(out["glow_teacher_prompt"], tokens)
    np.testing.assert_array_equal(out["glow_teacher_prompt_mask"], mask)
    assert out["prompt"] == "" and 2 < mask.sum() < 48
    for key, value in transform(dict(sample)).items():
        np.testing.assert_array_equal(out[key], value)
    # Inference never carries it, and neither does a transform without a teacher tokenizer.
    assert "glow_teacher_prompt" not in with_teacher({"prompt": task, "oracle_plan_index": 0})
    assert "glow_teacher_prompt" not in transform(dict(sample))
    # A chained sample's teacher prompt is still its own task string.
    own = _by_index(pack, 17)
    own_ep = next(e for e, p in pack.episodes.items() if p.task_index == own.task_index)
    chained = _mi.GlowModuleInputs(pack, tokenizer, teacher_tokenizer, chain_p=1.0, chain_tasks=GOAL_TASKS)
    out = chained({"prompt": own.task, "episode_index": np.int64(own_ep), "frame_index": np.int64(0)})
    np.testing.assert_array_equal(out["glow_teacher_prompt"], teacher_tokenizer.tokenize(own.task)[0])
    assert out["glow_ints"][_mi.INT_NUM_STAGES] == 2


def test_samples_make_a_valid_observation(pack, transform):
    from openpi.models import model as _model

    samples = []
    for i, t in enumerate(list(pack.tasks.values())[:3]):
        s = transform({"prompt": t.task, "oracle_plan_index": 0})
        s |= {
            "image": {"base_0_rgb": np.zeros((224, 224, 3), np.uint8)},
            "image_mask": {"base_0_rgb": np.bool_(True)},
            "state": np.full(32, i, np.float32),
            "tokenized_prompt": np.zeros(8, np.int32),
            "tokenized_prompt_mask": np.zeros(8, bool),
        }
        s.pop("prompt")
        samples.append(s)
    import jax

    batch = jax.tree.map(lambda *xs: np.stack(xs), *samples)
    obs = _model.Observation.from_dict(batch)
    assert obs.glow_text.shape == (3, _mi.TEXT_LEN) and obs.glow_floats.shape == (3, _mi.FLOATS_LEN)
    assert obs.glow_ints.shape == (3, _mi.INTS_LEN)


def test_module_configs(monkeypatch):
    import openpi.training.config as _config
    from openpi.glow.model_config import GlowModulePi0Config

    plan_prompt = _config.get_config("pi05_libero_plan_prompt")
    switches = {
        "pi05_libero_glow_st": {},
        "pi05_libero_glow_s_only": {"module_temporal": False},
        "pi05_libero_glow_nogrounding": {"module_grounding": False},
        "pi05_libero_glow_nosubgoal": {"module_subgoal_text": False},
        "pi05_libero_glow_chain": {"module_text_anchor": True, "module_token_scale": "words"},
        "pi05_libero_glow_chain_exec": {"module_text_anchor": True, "module_token_scale": "words"},
    }
    data = {
        "pi05_libero_glow_chain": {
            "module_chain_p": 0.5,
            "module_chain_tasks": GOAL_TASKS,
            "module_stage_order": "sorted",
        },
        "pi05_libero_glow_chain_exec": {
            "module_chain_p": 0.5,
            "module_chain_tasks": GOAL_TASKS,
            "module_stage_order": "execution",
        },
    }
    monkeypatch.setattr(_config.DataConfigFactory, "_load_norm_stats", lambda self, d, a: None)
    for name in CONFIGS:
        cfg = _config.get_config(name)
        assert isinstance(cfg.model, GlowModulePi0Config) and cfg.model.max_token_len == 8
        expected = dataclasses.replace(GlowModulePi0Config(), **switches[name])
        for field in (
            "module_spatial",
            "module_temporal",
            "module_grounding",
            "module_subgoal_text",
            "module_text_anchor",
        ):
            assert getattr(cfg.model, field) == getattr(expected, field), (name, field)
        assert cfg.data.module_inputs and cfg.data.plan_pack_path == plan_prompt.data.plan_pack_path
        assert cfg.data.train_episodes == plan_prompt.data.train_episodes == "0:807"
        for field, value in data.get(
            name, {"module_chain_p": 0.0, "module_chain_tasks": (), "module_stage_order": "sorted"}
        ).items():
            assert getattr(cfg.data, field) == value, (name, field)
        assert cfg.lr_schedule == plan_prompt.lr_schedule and cfg.batch_size == plan_prompt.batch_size
        # The transform the config builds carries the data fields.
        monkeypatch.setenv(_plan_pack.ENV_VAR, str(GOAL_PACK))
        dc = dataclasses.replace(cfg.data, plan_pack_path=None).create(cfg.assets_dirs, cfg.model)
        module = next(t for t in dc.model_transforms.inputs if isinstance(t, _mi.GlowModuleInputs))
        assert (module.chain_p, module.chain_tasks, module.stage_order) == (
            cfg.data.module_chain_p,
            cfg.data.module_chain_tasks,
            cfg.data.module_stage_order,
        )
        assert set(module.partners) == set(module.chain_tasks)
        # The text control's data and recipe.
        for field in ("batch_size", "num_train_steps", "lr_schedule", "optimizer", "ema_decay", "weight_loader"):
            assert getattr(cfg, field) == getattr(plan_prompt, field), (name, field)
        assert not cfg.data.plan_as_prompt
    cfg = _config.get_config("pi05_libero_glow_st")
    local = dataclasses.replace(cfg.data, plan_pack_path=str(GOAL_PACK))
    kinds = [type(t).__name__ for t in local.create(pathlib.Path("."), cfg.model).model_transforms.inputs]
    assert kinds.index("GlowModuleInputs") + 1 == kinds.index("TokenizePrompt")
    assert "GlowPlanAsPrompt" not in kinds and "GlowPlanState" not in kinds
    with pytest.raises(ValueError, match="GlowModulePi0Config"):
        local.create(pathlib.Path("."), plan_prompt.model)
    with pytest.raises(ValueError, match="exclude"):
        dataclasses.replace(local, plan_as_prompt=True).create(pathlib.Path("."), cfg.model)
    with pytest.raises(ValueError, match="stage_order"):
        dataclasses.replace(local, module_stage_order="random").create(pathlib.Path("."), cfg.model)


def test_align_configs(monkeypatch):
    import openpi.training.config as _config

    monkeypatch.setattr(_config.DataConfigFactory, "_load_norm_stats", lambda self, d, a: None)
    st = _config.get_config("pi05_libero_glow_st")
    align = _config.get_config("pi05_libero_glow_st_align")
    anchor = _config.get_config("pi05_libero_glow_st_anchor")
    chain = _config.get_config("pi05_libero_glow_chain")
    assert align.model.module_text_anchor and align.model.freeze_pretrained and align.model.module_distill_weight > 0
    assert anchor.model.module_text_anchor and not anchor.model.freeze_pretrained
    assert anchor.model.module_distill_weight == 0 and not st.model.module_text_anchor
    # Both enter the prefix at a word token's size; the unanchored row keeps the language scale.
    assert align.model.module_token_scale == anchor.model.module_token_scale == "words"
    assert st.model.module_token_scale == "language"
    # Same data and recipe as the unanchored row; the chain row is _st_anchor's model with chains.
    for cfg in (align, anchor):
        assert dataclasses.replace(cfg.data) == st.data
        for field in ("batch_size", "num_train_steps", "lr_schedule", "optimizer", "weight_loader"):
            assert getattr(cfg, field) == getattr(st, field), field
    assert chain.model == anchor.model
    assert dataclasses.replace(chain.data, module_chain_p=0.0, module_chain_tasks=()) == anchor.data

    def module_transform(cfg):
        local = dataclasses.replace(cfg.data, plan_pack_path=str(GOAL_PACK))
        inputs = local.create(pathlib.Path("."), cfg.model).model_transforms.inputs
        return next(t for t in inputs if isinstance(t, _mi.GlowModuleInputs))

    # Only a distilling model gets the teacher prompt.
    assert module_transform(align).teacher_tokenizer is not None
    assert module_transform(anchor).teacher_tokenizer is None and module_transform(st).teacher_tokenizer is None


@pytest.mark.parametrize(
    "config_name", ["pi05_libero_glow_st", "pi05_libero_glow_st_align", "pi05_libero_glow_chain", "pi05_libero_glow_v2"]
)
def test_module_loader_end_to_end(monkeypatch, config_name):
    home = os.environ.get("HF_LEROBOT_HOME")
    if not home or not (pathlib.Path(home) / "libero_staging").exists():
        pytest.skip("local LeRobot copy not present")
    import openpi.training.config as _config
    import openpi.training.data_loader as _data

    # A Goal-only subset: not episodes 0..n-1, so the loader restricts the whole dataset's rows.
    monkeypatch.setenv("OPENPI_TRAIN_EPISODES", "379:383")
    monkeypatch.setattr(_config.DataConfigFactory, "_load_norm_stats", lambda self, d, a: None)
    cfg = _config.get_config(config_name)
    cfg = dataclasses.replace(
        cfg,
        data=dataclasses.replace(cfg.data, repo_id="libero_staging", plan_pack_path=str(GOAL_PACK)),
        batch_size=8,
        num_workers=0,
    )
    chained = 0
    loader = _data.create_data_loader(cfg, framework="pytorch", shuffle=True, num_batches=3, skip_norm_stats=True)
    for obs, _ in loader:
        # The prompt is BOS + "\n" only; the goal is in the module arrays.
        assert obs.tokenized_prompt.shape == (8, 8)
        assert (np.asarray(obs.tokenized_prompt_mask).sum(axis=1) == 2).all()
        ints = np.asarray(obs.glow_ints)
        assert ints.shape == (8, _mi.INTS_LEN) and np.asarray(obs.glow_text).shape == (8, _mi.TEXT_LEN)
        assert np.asarray(obs.glow_floats).shape == (8, _mi.FLOATS_LEN)
        two = ints[:, _mi.INT_NUM_STAGES] == 2
        chained += int(two.sum())
        if cfg.data.module_chain_p == 0:
            assert (ints[:, _mi.INT_NUM_STAGES] == 1).all() and (ints[:, _mi.INT_STAGE] == 0).all()
        else:
            assert (ints[~two, _mi.INT_STAGE] == 0).all() and (ints[two, _mi.INT_STAGE] <= 1).all()
            # A chained sample has two runtime keys; an unchained one has one.
            rk1 = slice(*_mi.TEXT_SEGMENTS["rk1"])
            assert (np.asarray(obs.glow_text_mask)[:, rk1].any(axis=1) == two).all()
        assert np.asarray(obs.glow_text_mask).any(axis=1).all()
        if cfg.model.module_distill_weight > 0:
            # The teacher's prompt: BOS, the task string, newline.
            assert obs.glow_teacher_prompt.shape == (8, cfg.model.module_teacher_prompt_len)
            assert (np.asarray(obs.glow_teacher_prompt_mask).sum(axis=1) > 2).all()
        else:
            assert obs.glow_teacher_prompt is None
    if cfg.data.module_chain_p > 0:
        assert 0 < chained < 24


def test_remaining_order_shows_the_active_stage_first(pack, tokenizer):
    task = _by_index(pack, 0)  # two stages
    remaining = _mi.GlowModuleInputs(pack, tokenizer, stage_order="remaining")
    first, second = (remaining.encode(task.task, k)["glow_ints"] for k in (0, 1))
    assert first[_mi.INT_SPATIAL_SLOT].tolist() == [0, 1, 2, 3] and first[_mi.INT_SHOWN] == 2
    assert second[_mi.INT_SPATIAL_SLOT].tolist() == [1, 0, 2, 3] and second[_mi.INT_SHOWN] == 1
    # The text, floats and the Temporal ints stay in execution order: only the Spatial view changes.
    execution = _mi.GlowModuleInputs(pack, tokenizer, stage_order="execution").encode(task.task, 1)
    out = remaining.encode(task.task, 1)
    for key in ("glow_text", "glow_text_mask", "glow_floats"):
        np.testing.assert_array_equal(out[key], execution[key])
    assert execution["glow_ints"][_mi.INT_SHOWN] == 2
    for order in ("sorted", "execution"):
        m = _mi.GlowModuleInputs(pack, tokenizer, stage_order=order)
        assert all(
            m.encode(t.task, k)["glow_ints"][_mi.INT_SHOWN] == t.num_plans
            for t in pack.tasks.values()
            for k in range(t.num_plans)
        )


def test_head_words(pack, transform, tokenizer):
    stage = _by_index(pack, 0).stages[1]  # "put the yellow and white mug on the right plate"
    words = _mi.head_words(stage, 1)
    assert words == {"op": "put", "object": "mug", "target": "plate", "atom": "on", "ordinal": "second", "count": "two"}
    assert _mi.head_noun("black bowl (left)") == "bowl" and _mi.head_noun("") is None
    turnon = next(t for t in pack.tasks.values() if t.stages[0].predicate == "turnon")
    assert _mi.head_words(turnon.stages[0], 0)["atom"] == "on"
    for task in pack.tasks.values():
        for k in range(task.num_plans):
            ints = transform.encode(task.task, k)["glow_ints"]
            for j in range(_mi.K_MAX):
                if j < task.num_plans:
                    words = _mi.head_words(task.stages[j], j)
                else:
                    words = {"ordinal": _mi.ORDINAL_WORDS[j], "count": _mi.COUNT_WORDS[j]}
                for field in _mi.HEAD_FIELDS:
                    word = words.get(field)
                    want = tokenizer.encode_text(word)[0] if word else 0
                    assert ints[_mi.head_index(j, field)] == want, (task.task_index, k, j, field)


def test_stage_clock(pack, chain, transform, tokenizer):
    task = _by_index(pack, 0)
    out = transform({"prompt": task.task, "oracle_plan_index": 1, "plan_stage_frames": np.int64(37)})
    assert out["glow_ints"][_mi.INT_STAGE_FRAMES] == 37 and "plan_stage_frames" not in out
    assert transform({"prompt": task.task, "oracle_plan_index": 1})["glow_ints"][_mi.INT_STAGE_FRAMES] == -1
    clocked = _mi.GlowModuleInputs(pack, tokenizer, stage_clock=True)
    with pytest.raises(ValueError, match="plan_stage_frames"):
        clocked({"prompt": task.task, "oracle_plan_index": 1})
    # Training: frames since the labeled stage began.
    ep = next(e for e, rows in sorted(chain.items()) if len(rows) == 2)
    start = int(chain[ep][1]["start_frame"])
    task = _by_index(pack, int(chain[ep][0]["task_index"]))
    out = clocked({"prompt": task.task, "episode_index": np.int64(ep), "frame_index": np.int64(start + 9)})
    assert out["glow_ints"][_mi.INT_STAGE] == 1 and out["glow_ints"][_mi.INT_STAGE_FRAMES] == 9


def test_episode_grounding(pack, raw, tokenizer):
    module = _mi.GlowModuleInputs(pack, tokenizer, episode_grounding=True)
    plain = _mi.GlowModuleInputs(pack, tokenizer)
    own_count = 0
    for ep_key in ("0", "3", "400", "700"):
        entry = raw["episodes"][ep_key]
        task = _by_index(pack, entry["task_index"])
        sample = {"prompt": task.task, "episode_index": np.int64(int(ep_key)), "frame_index": np.int64(0)}
        floats = module(dict(sample))["glow_floats"]
        means = plain(dict(sample))["glow_floats"]
        for j, anchors in enumerate(entry["anchors"]):
            obj, obj_valid, tgt, tgt_valid = _mi.float_slices(j)
            for sl, valid, own in ((obj, obj_valid, anchors["object"]), (tgt, tgt_valid, anchors["target"])):
                if own is None:  # no episode anchor: the per-task one
                    np.testing.assert_array_equal(floats[sl], means[sl])
                else:
                    np.testing.assert_allclose(floats[sl], own, rtol=1e-6)
                    assert floats[valid] == 1.0
                    own_count += 1
    assert own_count > 0
    # Inference reads the per-task anchors.
    task = _by_index(pack, 0)
    np.testing.assert_array_equal(
        module({"prompt": task.task, "oracle_plan_index": 0})["glow_floats"], plain.encode(task.task, 0)["glow_floats"]
    )
    # A chain keeps the own stage's episode anchor and the partner's per-task anchor.
    own = _by_index(pack, 17)
    ep = next(e for e, p in pack.episodes.items() if p.task_index == 17 and p.anchors[0][0] is not None)
    chained = _mi.GlowModuleInputs(
        pack, tokenizer, episode_grounding=True, chain_p=1.0, chain_tasks=GOAL_TASKS, rng=np.random.default_rng(1)
    )
    out = chained({"prompt": own.task, "episode_index": np.int64(ep), "frame_index": np.int64(0)})
    k = int(out["glow_ints"][_mi.INT_STAGE])
    np.testing.assert_allclose(out["glow_floats"][_mi.float_slices(k)[0]], pack.episodes[ep].anchors[0][0], rtol=1e-6)
    partner_rk = _segment(out, f"rk{1 - k}")
    partner = next(t for t in pack.tasks.values() if tokenizer.encode_text(t.stages[0].runtime_key) == partner_rk)
    if partner.stages[0].object_xyz is not None:
        np.testing.assert_allclose(
            out["glow_floats"][_mi.float_slices(1 - k)[0]], partner.stages[0].object_xyz, rtol=1e-6
        )


def test_pack_episode_anchors(raw, tmp_path):
    pack = _plan_pack.PlanPack.load(GOAL_PACK)
    assert all(e.anchors is not None and len(e.anchors) == len(e.stage_starts) for e in pack.episodes.values())
    # A pack without them loads (None); one whose anchors miss a stage does not.
    bare = json.loads(json.dumps(raw))
    for e in bare["episodes"].values():
        e.pop("anchors")
    (tmp_path / "bare.json").write_text(json.dumps(bare), encoding="utf-8")
    assert all(e.anchors is None for e in _plan_pack.PlanPack.load(tmp_path / "bare.json").episodes.values())
    broken = json.loads(json.dumps(raw))
    broken["episodes"]["0"]["anchors"] = broken["episodes"]["0"]["anchors"][:1]
    (tmp_path / "broken.json").write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="stage starts"):
        _plan_pack.PlanPack.load(tmp_path / "broken.json")


def test_v2_configs(monkeypatch):
    import openpi.training.config as _config

    monkeypatch.setattr(_config.DataConfigFactory, "_load_norm_stats", lambda self, d, a: None)
    anchor = _config.get_config("pi05_libero_glow_st_anchor")
    parity = _config.get_config("pi05_libero_glow_st_anchor_fv")
    assert parity.freeze_vlm and not anchor.freeze_vlm
    assert dataclasses.replace(parity, name=anchor.name, freeze_vlm=False) == anchor
    v2 = _config.get_config("pi05_libero_glow_v2")
    assert v2.freeze_vlm and v2.model.module_anchor == "head" and v2.model.module_stage_clock
    assert (v2.data.module_stage_order, v2.data.module_stage_clock, v2.data.module_episode_grounding) == (
        "remaining",
        True,
        True,
    )
    assert (v2.data.decision_window, v2.data.decision_repeat) == (40, 3)
    for name, switches in (
        ("pi05_libero_glow_v2_nogrounding", {"module_grounding": False}),
        ("pi05_libero_glow_v2_noatom", {"module_atom": False}),
        (
            "pi05_libero_glow_v2_grounding_only",
            {"module_subgoal_text": False, "module_nouns": False, "module_atom": False},
        ),
    ):
        cfg = _config.get_config(name)
        assert cfg.model == dataclasses.replace(v2.model, **switches) and cfg.data == v2.data, name
    text = _config.get_config("pi05_libero_plan_prompt_fv")
    stock = _config.get_config("pi05_libero_plan_prompt")
    assert text.freeze_vlm and text.model == stock.model
    assert dataclasses.replace(text.data, decision_window=0, decision_repeat=1) == stock.data
    # The data config carries the stage starts for oversampling and the transform flags.
    local = dataclasses.replace(v2.data, plan_pack_path=str(GOAL_PACK))
    dc = local.create(pathlib.Path("."), v2.model)
    assert (dc.decision_window, dc.decision_repeat) == (40, 3) and len(dc.episode_stage_starts) == 807
    module = next(t for t in dc.model_transforms.inputs if isinstance(t, _mi.GlowModuleInputs))
    assert module.stage_order == "remaining" and module.stage_clock and module.episode_grounding
    # The clock must match the model's.
    with pytest.raises(ValueError, match="module_stage_clock"):
        dataclasses.replace(local, module_stage_clock=False).create(pathlib.Path("."), v2.model)
    with pytest.raises(ValueError, match="decision_repeat"):
        dataclasses.replace(local, decision_repeat=0).create(pathlib.Path("."), v2.model)


def test_stage_clock_through_policy_inputs(monkeypatch, pack):
    """The stage clock as examples/libero sends it reaches a clocked config's module arrays."""
    import openpi.training.config as _config
    from openpi import transforms as _transforms
    from openpi_client import msgpack_numpy

    monkeypatch.setattr(_config.DataConfigFactory, "_load_norm_stats", lambda self, d, a: None)
    cfg = _config.get_config("pi05_libero_glow_v2")
    dc = dataclasses.replace(cfg.data, plan_pack_path=str(GOAL_PACK)).create(pathlib.Path("."), cfg.model)
    served = _transforms.compose(
        [
            *dc.data_transforms.inputs,
            _transforms.Normalize(None, use_quantiles=dc.use_quantile_norm),
            *dc.model_transforms.inputs,
        ]
    )
    task = _by_index(pack, 0)
    element = {
        "observation/image": np.zeros((224, 224, 3), np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), np.uint8),
        "observation/state": np.zeros(8),
        "prompt": task.task,
        "prev_plan_index": 1,
        "oracle_plan_index": 1,
        "plan_stage_frames": 25,
    }
    out = served(msgpack_numpy.unpackb(msgpack_numpy.packb(element)))
    ints = out["glow_ints"]
    assert ints[_mi.INT_STAGE] == 1 and ints[_mi.INT_STAGE_FRAMES] == 25 and ints[_mi.INT_SHOWN] == 1
    assert "plan_stage_frames" not in out
    with pytest.raises(ValueError, match="plan_stage_frames"):
        served(msgpack_numpy.unpackb(msgpack_numpy.packb({k: v for k, v in element.items() if k != "plan_stage_frames"})))
