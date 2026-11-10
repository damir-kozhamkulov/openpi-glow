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
    out = transform({"prompt": two.task, "oracle_plan_index": np.int64(1)})
    assert out["prompt"] == "" and "oracle_plan_index" not in out
    np.testing.assert_array_equal(out["glow_text"], transform.encode(two.task, 1)["glow_text"])
    # Steering: the steering task's plan, the stage clamped to its stages.
    out = transform({"prompt": two.task, "oracle_plan_index": 1, "plan_task": one.task})
    for key, value in transform.encode(one.task, 0).items():
        np.testing.assert_array_equal(out[key], value)
    with pytest.raises(ValueError, match="oracle"):
        transform({"prompt": two.task})
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
    cfg = _config.get_config("pi05_libero_glow_v2t_fixed")
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


def test_data_config_validation(monkeypatch):
    import openpi.training.config as _config

    monkeypatch.setattr(_config.DataConfigFactory, "_load_norm_stats", lambda self, d, a: None)
    cfg = _config.get_config("pi05_libero_glow_v2t_fixed")
    stock = _config.get_config("pi05_libero")
    assert cfg.model.max_token_len == 8 and cfg.data.train_episodes == "0:807"
    local = dataclasses.replace(cfg.data, plan_pack_path=str(GOAL_PACK))
    kinds = [type(t).__name__ for t in local.create(pathlib.Path("."), cfg.model).model_transforms.inputs]
    assert kinds.index("GlowModuleInputs") + 1 == kinds.index("TokenizePrompt")
    # The pack path falls back to the environment variable.
    monkeypatch.setenv(_plan_pack.ENV_VAR, str(GOAL_PACK))
    dc = dataclasses.replace(cfg.data, plan_pack_path=None).create(cfg.assets_dirs, cfg.model)
    module = next(t for t in dc.model_transforms.inputs if isinstance(t, _mi.GlowModuleInputs))
    assert (module.chain_p, module.chain_tasks, module.stage_order) == (0.0, (), "sorted")
    with pytest.raises(ValueError, match="GlowModulePi0Config"):
        local.create(pathlib.Path("."), stock.model)
    with pytest.raises(ValueError, match="stage_order"):
        dataclasses.replace(local, module_stage_order="random").create(pathlib.Path("."), cfg.model)
    # The clock must match the model's.
    with pytest.raises(ValueError, match="module_stage_clock"):
        dataclasses.replace(local, module_stage_clock=True).create(pathlib.Path("."), cfg.model)
    with pytest.raises(ValueError, match="decision_repeat"):
        dataclasses.replace(local, decision_repeat=0).create(pathlib.Path("."), cfg.model)


@pytest.mark.parametrize(
    "config_name", ["pi05_libero_glow_v2t_gate", "pi05_libero_glow_v2t_fixed", "pi05_libero_glow_monitor"]
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
    loader = _data.create_data_loader(cfg, framework="pytorch", shuffle=True, num_batches=3, skip_norm_stats=True)
    for obs, _ in loader:
        # The prompt is BOS + "\n" only; the goal is in the module arrays.
        assert obs.tokenized_prompt.shape == (8, 8)
        assert (np.asarray(obs.tokenized_prompt_mask).sum(axis=1) == 2).all()
        ints = np.asarray(obs.glow_ints)
        assert ints.shape == (8, _mi.INTS_LEN) and np.asarray(obs.glow_text).shape == (8, _mi.TEXT_LEN)
        assert np.asarray(obs.glow_floats).shape == (8, _mi.FLOATS_LEN)
        # A Goal sample has one stage; a counterfactual one has another task's plan chained with a
        # partner (two runtime keys), stage 0 active, and no flow term.
        two = ints[:, _mi.INT_NUM_STAGES] == 2
        assert (ints[~two, _mi.INT_NUM_STAGES] == 1).all() and (ints[:, _mi.INT_STAGE] == 0).all()
        rk1 = slice(*_mi.TEXT_SEGMENTS["rk1"])
        assert (np.asarray(obs.glow_text_mask)[:, rk1].any(axis=1) == two).all()
        np.testing.assert_array_equal(np.asarray(obs.glow_flow_weight) == 0, two)
        assert np.asarray(obs.glow_text_mask).any(axis=1).all()
        if cfg.model.module_distill_weight > 0:
            # The teacher's prompt: BOS, the task string, newline.
            assert obs.glow_teacher_prompt.shape == (8, cfg.model.module_teacher_prompt_len)
            assert (np.asarray(obs.glow_teacher_prompt_mask).sum(axis=1) > 2).all()
        else:
            assert obs.glow_teacher_prompt is None


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
    assert words == {
        "op": "put",
        "object": "mug",
        "target": "plate",
        "atom": "on",
        "ordinal": "second",
        "count": "two",
        "key_predicate": "on",
        "key_object": "mug",
    }
    assert _mi.key_words("on(moka_pot_1, flat_stove_1_cook_region)") == ("on", "pot")
    assert _mi.key_words("open(wooden_cabinet_1_middle_region)") == ("open", "middle")
    assert _mi.key_words("turnon(flat_stove_1)") == ("on", "stove")
    assert all(all(_mi.key_words(s.runtime_key)) for t in pack.tasks.values() for s in t.stages)
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


def test_stage_clock_through_policy_inputs(monkeypatch, pack):
    """The stage clock as examples/libero sends it reaches a clocked config's module arrays."""
    import openpi.training.config as _config
    from openpi import transforms as _transforms
    from openpi_client import msgpack_numpy

    monkeypatch.setattr(_config.DataConfigFactory, "_load_norm_stats", lambda self, d, a: None)
    cfg = _config.get_config("pi05_libero_glow_v2t_gate")
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
        "oracle_plan_index": 1,
        "plan_stage_frames": 25,
    }
    out = served(msgpack_numpy.unpackb(msgpack_numpy.packb(element)))
    ints = out["glow_ints"]
    assert ints[_mi.INT_STAGE] == 1 and ints[_mi.INT_STAGE_FRAMES] == 25 and ints[_mi.INT_SHOWN] == 2
    assert "plan_stage_frames" not in out
    with pytest.raises(ValueError, match="plan_stage_frames"):
        served(msgpack_numpy.unpackb(msgpack_numpy.packb({k: v for k, v in element.items() if k != "plan_stage_frames"})))


def test_teacher_prompt_rule(pack):
    # A Goal stage's instruction is its stock prompt; a LIBERO-10 stage keeps the task string.
    goal = _by_index(pack, 17)
    assert _mi.teacher_prompt(goal.stages[0], "put the bowl on the stove then open the top drawer") == goal.task
    l10 = _by_index(pack, 0)
    assert all(_mi.teacher_prompt(s, l10.task) == l10.task for s in l10.stages)


def _counterfactual(pack, p=1.0, seed=0, **kw):
    from openpi.models import tokenizer as _tokenizer

    return _mi.GlowModuleInputs(
        pack,
        _tokenizer.PaligemmaTokenizer(8),
        _tokenizer.PaligemmaTokenizer(48),
        counterfactual_p=p,
        counterfactual_window=40,
        counterfactual_tasks=GOAL_TASKS,
        rng=np.random.default_rng(seed),
        **kw,
    )


def test_counterfactual_samples(pack):
    t = _counterfactual(pack)
    teacher = t.teacher_tokenizer
    own = _by_index(pack, 17)
    ep = next(e for e, p in pack.episodes.items() if p.task_index == own.task_index)
    actives = collections.Counter()
    for frame in (0, 39):
        for _ in range(60):
            out = t({"prompt": own.task, "episode_index": np.int64(ep), "frame_index": np.int64(frame)})
            ints = out["glow_ints"]
            assert ints[_mi.INT_STAGE] == 0 and ints[_mi.INT_NUM_STAGES] == 2 and out["glow_flow_weight"] == 0.0
            def runtime_key_is(name, task, out=out):
                return _segment(out, name) == t.tokenizer.encode_text(task.stages[0].runtime_key)

            active = next(a for a in (_by_index(pack, i) for i in GOAL_TASKS) if runtime_key_is("rk0", a))
            partner = next(a for a in t.counter_partners[active.task_index] if runtime_key_is("rk1", a))
            assert active.task_index != own.task_index
            # The plan is (another task, one of its partners), stage 0 active, the per-task anchors
            # and the sample's own clock; the teacher reads the active task's instruction.
            expected = t.encode_stages((active.stages[0], partner.stages[0]), 0, stage_frames=frame)
            for key, value in expected.items():
                np.testing.assert_array_equal(out[key], value)
            np.testing.assert_array_equal(out["glow_teacher_prompt"], teacher.tokenize(active.task)[0])
            actives[active.task_index] += 1
    assert set(actives) == set(GOAL_TASKS) - {own.task_index}
    # Past the window, a LIBERO-10 sample, or an inference sample: the own plan, weight 1, the own
    # task's teacher prompt; inference samples carry neither field.
    late = t({"prompt": own.task, "episode_index": np.int64(ep), "frame_index": np.int64(40)})
    assert late["glow_flow_weight"] == 1.0 and late["glow_ints"][_mi.INT_NUM_STAGES] == 1
    np.testing.assert_array_equal(late["glow_teacher_prompt"], teacher.tokenize(own.task)[0])
    l10 = _by_index(pack, 0)
    l10_ep = next(e for e, p in pack.episodes.items() if p.task_index == l10.task_index)
    out = t({"prompt": l10.task, "episode_index": np.int64(l10_ep), "frame_index": np.int64(0)})
    assert out["glow_flow_weight"] == 1.0 and out["glow_ints"][_mi.INT_NUM_STAGES] == l10.num_plans
    served = t({"prompt": own.task, "oracle_plan_index": 0})
    assert "glow_flow_weight" not in served and "glow_teacher_prompt" not in served
    # About half the decision frames at p 0.5; none at p 0, which adds no weight field at all.
    half = _counterfactual(pack, p=0.5, seed=1)
    sample = {"prompt": own.task, "episode_index": np.int64(ep), "frame_index": np.int64(5)}
    n = sum(half(dict(sample))["glow_flow_weight"] == 0.0 for _ in range(400))
    assert 150 < n < 250
    assert "glow_flow_weight" not in _counterfactual(pack, p=0.0)(dict(sample))


def test_counterfactual_validation(pack, tokenizer):
    from openpi.models import tokenizer as _tokenizer

    with pytest.raises(ValueError, match="teacher"):
        _mi.GlowModuleInputs(
            pack, tokenizer, counterfactual_p=0.5, counterfactual_window=40, counterfactual_tasks=GOAL_TASKS
        )
    with pytest.raises(ValueError, match="counterfactual_window"):
        _mi.GlowModuleInputs(
            pack, tokenizer, _tokenizer.PaligemmaTokenizer(48), counterfactual_p=0.5, counterfactual_tasks=GOAL_TASKS
        )
    with pytest.raises(ValueError, match="counterfactual_p"):
        _counterfactual(pack, p=1.5)
    one_stage_l10 = next(t.task_index for t in pack.tasks.values() if t.task_index < 10 and t.num_plans == 1)
    with pytest.raises(ValueError, match="LIBERO-Goal instruction"):
        _mi.GlowModuleInputs(
            pack,
            tokenizer,
            _tokenizer.PaligemmaTokenizer(48),
            counterfactual_p=0.5,
            counterfactual_window=40,
            counterfactual_tasks=(one_stage_l10, *GOAL_TASKS),
        )


def test_ordering_configs(monkeypatch):
    import openpi.training.config as _config

    monkeypatch.setattr(_config.DataConfigFactory, "_load_norm_stats", lambda self, d, a: None)
    gate = _config.get_config("pi05_libero_glow_v2t_gate")
    fixed = _config.get_config("pi05_libero_glow_v2t_fixed")
    for cfg in (gate, fixed):
        # The align recipe: only the encoders train, distillation 1.
        assert not cfg.freeze_vlm and cfg.model.freeze_pretrained and cfg.model.module_distill_weight == 1.0
        assert cfg.model.module_text_anchor and cfg.model.module_anchor == "head"
        assert cfg.model.module_token_scale == "words" and not cfg.model.module_ctx_active
        assert (cfg.data.decision_window, cfg.data.decision_repeat) == (40, 3)
    # The gate row: a clocked Temporal encoder anchored to the object noun, scaling the Spatial groups.
    assert gate.model.module_temporal_gate == 0.25 and gate.model.module_stage_clock
    assert gate.model.module_temporal_active_anchor == "object" and not gate.model.module_temporal_key_only
    # The fixed row: the Temporal encoder on the active runtime key alone, a constant turn-down.
    assert fixed.model == dataclasses.replace(
        gate.model,
        module_temporal_gate=0.0,
        module_temporal_key_only=True,
        module_stage_clock=False,
        module_temporal_active_anchor="ordinal",
        module_stage_gate=0.06,
    )
    assert fixed.data == dataclasses.replace(gate.data, module_stage_clock=False)
    assert fixed.lr_schedule == gate.lr_schedule
    for cfg in (gate, fixed):
        dc = dataclasses.replace(cfg.data, plan_pack_path=str(GOAL_PACK)).create(pathlib.Path("."), cfg.model)
        assert (dc.decision_window, dc.decision_repeat) == (40, 3) and len(dc.episode_stage_starts) == 807
        module = next(t for t in dc.model_transforms.inputs if isinstance(t, _mi.GlowModuleInputs))
        assert module.teacher_tokenizer is not None and module.counterfactual_p == 0.5
        assert module.counterfactual_window == 40 and module.counterfactual_tasks == GOAL_TASKS
        assert module.stage_order == "sorted" and module.episode_grounding
        assert module.stage_clock == cfg.model.module_stage_clock
    # Counterfactual samples need the teacher.
    with pytest.raises(ValueError, match="teacher"):
        dataclasses.replace(gate.data, plan_pack_path=str(GOAL_PACK)).create(
            pathlib.Path("."), dataclasses.replace(gate.model, module_distill_weight=0.0)
        )


def _two_stage_episode(pack):
    ep, plan = next((e, p) for e, p in sorted(pack.episodes.items()) if len(p.stage_starts) == 2)
    return ep, plan, next(t for t in pack.tasks.values() if t.task_index == plan.task_index)


def test_done_labels_follow_the_next_stage_start():
    ep = _plan_pack.EpisodePlan(task_index=0, n_frames=100, stage_starts=(0, 50))
    assert [_mi.done_labels(ep, f)[0].tolist() for f in (0, 49, 50, 99)] == [[0, 0], [0, 0], [1, 0], [1, 0]]
    assert _mi.done_labels(ep, 84)[1].tolist() == [True, True]
    assert _mi.done_labels(ep, 85)[1].tolist() == [True, False]


def test_monitor_training_samples_carry_done_labels(pack, tokenizer):
    t = _mi.GlowModuleInputs(pack, tokenizer, monitor=True)
    ep, plan, task = _two_stage_episode(pack)
    switch = plan.stage_starts[1]
    for frame, done in ((0, [0, 0]), (switch, [1, 0])):
        out = t({"prompt": task.task, "episode_index": np.int64(ep), "frame_index": np.int64(frame)})
        assert out["glow_done"].tolist() == [*map(float, done), 0.0, 0.0]
        assert out["glow_done_mask"].tolist() == [True, True, False, False]
        assert bool(out["glow_stage_given"])
        assert "glow_prev_done" not in out
    # Without the monitor nothing is added.
    plain = _mi.GlowModuleInputs(pack, tokenizer)({"prompt": task.task, "episode_index": np.int64(ep), "frame_index": np.int64(0)})
    assert not {"glow_done", "glow_done_mask", "glow_stage_given"} & set(plain)


def test_monitor_counterfactual_samples_are_not_done(pack):
    t = _counterfactual(pack, monitor=True)
    own = _by_index(pack, 17)
    ep = next(e for e, p in pack.episodes.items() if p.task_index == own.task_index)
    out = t({"prompt": own.task, "episode_index": np.int64(ep), "frame_index": np.int64(3)})
    assert out["glow_flow_weight"] == 0.0
    assert out["glow_done"].tolist() == [0.0] * _mi.K_MAX
    assert out["glow_done_mask"].tolist() == [True, True, False, False]


def test_monitor_inference_without_a_stage(pack, tokenizer):
    t = _mi.GlowModuleInputs(pack, tokenizer, monitor=True)
    _, _, task = _two_stage_episode(pack)
    free = t({"prompt": task.task})
    assert not bool(free["glow_stage_given"]) and free["glow_ints"][_mi.INT_STAGE] == 0
    assert free["glow_prev_done"].tolist() == [0.0] * _mi.K_MAX
    np.testing.assert_array_equal(free["glow_text"], t.encode(task.task, 0)["glow_text"])
    latched = t({"prompt": task.task, "plan_prev_done": [1.0, 0.0]})
    assert latched["glow_prev_done"].tolist() == [1.0, 0.0, 0.0, 0.0]
    oracle = t({"prompt": task.task, "oracle_plan_index": 1, "plan_prev_done": [1.0, 0.0]})
    assert bool(oracle["glow_stage_given"]) and oracle["glow_ints"][_mi.INT_STAGE] == 1
    chain = [_by_index(pack, 17).task, _by_index(pack, 16).task]
    chained = t({"prompt": chain[0], "plan_chain": chain, "plan_prev_done": [0.0, 0.0]})
    assert chained["glow_ints"][_mi.INT_NUM_STAGES] == 2 and not bool(chained["glow_stage_given"])
    with pytest.raises(ValueError, match="plan_prev_done"):
        t({"prompt": task.task, "plan_prev_done": [1.0]})
    # A model without the monitor needs a stage, and ignores a latched one.
    plain = _mi.GlowModuleInputs(pack, tokenizer)
    with pytest.raises(ValueError, match="oracle_plan_index"):
        plain({"prompt": task.task})
    out = plain({"prompt": task.task, "oracle_plan_index": 0, "plan_prev_done": [1.0, 0.0]})
    assert "glow_prev_done" not in out and "plan_prev_done" not in out


def test_monitor_excludes_chain_augmentation(pack, tokenizer):
    with pytest.raises(ValueError, match="chain_p must be 0"):
        _mi.GlowModuleInputs(pack, tokenizer, monitor=True, chain_p=0.5, chain_tasks=GOAL_TASKS)


def test_monitor_config_resolves():
    from openpi.training import config as _config

    cfg = _config.get_config("pi05_libero_glow_monitor")
    gate = _config.get_config("pi05_libero_glow_v2t_gate")
    assert cfg.model.module_temporal_monitor and not cfg.model.module_temporal
    assert cfg.model.module_stage_gate == 0.06 and cfg.model.module_freeze_spatial
    assert cfg.data.module_chain_p == 0.0
    # The Spatial encoder is the gate row's, so the gate row's weights load into it.
    for field in ("module_grounding", "module_subgoal_text", "module_nouns", "module_atom", "module_text_anchor",
                  "module_anchor", "module_ctx_active", "module_width", "module_depth", "module_token_scale"):
        assert getattr(cfg.model, field) == getattr(gate.model, field), field
    assert cfg.data.module_stage_order == gate.data.module_stage_order


def test_monitor_request_through_policy_transforms(monkeypatch, pack):
    """A monitor-mode request (no stage, latched stages) through the served monitor config's input
    transforms, and the model's done probabilities through LiberoOutputs."""
    import openpi.training.config as _config
    from openpi import transforms as _transforms
    from openpi.policies import libero_policy
    from openpi_client import msgpack_numpy

    monkeypatch.setattr(_config.DataConfigFactory, "_load_norm_stats", lambda self, d, a: None)
    cfg = _config.get_config("pi05_libero_glow_monitor")
    dc = dataclasses.replace(cfg.data, plan_pack_path=str(GOAL_PACK)).create(pathlib.Path("."), cfg.model)
    served = _transforms.compose(
        [
            *dc.data_transforms.inputs,
            _transforms.Normalize(None, use_quantiles=dc.use_quantile_norm),
            *dc.model_transforms.inputs,
        ]
    )
    a, b = _by_index(pack, 17), _by_index(pack, 16)
    element = {
        "observation/image": np.zeros((224, 224, 3), np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), np.uint8),
        "observation/state": np.zeros(8),
        "prompt": a.task,
        "plan_chain": [a.task, b.task],
        "plan_prev_done": [1.0, 0.0],
    }
    out = served(msgpack_numpy.unpackb(msgpack_numpy.packb(element)))
    assert not bool(out["glow_stage_given"])
    assert out["glow_prev_done"].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert out["glow_ints"][_mi.INT_NUM_STAGES] == 2
    done = np.array([0.9, 0.1, 0.0, 0.0], np.float32)
    back = libero_policy.LiberoOutputs()({"actions": np.zeros((10, 32)), "glow_done": done})
    np.testing.assert_array_equal(back["glow_done"], done)
