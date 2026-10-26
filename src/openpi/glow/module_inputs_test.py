"""The planning-module arrays the data pipeline produces for the GLOW module encoders.

Needs the LIBERO-10 + Goal plan pack (glow_repo/derived/plan_pack_libero10_goal.json) and its
label chains (plan_chain.csv, goal_plan_chain.csv) - skipped otherwise. The end-to-end loader
check additionally needs the local LeRobot copy (HF_LEROBOT_HOME with repo `libero_staging`).
"""

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
CONFIGS = ("pi05_libero_glow_st", "pi05_libero_glow_s_only", "pi05_libero_glow_nogrounding", "pi05_libero_glow_nosubgoal")

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


def test_layout():
    spans = sorted(_mi.TEXT_SEGMENTS.values())
    assert spans[0][0] == 0 and spans[-1][1] == _mi.TEXT_LEN
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:], strict=False))


def test_encode_every_stage(pack, raw, transform, tokenizer):
    """Every stage of every task, against the pack JSON read independently."""
    for task_string, entry in raw["tasks"].items():
        stages = sorted(entry["stages"], key=lambda s: s["plan_index"])
        for k, st in enumerate(stages):
            a = transform.encode(task_string, k)
            assert a["glow_text"].shape == a["glow_text_mask"].shape == (_mi.TEXT_LEN,)
            assert a["glow_ints"].shape == (_mi.INTS_LEN,) and a["glow_floats"].shape == (_mi.FLOATS_LEN,)
            assert _segment(a, "subgoal") == tokenizer.encode_text(st["subgoal_operation"])
            assert _segment(a, "object") == tokenizer.encode_text(st["object_noun"])
            assert _segment(a, "target") == tokenizer.encode_text(st["target_noun"])
            for j in range(_mi.K_MAX):
                expected = tokenizer.encode_text(stages[j]["runtime_key"]) if j < len(stages) else []
                assert _segment(a, f"rk{j}") == expected
            # Valid tokens fill each segment from its start.
            for start, stop in _mi.TEXT_SEGMENTS.values():
                m = a["glow_text_mask"][start:stop]
                assert m.tolist() == sorted(m.tolist(), reverse=True)
                assert not a["glow_text"][start:stop][~m].any()
            predicate = st["atom"].split("(")[0].lower()
            assert a["glow_ints"].tolist() == [_mi.PREDICATES.index(predicate) + 1, k, entry["K"]]
            for role, xyz, valid in (
                ("object", _mi.FLOAT_OBJECT, _mi.FLOAT_OBJECT_VALID),
                ("target", _mi.FLOAT_TARGET, _mi.FLOAT_TARGET_VALID),
            ):
                anchor = st[f"{role}_anchor"]
                if anchor is None:
                    assert a["glow_floats"][valid] == 0 and not a["glow_floats"][xyz].any()
                else:
                    assert a["glow_floats"][valid] == 1
                    np.testing.assert_allclose(a["glow_floats"][xyz], anchor["normalized"], rtol=1e-6)


def test_training_stage_at_boundaries(pack, chain, transform):
    ep = next(e for e, rows in sorted(chain.items()) if len(rows) == 2)
    rows = chain[ep]
    task = next(t.task for t in pack.tasks.values() if t.task_index == int(rows[0]["task_index"]))
    boundary = int(rows[1]["start_frame"])
    for frame, k in ((0, 0), (boundary - 1, 0), (boundary, 1)):
        out = transform({"prompt": task, "episode_index": np.int64(ep), "frame_index": np.int64(frame), "state": 1})
        assert out["prompt"] == "" and out["state"] == 1
        assert "episode_index" not in out and "frame_index" not in out
        expected = transform.encode(task, k)
        for key, value in expected.items():
            np.testing.assert_array_equal(out[key], value)


def test_inference_and_steering(pack, transform):
    two = next(t for t in pack.tasks.values() if t.num_plans == 2)
    one = next(t for t in pack.tasks.values() if t.num_plans == 1)
    out = transform({"prompt": two.task, "oracle_plan_index": np.int64(1), "prev_plan_index": np.int64(0)})
    assert out["prompt"] == "" and "prev_plan_index" not in out and "oracle_plan_index" not in out
    np.testing.assert_array_equal(out["glow_text"], transform.encode(two.task, 1)["glow_text"])
    # Steering: the steering task's module, the stage clamped to its stages.
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


def test_teacher_prompt(pack, chain, transform, tokenizer):
    from openpi.models import tokenizer as _tokenizer

    teacher_tokenizer = _tokenizer.PaligemmaTokenizer(48)
    with_teacher = _mi.GlowModuleInputs(pack, tokenizer, teacher_tokenizer)
    ep = next(e for e, rows in sorted(chain.items()) if len(rows) == 2)
    task = next(t.task for t in pack.tasks.values() if t.task_index == int(chain[ep][0]["task_index"]))
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


def test_module_configs(monkeypatch):
    import openpi.training.config as _config
    from openpi.glow.model_config import GlowModulePi0Config

    plan_prompt = _config.get_config("pi05_libero_plan_prompt")
    switches = {
        "pi05_libero_glow_st": {},
        "pi05_libero_glow_s_only": {"module_temporal": False},
        "pi05_libero_glow_nogrounding": {"module_grounding": False},
        "pi05_libero_glow_nosubgoal": {"module_subgoal_text": False},
    }
    monkeypatch.setattr(_config.DataConfigFactory, "_load_norm_stats", lambda self, d, a: None)
    for name in CONFIGS:
        cfg = _config.get_config(name)
        assert isinstance(cfg.model, GlowModulePi0Config) and cfg.model.max_token_len == 8
        expected = dataclasses.replace(GlowModulePi0Config(), **switches[name])
        for field in ("module_spatial", "module_temporal", "module_grounding", "module_subgoal_text"):
            assert getattr(cfg.model, field) == getattr(expected, field), (name, field)
        # The text control's data and recipe.
        for field in ("batch_size", "num_train_steps", "lr_schedule", "optimizer", "ema_decay", "weight_loader"):
            assert getattr(cfg, field) == getattr(plan_prompt, field), (name, field)
        assert cfg.data.module_inputs and not cfg.data.plan_as_prompt
        assert cfg.data.plan_pack_path == plan_prompt.data.plan_pack_path
        assert cfg.data.train_episodes == plan_prompt.data.train_episodes
    cfg = _config.get_config("pi05_libero_glow_st")
    local = dataclasses.replace(cfg.data, plan_pack_path=str(GOAL_PACK))
    kinds = [type(t).__name__ for t in local.create(pathlib.Path("."), cfg.model).model_transforms.inputs]
    assert kinds.index("GlowModuleInputs") + 1 == kinds.index("TokenizePrompt")
    assert "GlowPlanAsPrompt" not in kinds and "GlowPlanState" not in kinds
    with pytest.raises(ValueError, match="GlowModulePi0Config"):
        local.create(pathlib.Path("."), plan_prompt.model)
    with pytest.raises(ValueError, match="exclude"):
        dataclasses.replace(local, plan_as_prompt=True).create(pathlib.Path("."), cfg.model)


def test_align_configs(monkeypatch):
    import openpi.training.config as _config

    monkeypatch.setattr(_config.DataConfigFactory, "_load_norm_stats", lambda self, d, a: None)
    st = _config.get_config("pi05_libero_glow_st")
    align = _config.get_config("pi05_libero_glow_st_align")
    anchor = _config.get_config("pi05_libero_glow_st_anchor")
    assert align.model.module_text_anchor and align.model.freeze_pretrained and align.model.module_distill_weight > 0
    assert anchor.model.module_text_anchor and not anchor.model.freeze_pretrained
    assert anchor.model.module_distill_weight == 0 and not st.model.module_text_anchor
    # Both enter the prefix at a word token's size; the unanchored row keeps the language scale.
    assert align.model.module_token_scale == anchor.model.module_token_scale == "words"
    assert st.model.module_token_scale == "language"
    # Same data and recipe as the unanchored row.
    for cfg in (align, anchor):
        assert dataclasses.replace(cfg.data) == st.data
        for field in ("batch_size", "num_train_steps", "lr_schedule", "optimizer", "weight_loader"):
            assert getattr(cfg, field) == getattr(st, field), field

    def module_transform(cfg):
        local = dataclasses.replace(cfg.data, plan_pack_path=str(GOAL_PACK))
        inputs = local.create(pathlib.Path("."), cfg.model).model_transforms.inputs
        return next(t for t in inputs if isinstance(t, _mi.GlowModuleInputs))

    # Only a distilling model gets the teacher prompt.
    assert module_transform(align).teacher_tokenizer is not None
    assert module_transform(anchor).teacher_tokenizer is None and module_transform(st).teacher_tokenizer is None


@pytest.mark.parametrize("config_name", ["pi05_libero_glow_st", "pi05_libero_glow_st_align"])
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
    pack = _plan_pack.PlanPack.load(GOAL_PACK)
    goal_tasks = {t.task_index for t in pack.tasks.values() if t.task_index >= 10}
    loader = _data.create_data_loader(cfg, framework="pytorch", shuffle=True, num_batches=3, skip_norm_stats=True)
    for obs, _ in loader:
        # The prompt is BOS + "\n" only; the goal is in the module arrays.
        assert obs.tokenized_prompt.shape == (8, 8)
        assert (np.asarray(obs.tokenized_prompt_mask).sum(axis=1) == 2).all()
        ints = np.asarray(obs.glow_ints)
        assert (ints[:, _mi.INT_NUM_STAGES] == 1).all() and (ints[:, _mi.INT_STAGE] == 0).all()
        assert np.asarray(obs.glow_text_mask).any(axis=1).all()
        if cfg.model.module_distill_weight > 0:
            # The teacher's prompt: BOS, the task string, newline.
            assert obs.glow_teacher_prompt.shape == (8, cfg.model.module_teacher_prompt_len)
            assert (np.asarray(obs.glow_teacher_prompt_mask).sum(axis=1) > 2).all()
        else:
            assert obs.glow_teacher_prompt is None
    assert goal_tasks  # Goal episodes are single-stage tasks 10-19
