"""The plan state the data pipeline produces: previous plan in the prompt, current plan as target.

Needs the plan pack (OPENPI_PLAN_PACK, else glow_repo/derived/plan_pack.json relative to this
checkout) and its inputs (plan_chain.csv, tasks.jsonl, norm_stats.json) - skipped otherwise.
The end-to-end loader check additionally needs the local LeRobot copy (HF_LEROBOT_HOME with
repo `libero_staging`) and is skipped without it.
"""

from collections import defaultdict
import csv
import dataclasses
import json
import os
import pathlib

import numpy as np
import pytest

from openpi.glow import plan_pack as _plan_pack
from openpi.glow import plan_prompt as _plan_prompt
import openpi.shared.normalize as _normalize
import openpi.transforms as _transforms

REPO = pathlib.Path(__file__).resolve().parents[4]  # glow_repo (openpi is its submodule)
PACK = pathlib.Path(os.environ.get(_plan_pack.ENV_VAR) or REPO / "derived/plan_pack.json")
MAX_TOKEN_LEN = 200  # pi05

pytestmark = pytest.mark.skipif(
    not (PACK.exists() and (REPO / "derived/plan_chain.csv").exists()), reason="plan pack / label factory not present"
)


@pytest.fixture(scope="module")
def pack():
    return _plan_pack.PlanPack.load(PACK)


@pytest.fixture(scope="module")
def raw():
    return json.loads(PACK.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tasks():
    out = {}
    with open(REPO / "data/meta/tasks.jsonl", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            out[e["task_index"]] = e["task"]
    return out


@pytest.fixture(scope="module")
def chain():
    out = defaultdict(list)
    with open(REPO / "derived/plan_chain.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[int(r["episode_index"])].append(r)
    for rows in out.values():
        rows.sort(key=lambda r: int(r["plan_index"]))
    return out


@pytest.fixture(scope="module")
def tokenizer():
    from openpi.models import tokenizer as _tokenizer

    return _tokenizer.PaligemmaTokenizer(MAX_TOKEN_LEN)


@pytest.fixture(scope="module")
def first_ep(chain):
    out = {}
    for ep in sorted(chain):
        out.setdefault(int(chain[ep][0]["task_index"]), ep)
    return out


def _stage_at(chain, ep, frame):
    return max(i for i, r in enumerate(chain[ep]) if int(r["start_frame"]) <= frame)


def _expected_plan(raw, tasks, chain, ep, k, nouns=True, anchors=True):
    """The plan string rebuilt from plan_chain.csv and the pack's anchors, independently of render_plan."""
    rows = chain[ep]
    task = tasks[int(rows[0]["task_index"])]
    st = raw["tasks"][task]["stages"][k]
    r = rows[k]
    s = f"Stage {k + 1} of {len(rows)} [{r['runtime_key']}]: {r['subgoal_operation']}. {r['atom']}"
    for role, noun, anchor in (
        ("object", r["object_noun"], st["object_anchor"]),
        ("target", r["target_noun"], st["target_anchor"]),
    ):
        parts = [noun] if nouns and noun else []
        if anchors and anchor:
            parts.append(" ".join(map(str, anchor["bins"])))
        if parts:
            s += f"; {role} " + " ".join(parts)
    return s


def _train_sample(task, ep, frame):
    return {"prompt": task, "episode_index": np.int64(ep), "frame_index": np.int64(frame)}


def test_bins_match_openpi_normalize_and_digitize(raw):
    ns_path = REPO / raw["provenance"]["norm_stats"]["path"]
    if not ns_path.exists():
        pytest.skip("norm_stats not present")
    norm = _transforms.Normalize(_normalize.load(ns_path.parent), use_quantiles=True)
    n = 0
    for entry in raw["tasks"].values():
        for st in entry["stages"]:
            for name in ("object_anchor", "target_anchor"):
                a = st[name]
                if a is None:
                    continue
                state = np.zeros(8, dtype=np.float32)
                state[:3] = np.asarray(a["xyz"], dtype=np.float32)
                x = norm({"state": state})["state"][:3]
                bins = np.clip(np.digitize(x, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1, 0, 255).tolist()
                assert bins == a["bins"], (name, st["runtime_key"])
                n += 1
    assert n == 36


def test_render_plan(pack, raw, tasks, chain, first_ep):
    for ti, ep in first_ep.items():
        for k in range(len(chain[ep])):
            for nouns in (True, False):
                for anchors in (True, False):
                    got = pack.render_plan(tasks[ti], k, nouns=nouns, anchors=anchors)
                    assert got == _expected_plan(raw, tasks, chain, ep, k, nouns=nouns, anchors=anchors)
    # Task 6's two stages share instruction, atom and nouns; the runtime_key tells them apart.
    six = [pack.render_plan(tasks[6], k, nouns=False, anchors=False) for k in range(2)]
    assert six[0] != six[1] and "moka_pot_2" in six[0] and "moka_pot_1" in six[1]
    # Close stages carry no target noun or bins: no target fragment.
    assert "; target" not in pack.render_plan(tasks[2], 1)
    # Without grounding no plan carries an object or target fragment at all.
    assert not any(
        "; object" in pack.render_plan(t, k, nouns=False, anchors=False)
        or "; target" in pack.render_plan(t, k, nouns=False, anchors=False)
        for t, e in pack.tasks.items()
        for k in range(e.num_plans)
    )
    with pytest.raises(IndexError):
        pack.render_plan(tasks[0], 2)
    with pytest.raises(KeyError):
        pack.render_plan("put the bowl on the plate", 0)


def test_training_plan_state_on_boundary_frames(pack, raw, tasks, chain, first_ep):
    delta = 5
    t = _plan_prompt.GlowPlanState(pack, delta=delta, corrupt_p=0.0)
    checked = transitions = 0
    for ep in first_ep.values():
        rows = chain[ep]
        task = tasks[int(rows[0]["task_index"])]
        n = int(rows[-1]["end_frame"]) + 1
        probes = {0, 1, delta - 1, delta, n - 1, n // 2}
        for r in rows[1:]:
            s = int(r["start_frame"])
            probes |= {s - 1, s, s + delta - 1, s + delta}
        for fr in sorted(probes):
            out = t(_train_sample(task, ep, fr))
            k_prev, k_now = _stage_at(chain, ep, max(0, fr - delta)), _stage_at(chain, ep, fr)
            assert out["prompt"] == f"{task}. Was: " + _expected_plan(raw, tasks, chain, ep, k_prev)
            assert out["glow_plan_target"] == "Now: " + _expected_plan(raw, tasks, chain, ep, k_now)
            assert "episode_index" not in out and "frame_index" not in out and "glow_plan_candidates" not in out
            checked += 1
        # Over the whole episode, exactly the frames start .. start + delta - 1 of every later
        # stage see the previous stage as input.
        for fr in range(n):
            out = t(_train_sample(task, ep, fr))
            transitions += out["prompt"].split(". Was: ", 1)[1] != out["glow_plan_target"].removeprefix("Now: ")
    assert checked >= 60
    assert transitions == delta * sum(len(chain[ep]) - 1 for ep in first_ep.values())
    # Frame 0's input is stage 1, as the client's first query.
    ep0 = first_ep[0]
    assert t(_train_sample(tasks[0], ep0, 0))["prompt"].startswith(f"{tasks[0]}. Was: Stage 1 of 2 ")
    with pytest.raises(ValueError):
        t(_train_sample(tasks[1], ep0, 0))  # episode of task 0 under task 1's prompt


def test_corruption(pack, tasks, chain, first_ep):
    t = _plan_prompt.GlowPlanState(pack, delta=5, corrupt_p=0.25)
    n = 4000
    for ti in (0, 9):
        ep = first_ep[ti]
        frame = int(chain[ep][-1]["end_frame"])  # well inside the last stage
        task = tasks[ti]
        true_prev = "Was: " + pack.render_plan(task, pack.task(task).num_plans - 1)
        outs = [t(_train_sample(task, ep, frame)) for _ in range(n)]
        corrupted = sum(not o["prompt"].endswith(true_prev) for o in outs)
        if pack.task(task).num_plans == 1:
            assert corrupted == 0
        else:
            assert abs(corrupted / n - 0.25) < 0.03, corrupted / n
            # A corrupted input is always another stage of this task, never the target.
            others = {f"{task}. Was: " + pack.render_plan(task, k) for k in range(pack.task(task).num_plans - 1)}
            assert {o["prompt"] for o in outs if not o["prompt"].endswith(true_prev)} <= others
        # The target is never corrupted.
        assert {o["glow_plan_target"] for o in outs} == {
            "Now: " + pack.render_plan(task, pack.task(task).num_plans - 1)
        }
    with pytest.raises(ValueError):
        _plan_prompt.GlowPlanState(pack, delta=5, corrupt_p=1.5)
    with pytest.raises(ValueError):
        _plan_prompt.GlowPlanState(pack, delta=0, corrupt_p=0.0)


def test_inference_keys(pack, tasks):
    t = _plan_prompt.GlowPlanState(pack, delta=5, corrupt_p=0.25)
    out = t({"prompt": tasks[0], "prev_plan_index": np.int64(1)})
    assert out["prompt"] == f"{tasks[0]}. Was: " + pack.render_plan(tasks[0], 1)
    assert out["glow_plan_candidates"] == ["Now: " + pack.render_plan(tasks[0], k) for k in range(2)]
    assert "glow_plan_target" not in out and "prev_plan_index" not in out
    # Corruption is training-only: inference is deterministic.
    assert all(
        t({"prompt": tasks[0], "prev_plan_index": 0})["prompt"].endswith(pack.render_plan(tasks[0], 0))
        for _ in range(200)
    )
    out = t({"prompt": tasks[0], "prev_plan_index": 0, "oracle_plan_index": np.int64(1)})
    assert out["glow_plan_target"] == "Now: " + pack.render_plan(tasks[0], 1)
    assert "glow_plan_candidates" not in out and "oracle_plan_index" not in out
    assert len(t({"prompt": tasks[9], "prev_plan_index": 0})["glow_plan_candidates"]) == 1
    with pytest.raises(ValueError):
        t({"prompt": tasks[0]})
    with pytest.raises(KeyError):
        t({"prompt": "put the bowl on the plate", "prev_plan_index": 0})
    with pytest.raises(IndexError):
        t({"prompt": tasks[0], "prev_plan_index": 2})


def _check_row(tok, tokens, mask, ar, loss, prompt, block):
    n_p = int((mask & ~loss).sum())
    n_b = int(loss.sum())
    stock_tokens, stock_mask = tok.tokenize(prompt, None)
    assert int(stock_mask.sum()) == n_p and np.array_equal(tokens[:n_p], stock_tokens[:n_p])
    assert np.array_equal(tokens[n_p : n_p + n_b], tok.encode_answer(block))
    assert tokens[n_p + n_b - 1] == 1  # eos
    assert ar[n_p : n_p + n_b].all() and not ar[:n_p].any() and not ar[n_p + n_b :].any()
    assert mask[: n_p + n_b].all() and not mask[n_p + n_b :].any()


def test_block_layout(pack, tasks, chain, first_ep, tokenizer):
    tp = _transforms.TokenizePrompt(tokenizer)
    tt = _plan_prompt.GlowTokenizePlan(tokenizer)
    t = _plan_prompt.GlowPlanState(pack, delta=5, corrupt_p=0.0)
    ep0 = first_ep[0]

    plan = t(_train_sample(tasks[0], ep0, 0))
    sample = tt(tp(dict(plan)))
    assert sample["tokenized_prompt"].shape == (MAX_TOKEN_LEN,) and sample["token_loss_mask"].dtype == bool
    assert "glow_plan_target" not in sample
    _check_row(
        tokenizer,
        sample["tokenized_prompt"],
        sample["tokenized_prompt_mask"],
        sample["token_ar_mask"],
        sample["token_loss_mask"],
        plan["prompt"],
        plan["glow_plan_target"],
    )

    plan = t({"prompt": tasks[0], "prev_plan_index": 0})
    sample = tt(tp(dict(plan)))
    assert sample["tokenized_prompt"].shape == (2 * MAX_TOKEN_LEN,)
    for k in range(2):
        row = slice(k * MAX_TOKEN_LEN, (k + 1) * MAX_TOKEN_LEN)
        _check_row(
            tokenizer,
            sample["tokenized_prompt"][row],
            sample["tokenized_prompt_mask"][row],
            sample["token_ar_mask"][row],
            sample["token_loss_mask"][row],
            plan["prompt"],
            plan["glow_plan_candidates"][k],
        )
    assert sample["tokenized_prompt"].shape == sample["token_ar_mask"].shape == sample["token_loss_mask"].shape

    with pytest.raises(ValueError):
        tt(tp({"prompt": tasks[0]}))  # neither a target nor candidates
    with pytest.raises(ValueError):
        tt(tp({"prompt": "x " * 400, "glow_plan_target": "Now: y"}))  # prompt filled max_token_len


def test_inference_sample_is_a_valid_observation(pack, tasks, tokenizer):
    """Candidate rows survive Policy.infer's batching and Observation's shape checks."""
    import openpi.models.model as _model

    tp = _transforms.TokenizePrompt(tokenizer)
    tt = _plan_prompt.GlowTokenizePlan(tokenizer)
    t = _plan_prompt.GlowPlanState(pack, delta=5, corrupt_p=0.0)
    for extra, width in (({}, 2 * MAX_TOKEN_LEN), ({"oracle_plan_index": 1}, MAX_TOKEN_LEN)):
        sample = tt(tp(t({"prompt": tasks[0], "prev_plan_index": 0, **extra})))
        sample = {
            **{k: sample[k] for k in ("tokenized_prompt", "tokenized_prompt_mask", "token_ar_mask", "token_loss_mask")},
            "image": {
                k: np.zeros((224, 224, 3), np.float32) for k in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            },
            "image_mask": dict.fromkeys(("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"), np.True_),
            "state": np.zeros(32, np.float32),
        }
        batched = {
            k: ({kk: np.asarray(vv)[None] for kk, vv in v.items()} if isinstance(v, dict) else np.asarray(v)[None])
            for k, v in sample.items()
        }
        obs = _model.Observation.from_dict(batched)
        assert obs.tokenized_prompt.shape == (1, width)


def test_client_round_trip(pack, tasks, tokenizer):
    """What examples/libero's PlanEcho sends is what LiberoInputs and the GLOW transforms accept,
    and the choice it reads back becomes the next prompt's previous plan."""
    import sys

    import openpi.models.model as _model
    from openpi.policies import libero_policy

    sys.path.insert(0, str(REPO / "openpi/examples/libero"))
    try:
        import glow_stage_tracker as gst
    finally:
        sys.path.pop(0)

    inputs = libero_policy.LiberoInputs(model_type=_model.ModelType.PI05)
    t = _plan_prompt.GlowPlanState(pack, delta=5, corrupt_p=0.0)
    tp = _transforms.TokenizePrompt(tokenizer)
    tt = _plan_prompt.GlowTokenizePlan(tokenizer)

    def serve(request):
        element = {
            "observation/image": np.zeros((224, 224, 3), np.uint8),
            "observation/wrist_image": np.zeros((224, 224, 3), np.uint8),
            "observation/state": np.zeros(8, np.float32),
            "prompt": tasks[0],
            **request,
        }
        planned = t(inputs(element))
        return planned["prompt"], tt(tp(dict(planned)))["tokenized_prompt"].shape

    echo = gst.PlanEcho("self", pack.task(tasks[0]).num_plans)
    prompt, shape = serve(echo.request())
    assert shape == (2 * MAX_TOKEN_LEN,) and "Was: Stage 1 of 2" in prompt
    echo.update({"plan_choice": np.asarray(1)})  # as msgpack returns it
    prompt, _ = serve(echo.request())
    assert "Was: Stage 2 of 2" in prompt

    oracle = gst.PlanEcho("oracle", 2)
    _, shape = serve(oracle.request())
    assert shape == (MAX_TOKEN_LEN,)


def test_token_budget(pack, tokenizer):
    """The longest prompt + block over every (task, previous stage, current stage) fits max_token_len."""
    longest, where = 0, None
    for task, e in pack.tasks.items():
        for prev in range(e.num_plans):
            prompt = f"{task}. Was: " + pack.render_plan(task, prev)
            n_p = int(tokenizer.tokenize(prompt)[1].sum())
            for k in range(e.num_plans):
                n = n_p + len(tokenizer.encode_answer("Now: " + pack.render_plan(task, k)))
                if n > longest:
                    longest, where = n, (e.task_index, prev, k)
    # GlowTokenizePlan refuses rather than truncates, so this bound is what keeps training alive.
    assert longest < MAX_TOKEN_LEN, (longest, where)
    print(f"longest prompt + plan block: {longest} of {MAX_TOKEN_LEN} tokens (task, prev, now) = {where}")


def test_unified_configs():
    import openpi.training.config as _config

    cfg = _config.get_config("pi05_libero_glow_unified")
    ng = _config.get_config("pi05_libero_glow_unified_nogrounding")
    stock = _config.get_config("pi05_libero")
    assert cfg.data.plan_delta == 5 and cfg.data.plan_corrupt_p == 0.25 and cfg.model.subtask_ce_weight == 0.1
    assert cfg.data.plan_nouns and cfg.data.plan_anchors and not ng.data.plan_nouns and not ng.data.plan_anchors
    # The ablation differs from the main row in the grounding flags only.
    assert dataclasses.replace(ng.data, plan_nouns=True, plan_anchors=True) == cfg.data
    assert dataclasses.replace(ng, name=cfg.name, data=cfg.data) == cfg
    # Everything outside model and data is pi05_libero's recipe.
    assert dataclasses.replace(cfg, name=stock.name, model=stock.model, data=stock.data) == stock
    with pytest.raises(ValueError):
        dataclasses.replace(cfg.data, plan_corrupt_p=None).create(pathlib.Path("."), cfg.model)


def test_loader_end_to_end(pack):
    home = os.environ.get("HF_LEROBOT_HOME")
    if not home or not (pathlib.Path(home) / "libero_staging").exists():
        pytest.skip("local LeRobot copy not present")
    import openpi.training.config as _config
    import openpi.training.data_loader as _data

    os.environ["OPENPI_TRAIN_EPISODES"] = "0:379"
    os.environ[_plan_pack.ENV_VAR] = str(PACK)
    _config.DataConfigFactory._load_norm_stats = lambda self, d, a: None  # noqa: SLF001  (Windows path quirk)
    cfg = _config.get_config("pi05_libero_glow_unified")
    cfg = dataclasses.replace(
        cfg, data=dataclasses.replace(cfg.data, repo_id="libero_staging"), batch_size=4, num_workers=0
    )
    loader = _data.create_data_loader(cfg, framework="pytorch", shuffle=True, num_batches=2, skip_norm_stats=True)
    for obs, actions in loader:
        assert obs.tokenized_prompt.shape == (4, MAX_TOKEN_LEN) and obs.token_ar_mask.shape == (4, MAX_TOKEN_LEN)
        assert actions.shape == (4, 10, 32)
        # Every training sample carries a plan block.
        assert (np.asarray(obs.token_loss_mask).sum(axis=1) > 0).all()
