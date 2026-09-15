"""T0: the plan clause and subtask block the data pipeline produces.

Needs the plan pack (OPENPI_PLAN_PACK, else glow_repo/derived/plan_pack.json relative to this
checkout) and its inputs (plan_chain.csv, tasks.jsonl, norm_stats.json) - skipped otherwise.
The end-to-end loader check additionally needs the local LeRobot copy (HF_LEROBOT_HOME with
repo `libero_staging`) and is skipped without it.
"""

import csv
import dataclasses
import json
import os
import pathlib
from collections import defaultdict

import numpy as np
import pytest

import openpi.shared.normalize as _normalize
import openpi.transforms as _transforms
from openpi.glow import plan_pack as _plan_pack
from openpi.glow import plan_prompt as _plan_prompt

REPO = pathlib.Path(__file__).resolve().parents[4]  # glow_repo (openpi is its submodule)
PACK = pathlib.Path(os.environ.get(_plan_pack.ENV_VAR) or REPO / "derived/plan_pack.json")

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


def _expected(raw, tasks, chain, ep, frame, anchors=True):
    rows = chain[ep]
    k = max(i for i, r in enumerate(rows) if int(r["start_frame"]) <= frame)
    task = tasks[int(rows[0]["task_index"])]
    st = raw["tasks"][task]["stages"][k]
    s = f"{task}. Plan {k + 1} of {len(rows)}: {rows[k]['naming_instruction']}"
    if anchors:
        if st["object_anchor"]:
            s += "; object " + " ".join(map(str, st["object_anchor"]["bins"]))
        if st["target_anchor"]:
            s += "; target " + " ".join(map(str, st["target_anchor"]["bins"]))
    if k + 1 < len(rows):
        s += f". Next: {rows[k + 1]['naming_instruction']}"
    return s, k, task


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


def test_clause_strings_on_boundary_frames(pack, raw, tasks, chain):
    t = _plan_prompt.GlowPlanPrompt(pack, dropout_p=0.0)
    first_ep = {}
    for ep in sorted(chain):
        first_ep.setdefault(int(chain[ep][0]["task_index"]), ep)
    checked = 0
    for ep in first_ep.values():
        rows = chain[ep]
        n = int(rows[-1]["end_frame"]) + 1
        probes = {0, n - 1, n // 2}
        for r in rows[1:]:
            probes |= {int(r["start_frame"]) - 1, int(r["start_frame"])}
        for fr in sorted(probes):
            exp, k, task = _expected(raw, tasks, chain, ep, fr)
            out = t({"prompt": task, "episode_index": np.int64(ep), "frame_index": np.int64(fr)})
            assert out["prompt"] == exp
            assert f"Plan {k + 1} of " in out["prompt"]
            assert "glow_subtask" not in out and "episode_index" not in out and "frame_index" not in out
            checked += 1
    assert checked >= 40
    # K = 1 task: no Next; final stage of a K = 2 task: no Next; first stage: Next present.
    s9, _, _ = _expected(raw, tasks, chain, first_ep[9], 0)
    assert "Plan 1 of 1" in s9 and "Next:" not in s9
    ep0 = first_ep[0]
    s_last, k_last, _ = _expected(raw, tasks, chain, ep0, int(chain[ep0][-1]["end_frame"]))
    assert k_last == 1 and "Plan 2 of 2" in s_last and "Next:" not in s_last
    s_first, _, _ = _expected(raw, tasks, chain, ep0, 0)
    assert "Plan 1 of 2" in s_first and ". Next: " in s_first
    out = _plan_prompt.GlowPlanPrompt(pack, dropout_p=0.0, anchors=False)(
        {"prompt": tasks[0], "episode_index": np.int64(ep0), "frame_index": np.int64(0)}
    )
    assert out["prompt"] == _expected(raw, tasks, chain, ep0, 0, anchors=False)[0]


def test_dropout(pack, tasks, chain):
    eps = sorted(chain)
    t1 = _plan_prompt.GlowPlanPrompt(pack, dropout_p=1.0)
    for ep in eps[:400]:
        task = tasks[int(chain[ep][0]["task_index"])]
        out = t1({"prompt": task, "episode_index": np.int64(ep), "frame_index": np.int64(0)})
        assert out["prompt"] == task
        assert out["glow_subtask"] == pack.subtask(task, 0)
    t3 = _plan_prompt.GlowPlanPrompt(pack, dropout_p=0.3)
    n = 2000
    # A dropped sample is exactly one that carries the subtask target instead of the clause.
    dropped = sum(
        "glow_subtask" in t3({"prompt": tasks[0], "episode_index": np.int64(eps[0]), "frame_index": np.int64(0)})
        for _ in range(n)
    )
    assert abs(dropped / n - 0.3) < 0.05


def test_inference_keys(pack, raw, tasks, chain):
    t = _plan_prompt.GlowPlanPrompt(pack, dropout_p=0.0)
    ep0 = min(ep for ep in chain if int(chain[ep][0]["task_index"]) == 0)
    assert t({"prompt": tasks[0], "plan_index": 1})["prompt"] == _expected(raw, tasks, chain, ep0, int(chain[ep0][-1]["end_frame"]))[0]
    out = t({"prompt": tasks[0], "plan_index": _plan_pack.PLAN_OFF})
    assert out["prompt"] == tasks[0] and "glow_subtask" not in out
    with pytest.raises(ValueError):
        t({"prompt": tasks[0]})
    with pytest.raises(KeyError):
        t({"prompt": "put the bowl on the plate", "plan_index": 0})
    with pytest.raises(IndexError):
        t({"prompt": tasks[0], "plan_index": 2})


def test_subtask_block_layout(pack, tasks, chain):
    from openpi.models import tokenizer as _tokenizer

    tok = _tokenizer.PaligemmaTokenizer(200)
    tp = _transforms.TokenizePrompt(tok)
    tt = _plan_prompt.GlowTokenizeSubtask(tok)
    ep0 = min(ep for ep in chain if int(chain[ep][0]["task_index"]) == 0)
    longest = max(
        int(tok.tokenize(pack.render_clause(task, k))[1].sum()) for task, e in pack.tasks.items() for k in range(e.num_plans)
    )
    assert longest < 120
    sample = tt(tp(_plan_prompt.GlowPlanPrompt(pack, dropout_p=0.0)(
        {"prompt": tasks[0], "episode_index": np.int64(ep0), "frame_index": np.int64(0)})))
    assert sample["token_loss_mask"].sum() == 0 and sample["token_ar_mask"].sum() == 0
    sample = tt(tp(_plan_prompt.GlowPlanPrompt(pack, dropout_p=1.0)(
        {"prompt": tasks[0], "episode_index": np.int64(ep0), "frame_index": np.int64(0)})))
    n_p = int((sample["tokenized_prompt_mask"] & ~sample["token_loss_mask"]).sum())
    n_b = int(sample["token_loss_mask"].sum())
    assert n_b > 0
    assert sample["tokenized_prompt"][n_p + n_b - 1] == 1  # eos
    assert sample["token_ar_mask"][n_p : n_p + n_b].all()
    assert sample["tokenized_prompt_mask"][: n_p + n_b].all() and not sample["tokenized_prompt_mask"][n_p + n_b :].any()
    stock_tokens, stock_mask = tok.tokenize(tasks[0], None)
    assert int(stock_mask.sum()) == n_p and np.array_equal(sample["tokenized_prompt"][:n_p], stock_tokens[:n_p])


def test_loader_end_to_end(pack):
    home = os.environ.get("HF_LEROBOT_HOME")
    if not home or not (pathlib.Path(home) / "libero_staging").exists():
        pytest.skip("local LeRobot copy not present")
    import openpi.training.config as _config
    import openpi.training.data_loader as _data

    os.environ["OPENPI_TRAIN_EPISODES"] = "0:379"
    os.environ[_plan_pack.ENV_VAR] = str(PACK)
    _config.DataConfigFactory._load_norm_stats = lambda self, d, a: None  # noqa: SLF001  (Windows path quirk)
    cfg = _config.get_config("pi05_libero_glow")
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, repo_id="libero_staging"), batch_size=4, num_workers=0)
    loader = _data.create_data_loader(cfg, framework="pytorch", shuffle=True, num_batches=2, skip_norm_stats=True)
    for obs, actions in loader:
        assert obs.tokenized_prompt.shape == (4, 200) and obs.token_ar_mask.shape == (4, 200)
        assert obs.token_loss_mask.shape == (4, 200) and actions.shape == (4, 10, 32)
