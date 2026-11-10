"""glow_monitor_replay's pure parts: traces, latches, episode classes, requests, weight parity, demos."""

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "examples" / "libero"))
import glow_monitor_replay as rp  # noqa: E402
import glow_query_dump as gqd  # noqa: E402

TRACE = (
    "INFO:root:plan trace: task=2 episode=5 success=False chain=[0, 1] partner_success=True "
    "chain_success=False frames=600 order=0,1 monitor_latched=[1, 1] mode=monitor plan=1 tracker=0 of 2 "
    "switches=[(5, 0, 1)] tracker_releases=[219] atoms_first_met=[None, 224] first_stage=1"
)


def test_parse_trace():
    t = rp.parse_trace(TRACE)
    assert t["task"] == "2" and t["chain"] == [0, 1] and t["switches"] == [(5, 0, 1)]
    assert t["atoms_first_met"] == [None, 224] and t["frames"] == "600"
    assert rp.parse_trace("no trace here") == {}


def test_latched_frame_needs_consecutive_queries():
    frames = [0, 5, 10, 15, 20]
    assert rp.latched_frame(frames, [0.8, 0.1, 0.8, 0.9, 0.2], 0.7, 2) == 15
    assert rp.latched_frame(frames, [0.8, 0.1, 0.8, 0.1, 0.8], 0.7, 2) is None
    assert rp.latched_frame(frames, [0.0, 0.71, 0.0, 0.0, 0.0], 0.7, 1) == 5


def test_first_met_and_premature():
    assert rp.first_met(rp.parse_trace(TRACE)) is None
    assert rp.first_met({"atoms_first_met": [125, 384], "frames": "384"}) == 125
    assert rp.first_met({"atoms_first_met": [384, 384], "frames": "384"}) is None
    assert rp.premature(5, None) and rp.premature(100, 125)
    assert not rp.premature(130, 125) and not rp.premature(None, None)


def test_element_from_restores_arrays_and_swaps_images(tmp_path):
    dump = gqd.QueryDump(tmp_path)
    element = {
        gqd.IMAGE_KEY: np.zeros((4, 4, 3), np.uint8),
        gqd.WRIST_KEY: np.ones((4, 4, 3), np.uint8),
        gqd.STATE_KEY: np.arange(8.0),
        "prompt": "p",
        "plan_prev_done": np.array([0.0, 1.0]),
        "plan_chain": ["a", "b"],
    }
    dump.add(0, element, {"glow_done": np.array([0.2, 0.0])})
    ep = gqd.load(dump.save(0, 0, "p", TRACE))
    out = rp.element_from(ep, 0)
    assert out["plan_prev_done"].dtype == np.float32 and out["plan_prev_done"].tolist() == [0.0, 1.0]
    assert out["plan_chain"] == ["a", "b"] and out["prompt"] == "p"
    assert out[gqd.WRIST_KEY][0, 0, 0] == 1
    swapped = rp.element_from(ep, 0, image=np.full((4, 4, 3), 7, np.uint8), wrist=np.full((4, 4, 3), 9, np.uint8))
    assert swapped[gqd.IMAGE_KEY][0, 0, 0] == 7 and swapped[gqd.WRIST_KEY][0, 0, 0] == 9


def test_run_episodes_groups_by_directory_in_order(tmp_path):
    for run, name in (
        ("a", "task1_ep0.npz"),
        ("a", "task0_ep10.npz"),
        ("a", "task0_ep2.npz"),
        ("b/c", "task3_ep1.npz"),
    ):
        (tmp_path / run).mkdir(parents=True, exist_ok=True)
        (tmp_path / run / name).write_bytes(b"")
    runs = rp.run_episodes(tmp_path)
    assert list(runs) == ["a", str(pathlib.Path("b/c"))]
    assert [p.name for p in runs["a"]] == ["task0_ep2.npz", "task0_ep10.npz", "task1_ep0.npz"]


def test_parity(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    base = {
        "x.vision_tower.w": torch.ones(2, 2),
        "x.multi_modal_projector.b": torch.zeros(3),
        "x.lm_head.weight": torch.ones(4, 2),
        "y.action_expert.w": torch.ones(1),
    }
    other = dict(base)
    other["x.multi_modal_projector.b"] = torch.tensor([0.0, 0.5, 0.0])
    other["y.action_expert.w"] = torch.zeros(1)
    del other["x.lm_head.weight"]
    save_file(base, str(tmp_path / "a.safetensors"))
    save_file(other, str(tmp_path / "b.safetensors"))
    out = rp.parity(str(tmp_path / "a.safetensors"), str(tmp_path / "b.safetensors"))
    assert out["compared"] == 2 and out["identical"] == 1
    assert out["max_abs_diff"] == pytest.approx(0.5)
    assert out["only_policy"] == ["x.lm_head.weight"] and out["only_weights"] == []
    assert out["differ"] == [["x.multi_modal_projector.b", 0.5]]


def test_demos_index_one_task_episodes_below_the_chains(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text(
        json.dumps({"data_path": "d/{episode_chunk}/{episode_index}", "chunks_size": 1000})
    )
    lines = [
        {"episode_index": 0, "tasks": ["t0"]},
        {"episode_index": 1, "tasks": ["t0"]},
        {"episode_index": 2, "tasks": ["t1"]},
        {"episode_index": 3, "tasks": ["t0"]},
        {"episode_index": rp.GOAL_END_EPISODE, "tasks": ["t1"]},
        {"episode_index": rp.GOAL_END_EPISODE + 1, "tasks": ["t1", "t0"]},
    ]
    (tmp_path / "meta/episodes.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    for x in lines:
        if x["episode_index"] != 1:  # episode 1's file is missing: not indexed
            path = tmp_path / f"d/0/{x['episode_index']}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
    demos = rp.Demos(tmp_path, per_task=2)
    assert demos.by_task["t0"] == [0, 3] and demos.by_task["t1"] == [2]
    assert demos.episode("t0", 3) == 3 and demos.episode("t1", 5) == 2 and demos.episode("t9", 0) is None
    cached = rp.Demos(tmp_path, per_task=2, cache=tmp_path / "cache", parallel=2)
    assert sorted(p.name for p in (tmp_path / "cache/d/0").iterdir()) == ["0", "2", "3"]
    assert cached.path(3) == tmp_path / "cache/d/0/3"


def test_summarize_counts():
    rows = [
        {
            "met": 10,
            "early_served": True,
            "premature_served": True,
            "premature_m1": False,
            "fidelity": 0.0,
            "token_gap": 0.1,
        },
        {"met": None, "early_demo": True, "premature_recorded": True, "fidelity": 0.002, "token_gap": 0.3},
    ]
    s = rp.summarize(rows, 40)
    assert s["episodes"] == 2 and s["first_met"] == 1
    assert (s["early_false_served"], s["early_false_demo"], s["premature_served"], s["premature_m1"]) == (1, 1, 1, 0)
    assert s["premature_recorded"] == 1
    assert s["fidelity"]["max"] == pytest.approx(0.002) and s["token_gap"]["median"] == pytest.approx(0.2)
