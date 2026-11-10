"""QueryDump: one episode's queries written and read back.

Needs nothing but this directory: `pytest examples/libero/glow_query_dump_test.py`.
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import glow_query_dump as gqd


def _element(i):
    return {
        gqd.IMAGE_KEY: np.full((4, 4, 3), i, dtype=np.uint8),
        gqd.WRIST_KEY: np.full((4, 4, 3), 10 + i, dtype=np.uint8),
        gqd.STATE_KEY: np.arange(8, dtype=np.float64) + i,
        "prompt": "put the bowl on the stove",
        "plan_prev_done": np.array([0.0, float(i)]),
        "plan_chain": ["a", "b"],
    }


def test_round_trip(tmp_path):
    dump = gqd.QueryDump(tmp_path / "run")
    dump.add(0, _element(0), {"actions": np.zeros((10, 7)), "glow_done": np.array([0.1, 0.2, 0.0])})
    dump.add(5, _element(1), {"actions": np.zeros((10, 7))})
    path = dump.save(3, 7, "put the bowl on the stove", "plan trace: task=3 episode=7 success=True")
    assert path.name == "task3_ep7.npz"
    assert len(dump) == 0
    ep = gqd.load(path)
    assert ep["image"].shape == (2, 4, 4, 3) and ep["image"].dtype == np.uint8
    assert ep["wrist_image"][1, 0, 0, 0] == 11
    assert ep["state"].dtype == np.float32 and ep["state"][1, 0] == 1.0
    assert ep["frame"].tolist() == [0, 5]
    np.testing.assert_allclose(ep["done"][0], [0.1, 0.2, 0.0], rtol=1e-6)
    assert np.isnan(ep["done"][1]).all()
    assert ep["request"][1] == {
        "prompt": "put the bowl on the stove",
        "plan_prev_done": [0.0, 1.0],
        "plan_chain": ["a", "b"],
    }
    assert (ep["task"], ep["episode"]) == (3, 7)
    assert ep["prompt"] == "put the bowl on the stove"
    assert ep["trace"].startswith("plan trace: task=3")


def test_empty_episode_writes_nothing(tmp_path):
    dump = gqd.QueryDump(tmp_path)
    assert dump.save(0, 0, "x") is None
    assert list(tmp_path.iterdir()) == []


def test_save_starts_the_next_episode_empty(tmp_path):
    dump = gqd.QueryDump(tmp_path)
    dump.add(0, _element(0), {"glow_done": np.array([0.5])})
    dump.save(0, 0, "x")
    dump.add(0, _element(2), {"glow_done": np.array([0.9])})
    ep = gqd.load(dump.save(0, 1, "x"))
    assert ep["image"].shape[0] == 1 and ep["image"][0, 0, 0, 0] == 2
