"""glow_monitor_simtune's sim labels, episode listing, metrics, bars and multi-source training."""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "examples" / "libero"))
import glow_monitor_simtune as st  # noqa: E402
import glow_query_dump as gqd  # noqa: E402


def test_sim_labels():
    frames = np.array([0, 5, 10, 15, 20, 25])
    done, labeled = st.sim_labels(frames, [10, None], 2, 10, 4)
    assert done[:, 0].tolist() == [0, 0, 0, 0, 1, 1]
    assert labeled[:, 0].tolist() == [True, True, False, False, True, True]
    assert done[:, 1].tolist() == [0] * 6 and labeled[:, 1].all()
    assert not labeled[:, 2:].any()


def test_sim_family_and_kept_queries():
    assert st.sim_family("10") == "libero10"
    assert st.sim_family("return-01") == "chain" and st.sim_family("chain-01") == "chain"
    assert st.sim_family("goal") == "goal"
    assert st.kept_queries(5, True, 2).tolist() == [0, 1, 2, 3, 4]
    assert st.kept_queries(5, False, 2).tolist() == [0, 2, 4]


def _episode(dump, frames, trace):
    for f in frames:
        dump.add(
            f,
            {
                gqd.IMAGE_KEY: np.zeros((2, 2, 3), np.uint8),
                gqd.WRIST_KEY: np.zeros((2, 2, 3), np.uint8),
                gqd.STATE_KEY: np.zeros(8),
                "prompt": "p",
            },
            {"glow_done": np.zeros(2)},
        )
    return trace


def test_list_sim_episodes(tmp_path):
    for run, ep, met in (("return-01", 0, "[12, None]"), ("return-01", 8, "[3, 40]"), ("goal-steer", 0, "[1]")):
        dump = gqd.QueryDump(tmp_path / run)
        trace = f"plan trace: task=0 episode={ep} success=True frames=40 atoms_first_met={met} first_stage=0"
        _episode(dump, [0, 5, 10, 15], trace)
        dump.save(0, ep, "p", trace)
    eps = st.list_sim_episodes(tmp_path, {"goal-steer"}, {8, 9}, 2)
    assert [(e.run, e.episode, e.heldout) for e in eps] == [("return-01", 0, False), ("return-01", 8, True)]
    assert eps[0].met == [12, None] and eps[1].met == [3, 40]
    assert eps[0].queries.tolist() == [0, 2] and eps[0].frames.tolist() == [0, 10]
    assert eps[1].queries.tolist() == [0, 1, 2, 3] and eps[0].family == "chain"


def test_cache_files(tmp_path):
    src = tmp_path / "src"
    (src / "a").mkdir(parents=True)
    (src / "a/x.npz").write_bytes(b"1")
    out = st.cache_files([src / "a/x.npz"], src, tmp_path / "dst", parallel=2)
    assert out == [tmp_path / "dst/a/x.npz"] and out[0].read_bytes() == b"1"


def test_plan_id_ignores_frame_fields():
    a = st.plan_id({"prompt": "p", "plan_chain": ["a", "b"], "plan_prev_done": [0.0, 1.0]})
    b = st.plan_id({"prompt": "p", "plan_chain": ["a", "b"], "plan_prev_done": [1.0, 1.0]})
    assert a == b and a != st.plan_id({"prompt": "p", "plan_chain": ["b", "a"]})


def test_boundary_events():
    frames = np.array([0, 5, 10, 15, 20])
    probs = np.zeros((5, 3))
    probs[1:, 0] = 0.9  # latches at 10
    ev = st.boundary_events(frames, probs, [12, None], 2, 0.7, 2)
    assert ev == [{"stage": 0, "met": 12, "latch": 10, "premature": True, "missed": False, "lag": None}]
    probs[:, 0] = 0.0
    probs[3:, 0] = 0.9  # latches at 20
    ev = st.boundary_events(frames, probs, [12, None], 2, 0.7, 2)
    assert ev[0]["lag"] == 8 and not ev[0]["premature"]
    ev = st.boundary_events(frames, np.zeros((5, 3)), [12, None], 2, 0.7, 2)
    assert ev[0]["missed"]


def test_sim_metrics_and_passes():
    class A:
        threshold, consecutive, start_window = 0.7, 2, 40

    frames = np.array([0, 5, 10, 15, 20, 25])
    good = st.SimEpisode("return-01", pathlib.Path("g"), 0, 8, "chain", True, [10, None], np.arange(6), frames)
    bad = st.SimEpisode("return-01", pathlib.Path("b"), 1, 9, "chain", True, [None, None], np.arange(6), frames)
    pg = np.zeros((6, 3))
    pg[4:, 0] = 0.95  # latches at 25: lag 15
    pb = np.zeros((6, 3))
    pb[:2, 0] = 0.95  # latches at 5 with the atom never holding: premature and early
    labels = {e.path: st.sim_labels(e.frames, e.met, 2, 10, 3) for e in (good, bad)}
    m = st.sim_metrics([good, bad], {good.path: pg, bad.path: pb}, labels, A)
    chain = m["chain"]
    assert (chain["boundaries"], chain["premature"], chain["missed"], chain["early_false_done"]) == (2, 1, 0, 1)
    assert chain["lag_median"] == 15.0
    assert st.sim_passes(m, m) == {"premature": False, "premature_vs_m1": False, "missed": True, "lag": True}
    better = {"chain": {**chain, "premature_rate": 0.0}}
    assert st.sim_passes(better, m)["premature_vs_m1"]


def test_train_from_two_sources_and_init():
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
    from openpi.glow import module_encoders as _enc

    class A:
        seed, width, depth, heads, weight_decay, warmup, batch_size, swap_p = 0, 32, 1, 2, 0.0, 2, 8, 0.3

    k_max, d_text, d_img = 4, 16, 24
    device = torch.device("cpu")

    def source(n, seed):
        g = torch.Generator().manual_seed(seed)
        s = st._m1.Store()
        s.images = torch.randn(n, 2, 4, d_img, generator=g).to(torch.bfloat16)
        s.image_mask = torch.ones(2, dtype=torch.bool)
        s.keys = torch.randn(2, k_max, d_text, generator=g)
        s.key_heads = torch.randn(2, k_max, 2, d_text, generator=g)
        s.valid = torch.tensor([[True, True, False, False], [True, False, False, False]])
        done = (torch.rand(n, k_max, generator=g) > 0.5).float()
        lab = torch.zeros(n, k_max, dtype=torch.bool)
        lab[:, 0] = True
        return st.Source(s, np.arange(n) % 2, done, lab, np.arange(n))

    a, b = source(20, 1), source(12, 2)
    init = _enc.ProgressMonitor(d_text, d_img, 32, 1, keys=True, heads=2).state_dict()
    same = st.train("t", [a, b], [0.5, 0.5], init, 0, 1e-3, A, device)
    assert all(torch.equal(same.state_dict()[k], v) for k, v in init.items())
    trained = st.train("t", [a, b], [0.5, 0.5], init, 5, 1e-3, A, device)
    assert any(not torch.equal(trained.state_dict()[k], v) for k, v in init.items())
    assert not trained.training
