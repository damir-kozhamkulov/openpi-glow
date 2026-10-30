"""glow_chain_demo: pair rule, sharding, the episode state machine, the return to the start pose,
the recorder. Pure Python and numpy, so `pytest examples/libero/glow_chain_demo_test.py` runs
anywhere."""

import json
import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import glow_chain_demo as gcd  # noqa: E402


class Episode:
    """A scripted episode: the first atom holds on frames [first_at, first_until), the second from
    second_at; the gripper action is closed until release_at; the return takes home_steps."""

    def __init__(self, first_at=None, second_at=None, release_at=0, home_steps=10, first_until=None,
                 first_goal=True, second_goal=True):
        self.frame = 0
        self.first_at, self.second_at, self.release_at = first_at, second_at, release_at
        self.home_steps, self.first_until = home_steps, first_until
        self.first_goal_ok, self.second_goal_ok = first_goal, second_goal

    def first(self):
        if self.first_at is None or self.frame < self.first_at:
            return False
        return self.first_until is None or self.frame < self.first_until

    def second(self):
        return self.second_at is not None and self.frame >= self.second_at

    def gripper(self):
        """Gripper command of the action that produces the current frame."""
        return -1.0 if self.release_at is not None and self.frame >= self.release_at else 1.0

    def run(self, max_steps=300, release_cap=40):
        return gcd.ChainRun(
            self.first, self.second, lambda: self.first_goal_ok, lambda: self.second_goal_ok, max_steps, release_cap
        )


def play(ep, run, limit=3000):
    """Drive the run as the client does; returns the statuses seen."""
    seen = [run.start()]
    homing = 0
    while run.status == "running" and ep.frame < limit:
        phase = run.phase  # the phase that chose this action
        ep.frame += 1
        seen.append(run.step(ep.gripper() if phase == 0 else gcd.GRIPPER_OPEN))
        if phase == 1 and run.status == "running":
            homing += 1
            if homing == ep.home_steps:
                seen.append(run.homed())
    return seen


def test_kept_episode_switches_on_release_and_labels_the_return_as_stage_two():
    ep = Episode(first_at=40, release_at=40, second_at=120, home_steps=10)
    run = ep.run()
    seen = play(ep, run)
    assert seen.count("switch") == 1
    assert run.status == "kept"
    assert (run.first_met, run.switch_frame, run.released) == (40, 40, True)
    assert run.homed_frame == 50  # ten return actions produce frames 41..50
    assert run.n_frames == 120
    assert run.stage_starts() == [0, 40]


def test_switch_waits_for_the_gripper_to_open():
    ep = Episode(first_at=40, release_at=47, second_at=120)
    run = ep.run()
    play(ep, run)
    assert (run.first_met, run.switch_frame, run.released) == (40, 47, True)


def test_switch_after_the_release_cap_without_a_release():
    ep = Episode(first_at=40, release_at=None, second_at=150)
    run = ep.run(release_cap=10)
    play(ep, run)
    assert (run.switch_frame, run.released) == (50, False)
    assert run.status == "kept"


def test_first_atom_lost_during_the_return_rejects():
    ep = Episode(first_at=40, release_at=40, first_until=45, second_at=120)
    run = ep.run()
    play(ep, run)
    assert run.status == "first_lost"


def test_numpy_booleans_from_the_simulator_stay_python_values(tmp_path):
    # The cap path with the atom not holding at that frame: released must be a Python bool.
    ep = Episode(first_at=40, first_until=45, release_at=None, second_at=150)
    run = gcd.ChainRun(
        lambda: np.bool_(ep.first()), lambda: np.bool_(ep.second()), lambda: np.True_, lambda: np.True_, 300, 10
    )
    play(ep, run)
    assert run.released is False and type(run.released) is bool
    path = tmp_path / "attempts.jsonl"
    gcd.append_attempt(path, {"released": run.released, "flag": np.True_, "n": np.int64(3), "err": np.float32(1.5)})
    assert json.loads(path.read_text(encoding="utf-8")) == {"released": False, "flag": True, "n": 3, "err": 1.5}
    with pytest.raises(TypeError):
        gcd.append_attempt(path, {"bad": object()})


def test_goal_at_start_rejects():
    for ep in (Episode(first_at=0, second_at=50), Episode(first_at=30, second_at=0)):
        assert ep.run().start() == "goal_at_start"


def test_second_atom_before_first_rejects():
    ep = Episode(first_at=60, second_at=20)
    run = ep.run()
    play(ep, run)
    assert (run.status, run.n_frames) == ("second_first", 20)


def test_phase_budgets():
    ep = Episode()
    run = ep.run(max_steps=50)
    play(ep, run)
    assert (run.status, run.n_frames) == ("first_timeout", 50)

    # The second phase's budget counts from the end of the return.
    ep = Episode(first_at=30, release_at=30, home_steps=10)
    run = ep.run(max_steps=50)
    play(ep, run)
    assert run.status == "second_timeout"
    assert run.n_frames == run.homed_frame + 50

    ep = Episode(first_at=30, release_at=30, home_steps=10, second_at=40 + 49)
    run = ep.run(max_steps=50)
    play(ep, run)
    assert run.status == "kept"


def test_second_task_undoing_the_first_rejects():
    ep = Episode(first_at=30, release_at=30, second_at=90, first_goal=False)
    run = ep.run()
    play(ep, run)
    assert run.status == "first_undone"
    with pytest.raises(RuntimeError):
        run.stage_starts()


def test_misuse_raises():
    ep = Episode(first_at=5, release_at=5, second_at=30)
    run = ep.run()
    with pytest.raises(RuntimeError):
        run.homed()
    play(ep, run)
    with pytest.raises(RuntimeError):
        run.step(-1.0)


def _axisangle_quat(v):
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = v / angle
    return np.concatenate([axis * math.sin(angle / 2), [math.cos(angle / 2)]])


def test_rotation_error():
    q = np.array([0.0, 0.0, 0.0, 1.0])
    assert np.allclose(gcd.rotation_error(q, q), 0)
    target = _axisangle_quat(np.array([0.0, 0.0, 0.3]))
    assert np.allclose(gcd.rotation_error(q, target), [0, 0, 0.3])
    # World frame: applying the error on the left of the current orientation gives the target.
    cur = _axisangle_quat(np.array([0.4, -0.2, 0.1]))
    err = gcd.rotation_error(cur, target)
    assert np.allclose(np.abs(gcd._quat_mul(_axisangle_quat(err), cur)), np.abs(target), atol=1e-9)


HOME_POS, HOME_QUAT = np.array([-0.21, 0.0, 1.18]), np.array([1.0, 0.0, 0.0, 0.0])


def _move(pos, quat, action, blocked=lambda p: p):
    """Kinematic stand-in for the controller: 60% of each commanded delta, then any obstacle."""
    pos = blocked(pos + 0.6 * action[:3] * gcd.OSC_POS_SCALE)
    return pos, gcd._quat_mul(_axisangle_quat(0.6 * action[3:6] * gcd.OSC_ROT_SCALE), quat)


def _drive(plan, pos, quat, blocked=lambda p: p, steps=300):
    """Run a ReturnPlan until it reports the start pose reached; returns (reached, steps, poses, actions)."""
    poses, actions = [], []
    for step in range(steps):
        action, reached = plan.action(pos, quat)
        if reached:
            return True, step, poses, actions
        pos, quat = _move(pos, quat, action, blocked)
        poses.append((pos.copy(), quat.copy()))
        actions.append(action)
    return False, steps, poses, actions


def test_return_goes_up_sideways_under_the_start_then_into_it():
    start = np.array([0.0, -0.20, 1.23])  # placed on the cabinet top, off to the side
    quat = gcd._quat_mul(_axisangle_quat(np.array([0.0, 0.0, 0.5])), HOME_QUAT)
    plan = gcd.ReturnPlan(start, quat, HOME_POS, HOME_QUAT)
    targets = [t for t, _ in plan.targets]
    assert np.allclose(targets[0], start + [0, 0, 0.06])
    assert np.allclose(targets[1], [start[0], HOME_POS[1], start[2] + 0.06])
    assert np.allclose(targets[2], HOME_POS - [0, 0, 0.12])
    assert np.allclose(targets[3], HOME_POS)
    reached, steps, poses, actions = _drive(plan, start.copy(), quat)
    assert reached and steps < 150 and plan.skipped == []
    assert all(a[6] == gcd.GRIPPER_OPEN for a in actions)
    # The wrist keeps its orientation until the last stage, then turns to the start orientation.
    stage_of = []
    plan2 = gcd.ReturnPlan(start, quat, HOME_POS, HOME_QUAT)
    pos, q = start.copy(), quat.copy()
    for _ in range(steps):
        action, _ = plan2.action(pos, q)
        stage_of.append(plan2.stage)
        if plan2.stage < 3:
            assert np.linalg.norm(gcd.rotation_error(q, quat)) < 1e-6  # still the orientation it started with
        pos, q = _move(pos, q, action)
    assert stage_of == sorted(stage_of) and stage_of[-1] == 3
    # The lowest point of the return is the via point under the start, not the table.
    assert min(p[2] for p, _ in poses) > HOME_POS[2] - 0.12 - 0.02


def test_rise_to_start_height_option():
    low = np.array([0.10, 0.20, 0.95])  # placed on the table
    assert gcd.ReturnPlan(low, HOME_QUAT, HOME_POS, HOME_QUAT).targets[0][0][2] == pytest.approx(1.01)
    plan = gcd.ReturnPlan(low, HOME_QUAT, HOME_POS, HOME_QUAT, rise_to_start_height=True)
    assert plan.targets[0][0][2] == pytest.approx(HOME_POS[2])


def test_blocked_stage_is_given_up_and_the_return_goes_on():
    # Inside a drawer: nothing can rise above z 1.0 while x > 0.0.
    def drawer(p):
        if p[0] > 0.0:
            p[2] = min(p[2], 1.0)
        return p

    start = np.array([0.20, 0.0, 0.97])
    plan = gcd.ReturnPlan(start, HOME_QUAT, HOME_POS, HOME_QUAT, stage_steps=20)
    reached, steps, poses, _ = _drive(plan, start.copy(), HOME_QUAT, drawer)
    assert reached
    assert "rise" in plan.skipped
    assert all(p[0] <= 1e-9 or p[2] <= 1.0 + 1e-9 for p, _ in poses)


def test_already_at_the_start():
    plan = gcd.ReturnPlan(HOME_POS, HOME_QUAT, HOME_POS, HOME_QUAT)
    reached, steps, _, _ = _drive(plan, HOME_POS.copy(), HOME_QUAT)
    assert reached and steps < 60


def test_chain_pairs_follow_the_moved_object_rule():
    moved = {0: "bowl", 1: "wine", 2: "bowl", 3: None, 4: "plate"}
    pairs = gcd.chain_pairs(5, moved.get, lambda i, j: True)
    assert (0, 2) not in pairs  # both move the bowl
    assert (0, 3) in pairs and (2, 3) in pairs  # None conflicts with nothing
    assert all(i < j for i, j in pairs)
    assert len(pairs) == 9
    assert gcd.chain_pairs(5, moved.get, lambda i, j: False) == []


def test_shards_partition_the_pairs():
    pairs = list(range(38))
    shards = [gcd.shard(pairs, i, 4) for i in range(4)]
    assert sorted(p for s in shards for p in s) == pairs
    assert max(map(len, shards)) - min(map(len, shards)) <= 1
    with pytest.raises(ValueError):
        gcd.shard(pairs, 4, 4)


def test_recorder_round_trip(tmp_path):
    rec = gcd.EpisodeRecorder()
    for f in range(6):
        rec.add(np.full((256, 256, 3), f, np.uint8), np.zeros((256, 256, 3), np.uint8), np.arange(8) + f, np.ones(7))
    path = tmp_path / gcd.EPISODE_FILE.format(first=3, second=7, init=10)
    rec.save(path, [0, 4])
    data = np.load(path)
    assert data["image"].shape == (6, 256, 256, 3) and data["image"].dtype == np.uint8
    assert data["image"][5, 0, 0, 0] == 5
    assert data["state"].dtype == np.float32 and data["state"][2, 0] == 2
    assert data["actions"].shape == (6, 7)
    assert data["stage_starts"].tolist() == [0, 4]
    with pytest.raises(ValueError):
        rec.save(tmp_path / "bad.npz", [0, 6])


def test_attempt_lines(tmp_path):
    path = tmp_path / gcd.ATTEMPTS_FILE.format(index=0, count=4)
    gcd.append_attempt(path, {"status": "kept", "first": 1})
    gcd.append_attempt(path, {"status": "first_timeout", "first": 2})
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert [x["status"] for x in lines] == ["kept", "first_timeout"]
    assert path.name == "attempts-shard00-of-04.jsonl"


def test_chain_task_string():
    assert gcd.chain_task("turn on the stove", "put the bowl on the plate") == (
        "turn on the stove then put the bowl on the plate"
    )
