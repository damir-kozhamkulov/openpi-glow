"""PlanEcho: the plan state one evaluation episode sends and keeps, per mode.

Needs nothing but this directory: `pytest examples/libero/glow_stage_tracker_test.py`.
"""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import glow_stage_tracker as gst


def _grasp_and_release(echo, hold):
    for _ in range(hold):
        echo.observe(1.0)
    echo.observe(-1.0)


def test_update_needs_a_request():
    with pytest.raises(RuntimeError):
        gst.PlanEcho("oracle", 2).update({})


def test_oracle_mode_follows_the_tracker():
    echo = gst.PlanEcho("oracle", 2)
    assert echo.request() == {"oracle_plan_index": 0, "plan_stage_frames": 0}
    echo.update({})
    _grasp_and_release(echo, gst.HOLD_MIN)
    assert echo.request() == {"oracle_plan_index": 1, "plan_stage_frames": 0}
    assert echo.update({}) == 1
    assert echo.request() == {"oracle_plan_index": 1, "plan_stage_frames": 0}
    echo.update({})
    assert echo.agreements == echo.queries == 3
    assert echo.switches == [(gst.HOLD_MIN + 1, 0, 1)]


def test_stage_clock_counts_frames_since_the_sent_stage_changed():
    echo = gst.PlanEcho("oracle", 2)
    assert echo.request()["plan_stage_frames"] == 0
    echo.update({})
    for _ in range(7):
        echo.observe(-1.0)
    assert echo.request()["plan_stage_frames"] == 7
    echo.update({})
    _grasp_and_release(echo, gst.HOLD_MIN)  # the tracker moves to stage 1: the clock restarts
    req = echo.request()
    assert req["oracle_plan_index"] == 1 and req["plan_stage_frames"] == 0
    echo.update({})
    echo.observe(-1.0)
    assert echo.request()["plan_stage_frames"] == 1
    assert "plan_stage_frames" not in gst.PlanEcho("monitor", 2).request()  # no stage sent, no clock


def test_forced_mode():
    with pytest.raises(ValueError):
        gst.PlanEcho("forced", 2)
    with pytest.raises(ValueError):
        gst.PlanEcho("bogus", 2)
    echo = gst.PlanEcho("forced", 1, forced_index=1)  # clamped to the one-stage task's stage
    assert echo.request() == {"oracle_plan_index": 0, "plan_stage_frames": 0}
    echo = gst.PlanEcho("forced", 2, forced_index=1)
    assert echo.request() == {"oracle_plan_index": 1, "plan_stage_frames": 0}
    assert echo.update({}) == 1
    assert echo.request() == {"oracle_plan_index": 1, "plan_stage_frames": 0}
    assert echo.switches == [(0, 0, 1)] and echo.regressions == 0
    assert echo.agreements == 0  # the tracker is still at stage 1


def test_steering_counts_plans_and_tracks_the_environment():
    """A steering task's plans index num_plans; the tracker counts the environment task's stages."""
    echo = gst.PlanEcho("forced", 1, forced_index=0, tracker_plans=2)
    assert echo.num_plans == 1 and echo.tracker.num_plans == 2
    for t in range(gst.HOLD_MIN + 2):
        echo.observe(1.0 if t <= gst.HOLD_MIN else -1.0)  # one qualifying grasp-release cycle
    assert echo.tracker.plan_index == 1  # the environment advanced; the steering plan stays at 0
    assert echo.request() == {"oracle_plan_index": 0, "plan_stage_frames": 0}
    assert echo.update({}) == 0 and echo.agreements == 0
    assert gst.PlanEcho("oracle", 2).tracker.num_plans == 2  # default: the plans' own count


def test_plan_task_for_is_a_derangement():
    names = ["a", "b", "c"]
    assert [gst.plan_task_for(names, i, 0) for i in range(3)] == names  # offset 0: own plans
    shifted = [gst.plan_task_for(names, i, 1) for i in range(3)]
    assert shifted == ["b", "c", "a"] and all(s != n for s, n in zip(shifted, names))
    assert gst.plan_task_for(names, 2, 4) == "a"  # wraps
    with pytest.raises(ValueError):
        gst.plan_task_for(names, 0, -1)
    with pytest.raises(ValueError):
        gst.plan_task_for([], 0, 1)


def test_parse_atom():
    assert gst.parse_atom("on(moka_pot_1, flat_stove_1_cook_region)") == (
        "on",
        "moka_pot_1",
        "flat_stove_1_cook_region",
    )
    assert gst.parse_atom(" Close(microwave_1) ") == ("close", "microwave_1")
    for bad in ("on moka_pot_1", "on(a, )", "(a)", "on(a"):
        with pytest.raises(ValueError):
            gst.parse_atom(bad)


def test_parse_order_and_stage_order_for():
    assert gst.parse_order("1,0") == [1, 0]
    for bad in ("", "1,1", "-1,0", "a,b"):
        with pytest.raises(ValueError):
            gst.parse_order(bad)
    assert gst.stage_order_for([1, 0], 2) == [1, 0]
    assert gst.stage_order_for([1, 0], 1) == [0]  # a one-stage task keeps its stage
    assert gst.stage_order_for([2, 0, 1], 2) == [0, 1]
    with pytest.raises(ValueError):
        gst.stage_order_for([1], 2)  # stage 0 left out


def _atoms(holds):
    return gst.StageAtoms([("on", "a_1", "b_1"), ("on", "c_1", "b_1")], lambda atom: holds[atom[1]])


def test_stage_atoms_record_when_each_atom_first_held():
    holds = {"a_1": False, "c_1": False}
    atoms = _atoms(holds)
    atoms.observe(0)
    assert atoms.first_met == [None, None] and atoms.first() == "none"
    assert atoms.stage([1, 0]) == 1 and atoms.stage([0, 1]) == 0
    holds["c_1"] = True
    atoms.observe(7)
    assert atoms.first_met == [None, 7] and atoms.first() == "1"
    assert atoms.stage([1, 0]) == 0 and atoms.stage([0, 1]) == 0
    holds["c_1"] = False  # undone later: first_met keeps the first time
    holds["a_1"] = True
    atoms.observe(9)
    assert atoms.first_met == [9, 7] and atoms.first() == "1"
    assert atoms.stage([1, 0]) == 0  # all met: the order's last stage
    assert atoms.trace() == "atoms_first_met=[9, 7] first_stage=1"
    together = _atoms({"a_1": True, "c_1": True})
    together.observe(3)
    assert together.first() == "together"
    with pytest.raises(ValueError):
        gst.StageAtoms([], lambda atom: True)


def test_atoms_mode_sends_the_ordered_stage():
    holds = {"a_1": False, "c_1": False}
    atoms = _atoms(holds)
    with pytest.raises(ValueError):
        gst.PlanEcho("atoms", 2, atoms=atoms)  # no order
    with pytest.raises(ValueError):
        gst.PlanEcho("atoms", 1, atoms=atoms, order=[0])  # not the environment task's stages
    echo = gst.PlanEcho("atoms", 2, atoms=atoms, order=[1, 0])
    assert echo.request() == {"oracle_plan_index": 1, "plan_stage_frames": 0}
    assert echo.update({}) == 1
    holds["c_1"] = True
    atoms.observe(5)
    assert echo.request()["oracle_plan_index"] == 0
    assert echo.update({}) == 0 and echo.regressions == 1
    assert echo.trace().startswith("order=1,0 mode=atoms plan=0 ")


def test_moved_object():
    assert gst.moved_object("in(butter_1, basket_1_contain_region)") == "butter_1"
    assert gst.moved_object("on(akita_black_bowl_1, plate_1)") == "akita_black_bowl_1"
    assert gst.moved_object("open(wooden_cabinet_1_middle_region)") is None
    assert gst.moved_object("turnon(flat_stove_1)") is None


def test_chain_partner_for():
    names = ["bowl on plate", "wine on rack", "bowl in drawer", "turn on the stove", "bowl on stove"]
    moved = {0: "bowl_1", 1: "wine_1", 2: "bowl_1", 3: None, 4: "bowl_1"}
    same = lambda i: True  # noqa: E731
    # Offset 1 from task 0 skips task 2 (the bowl again) only when it is next: here 1 is the wine.
    assert gst.chain_partner_for(names, 0, 1, moved.get, same) == 1
    # From the drawer task, offset 1 is the stove (no moved object), fine.
    assert gst.chain_partner_for(names, 2, 1, moved.get, same) == 3
    # From the stove-bowl task, offset 1 wraps to 0 (bowl: conflict) then 1 (wine).
    assert gst.chain_partner_for(names, 4, 1, moved.get, same) == 1
    # A task moving nothing pairs with anything, never itself.
    assert gst.chain_partner_for(names, 3, 1, moved.get, same) == 4
    assert gst.chain_partner_for(names, 3, 5, moved.get, same) == 4  # offset n = itself, skipped
    # Scene filter.
    assert gst.chain_partner_for(names, 0, 1, moved.get, lambda i: i != 1) == 3
    with pytest.raises(ValueError, match="none in its scene"):
        gst.chain_partner_for(names, 0, 1, moved.get, lambda i: i in (0, 2, 4))
    with pytest.raises(ValueError, match="offset"):
        gst.chain_partner_for(names, 0, 0, moved.get, same)
    with pytest.raises(ValueError, match="at least two"):
        gst.chain_partner_for(["only"], 0, 1, moved.get, same)


def test_atoms_mode_over_a_chain():
    """A two-stage chain of one-stage tasks: the pointer follows the first unmet atom of the order."""
    held = {("on", "bowl_1", "plate_1"): False, ("open", "drawer_1"): False}
    atoms = gst.StageAtoms([("on", "bowl_1", "plate_1"), ("open", "drawer_1")], lambda a: held[a])
    echo = gst.PlanEcho("atoms", 2, atoms=atoms, order=[0, 1], tracker_plans=1)
    atoms.observe(0)
    assert echo.request()["oracle_plan_index"] == 0
    echo.update({})
    held[("on", "bowl_1", "plate_1")] = True
    echo.observe(-1.0)
    atoms.observe(1)
    assert echo.request()["oracle_plan_index"] == 1
    echo.update({})
    assert atoms.first() == "0" and echo.tracker.num_plans == 1
    assert "order=0,1 mode=atoms plan=1" in echo.trace()


def test_release_gate():
    """With release_cap a stage is done once its atom holds after an open-gripper action (or the
    cap runs out), as the chain demos switch; first_met still records the first hold."""
    held = {("on", "bowl_1", "plate_1"): False, ("open", "drawer_1"): False}
    atoms = gst.StageAtoms(
        [("on", "bowl_1", "plate_1"), ("open", "drawer_1")], lambda a: held[a], release_cap=40
    )
    atoms.observe(0)
    assert atoms.stage([0, 1]) == 0
    held[("on", "bowl_1", "plate_1")] = True
    atoms.observe(10, 1.0)  # placed, gripper still closed
    assert atoms.first_met == [10, None] and atoms.released == [None, None] and atoms.stage([0, 1]) == 0
    atoms.observe(11, 0.5)
    assert atoms.stage([0, 1]) == 0
    atoms.observe(12, -1.0)  # let go
    assert atoms.released == [12, None] and atoms.stage([0, 1]) == 1
    assert atoms.trace() == "atoms_first_met=[10, None] first_stage=0 atoms_released=[12, None]"
    # The cap: never opened, done release_cap frames after the first hold.
    capped = gst.StageAtoms([("on", "bowl_1", "plate_1")], lambda a: True, release_cap=3)
    capped.observe(1, 1.0)
    capped.observe(3, 1.0)
    assert capped.released == [None]
    capped.observe(4, 1.0)
    assert capped.released == [4]
    # Opening while the atom no longer holds does not count; an atom true before any action does.
    lost = {"x": True}
    flaky = gst.StageAtoms([("on", "x", "y")], lambda a: lost["x"], release_cap=40)
    flaky.observe(5, 1.0)
    lost["x"] = False
    flaky.observe(6, -1.0)
    assert flaky.released == [None] and flaky.first_met == [5]
    start = gst.StageAtoms([("on", "x", "y")], lambda a: True, release_cap=40)
    start.observe(0)
    assert start.released == [0]
    # Without release_cap the first hold is enough (the earlier rule).
    plain = gst.StageAtoms([("on", "x", "y")], lambda a: True)
    plain.observe(2, 1.0)
    assert plain.done(0) and plain.released == [None]
    with pytest.raises(ValueError):
        gst.StageAtoms([("on", "x", "y")], lambda a: True, release_cap=-1)


def test_monitor_latch_needs_consecutive_queries_and_never_unlatches():
    latch = gst.MonitorLatch(2, threshold=0.7, consecutive=2)
    assert latch.update([0.9, 0.0, 0.0, 0.0]) == 0  # once above: not yet
    assert latch.update([0.2, 0.0, 0.0, 0.0]) == 0  # the run is broken
    assert latch.update([0.8, 0.0, 0.0, 0.0]) == 0
    assert latch.update([0.95, 0.1, 0.0, 0.0]) == 1  # two in a row: stage 0 done
    assert latch.request() == [1.0, 0.0]
    assert latch.update([0.0, 0.0, 0.0, 0.0]) == 1  # stays done
    latch.update([0.0, 0.9, 0.0, 0.0])
    assert latch.update([0.0, 0.9, 0.0, 0.0]) == 1  # all done: the last stage
    assert latch.latched == [True, True] and latch.last == [0.0, 0.9]
    with pytest.raises(ValueError, match="2 stages"):
        latch.update([0.1])
    with pytest.raises(ValueError):
        gst.MonitorLatch(2, threshold=1.0)


def test_monitor_mode_sends_no_stage_and_latches_the_response():
    echo = gst.PlanEcho("monitor", 2, monitor={"threshold": 0.5, "consecutive": 1})
    req = echo.request()
    assert "oracle_plan_index" not in req and req["plan_prev_done"] == [0.0, 0.0]
    assert echo.update({"actions": None, "glow_done": [0.1, 0.0, 0.0, 0.0]}) == 0
    assert not echo.monitor_done(0)
    echo.request()
    assert echo.update({"actions": None, "glow_done": [0.9, 0.0, 0.0, 0.0]}) == 1
    assert echo.monitor_done(0) and echo.request()["plan_prev_done"] == [1.0, 0.0]
    assert echo.switches == [(0, 0, 1)]
    assert "monitor_latched=[1, 0]" in echo.trace()
    echo.request()
    with pytest.raises(ValueError, match="glow_done"):
        echo.update({"actions": None})
    # The other modes never latch.
    assert not gst.PlanEcho("oracle", 2).monitor_done(0)
