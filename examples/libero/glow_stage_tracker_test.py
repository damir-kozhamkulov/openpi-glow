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


def test_self_mode_echoes_the_choice():
    echo = gst.PlanEcho("self", 2)
    assert echo.request() == {"prev_plan_index": 0}
    assert echo.update({"plan_choice": 0}) == 0
    for _ in range(5):
        echo.observe(-1.0)
    echo.request()
    assert echo.update({"plan_choice": 1}) == 1
    assert echo.request() == {"prev_plan_index": 1}
    echo.update({"plan_choice": 0})  # a regression
    assert echo.switches == [(5, 0, 1), (5, 1, 0)]
    assert echo.regressions == 1
    # The tracker saw no release, so only the stage-1 queries agree with it.
    assert (echo.queries, echo.agreements) == (3, 2)
    assert "mode=self plan=0 tracker=0 of 2 queries=3 agreement=2/3" in echo.trace()


def test_self_mode_rejects_bad_responses():
    echo = gst.PlanEcho("self", 2)
    echo.request()
    with pytest.raises(ValueError):
        echo.update({"plan_choice": 2})
    with pytest.raises(RuntimeError):
        gst.PlanEcho("self", 2).update({"plan_choice": 0})  # no request first


def test_oracle_mode_follows_the_tracker():
    echo = gst.PlanEcho("oracle", 2)
    assert echo.request() == {"prev_plan_index": 0, "oracle_plan_index": 0}
    echo.update({"plan_choice": 0})  # the server's choice is ignored: the oracle is the plan
    _grasp_and_release(echo, gst.HOLD_MIN)
    assert echo.request() == {"prev_plan_index": 0, "oracle_plan_index": 1}
    assert echo.update({"plan_choice": 0}) == 1
    assert echo.request() == {"prev_plan_index": 1, "oracle_plan_index": 1}
    echo.update({})
    assert echo.agreements == echo.queries == 3
    assert echo.switches == [(gst.HOLD_MIN + 1, 0, 1)]


def test_forced_mode():
    with pytest.raises(ValueError):
        gst.PlanEcho("forced", 2)
    with pytest.raises(ValueError):
        gst.PlanEcho("bogus", 2)
    echo = gst.PlanEcho("forced", 1, forced_index=1)  # clamped to the one-stage task's stage
    assert echo.request() == {"prev_plan_index": 0, "oracle_plan_index": 0}
    echo = gst.PlanEcho("forced", 2, forced_index=1)
    assert echo.request() == {"prev_plan_index": 0, "oracle_plan_index": 1}
    assert echo.update({"plan_choice": 0}) == 1
    assert echo.request() == {"prev_plan_index": 1, "oracle_plan_index": 1}
    assert echo.agreements == 0  # the tracker is still at stage 1


def test_steering_counts_plans_and_tracks_the_environment():
    """A steering task's plans index num_plans; the tracker counts the environment task's stages."""
    echo = gst.PlanEcho("forced", 1, forced_index=0, tracker_plans=2)
    assert echo.num_plans == 1 and echo.tracker.num_plans == 2
    for t in range(gst.HOLD_MIN + 2):
        echo.observe(1.0 if t <= gst.HOLD_MIN else -1.0)  # one qualifying grasp-release cycle
    assert echo.tracker.plan_index == 1  # the environment advanced; the steering plan stays at 0
    assert echo.request() == {"prev_plan_index": 0, "oracle_plan_index": 0}
    assert echo.update({}) == 0 and echo.agreements == 0
    assert gst.PlanEcho("self", 2).tracker.num_plans == 2  # default: the plans' own count


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
