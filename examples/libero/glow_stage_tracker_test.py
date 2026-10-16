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
