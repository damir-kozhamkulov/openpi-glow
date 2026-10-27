"""Steered-goal scoring in main.py: resolving the steering task, the scene check, the goal check.

`pytest examples/libero/main_test.py` runs anywhere: modules the client environment provides
(libero, imageio, openpi_client, ...) are stubbed when they cannot be imported, and the tests
marked `needs_libero` are then skipped. In the client environment
(PYTHONPATH=/app:/app/packages/openpi-client/src:/app/third_party/libero, MUJOCO_GL=osmesa) they
run against the real LIBERO benchmark, BDDL parser and simulator.
"""

# ruff: noqa: SLF001

import importlib
import pathlib
import sys
from unittest import mock

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

try:
    import main

    HAVE_LIBERO = True
except ImportError:
    HAVE_LIBERO = False
    for _name in (
        "imageio",
        "libero",
        "libero.libero",
        "libero.libero.envs",
        "openpi_client.image_tools",
        "openpi_client.websocket_client_policy",
        "tqdm",
        "tyro",
    ):
        try:
            importlib.import_module(_name)
        except ImportError:
            sys.modules[_name] = mock.MagicMock()
    import main

needs_libero = pytest.mark.skipif(not HAVE_LIBERO, reason="needs the LIBERO client environment")


def _problem(goal, objects=("bowl_1",), language="task"):
    return {
        "problem_name": "libero_kitchen_tabletop_manipulation",
        "fixtures": {"main_table": ["main_table"]},
        "regions": {"stove_region": {"target": "main_table"}},
        "objects": {name: [name] for name in objects},
        "scene_properties": {},
        "initial_state": [["on", "bowl_1", "stove_region"]],
        "goal_state": goal,
        "language_instruction": [language],
        "obj_of_interest": [obj for atom in goal for obj in atom[1:]],
    }


class _Suite:
    """A task suite whose BDDL paths are the task indices, parsed by `_parse`."""

    def __init__(self, problems):
        self.problems = problems

    def get_task_bddl_file_path(self, i):
        return i


@pytest.fixture
def suite(monkeypatch):
    problems = [
        _problem([["on", "bowl_1", "stove_region"]], language="a"),
        _problem([["open", "drawer_1"]], language="b"),
        _problem([["on", "bowl_1", "plate_1"]], objects=("bowl_1", "plate_1"), language="c"),
        _problem([["turnon", "stove_1"]], language="b"),
    ]
    monkeypatch.setattr(main.bddl_utils, "robosuite_parse_problem", lambda path: problems[path])
    return _Suite(problems)


def test_scene_differences_ignore_goal_language_and_objects_of_interest():
    a = _problem([["on", "bowl_1", "stove_region"]], language="a")
    b = _problem([["open", "drawer_1"]], language="b")
    assert main._scene_differences(a, b) == []
    c = _problem([["open", "drawer_1"]], objects=("bowl_1", "plate_1"))
    c["regions"] = {}
    assert main._scene_differences(a, c) == ["objects", "regions"]


def test_offset_steering_takes_the_shifted_task(suite):
    names = ["a", "b", "c", "b"]
    assert main._steered_problem(suite, names, 0, None, 1) is suite.problems[1]
    assert main._steered_problem(suite, names, 3, None, 1) is suite.problems[0]  # wraps
    assert main._steered_problem(suite, names, 1, None, 0) is suite.problems[1]  # offset 0: own goal


def test_offset_steering_refuses_another_scene(suite):
    with pytest.raises(ValueError, match=r"'c' is set in another scene .*\(objects differ\)"):
        main._steered_problem(suite, ["a", "b", "c", "b"], 1, None, 1)


def test_plan_task_steering_matches_language_in_the_scene(suite):
    names = ["a", "b", "c", "b"]
    assert main._steered_problem(suite, names, 0, "b", None) is suite.problems[1]  # first of two
    with pytest.raises(ValueError, match="not a task of this suite"):
        main._steered_problem(suite, names, 0, "put the moka pot on the stove", None)
    with pytest.raises(ValueError, match="another scene"):
        main._steered_problem(suite, names, 0, "c", None)


def test_plan_task_steering_skips_a_same_named_task_in_another_scene(suite):
    suite.problems[1] = _problem([["open", "drawer_1"]], objects=("bowl_1", "plate_1"), language="b")
    assert main._steered_problem(suite, ["a", "b", "c", "b"], 0, "b", None) is suite.problems[3]


def test_steered_goal_is_the_conjunction_of_libero_predicates():
    holds = {("on", "bowl_1", "plate_1"): True, ("open", "drawer_1"): False}
    seen = []

    def eval_predicate(state):
        seen.append(tuple(state))
        return holds[tuple(state)]

    env = mock.MagicMock()
    env.env._eval_predicate = eval_predicate
    goal = main.SteeredGoal(env, {"goal_state": [["on", "bowl_1", "plate_1"], ["open", "drawer_1"]]})
    assert goal.check() is False
    holds[("open", "drawer_1")] = True
    assert goal.check() is True
    assert seen[-2:] == [("on", "bowl_1", "plate_1"), ("open", "drawer_1")]
    with pytest.raises(ValueError, match="no goal"):
        main.SteeredGoal(env, {"goal_state": []})


def _libero_suite(name):
    task_suite = main.benchmark.get_benchmark_dict()[name]()
    names = [str(task_suite.get_task(i).language) for i in range(task_suite.n_tasks)]
    return task_suite, names


@needs_libero
def test_libero_goal_tasks_share_one_scene():
    task_suite, names = _libero_suite("libero_goal")
    for offset in range(task_suite.n_tasks):
        for task_id in range(task_suite.n_tasks):
            problem = main._steered_problem(task_suite, names, task_id, None, offset)
            steered_id = (task_id + offset) % task_suite.n_tasks
            path = task_suite.get_task_bddl_file_path(steered_id)
            assert problem["goal_state"] == main.bddl_utils.robosuite_parse_problem(path)["goal_state"]


@needs_libero
def test_libero_10_steering_is_refused():
    task_suite, names = _libero_suite("libero_10")

    def resolve_all():
        for task_id in range(task_suite.n_tasks):
            main._steered_problem(task_suite, names, task_id, None, 1)

    with pytest.raises(ValueError, match="another scene"):
        resolve_all()


@needs_libero
def test_steered_goal_scores_the_simulator_state():
    """Environment: open the middle drawer. Steering task: turn on the stove, made true by hand."""
    task_suite, names = _libero_suite("libero_goal")
    env_id = names.index("open the middle drawer of the cabinet")
    stove_id = names.index("turn on the stove")
    env, _ = main._get_libero_env(task_suite.get_task(env_id), 128, 7)
    try:
        env.reset()
        env.set_init_state(task_suite.get_task_init_states(env_id)[0])
        for _ in range(10):
            env.step(main.LIBERO_DUMMY_ACTION)
        own = main.SteeredGoal(env, main._steered_problem(task_suite, names, env_id, None, 0))
        stove = main.SteeredGoal(env, main._steered_problem(task_suite, names, env_id, names[stove_id], None))
        assert own.check() is bool(env.check_success()) is False
        assert stove.check() is False

        problem_env = env.env
        sim = problem_env.sim
        for joint in problem_env.get_object("flat_stove_1").joints:
            sim.data.qpos[sim.model.get_joint_qpos_addr(joint)] = 1.0
        sim.forward()
        assert stove.check() is True
        assert own.check() is bool(env.check_success()) is False
    finally:
        env.close()


def _pack_task(*keys):
    return {
        "K": len(keys),
        "stages": [{"plan_index": k, "runtime_key": key} for k, key in reversed(list(enumerate(keys)))],
    }


def test_stage_atoms_are_goal_atoms_in_stage_order():
    problem = _problem([["on", "bowl_1", "plate_1"], ["open", "drawer_1"]], objects=("bowl_1", "plate_1"))
    task = _pack_task("on(bowl_1, plate_1)", "open(drawer_1)")
    assert main.stage_atoms(task, problem) == [("on", "bowl_1", "plate_1"), ("open", "drawer_1")]
    with pytest.raises(ValueError, match="not in the task's BDDL goal"):
        main.stage_atoms(_pack_task("on(bowl_1, stove_region)"), problem)


def test_first_counts_order():
    import collections

    c = collections.Counter({"none": 1, "1": 3, "together": 2, "0": 4})
    assert main._first_counts(c) == "stage 0: 4, stage 1: 3, together: 2, none: 1"


@needs_libero
@pytest.mark.parametrize("suite_name", ["libero_10", "libero_goal"])
def test_every_pack_stage_atom_is_in_its_bddl_goal(suite_name):
    import json
    import os

    pack_path = os.environ.get("OPENPI_PLAN_PACK", "/app/glow/plan_pack_libero10_goal.json")
    if not pathlib.Path(pack_path).exists():
        pytest.skip(f"no plan pack at {pack_path}")
    with open(pack_path, encoding="utf-8") as f:
        pack = json.load(f)
    task_suite, names = _libero_suite(suite_name)
    for task_id, name in enumerate(names):
        problem = main.bddl_utils.robosuite_parse_problem(task_suite.get_task_bddl_file_path(task_id))
        atoms = main.stage_atoms(pack["tasks"][name], problem)
        assert len(atoms) == pack["tasks"][name]["K"]
        extra = {tuple(a) for a in problem["goal_state"]} - set(atoms)
        # The stages cover the whole goal, except that both moka pots must also sit on a lit stove.
        assert extra == ({("turnon", "flat_stove_1")} if name == "put both moka pots on the stove" else set()), name
