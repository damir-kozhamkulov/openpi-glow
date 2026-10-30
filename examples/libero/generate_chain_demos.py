"""Two-task LIBERO-Goal chain demos from a policy server (Python 3.8, the LIBERO client venv).

For every chainable pair of suite tasks {a, b} (same scene, different moved object; see
glow_chain_demo.chain_pairs) and every initial state tried, the episode runs twice from that
initial state, once per order: the first task's instruction is the prompt until its stage atom
holds and the gripper has let go, then a scripted return brings the arm up, sideways, under the
pose it started from and into it (glow_chain_demo.ReturnPlan), then the second task's
instruction is the prompt until its stage atom holds (glow_chain_demo.ChainRun). The server is a
stock language-prompted policy (pi05_libero); nothing GLOW-specific is sent. Kept episodes are
written as chain_<first>_<second>_init<k>.npz (frames as the training dataset stores them, plus the stage
starts); every attempt adds a line to attempts-shard<i>-of-<n>.jsonl. The env of a pair is built
from the lower task id's BDDL and its initial states; both tasks' goals are scored on it, which
the scene check makes meaningful.

The entrypoint passes --args.task-suite-name, --args.num-trials-per-task (here: initial states
tried per pair, from --args.init-start) and --args.video-out-path.
"""

import collections
import dataclasses
import json
import logging
import pathlib
from typing import Optional

import glow_chain_demo
import glow_stage_tracker
import imageio
from libero.libero import benchmark
from libero.libero.envs import bddl_utils
import main as _main
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    num_steps_wait: int = 10
    seed: int = 7
    task_suite_name: str = "libero_goal"
    # Plan pack: each task's stage atom (runtime_key). Every suite task must be in it with one stage.
    plan_pack: str = "/app/glow/plan_pack_libero10_goal.json"
    # Initial states tried per pair: init_start .. init_start + num_trials_per_task - 1. The evals
    # use states 0..TRIALS-1, so generation starts above them.
    init_start: int = 10
    num_trials_per_task: int = 2
    # Stop a pair once this many initial states were kept in both orders (None: try them all).
    paired_target: Optional[int] = None
    max_steps_per_phase: int = 300  # libero_goal's eval budget
    # Frames after the first atom held to wait for an open-gripper action before the return starts.
    release_cap: int = 40
    # Steps the return (rise, sideways, under the start, into it) may take before the second task starts
    # wherever the arm is.
    max_homing_steps: int = 150
    out_dir: str = "/data/gen/chains"
    shard_index: int = 0
    shard_count: int = 1
    video_out_path: str = "/data/libero/videos/chains"
    # Videos of rejected attempts saved per pair (kept episodes always get one).
    failure_videos_per_pair: int = 3


def generate(args: Args) -> None:
    np.random.seed(args.seed)
    if args.task_suite_name != "libero_goal":
        raise ValueError("chain demos are LIBERO-Goal only (one shared scene, one-stage tasks)")
    if args.num_trials_per_task < 1 or args.init_start < 0:
        raise ValueError("need init_start >= 0 and num_trials_per_task >= 1")
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = pathlib.Path(args.video_out_path)
    video_dir.mkdir(parents=True, exist_ok=True)
    attempts_path = out_dir / glow_chain_demo.ATTEMPTS_FILE.format(index=args.shard_index, count=args.shard_count)

    with open(args.plan_pack, encoding="utf-8") as f:
        pack = json.load(f)
    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    names = [str(task_suite.get_task(i).language) for i in range(task_suite.n_tasks)]
    problems = [bddl_utils.robosuite_parse_problem(task_suite.get_task_bddl_file_path(i)) for i in range(len(names))]
    atoms = []
    for i, name in enumerate(names):
        if name not in pack["tasks"] or int(pack["tasks"][name]["K"]) != 1:
            raise ValueError("task {!r} must be a one-stage task of the plan pack {}".format(name, args.plan_pack))
        atoms.append(_main.stage_atoms(pack["tasks"][name], problems[i])[0])

    pairs = glow_chain_demo.chain_pairs(
        len(names),
        moved=lambda i: glow_stage_tracker.moved_object(pack["tasks"][names[i]]["stages"][0]["runtime_key"]),
        same_scene=lambda i, j: not _main._scene_differences(problems[i], problems[j]),  # noqa: SLF001
    )
    mine = glow_chain_demo.shard(pairs, args.shard_index, args.shard_count)
    inits = list(range(args.init_start, args.init_start + args.num_trials_per_task))
    logging.info(
        f"chain demos: {len(pairs)} pairs in the suite, shard {args.shard_index}/{args.shard_count} runs {len(mine)}: "
        f"{mine}; initial states {inits[0]}..{inits[-1]}; paired target {args.paired_target}; out {out_dir}"
    )

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    totals = collections.Counter()
    aborted = 0
    for a, b in mine:
        env, _ = _main._get_libero_env(task_suite.get_task(a), _main.LIBERO_ENV_RESOLUTION, args.seed)  # noqa: SLF001
        initial_states = task_suite.get_task_init_states(a)
        if inits[-1] >= len(initial_states):
            raise ValueError(f"task {a} has {len(initial_states)} initial states, asked for up to {inits[-1]}")
        problem_env = env.env

        def holds(atom, problem_env=problem_env):
            return problem_env._eval_predicate(list(atom))  # noqa: SLF001

        goals = {a: _main.SteeredGoal(env, problems[a]), b: _main.SteeredGoal(env, problems[b])}
        pair_counts, paired, failure_videos = collections.Counter(), 0, 0
        for k in inits:
            kept_orders = 0
            for first, second in ((a, b), (b, a)):
                run = glow_chain_demo.ChainRun(
                    lambda first=first: holds(atoms[first]),
                    lambda second=second: holds(atoms[second]),
                    goals[first].check,
                    goals[second].check,
                    args.max_steps_per_phase,
                    args.release_cap,
                )
                recorder = glow_chain_demo.EpisodeRecorder()
                replay = []
                homing = {}
                try:
                    _run_episode(
                        args, env, initial_states[k], client, names[first], names[second], run, recorder, replay, homing
                    )
                except Exception as e:  # noqa: BLE001
                    logging.error(f"Caught exception: {e}")
                    run.fail()
                    if isinstance(e, _websocket_client_policy.PolicyServerUnreachable):
                        aborted += 1
                status = run.status
                record = {
                    "env_task": a,
                    "first": first,
                    "second": second,
                    "first_task": names[first],
                    "second_task": names[second],
                    "init": k,
                    "status": status,
                    "n_frames": run.n_frames,
                    "switch_frame": run.switch_frame,
                    "first_met": run.first_met,
                    "released": run.released,
                    "homed_frame": run.homed_frame,
                    "shard": args.shard_index,
                    **homing,
                }
                if status == "kept":
                    path = out_dir / glow_chain_demo.EPISODE_FILE.format(first=first, second=second, init=k)
                    recorder.save(path, run.stage_starts())
                    record["file"] = path.name
                    kept_orders += 1
                glow_chain_demo.append_attempt(attempts_path, record)
                pair_counts[status] += 1
                totals[status] += 1
                if replay and (status == "kept" or failure_videos < args.failure_videos_per_pair):
                    failure_videos += status != "kept"
                    imageio.mimwrite(
                        video_dir / f"chain_{first:02d}_{second:02d}_init{k:02d}_{status}.mp4",
                        [np.asarray(x) for x in replay],
                        fps=10,
                    )
                logging.info(
                    f"chain attempt: first={first} second={second} init={k} status={status} "
                    f"frames={run.n_frames} switch={run.switch_frame} first_met={run.first_met} "
                    f"released={run.released} homed={run.homed_frame} homing={homing}"
                )
            paired += kept_orders == 2
            totals["paired_inits"] += kept_orders == 2
            if args.paired_target is not None and paired >= args.paired_target:
                break
        logging.info(
            f"chain pair: {a} {names[a]!r} + {b} {names[b]!r}: paired initial states {paired}; {dict(pair_counts)}"
        )

    attempts = sum(totals[s] for s in ("kept", *glow_chain_demo.REJECT_REASONS))
    logging.info(
        f"Chain demos: {totals['kept']} kept of {attempts} attempts; paired initial states {totals['paired_inits']}; "
        f"rejected {{{', '.join(f'{s}: {totals[s]}' for s in glow_chain_demo.REJECT_REASONS)}}}"
    )
    if aborted:
        raise RuntimeError(
            f"RUN INVALID: {aborted} attempts were aborted because the policy server was unreachable; "
            f"their rejections are not policy results."
        )


def _run_episode(args, env, initial_state, client, first_task, second_task, run, recorder, replay, homing):
    """Run one chain episode into `recorder` / `replay`; `homing` receives the return's steps,
    whether it reached the start pose and its remaining position error."""
    env.reset()
    obs = env.set_init_state(initial_state)
    for _ in range(args.num_steps_wait):
        obs, _, _, _ = env.step(_main.LIBERO_DUMMY_ACTION)
    home_pos, home_quat = np.array(obs["robot0_eef_pos"]), np.array(obs["robot0_eef_quat"])
    if run.start() != "running":
        return
    action_plan = collections.deque()
    plan = None
    homing_steps = 0
    while run.status == "running":
        home_action = None
        if run.phase == 1:
            if plan is None:
                plan = glow_chain_demo.ReturnPlan(
                    np.array(obs["robot0_eef_pos"]), np.array(obs["robot0_eef_quat"]), home_pos, home_quat
                )
            home_action, reached = plan.action(obs["robot0_eef_pos"], obs["robot0_eef_quat"])
            if reached or homing_steps >= args.max_homing_steps:
                homing.update(
                    homing_steps=homing_steps,
                    homing_reached=reached,
                    homing_error_cm=round(float(np.linalg.norm(obs["robot0_eef_pos"] - home_pos)) * 100, 2),
                    homing_stage=glow_chain_demo.ReturnPlan.STAGES[plan.stage],
                    homing_skipped=plan.skipped,
                )
                if run.homed() != "running":
                    return
                home_action = None
        # Rotated 180 degrees: the training dataset's orientation (see main.py).
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        state = np.concatenate(
            (obs["robot0_eef_pos"], _main._quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])  # noqa: SLF001
        )
        model_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
        replay.append(model_img)
        if home_action is not None:
            action = home_action
            homing_steps += 1
        else:
            if not action_plan:
                element = {
                    "observation/image": model_img,
                    "observation/wrist_image": image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    ),
                    "observation/state": state,
                    "prompt": first_task if run.phase == 0 else second_task,
                }
                action_chunk = client.infer(element)["actions"]
                if len(action_chunk) < args.replan_steps:
                    raise ValueError(f"policy predicts {len(action_chunk)} steps, fewer than replan_steps")
                action_plan.extend(action_chunk[: args.replan_steps])
            action = action_plan.popleft()
        action = np.asarray(action, dtype=float)
        recorder.add(img, wrist_img, state, action)
        obs, _, _, _ = env.step(action.tolist())
        if run.step(action[6]) == "switch":
            action_plan.clear()  # the first task's remaining actions are not executed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(generate)
