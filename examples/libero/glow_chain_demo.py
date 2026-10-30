"""Two-task chain demos on LIBERO-Goal (Python 3.8): which pairs run, how one episode advances,
which episodes are kept, the scripted return to the start pose between the tasks, and how a kept
episode is written.

A chain episode runs a first task's instruction until that task's stage atom (its runtime_key,
one atom of its BDDL goal) holds and the gripper has let go, then returns the arm to the pose it
started from (scripted, `ReturnPlan`: up, sideways, under the start, into it), then runs a
second task's instruction until its atom holds. Both orders of a pair start from the same initial
state, so before either goal is met only the order tells them apart. `ChainRun` follows one episode frame by frame: frame f is the
observation before the f-th executed action, and an atom checked after that action belongs to
frame f + 1. Its phases:
  0  first task, policy           ends on the first frame with the atom held after an open-gripper
                                  action, or `release_cap` frames after the atom first held
  1  return to the start pose     scripted; ends when the client calls homed()
  2  second task, policy          ends when the second atom holds
The stage label of a kept episode is 0 on frames [0, switch_frame) and 1 on
[switch_frame, n_frames): the return belongs to the second stage.

An episode is kept when:
  - neither atom holds when the policy takes over,
  - the second atom does not hold before the first,
  - the first atom still holds once the arm is back at the start pose,
  - each policy phase ends within its step budget,
  - at the end both tasks' full BDDL goals hold (the second task did not undo the first).
"""

import json
import math
import pathlib

import numpy as np

ATTEMPTS_FILE = "attempts-shard{index:02d}-of-{count:02d}.jsonl"
EPISODE_FILE = "chain_{first:02d}_{second:02d}_init{init:02d}.npz"

REJECT_REASONS = ("goal_at_start", "second_first", "first_timeout", "first_lost", "second_timeout", "first_undone", "error")

# LIBERO's OSC_POSE controller: an action of 1 moves the position target 5 cm and the orientation
# target 0.5 rad, in the world frame.
OSC_POS_SCALE = 0.05
OSC_ROT_SCALE = 0.5
GRIPPER_OPEN = -1.0


def chain_pairs(num_tasks, moved, same_scene):
    """Unordered pairs (i, j), i < j, of suite tasks that can share a chain: set in the same scene
    (`same_scene(i, j)`) and moving different objects (`moved(i)`, None moves none)."""
    pairs = []
    for i in range(num_tasks):
        for j in range(i + 1, num_tasks):
            if not same_scene(i, j):
                continue
            mi, mj = moved(i), moved(j)
            if mi is None or mj is None or mi != mj:
                pairs.append((i, j))
    return pairs


def shard(items, index, count):
    """Every count-th item starting at index (round-robin, so shards stay balanced)."""
    if count < 1 or not 0 <= index < count:
        raise ValueError("shard index {} outside 0..{}".format(index, count - 1))
    return list(items)[index::count]


def chain_task(first, second):
    """The plan-pack task string of a chain: what the dataset's tasks.jsonl names it."""
    return "{} then {}".format(first, second)


class ChainRun:
    """One chain episode's progress. `first_holds` / `second_holds` read the two stage atoms on the
    simulator; `first_goal` / `second_goal` the two full BDDL goals (checked once, at the end)."""

    def __init__(self, first_holds, second_holds, first_goal, second_goal, max_steps_per_phase, release_cap=40):
        if max_steps_per_phase < 1 or release_cap < 0:
            raise ValueError("max_steps_per_phase must be >= 1 and release_cap >= 0")
        self._first_holds = first_holds
        self._second_holds = second_holds
        self._first_goal = first_goal
        self._second_goal = second_goal
        self.max_steps_per_phase = max_steps_per_phase
        self.release_cap = release_cap
        self.phase = 0
        self.first_met = None  # frame the first atom first held
        self.switch_frame = None  # frame the return to the start pose began (stage 1 from here)
        self.homed_frame = None  # frame the second task's policy took over
        self.released = None  # whether the switch followed an open-gripper action (else the cap)
        self.n_frames = 0
        self.status = "running"  # running | kept | one of REJECT_REASONS

    def start(self):
        """Check the atoms when the policy takes over (frame 0); returns the status."""
        if bool(self._first_holds()) or bool(self._second_holds()):
            self.status = "goal_at_start"
        return self.status

    def step(self, gripper_command):
        """Record one executed action (frame n_frames) and check the atoms on the resulting state.

        Returns "switch" when the return to the start pose begins at the next frame, else the
        status. Raises once the episode has ended."""
        if self.status != "running":
            raise RuntimeError("episode already ended: {}".format(self.status))
        self.n_frames += 1
        frame = self.n_frames  # the frame the new state belongs to
        if self.phase == 0:
            holds = bool(self._first_holds())  # the simulator's predicates return numpy booleans
            if holds and self.first_met is None:
                self.first_met = frame
            if self.first_met is not None:
                released = holds and float(gripper_command) <= 0
                if released or frame - self.first_met >= self.release_cap:
                    self.phase, self.switch_frame, self.released = 1, frame, released
                    return "switch"
            if bool(self._second_holds()):
                self.status = "second_first"
            elif self.n_frames >= self.max_steps_per_phase:
                self.status = "first_timeout"
            return self.status
        if self.phase == 1:
            return self.status
        if bool(self._second_holds()):
            self.status = "kept" if bool(self._first_goal()) and bool(self._second_goal()) else "first_undone"
        elif self.n_frames - self.homed_frame >= self.max_steps_per_phase:
            self.status = "second_timeout"
        return self.status

    def homed(self):
        """The arm is back at the start pose (or the return ran out of steps): the second task's
        policy takes over, if the first atom still holds. Returns the status."""
        if self.phase != 1:
            raise RuntimeError("homed() outside the return phase")
        self.phase, self.homed_frame = 2, self.n_frames
        if not bool(self._first_holds()):
            self.status = "first_lost"
        return self.status

    def fail(self):
        """The episode broke off (server or simulator error)."""
        self.status = "error"

    def stage_starts(self):
        """Stage start frames of a kept episode: [0, switch_frame]."""
        if self.status != "kept":
            raise RuntimeError("only a kept episode has stage labels, this one is {}".format(self.status))
        return [0, self.switch_frame]


def _quat_mul(a, b):
    """Hamilton product of xyzw quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def rotation_error(quat, target_quat):
    """Axis-angle (world frame) that rotates orientation `quat` onto `target_quat`, both xyzw."""
    q = _quat_mul(np.asarray(target_quat, float), np.asarray(quat, float) * np.array([-1, -1, -1, 1]))
    q = q / np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    s = np.linalg.norm(q[:3])
    if s < 1e-9:
        return np.zeros(3)
    return q[:3] / s * 2.0 * math.atan2(s, q[3])


def _pose_action(pos, quat, target_pos, target_quat, max_pos, max_rot):
    """OSC_POSE action straight toward a pose, gripper open, each component clipped."""
    action = np.zeros(7)
    action[:3] = np.clip((np.asarray(target_pos, float) - np.asarray(pos, float)) / OSC_POS_SCALE, -max_pos, max_pos)
    action[3:6] = np.clip(rotation_error(quat, target_quat) / OSC_ROT_SCALE, -max_rot, max_rot)
    action[6] = GRIPPER_OPEN
    return action


class ReturnPlan:
    """The scripted return between the two tasks, in four stages:
      rise     straight up `rise` (to at least the start height with `rise_to_start_height`)
      side     sideways to the start's y, at that height
      below    to a point `below` under the start pose
      start    up into the start pose
    The wrist keeps the orientation it had when the return began until the last stage, which turns
    it to the start orientation. The order follows how the policy itself leaves the far side of the
    workspace: after placing on the cabinet top the elbow is folded near its joint limit, and a move
    toward the robot at that height (or a turn of the wrist) drives it into the limit and stalls,
    while going sideways first and then toward the robot from below does not. A stage ends within
    `reach_tol`, or after `stage_steps` actions (the arm may be blocked, e.g. inside a drawer), and
    the return goes on with the next one; only the last stage has to be reached."""

    STAGES = ("rise", "side", "below", "start")

    def __init__(self, pos, quat, start_pos, start_quat, rise=0.06, below=0.12, rise_to_start_height=False,
                 reach_tol=0.015, rot_tol=0.1, stage_steps=30, max_pos=0.5, max_rot=0.3):
        pos, start_pos = np.asarray(pos, float), np.asarray(start_pos, float)
        top = pos[2] + rise
        if rise_to_start_height:
            top = max(top, start_pos[2])
        self.targets = [
            (np.array([pos[0], pos[1], top]), np.asarray(quat, float)),
            (np.array([pos[0], start_pos[1], top]), np.asarray(quat, float)),
            (start_pos - np.array([0.0, 0.0, below]), np.asarray(quat, float)),
            (start_pos, np.asarray(start_quat, float)),
        ]
        self.reach_tol, self.rot_tol, self.stage_steps = reach_tol, rot_tol, stage_steps
        self.max_pos, self.max_rot = max_pos, max_rot
        self.stage = 0
        self._steps_here = 0
        self.skipped = []  # stages given up after stage_steps

    def action(self, pos, quat):
        """The next action and whether the start pose is reached."""
        pos = np.asarray(pos, float)
        while True:
            target_pos, target_quat = self.targets[self.stage]
            last = self.stage == len(self.targets) - 1
            close = np.linalg.norm(target_pos - pos) < self.reach_tol
            if last and close and np.linalg.norm(rotation_error(quat, target_quat)) < self.rot_tol:
                return _pose_action(pos, quat, target_pos, target_quat, self.max_pos, self.max_rot), True
            if not last and (close or self._steps_here >= self.stage_steps):
                if not close:
                    self.skipped.append(self.STAGES[self.stage])
                self.stage, self._steps_here = self.stage + 1, 0
                continue
            self._steps_here += 1
            return _pose_action(pos, quat, target_pos, target_quat, self.max_pos, self.max_rot), False


class EpisodeRecorder:
    """Frames of one episode as the training dataset stores them: images 256x256 uint8 already
    rotated 180 degrees, state (8,), the executed action (7,)."""

    def __init__(self):
        self.image, self.wrist_image, self.state, self.actions = [], [], [], []

    def add(self, image, wrist_image, state, action):
        self.image.append(np.asarray(image, dtype=np.uint8))
        self.wrist_image.append(np.asarray(wrist_image, dtype=np.uint8))
        self.state.append(np.asarray(state, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))

    def __len__(self):
        return len(self.actions)

    def save(self, path, stage_starts):
        n = len(self.actions)
        if not (len(self.image) == len(self.wrist_image) == len(self.state) == n):
            raise ValueError("unequal frame lists")
        if not stage_starts or stage_starts[0] != 0 or stage_starts[-1] >= n:
            raise ValueError("stage starts {} do not fit {} frames".format(stage_starts, n))
        np.savez_compressed(
            str(path),
            image=np.stack(self.image),
            wrist_image=np.stack(self.wrist_image),
            state=np.stack(self.state),
            actions=np.stack(self.actions),
            stage_starts=np.asarray(stage_starts, dtype=np.int64),
        )


def _json_default(value):
    """numpy scalars (the simulator's predicates and arrays hand them out) as Python values."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError("Object of type {} is not JSON serializable".format(type(value).__name__))


def append_attempt(path, record):
    """One JSON line per attempt, kept or rejected."""
    with pathlib.Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True, default=_json_default) + "\n")
