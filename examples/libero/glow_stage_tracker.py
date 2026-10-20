"""GLOW's plan state on the client side (Python 3.8).

`StageTracker`: a proprioceptive stage tracker. The active plan advances on the frame after every
gripper release whose grasp was held for at least `hold_min` frames, and is capped at the task's
last plan. The only input is the gripper command of the actions the client has executed, so it
is evaluation-legal. Replayed over the LIBERO-10 training tapes it reproduces the label factory's
plan chain on 99.89% of frames (378/379 episodes exact); a regrasp guard was measured to make
that worse, so there is none.

`PlanEcho`: what one episode sends a GLOW server and what it keeps. Every query carries the
previous plan (`prev_plan_index`, stage 1 at reset); the plan that query settles on becomes the
next query's previous plan. Modes:
  self    the server scores every stage and returns its choice (`plan_choice`)
  oracle  the tracker's stage is sent as `oracle_plan_index` and used as the plan
  forced  a fixed stage is sent as `oracle_plan_index` (counterfactual: do actions follow it?)
The tracker runs in every mode, so the model's plan can be compared against it. When the plans
come from another task than the environment's (steering), `tracker_plans` is the environment
task's stage count and `num_plans` the plan task's.
"""

HOLD_MIN = 30
PLAN_MODES = ("self", "oracle", "forced")


def plan_task_for(task_names, task_id, offset):
    """The task whose plans steer environment task `task_id`: suite task (task_id + offset) mod n.

    offset 0 is the environment's own task; any other offset in 1..n-1 is a derangement (no task
    keeps its own plans).
    """
    if not task_names:
        raise ValueError("task_names is empty")
    if offset < 0:
        raise ValueError("offset must be >= 0, got {}".format(offset))
    return task_names[(task_id + offset) % len(task_names)]


class StageTracker:
    def __init__(self, num_plans, hold_min=HOLD_MIN):
        if num_plans < 1:
            raise ValueError("num_plans must be >= 1")
        self.num_plans = num_plans
        self.hold_min = hold_min
        self.releases = []  # frames of qualifying releases
        self._t = 0
        self._prev = None
        self._grasp_at = None

    @property
    def frame(self):
        """Index of the next frame to be executed."""
        return self._t

    @property
    def plan_index(self):
        """Plan active for the next frame: releases seen so far, capped at the last plan."""
        return min(len(self.releases), self.num_plans - 1)

    def observe(self, gripper_command):
        """Record the gripper command of the action executed at the current frame."""
        g = float(gripper_command)
        t = self._t
        if self._prev is None:
            self._grasp_at = 0 if g > 0 else None
        elif self._prev <= 0 < g:
            self._grasp_at = t
        elif self._prev > 0 >= g and self._grasp_at is not None:
            if t - self._grasp_at >= self.hold_min:
                self.releases.append(t)
            self._grasp_at = None
        self._prev = g
        self._t += 1
        return self.plan_index


class PlanEcho:
    def __init__(self, mode, num_plans, forced_index=None, hold_min=HOLD_MIN, tracker_plans=None):
        if mode not in PLAN_MODES:
            raise ValueError("mode must be one of {}, got {!r}".format(PLAN_MODES, mode))
        if mode == "forced":
            if forced_index is None or forced_index < 0:
                raise ValueError("forced mode needs a stage index >= 0")
            # One index serves every task: a stage past a task's last one means its last stage.
            forced_index = min(forced_index, num_plans - 1)
        self.mode = mode
        self.num_plans = num_plans
        self.forced_index = forced_index
        self.tracker = StageTracker(num_plans if tracker_plans is None else tracker_plans, hold_min=hold_min)
        self.prev = 0
        self.queries = 0
        self.agreements = 0  # queries whose plan equals the tracker's stage
        self.switches = []  # (frame, from, to) of the plan the queries settled on
        self.regressions = 0  # switches to an earlier stage
        self._sent = None

    def request(self):
        """Keys to add to the next observation sent to the server."""
        req = {"prev_plan_index": self.prev}
        if self.mode == "oracle":
            req["oracle_plan_index"] = self.tracker.plan_index
        elif self.mode == "forced":
            req["oracle_plan_index"] = self.forced_index
        self._sent = req
        return req

    def update(self, response):
        """Take the server's response to the last request; returns the plan that query used."""
        if self._sent is None:
            raise RuntimeError("update() without a preceding request()")
        if self.mode == "self":
            plan = int(response["plan_choice"])
            if not 0 <= plan < self.num_plans:
                raise ValueError("server chose plan {} outside 0..{}".format(plan, self.num_plans - 1))
        else:
            plan = int(self._sent["oracle_plan_index"])
        self._sent = None
        self.queries += 1
        self.agreements += int(plan == self.tracker.plan_index)
        if plan != self.prev:
            self.switches.append((self.tracker.frame, self.prev, plan))
            self.regressions += int(plan < self.prev)
        self.prev = plan
        return plan

    def observe(self, gripper_command):
        """Record the gripper command of the action just executed (drives the tracker)."""
        return self.tracker.observe(gripper_command)

    def trace(self):
        """One-line episode summary for the stage-resolved analysis."""
        return (
            "mode={} plan={} tracker={} of {} queries={} agreement={}/{} switches={} "
            "regressions={} tracker_releases={}".format(
                self.mode,
                self.prev,
                self.tracker.plan_index,
                self.num_plans,
                self.queries,
                self.agreements,
                self.queries,
                self.switches,
                self.regressions,
                self.tracker.releases,
            )
        )
