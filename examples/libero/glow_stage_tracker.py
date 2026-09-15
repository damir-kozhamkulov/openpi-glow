"""Proprioceptive stage tracker for GLOW's plan state (client side, Python 3.8).

The active plan advances on the frame after every gripper release whose grasp was held for at
least `hold_min` frames, and is capped at the task's last plan. The only input is the gripper
command of the actions the client has executed, so it is evaluation-legal. Replayed over the
LIBERO-10 training tapes it reproduces the label factory's plan chain on 99.89% of frames
(378/379 episodes exact); a regrasp guard was measured to make that worse, so there is none.
"""

HOLD_MIN = 30


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
