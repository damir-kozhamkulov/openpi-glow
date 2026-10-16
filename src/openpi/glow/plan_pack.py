"""The plan pack: per-task plan chains with evaluation-legal anchors, and per-episode stage ranges.

Built by glow_repo/scripts/build_plan_pack.py (format "glow-plan-pack/1"). Training looks up the
plan state at any frame by (episode_index, frame_index); at evaluation the plan state comes from
the model's own previous prediction (or the client's stage tracker, as an oracle). Every plan
string, input or target, training or evaluation, is rendered by `render_plan`, so they cannot
drift.
"""

import bisect
import dataclasses
import json
import os
import pathlib

FORMAT = "glow-plan-pack/1"
DEFAULT_PATH = "/app/glow/plan_pack.json"
ENV_VAR = "OPENPI_PLAN_PACK"


@dataclasses.dataclass(frozen=True)
class Stage:
    plan_index: int
    runtime_key: str
    subgoal_operation: str
    subgoal_operation_source: str
    object_noun: str
    target_noun: str
    atom: str
    object_bins: tuple[int, int, int] | None
    target_bins: tuple[int, int, int] | None
    weak_anchor: bool


@dataclasses.dataclass(frozen=True)
class TaskPlan:
    task_index: int
    task: str
    stages: tuple[Stage, ...]

    @property
    def num_plans(self) -> int:
        return len(self.stages)


@dataclasses.dataclass(frozen=True)
class EpisodePlan:
    task_index: int
    n_frames: int
    stage_starts: tuple[int, ...]

    def plan_index(self, frame_index: int) -> int:
        if not 0 <= frame_index < self.n_frames:
            raise IndexError(f"frame {frame_index} outside episode of {self.n_frames} frames")
        return bisect.bisect_right(self.stage_starts, frame_index) - 1


def _bins(anchor) -> tuple[int, int, int] | None:
    return None if anchor is None else tuple(int(b) for b in anchor["bins"])


class PlanPack:
    def __init__(self, tasks: dict[str, TaskPlan], episodes: dict[int, EpisodePlan], provenance: dict):
        self.tasks = tasks
        self.episodes = episodes
        self.provenance = provenance

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "PlanPack":
        """Load from `path`, else $OPENPI_PLAN_PACK, else the in-image default."""
        path = pathlib.Path(path or os.environ.get(ENV_VAR) or DEFAULT_PATH)
        if not path.exists():
            raise FileNotFoundError(
                f"plan pack not found at {path}; build it with glow_repo/scripts/build_plan_pack.py "
                f"and point {ENV_VAR} at it"
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("format") != FORMAT:
            raise ValueError(f"{path}: expected format {FORMAT!r}, got {raw.get('format')!r}")
        tasks = {}
        for task, entry in raw["tasks"].items():
            stages = tuple(
                Stage(
                    plan_index=int(s["plan_index"]),
                    runtime_key=s["runtime_key"],
                    subgoal_operation=s["subgoal_operation"],
                    subgoal_operation_source=s["subgoal_operation_source"],
                    object_noun=s["object_noun"],
                    target_noun=s["target_noun"],
                    atom=s["atom"],
                    object_bins=_bins(s["object_anchor"]),
                    target_bins=_bins(s["target_anchor"]),
                    weak_anchor=bool(s["weak_anchor"]),
                )
                for s in sorted(entry["stages"], key=lambda s: s["plan_index"])
            )
            if [s.plan_index for s in stages] != list(range(entry["K"])):
                raise ValueError(f"{path}: task {task!r} stages are not 0..K-1")
            tasks[task] = TaskPlan(task_index=int(entry["task_index"]), task=task, stages=stages)
        episodes = {
            int(ep): EpisodePlan(
                task_index=int(e["task_index"]),
                n_frames=int(e["n_frames"]),
                stage_starts=tuple(int(f) for f in e["stage_starts"]),
            )
            for ep, e in raw["episodes"].items()
        }
        return cls(tasks, episodes, raw.get("provenance", {}))

    def task(self, prompt: str) -> TaskPlan:
        try:
            return self.tasks[prompt]
        except KeyError:
            raise KeyError(
                f"prompt {prompt!r} is not a plan-pack task; the pack covers {len(self.tasks)} LIBERO-10 "
                "task strings exactly as tasks.jsonl / the BDDL language field spell them"
            ) from None

    def plan_index(self, episode_index: int, frame_index: int) -> int:
        try:
            episode = self.episodes[episode_index]
        except KeyError:
            raise KeyError(
                f"episode {episode_index} has no plan chain; the pack covers {len(self.episodes)} episodes "
                "(LIBERO-10 = 0..378 of physical-intelligence/libero)"
            ) from None
        return episode.plan_index(frame_index)

    def render_plan(self, prompt: str, plan_index: int, *, nouns: bool = True, anchors: bool = True) -> str:
        """The plan state at `plan_index` of `prompt`'s task, without the task string:

            "Stage k of K [{runtime_key}]: {subgoal_operation}. {atom}; object {noun} x y z; target {noun} x y z"

        Temporal fields (stage k of K, runtime_key) first, so the stage decision precedes the
        fields it determines; then the Spatial fields (subgoal_operation, atom, grounding).
        `nouns` / `anchors` drop the grounded nouns / the digitized anchor bins; an object or
        target fragment left with neither is omitted, as is one for a stage without that role.
        """
        task = self.task(prompt)
        if not 0 <= plan_index < task.num_plans:
            raise IndexError(f"plan_index {plan_index} outside 0..{task.num_plans - 1} for {prompt!r}")
        stage = task.stages[plan_index]
        text = (
            f"Stage {plan_index + 1} of {task.num_plans} [{stage.runtime_key}]: {stage.subgoal_operation}. {stage.atom}"
        )
        for role, noun, bins in (
            ("object", stage.object_noun, stage.object_bins),
            ("target", stage.target_noun, stage.target_bins),
        ):
            text += _grounding_fragment(role, noun if nouns else "", bins if anchors else None)
        return text


def _grounding_fragment(role: str, noun: str, bins: tuple[int, int, int] | None) -> str:
    parts = [p for p in (noun, bins and " ".join(str(b) for b in bins)) if p]
    return f"; {role} " + " ".join(parts) if parts else ""
