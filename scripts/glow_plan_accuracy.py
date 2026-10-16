"""Offline plan accuracy of a GLOW unified checkpoint on the LIBERO-10 training episodes.

Replays each episode's recorded frames through the serving path (`create_trained_policy`: the
same transforms, weights and norm_stats an evaluation server uses) every `plan_delta` frames,
as the LIBERO client queries every replan interval, and compares the plan the model chooses with
the plan pack's label at that frame. No simulator: it measures the planner alone, before any
closed-loop evaluation is spent on it.

  --prev-source self    the previous plan is the model's own last choice (stage 1 at frame 0),
                        exactly what the client sends - the closed-loop planner on recorded frames
  --prev-source label   the previous plan is the label delta frames earlier - the planner without
                        its own feedback, as trained (minus the input corruption)

Reports per-query stage accuracy, per-transition switch error (model switch frame - labeled stage
start, in frames; negative = early), missed transitions, regressions and episodes that end short
of the last stage, overall and per task, and the gate the unified plan sets before simulator
evaluation (accuracy >= 98%, median |switch error| <= 10 frames, no episode stuck).

  python scripts/glow_plan_accuracy.py --checkpoint-dir <ckpt> --dataset-dir <lerobot repo dir> \
      --out plan_accuracy.json
"""

from collections.abc import Callable, Iterable, Sequence
import dataclasses
import io
import itertools
import json
import logging
import pathlib
import statistics
import time
from typing import Literal

import numpy as np

GATE_ACCURACY = 0.98
GATE_MEDIAN_ABS_SWITCH_ERROR = 10


@dataclasses.dataclass
class Args:
    checkpoint_dir: str
    # LeRobot dataset root: the directory holding meta/ and data/ (e.g. .../physical-intelligence/libero).
    dataset_dir: str
    config: str = "pi05_libero_glow_unified"
    # Episode range "start:stop" (LIBERO-10 = 0:379).
    episodes: str = "0:379"
    prev_source: Literal["self", "label"] = "self"
    # Denoising steps per query. The actions are discarded, so 1 keeps the run cheap; the plan
    # choice does not depend on it.
    num_steps: int = 1
    # JSON summary (and per-query records) written here when set.
    out: str | None = None


@dataclasses.dataclass(frozen=True)
class Query:
    frame: int
    label: int
    prev: int
    plan: int
    logp: tuple[float, ...]


def run_episode(
    infer: Callable[[dict], dict],
    prompt: str,
    frames: Iterable[tuple[int, dict]],
    labels: Callable[[int], int],
    *,
    delta: int,
    prev_source: str,
) -> list[Query]:
    """Query the policy at every frame the iterator yields (frame index, observation keys).

    `labels(frame)` is the labeled plan at a frame. The previous plan sent with a query is the
    last query's choice (self) or the label `delta` frames earlier (label); stage 1 at the start.
    """
    queries = []
    prev = 0
    for frame, element in frames:
        if prev_source == "label":
            prev = labels(max(0, frame - delta))
        response = infer({**element, "prompt": prompt, "prev_plan_index": prev})
        plan = int(np.asarray(response["plan_choice"]))
        logp = tuple(float(x) for x in np.asarray(response["plan_logp"]).reshape(-1))
        queries.append(Query(frame=frame, label=labels(frame), prev=prev, plan=plan, logp=logp))
        prev = plan
    return queries


def episode_metrics(queries: Sequence[Query], stage_starts: Sequence[int], num_plans: int) -> dict:
    """Accuracy and per-transition switch errors of one episode's queries."""
    switch_errors = []
    missed = 0
    for k in range(1, num_plans):
        model_switch = next((q.frame for q in queries if q.plan >= k), None)
        if model_switch is None:
            missed += 1
        else:
            switch_errors.append(model_switch - stage_starts[k])
    return {
        "queries": len(queries),
        "correct": sum(q.plan == q.label for q in queries),
        "switch_errors": switch_errors,
        "missed": missed,
        "regressions": sum(b.plan < a.plan for a, b in itertools.pairwise(queries)),
        "stuck": bool(queries) and queries[-1].plan < num_plans - 1,
    }


def summarize(per_episode: dict[int, dict], task_of: dict[int, int]) -> dict:
    """Overall and per-task totals, switch-error statistics and the gate verdict."""

    def totals(eps):
        ms = [per_episode[e] for e in eps]
        errors = [x for m in ms for x in m["switch_errors"]]
        queries = sum(m["queries"] for m in ms)
        out = {
            "episodes": len(ms),
            "queries": queries,
            "accuracy": sum(m["correct"] for m in ms) / queries if queries else float("nan"),
            "transitions": len(errors) + sum(m["missed"] for m in ms),
            "missed": sum(m["missed"] for m in ms),
            "early": sum(x < 0 for x in errors),
            "regressions": sum(m["regressions"] for m in ms),
            "stuck_episodes": sum(m["stuck"] for m in ms),
        }
        if errors:
            out.update(
                switch_error_median=statistics.median(errors),
                abs_switch_error_median=statistics.median(abs(x) for x in errors),
                abs_switch_error_p90=float(np.percentile(np.abs(errors), 90)),
                switch_error_min=min(errors),
                switch_error_max=max(errors),
            )
        return out

    overall = totals(list(per_episode))
    per_task = {t: totals([e for e in per_episode if task_of[e] == t]) for t in sorted(set(task_of.values()))}
    gate = {
        "accuracy": overall["accuracy"] >= GATE_ACCURACY,
        # Every transition made, and made on time.
        "switch_timing": overall["missed"] == 0
        and overall.get("abs_switch_error_median", 0) <= GATE_MEDIAN_ABS_SWITCH_ERROR,
        "no_stuck_episodes": overall["stuck_episodes"] == 0,
    }
    return {"overall": overall, "per_task": per_task, "gate": gate, "gate_pass": all(gate.values())}


def read_episode_frames(dataset_dir: pathlib.Path, episode: int, every: int):
    """(frame index, client-style observation keys) for every `every`-th frame of an episode.

    Images are the stored 256x256 frames; the policy's transforms resize them exactly as they
    resize the client's.
    """
    from PIL import Image
    import pyarrow.parquet as pq

    info = json.loads((dataset_dir / "meta/info.json").read_text(encoding="utf-8"))
    chunk = episode // int(info["chunks_size"])
    path = dataset_dir / info["data_path"].format(episode_chunk=chunk, episode_index=episode)
    table = pq.read_table(path, columns=["image", "wrist_image", "state", "frame_index", "task_index"])
    frame_index = table.column("frame_index").to_numpy()
    if not np.array_equal(frame_index, np.arange(len(frame_index))):
        raise ValueError(f"{path}: frame_index is not 0..n-1")
    task_index = int(table.column("task_index")[0].as_py())
    rows = range(0, len(frame_index), every)

    def frames():
        for r in rows:
            row = table.slice(r, 1).to_pylist()[0]
            yield (
                r,
                {
                    "observation/image": np.asarray(Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")),
                    "observation/wrist_image": np.asarray(
                        Image.open(io.BytesIO(row["wrist_image"]["bytes"])).convert("RGB")
                    ),
                    "observation/state": np.asarray(row["state"], dtype=np.float32),
                },
            )

    return task_index, len(frame_index), frames()


def main(args: Args) -> dict:
    from openpi.glow import plan_pack as _plan_pack
    from openpi.policies import policy_config
    from openpi.training import config as _config

    dataset_dir = pathlib.Path(args.dataset_dir)
    cfg = _config.get_config(args.config)
    delta = cfg.data.plan_delta
    pack = _plan_pack.PlanPack.load(cfg.data.plan_pack_path)
    tasks = {}
    with open(dataset_dir / "meta/tasks.jsonl", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            tasks[int(entry["task_index"])] = entry["task"]

    policy = policy_config.create_trained_policy(cfg, args.checkpoint_dir, sample_kwargs={"num_steps": args.num_steps})
    start, stop = (int(x) for x in args.episodes.split(":"))
    per_episode, task_of, records = {}, {}, {}
    t0 = time.monotonic()
    n_queries = 0
    for ep in range(start, stop):
        task_index, n_frames, frames = read_episode_frames(dataset_dir, ep, delta)
        episode = pack.episodes[ep]
        if episode.task_index != task_index or episode.n_frames != n_frames:
            raise ValueError(
                f"episode {ep}: dataset (task {task_index}, {n_frames} frames) disagrees with the plan pack"
            )
        prompt = tasks[task_index]
        queries = run_episode(
            policy.infer,
            prompt,
            frames,
            lambda f, ep=ep: pack.plan_index(ep, f),
            delta=delta,
            prev_source=args.prev_source,
        )
        per_episode[ep] = episode_metrics(queries, episode.stage_starts, pack.task(prompt).num_plans)
        task_of[ep] = task_index
        records[ep] = [dataclasses.asdict(q) for q in queries]
        n_queries += len(queries)
        m = per_episode[ep]
        logging.info(
            f"episode {ep} task {task_index}: {m['correct']}/{m['queries']} switch_errors={m['switch_errors']} "
            f"missed={m['missed']} regressions={m['regressions']} stuck={m['stuck']} "
            f"({n_queries / (time.monotonic() - t0):.1f} queries/s)"
        )

    summary = summarize(per_episode, task_of)
    summary["args"] = dataclasses.asdict(args)
    summary["plan_delta"] = delta
    o = summary["overall"]
    logging.info(f"PLAN ACCURACY ({args.prev_source}): {json.dumps(o)}")
    for t, s in summary["per_task"].items():
        logging.info(f"  task {t}: accuracy={s['accuracy']:.4f} missed={s['missed']} stuck={s['stuck_episodes']}")
    logging.info(f"GATE: {json.dumps(summary['gate'])} -> {'PASS' if summary['gate_pass'] else 'FAIL'}")
    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({**summary, "queries": records}, indent=1), encoding="utf-8")
        logging.info(f"wrote {out}")
    return summary


if __name__ == "__main__":
    import tyro

    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
