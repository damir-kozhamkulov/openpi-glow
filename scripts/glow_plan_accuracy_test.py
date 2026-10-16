"""glow_plan_accuracy: the metrics, the query loop, the episode reader and the wiring of main().

The reader and main() tests need the local LeRobot copy (HF_LEROBOT_HOME with repo
`libero_staging`) and the plan pack (glow_repo/derived/plan_pack.json); skipped otherwise.
main() runs with a stand-in policy that copies its input plan - the copy collapse the gate exists
to catch - so no checkpoint is needed.
"""

import os
import pathlib

import numpy as np
import pytest

from . import glow_plan_accuracy as gpa

REPO = pathlib.Path(__file__).resolve().parents[2]  # glow_repo (openpi is its submodule)
PACK = REPO / "derived/plan_pack.json"
DATASET = pathlib.Path(os.environ.get("HF_LEROBOT_HOME", "")) / "libero_staging"


def _queries(plans, starts, delta=5):
    def label(f):
        return max(k for k, s in enumerate(starts) if s <= f)

    out, prev = [], 0
    for i, plan in enumerate(plans):
        out.append(gpa.Query(frame=i * delta, label=label(i * delta), prev=prev, plan=plan, logp=(0.0,)))
        prev = plan
    return out


def test_episode_metrics():
    starts = (0, 23)
    # Queries at 0, 5, ..., 45; the label switches at 23 (first labeled query 25).
    on_time = gpa.episode_metrics(_queries([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], starts), starts, 2)
    assert on_time == {
        "queries": 10,
        "correct": 10,
        "switch_errors": [2],
        "missed": 0,
        "regressions": 0,
        "stuck": False,
    }
    early = gpa.episode_metrics(_queries([0, 0, 0, 0, 1, 1, 0, 1, 1, 1], starts), starts, 2)
    assert early["switch_errors"] == [-3] and early["regressions"] == 1 and early["correct"] == 8
    copy = gpa.episode_metrics(_queries([0] * 10, starts), starts, 2)
    assert copy["missed"] == 1 and copy["stuck"] and copy["switch_errors"] == [] and copy["correct"] == 5
    single = gpa.episode_metrics(_queries([0] * 4, (0,)), (0,), 1)
    assert single["missed"] == 0 and not single["stuck"] and single["correct"] == 4


def test_summarize_and_gate():
    starts = (0, 23)
    good = gpa.episode_metrics(_queries([0] * 5 + [1] * 5, starts), starts, 2)
    copy = gpa.episode_metrics(_queries([0] * 10, starts), starts, 2)
    s = gpa.summarize({0: good, 1: good, 2: good}, {0: 0, 1: 0, 2: 1})
    assert s["gate_pass"] and s["overall"]["accuracy"] == 1.0 and s["overall"]["abs_switch_error_median"] == 2
    assert set(s["per_task"]) == {0, 1} and s["per_task"][1]["episodes"] == 1
    s = gpa.summarize({0: good, 1: copy}, {0: 0, 1: 0})
    assert not s["gate_pass"]
    assert s["gate"] == {"accuracy": False, "switch_timing": False, "no_stuck_episodes": False}
    assert s["overall"]["missed"] == 1 and s["overall"]["transitions"] == 2


def test_run_episode_prev_sources():
    labels = {0: 0, 5: 0, 10: 1, 15: 1}.get
    frames = [(f, {"observation/state": np.zeros(8)}) for f in (0, 5, 10, 15)]
    sent = []

    def infer(element):
        sent.append(element["prev_plan_index"])
        return {"plan_choice": np.asarray(1), "plan_logp": np.asarray([-3.0, -0.1])}

    qs = gpa.run_episode(infer, "task", iter(frames), labels, delta=5, prev_source="self")
    assert sent == [0, 1, 1, 1] and [q.plan for q in qs] == [1, 1, 1, 1] and qs[0].logp == (-3.0, -0.1)
    assert [q.label for q in qs] == [0, 0, 1, 1]
    sent.clear()
    gpa.run_episode(infer, "task", iter(frames), labels, delta=5, prev_source="label")
    assert sent == [0, 0, 0, 1]  # the label 5 frames earlier, clipped to frame 0


@pytest.mark.skipif(not (DATASET / "meta/info.json").exists(), reason="local LeRobot copy not present")
def test_read_episode_frames():
    task_index, n, frames = gpa.read_episode_frames(DATASET, 0, 5)
    frames = list(frames)
    assert task_index == 0 and n == 214 and len(frames) == 43 and frames[-1][0] == 210
    first = frames[0][1]
    assert first["observation/image"].shape == (256, 256, 3) and first["observation/image"].dtype == np.uint8
    assert first["observation/wrist_image"].shape == (256, 256, 3) and first["observation/state"].shape == (8,)


@pytest.mark.skipif(
    not ((DATASET / "meta/info.json").exists() and PACK.exists()), reason="local LeRobot copy or plan pack not present"
)
def test_main_flags_copy_collapse(monkeypatch, tmp_path):
    from openpi.policies import policy_config

    class CopyPolicy:
        """Echoes its input plan: never advances."""

        def infer(self, element):
            return {"plan_choice": np.asarray(element["prev_plan_index"]), "plan_logp": np.zeros(2)}

    monkeypatch.setenv("OPENPI_PLAN_PACK", str(PACK))
    monkeypatch.setattr(policy_config, "create_trained_policy", lambda *a, **k: CopyPolicy())
    out = tmp_path / "acc.json"
    s = gpa.main(gpa.Args(checkpoint_dir="unused", dataset_dir=str(DATASET), episodes="0:3", out=str(out)))
    o = s["overall"]
    assert o["episodes"] == 3 and o["missed"] == o["transitions"] == 3 and o["stuck_episodes"] == 3
    assert not s["gate_pass"] and out.exists()


NORM_STATS = REPO / "data/assets/pi05_libero/physical-intelligence/libero/norm_stats.json"


@pytest.mark.skipif(
    not ((DATASET / "meta/info.json").exists() and PACK.exists() and NORM_STATS.exists()),
    reason="local LeRobot copy, plan pack or norm_stats not present",
)
def test_serving_path_end_to_end(monkeypatch, tmp_path):
    """A dummy-size GLOW checkpoint with random weights, served by create_trained_policy exactly as
    an evaluation server does (transforms, bf16 cast, candidate scoring, output transforms)."""
    import dataclasses
    import shutil

    import safetensors.torch
    from transformers.models.auto import CONFIG_MAPPING

    import openpi.models_pytorch.gemma_pytorch as gemma_pytorch
    import openpi.shared.download as download
    import openpi.training.config as _config

    orig = CONFIG_MAPPING["paligemma"]

    def tiny():
        cfg = orig()
        cfg.vision_config.hidden_size = 32
        cfg.vision_config.num_hidden_layers = 2
        cfg.vision_config.num_attention_heads = 4
        cfg.vision_config.patch_size = 56
        return cfg

    monkeypatch.setattr(gemma_pytorch, "CONFIG_MAPPING", {**CONFIG_MAPPING, "paligemma": tiny})
    monkeypatch.setenv("OPENPI_PLAN_PACK", str(PACK))
    # Windows paths read as URL schemes ("w:") to these two; the cluster never takes either path.
    real_download = download.maybe_download
    monkeypatch.setattr(
        download, "maybe_download", lambda p, **k: real_download(p, **k) if "://" in str(p) else pathlib.Path(p)
    )
    monkeypatch.setattr(_config.DataConfigFactory, "_load_norm_stats", lambda self, d, a: None)

    real = _config.get_config("pi05_libero_glow_unified")
    cfg = dataclasses.replace(
        real,
        model=dataclasses.replace(
            real.model, paligemma_variant="dummy", action_expert_variant="dummy", pytorch_compile_mode=None
        ),
    )
    monkeypatch.setattr(_config, "get_config", lambda name: cfg)
    ckpt = tmp_path / "ckpt"
    (ckpt / "assets/physical-intelligence/libero").mkdir(parents=True)
    shutil.copy(NORM_STATS, ckpt / "assets/physical-intelligence/libero/norm_stats.json")
    import torch

    torch.manual_seed(0)
    safetensors.torch.save_model(cfg.model.create_pytorch(), str(ckpt / "model.safetensors"))

    out = tmp_path / "acc.json"
    s = gpa.main(gpa.Args(checkpoint_dir=str(ckpt), dataset_dir=str(DATASET), episodes="0:1", out=str(out)))
    o = s["overall"]
    assert o["episodes"] == 1 and o["queries"] == 43  # 214 frames, every 5th
    assert 0.0 <= o["accuracy"] <= 1.0 and o["transitions"] == 1
    import json

    queries = json.loads(out.read_text(encoding="utf-8"))["queries"]["0"]
    # Both of task 0's stages were scored at every query, and the choice is their argmax.
    assert all(len(q["logp"]) == 2 and q["plan"] == int(np.argmax(q["logp"])) for q in queries)
    assert all(np.isfinite(q["logp"]).all() and max(q["logp"]) < 0 for q in queries)
