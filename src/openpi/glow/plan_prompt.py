"""Data transforms for GLOW's plan clause and subtask target.

`GlowPlanPrompt` runs after the data transforms and before the stock tokenizer. Training samples
carry `episode_index` / `frame_index` (kept by the GLOW repack) and the active plan is looked up
in the pack; inference samples carry `plan_index` from the client's stage tracker. Either way it
rewrites `prompt` in place, so the tokenizer, model inputs and checkpoint format are untouched.

With probability `dropout_p` a training sample keeps the stock prompt and instead carries the
active stage's instruction as `glow_subtask`, which `GlowTokenizeSubtask` appends to the token
sequence after the prompt as the cross-entropy target. `dropout_p = 1` reproduces the stock
pipeline byte for byte; `dropout_p = 0` never drops.
"""

import dataclasses
import os

import numpy as np

from openpi.glow import plan_pack as _plan_pack
from openpi.models import tokenizer as _tokenizer
from openpi.transforms import DataDict
from openpi.transforms import DataTransformFn

_RNG: np.random.Generator | None = None
_RNG_PID: int | None = None


def _rng() -> np.random.Generator:
    """One generator per data-loader process, seeded from that process's torch seed."""
    global _RNG, _RNG_PID
    pid = os.getpid()
    if _RNG is None or _RNG_PID != pid:
        import torch

        _RNG = np.random.default_rng(int(torch.initial_seed()) % (2**63))
        _RNG_PID = pid
    return _RNG


@dataclasses.dataclass(frozen=True)
class GlowPlanPrompt(DataTransformFn):
    pack: _plan_pack.PlanPack
    # Probability that a TRAINING sample is served without the clause (the sample then carries
    # the subtask target instead). No default on purpose: every config states it.
    dropout_p: float
    anchors: bool = True
    lookahead_anchors: bool = False

    def __post_init__(self):
        if not 0.0 <= self.dropout_p <= 1.0:
            raise ValueError(f"dropout_p must be in [0, 1], got {self.dropout_p}")

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.get("prompt")) is None:
            raise ValueError("GlowPlanPrompt: prompt is required")
        if not isinstance(prompt, str):
            prompt = prompt.item()

        if "episode_index" in data and "frame_index" in data:
            episode_index = int(data.pop("episode_index"))
            frame_index = int(data.pop("frame_index"))
            plan_index = self.pack.plan_index(episode_index, frame_index)
            expected_task = self.pack.task(prompt).task_index
            if self.pack.episodes[episode_index].task_index != expected_task:
                raise ValueError(
                    f"episode {episode_index} belongs to task {self.pack.episodes[episode_index].task_index}, "
                    f"but its prompt {prompt!r} is task {expected_task}"
                )
            training = True
            dropped = bool(_rng().random() < self.dropout_p)
        elif "plan_index" in data:
            plan_index = int(data.pop("plan_index"))
            training = False
            dropped = plan_index == _plan_pack.PLAN_OFF
        else:
            raise ValueError(
                "GlowPlanPrompt: sample carries neither (episode_index, frame_index) nor plan_index. "
                "Training needs the GLOW repack; evaluation needs the LIBERO client run with "
                "--args.plan-pack (or --args.plan-off for the plan-off diagnostic)."
            )

        if not dropped:
            data["prompt"] = self.pack.render_clause(
                prompt, plan_index, anchors=self.anchors, lookahead_anchors=self.lookahead_anchors
            )
        elif training:
            data["glow_subtask"] = self.pack.subtask(prompt, plan_index)
        return data


@dataclasses.dataclass(frozen=True)
class GlowTokenizeSubtask(DataTransformFn):
    """Append the subtask target after the tokenized prompt, FAST-style, with ar/loss masks.

    Runs after `TokenizePrompt`. Samples without `glow_subtask` get all-zero masks, so every
    sample carries the same keys. The block occupies positions that are padding in the stock
    layout: the flow pass masks them out, the CE pass attends to them causally.
    """

    tokenizer: _tokenizer.PaligemmaTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        tokens = np.array(data["tokenized_prompt"])
        mask = np.array(data["tokenized_prompt_mask"], dtype=bool)
        max_len = len(tokens)
        ar_mask = np.zeros(max_len, dtype=np.int32)
        loss_mask = np.zeros(max_len, dtype=bool)

        subtask = data.pop("glow_subtask", None)
        if subtask is not None:
            if not isinstance(subtask, str):
                subtask = subtask.item()
            n_prompt = int(mask.sum())
            if not mask[:n_prompt].all():
                raise ValueError("GlowTokenizeSubtask: prompt tokens are not a contiguous prefix")
            ids = self.tokenizer.encode_answer(subtask)
            end = n_prompt + len(ids)
            if end > max_len:
                raise ValueError(
                    f"GlowTokenizeSubtask: prompt ({n_prompt}) + subtask ({len(ids)}) tokens exceed "
                    f"max_token_len {max_len}; refusing to truncate"
                )
            tokens[n_prompt:end] = ids
            mask[n_prompt:end] = True
            ar_mask[n_prompt:end] = 1
            loss_mask[n_prompt:end] = True

        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }
