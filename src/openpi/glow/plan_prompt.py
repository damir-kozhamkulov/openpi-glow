"""Data transforms for GLOW's plan state: the previous plan as input, the current plan as target.

`GlowPlanState` runs after the data transforms and before the stock tokenizer. It rewrites
`prompt` to "{task}. Was: {plan at t - delta}" and attaches the current plan as an answer block
"Now: {plan at t}", which `GlowTokenizePlan` appends after the tokenized prompt with causal
(ar) and loss masks. Both plans are rendered by `PlanPack.render_plan`.

Training samples carry `episode_index` / `frame_index` (kept by the GLOW repack): the input is
the labeled plan `delta` frames earlier (clipped to frame 0), replaced with probability
`corrupt_p` by a different stage of the same task, and the target is the labeled plan at the
frame. Corruption is what makes the target depend on the images: uncorrupted, the input differs
from the target on under 2% of LIBERO-10 frames at delta = 5, and copying it solves the loss.

Inference samples carry `prev_plan_index` (the model's own previous prediction, sent back by
the client) and optionally `oracle_plan_index`:
- with an oracle the block is that stage's plan, laid out exactly as in training;
- without one the sample carries all K candidate blocks in plan-index order, and
  `GlowTokenizePlan` lays K token rows (prompt + candidate k) side by side, K * max_token_len
  tokens, for the model to score (Observation's shape checks admit no extra axis).
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

INPUT_PREFIX = "Was: "
TARGET_PREFIX = "Now: "


def _rng() -> np.random.Generator:
    """One generator per data-loader process, seeded from that process's torch seed."""
    global _RNG, _RNG_PID
    pid = os.getpid()
    if _RNG is None or _RNG_PID != pid:
        import torch

        _RNG = np.random.default_rng(int(torch.initial_seed()) % (2**63))
        _RNG_PID = pid
    return _RNG


def _scalar(x):
    return x if isinstance(x, str | int) else np.asarray(x).item()


@dataclasses.dataclass(frozen=True)
class GlowPlanState(DataTransformFn):
    pack: _plan_pack.PlanPack
    # Frames between the input plan and the target plan: the client's replan interval, so the
    # input is what the model itself predicted at its previous query.
    delta: int
    # Probability that a TRAINING sample's input plan is replaced by another stage of its task.
    # No default on purpose: every config states it.
    corrupt_p: float
    nouns: bool = True
    anchors: bool = True

    def __post_init__(self):
        if self.delta < 1:
            raise ValueError(f"delta must be >= 1, got {self.delta}")
        if not 0.0 <= self.corrupt_p <= 1.0:
            raise ValueError(f"corrupt_p must be in [0, 1], got {self.corrupt_p}")

    def render(self, prompt: str, plan_index: int) -> str:
        return self.pack.render_plan(prompt, plan_index, nouns=self.nouns, anchors=self.anchors)

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.get("prompt")) is None:
            raise ValueError("GlowPlanState: prompt is required")
        prompt = _scalar(prompt)
        task = self.pack.task(prompt)

        if "episode_index" in data and "frame_index" in data:
            episode_index = int(_scalar(data.pop("episode_index")))
            frame_index = int(_scalar(data.pop("frame_index")))
            if self.pack.episodes[episode_index].task_index != task.task_index:
                raise ValueError(
                    f"episode {episode_index} belongs to task {self.pack.episodes[episode_index].task_index}, "
                    f"but its prompt {prompt!r} is task {task.task_index}"
                )
            target = self.pack.plan_index(episode_index, frame_index)
            prev = self.pack.plan_index(episode_index, max(0, frame_index - self.delta))
            if task.num_plans > 1 and _rng().random() < self.corrupt_p:
                others = [k for k in range(task.num_plans) if k != prev]
                prev = others[int(_rng().integers(len(others)))]
            data["glow_plan_target"] = TARGET_PREFIX + self.render(prompt, target)
        elif "prev_plan_index" in data:
            prev = int(_scalar(data.pop("prev_plan_index")))
            oracle = data.pop("oracle_plan_index", None)
            if oracle is not None:
                data["glow_plan_target"] = TARGET_PREFIX + self.render(prompt, int(_scalar(oracle)))
            else:
                data["glow_plan_candidates"] = [TARGET_PREFIX + self.render(prompt, k) for k in range(task.num_plans)]
        else:
            raise ValueError(
                "GlowPlanState: sample carries neither (episode_index, frame_index) nor prev_plan_index. "
                "Training needs the GLOW repack; evaluation needs the LIBERO client run with "
                "--args.plan-pack."
            )

        data["prompt"] = f"{prompt}. {INPUT_PREFIX}{self.render(prompt, prev)}"
        return data


@dataclasses.dataclass(frozen=True)
class GlowTokenizePlan(DataTransformFn):
    """Append the plan block after the tokenized prompt, with causal (ar) and loss masks.

    Runs after `TokenizePrompt`. A target block yields the usual max_token_len arrays; K candidate
    blocks yield K * max_token_len, row k (tokens k * max_token_len onward) = prompt + candidate
    k. The block occupies
    positions that are padding in the stock layout. Never truncates: a prompt that filled
    max_token_len or a block that does not fit raises.
    """

    tokenizer: _tokenizer.PaligemmaTokenizer

    def _append(self, tokens: np.ndarray, mask: np.ndarray, block: str):
        tokens, mask = tokens.copy(), mask.copy()
        max_len = len(tokens)
        ar_mask = np.zeros(max_len, dtype=np.int32)
        loss_mask = np.zeros(max_len, dtype=bool)
        n_prompt = int(mask.sum())
        if not mask[:n_prompt].all():
            raise ValueError("GlowTokenizePlan: prompt tokens are not a contiguous prefix")
        if n_prompt >= max_len:
            raise ValueError(f"GlowTokenizePlan: prompt fills max_token_len {max_len} (possibly truncated)")
        ids = self.tokenizer.encode_answer(block)
        end = n_prompt + len(ids)
        if end > max_len:
            raise ValueError(
                f"GlowTokenizePlan: prompt ({n_prompt}) + plan block ({len(ids)}) tokens exceed "
                f"max_token_len {max_len}; refusing to truncate"
            )
        tokens[n_prompt:end] = ids
        mask[n_prompt:end] = True
        ar_mask[n_prompt:end] = 1
        loss_mask[n_prompt:end] = True
        return tokens, mask, ar_mask, loss_mask

    def __call__(self, data: DataDict) -> DataDict:
        tokens = np.asarray(data["tokenized_prompt"])
        mask = np.asarray(data["tokenized_prompt_mask"], dtype=bool)
        target = data.pop("glow_plan_target", None)
        candidates = data.pop("glow_plan_candidates", None)
        if (target is None) == (candidates is None):
            raise ValueError("GlowTokenizePlan: expected exactly one of a plan target or plan candidates")
        if target is not None:
            out = self._append(tokens, mask, _scalar(target))
        else:
            rows = [self._append(tokens, mask, _scalar(c)) for c in candidates]
            out = tuple(np.concatenate(col) for col in zip(*rows, strict=True))
        return {
            **data,
            "tokenized_prompt": out[0],
            "tokenized_prompt_mask": out[1],
            "token_ar_mask": out[2],
            "token_loss_mask": out[3],
        }
