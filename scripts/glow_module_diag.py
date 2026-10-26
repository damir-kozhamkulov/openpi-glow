"""Why a GLOW module model's actions read (or ignore) its encoder tokens. Forward only.

Three measurements on one checkpoint:

  collapse    The encoders' tokens for every (task, stage) of the plan pack, no images needed.
              Per token slot: mean pairwise cosine between distinct modules, the spread (mean
              distance of a module's token from the slot's mean, relative to its norm) and the
              closest pair (min 1 - cos). Spread ~0 / cosine ~1 means the slot is the same for
              every task: the encoder carries nothing. Also for LIBERO-10 and LIBERO-Goal apart.
  attention   Attention weights captured inside the joint prefix + suffix forward (the flow-loss
              forward), averaged over heads, per layer, grouped by query (images, module,
              language, actions) and key (the same groups; padding excluded), for every flow
              variant. Reports each key group's attention mass, its per-token mass relative to an
              average prefix token, and for the action queries a value-norm-weighted share
              (mass x ||v||). A plan-prompt checkpoint (pi05_libero_plan_prompt) has no module
              group; its language group is the plan text.
  flow        Paired flow loss on identical samples, noise, time and augmentation, one prefix per
              variant (FLOW_VARIANTS; --variants picks some):
                correct              the model's own prefix: module tokens before BOS, rescaled
                                     as its module_token_scale says (the language tokens' RMS,
                                     or its module text's word-token RMS)
                no_module            the module tokens removed (the empty prompt stays)
                task_prompt          no module tokens, the stock task-string prompt instead
                plan_text            no module tokens, the active plan as the whole prompt (as
                                     pi05_libero_plan_prompt renders it): the text upper bound
                post_bos             the module tokens right after BOS instead of before it
                word_scale           before BOS, rescaled to the prompt's word-token RMS instead
                                     (the language group of the empty prompt is BOS and newline)
                post_bos_word_scale  both
                head_pre_bos         instead of the encoders, one real word embedding per slot
                head_post_bos        (head_words), at a language token's scale, before / after BOS
              At the fine-tuning init (`--init`) correct - no_module is the cost of fresh encoder
              tokens and no_module - task_prompt that of dropping the task string; a variant that
              lands near no_module is harmless at init, one below it on LIBERO-Goal carries the
              goal.

`--init` loads the weights non-strictly for the model's NEW_PARAM_PREFIXES (as the trainer's
fine-tuning init does), seeding the fresh encoders with `--init-seed`.

  python scripts/glow_module_diag.py --weights-dir <dir with model.safetensors> \
      --norm-stats-dir <assets dir holding physical-intelligence/libero/norm_stats.json> \
      --dataset-root <lerobot repo dir> --episodes 200:379,600:807 --out module_diag.json
"""

import contextlib
import dataclasses
import json
import logging
import math
import pathlib
import re
import sys
import time
import types

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import glow_flow_probe as _flow_probe  # noqa: E402

GROUPS = ("images", "module", "language", "actions")
# name -> (prompt, prefix): prompt "empty" (the module config's BOS + newline), "task" (the task
# string) or "plan" (the plan text, as pi05_libero_plan_prompt renders it); prefix "own" (the
# model's), "stock" (no module tokens) or (tokens, placement, scale) for make_prefix.
FLOW_VARIANTS = {
    "correct": ("empty", "own"),
    "no_module": ("empty", "stock"),
    "task_prompt": ("task", "stock"),
    "plan_text": ("plan", "stock"),
    "post_bos": ("empty", ("module", "post_bos", "language")),
    "word_scale": ("empty", ("module", "pre_bos", "word")),
    "post_bos_word_scale": ("empty", ("module", "post_bos", "word")),
    "head_pre_bos": ("empty", ("head", "pre_bos", "raw")),
    "head_post_bos": ("empty", ("head", "post_bos", "raw")),
}
PREDICATE_WORDS = {"on": "on", "in": "in", "open": "open", "close": "close", "turnon": "on", "turnoff": "off"}
COUNT_WORDS = ("one", "two", "three", "four")
ORDINAL_WORDS = ("first", "second", "third", "fourth")
SPATIAL_SLOTS = ("OP", "OBJ", "TGT", "CTX")
TEMPORAL_SLOTS = ("T_SUMMARY", "T_ACTIVE")
# LIBERO-10 is task indices 0-9 and LIBERO-Goal 10-19 in physical-intelligence/libero.
SUITE_SIZE = 10
TASK_PROMPT_LEN = 48
# The config whose renderer and prompt length the plan_text variant uses.
PLAN_TEXT_CONFIG = "pi05_libero_plan_prompt"


@dataclasses.dataclass
class Args:
    weights_dir: str
    # Assets directory whose <repo_id>/norm_stats.json normalizes state and actions.
    norm_stats_dir: str
    # LeRobot dataset root: the directory holding meta/ and data/.
    dataset_root: str
    config: str = "pi05_libero_glow_st"
    repo_id: str = "physical-intelligence/libero"
    # Episodes, as OPENPI_TRAIN_EPISODES.
    episodes: str = "0:807"
    num_samples: int = 256
    batch_size: int = 4
    seed: int = 0
    # The weights are a fine-tuning init without the encoders (e.g. pi05_libero_ref_pytorch).
    init: bool = False
    init_seed: int = 0
    # FLOW_VARIANTS to run (default: all that apply; a model without module tokens runs `correct`).
    variants: tuple[str, ...] = ()
    precision: str = "float32"
    device: str = "cuda"
    out: str | None = None


def slot_names(model) -> list[str]:
    names = []
    if getattr(model, "glow_spatial_encoder", None) is not None:
        names += SPATIAL_SLOTS
    if getattr(model, "glow_temporal_encoder", None) is not None:
        names += TEMPORAL_SLOTS
    return names


def collapse_stats(tokens: np.ndarray) -> dict:
    """tokens [M, D] (one slot, M distinct modules) -> cosine / spread / closest-pair stats."""
    t = tokens / np.linalg.norm(tokens, axis=-1, keepdims=True).clip(min=1e-12)
    cos = t @ t.T
    off = ~np.eye(len(t), dtype=bool)
    mean = t.mean(axis=0)
    return {
        "modules": len(t),
        "mean_cos": float(cos[off].mean()),
        "min_1_minus_cos": float((1.0 - cos[off]).min()),
        "spread": float(np.linalg.norm(t - mean, axis=-1).mean()),
    }


def module_collapse(model, module, device) -> dict:
    """The encoders' tokens for every (task, stage) of the pack, compared across modules."""
    import torch

    keys, arrays = [], []
    for task in sorted(module.pack.tasks.values(), key=lambda t: t.task_index):
        for k in range(task.num_plans):
            keys.append((task.task_index, k))
            arrays.append(module.encode(task.task, k))
    batch = {name: torch.as_tensor(np.stack([a[name] for a in arrays])).to(device) for name in arrays[0]}
    with torch.no_grad():
        tokens = model.module_tokens(types.SimpleNamespace(**batch)).float().cpu().numpy()
    suites = np.asarray([t // SUITE_SIZE for t, _ in keys])
    out = {"modules": [list(k) for k in keys], "slots": {}}
    for s, name in enumerate(slot_names(model)):
        out["slots"][name] = {
            "all": collapse_stats(tokens[:, s]),
            "libero_10": collapse_stats(tokens[suites == 0, s]),
            "libero_goal": collapse_stats(tokens[suites == 1, s]),
        }
    return out


class AttentionRecorder:
    """Accumulates grouped attention mass from eager_attention_forward calls, one per layer."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        g = len(GROUPS)
        self.mass = np.zeros((num_layers, g, g))  # [layer, query group, key group], summed over queries
        self.vshare = np.zeros((num_layers, g))  # action queries, value-norm-weighted share per key group
        self.queries = np.zeros(g)  # valid queries per group, summed over samples
        self.keys = np.zeros(g)  # valid keys per group, summed over samples
        self.samples = 0
        self.layer = 0
        self.prefix = None  # [B, P] group id per prefix position (-1 = padding), set by the prefix hook
        self.groups = None  # [B, T] the same over prefix + suffix, built at the first layer

    def record(self, weights, value):
        import torch

        if self.groups is None:
            if self.prefix is None:
                return
            # The suffix (the action tokens) follows the prefix in the joint sequence.
            suffix = torch.full(
                (self.prefix.shape[0], weights.shape[-1] - self.prefix.shape[1]),
                GROUPS.index("actions"),
                dtype=self.prefix.dtype,
                device=self.prefix.device,
            )
            self.groups = torch.cat([self.prefix, suffix], dim=1)
            self.prefix = None
            self.layer = 0
            g = self.groups.cpu().numpy()
            for i in range(len(GROUPS)):
                self.queries[i] += (g == i).sum()
                self.keys[i] += (g == i).sum()
            self.samples += g.shape[0]
        groups = self.groups.to(weights.device)
        onehot = torch.stack([(groups == i) for i in range(len(GROUPS))], dim=-1).to(torch.float32)  # [B, T, G]
        w = weights.float().mean(dim=1)  # [B, Q, K], heads averaged
        key_mass = w @ onehot  # [B, Q, G]
        self.mass[self.layer] += torch.einsum("bqg,bqh->gh", onehot, key_mass).cpu().numpy()
        # Value norms: one KV head in pi05, so value is [B, 1, K, d].
        vnorm = value.float().norm(dim=-1).mean(dim=1)  # [B, K]
        contrib = (w * vnorm[:, None, :]) @ onehot  # [B, Q, G]
        contrib = contrib / contrib.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        actions = onehot[..., GROUPS.index("actions")]  # [B, Q]
        self.vshare[self.layer] += torch.einsum("bq,bqg->g", actions, contrib).cpu().numpy()
        self.layer = (self.layer + 1) % self.num_layers

    def summary(self) -> dict:
        mass = self.mass / self.queries[None, :, None].clip(min=1)
        vshare = self.vshare / self.queries[GROUPS.index("actions")].clip(min=1)
        keys_per_sample = self.keys / max(self.samples, 1)
        prefix = [GROUPS.index(g) for g in ("images", "module", "language")]
        out = {"keys_per_sample": dict(zip(GROUPS, keys_per_sample.tolist(), strict=True)), "by_query": {}}
        for qi, q in enumerate(GROUPS):
            if self.queries[qi] == 0:
                continue
            m = mass[:, qi]  # [layer, key group]
            prefix_mass = m[:, prefix].sum(-1)
            prefix_keys = keys_per_sample[prefix].sum()
            per_token = {}
            for ki, k in enumerate(GROUPS):
                if ki in prefix and keys_per_sample[ki] > 0:
                    # Mass per token of this group over the mass per token of the whole prefix.
                    per_token[k] = (
                        (m[:, ki] / keys_per_sample[ki]) / (prefix_mass / prefix_keys).clip(min=1e-12)
                    ).tolist()
            out["by_query"][q] = {
                "mass_per_layer": {k: m[:, ki].tolist() for ki, k in enumerate(GROUPS)},
                "mass_mean": {k: float(m[:, ki].mean()) for ki, k in enumerate(GROUPS)},
                "per_token_vs_prefix_per_layer": per_token,
                "per_token_vs_prefix_mean": {k: float(np.mean(v)) for k, v in per_token.items()},
            }
        out["actions_value_weighted_share_per_layer"] = {k: vshare[:, ki].tolist() for ki, k in enumerate(GROUPS)}
        out["actions_value_weighted_share_mean"] = {k: float(vshare[:, ki].mean()) for ki, k in enumerate(GROUPS)}
        return out


@contextlib.contextmanager
def capture_attention(recorder: AttentionRecorder):
    """Route the joint forward's eager attention through `recorder`."""
    from openpi.models_pytorch import gemma_pytorch

    mg = gemma_pytorch.modeling_gemma
    original = mg.eager_attention_forward

    def eager(module, query, key, value, attention_mask, scaling, *args, **kwargs):
        out, weights = original(module, query, key, value, attention_mask, scaling, *args, **kwargs)
        recorder.record(weights, value)
        return out, weights

    mg.eager_attention_forward = eager
    try:
        yield
    finally:
        mg.eager_attention_forward = original


def head_words(task, stage_index: int, slots: list[str]) -> list[str | None]:
    """One real word per module slot, for the head-token variants: the subgoal's verb (OP), the
    head nouns of the object (OBJ) and target (TGT, None without one), the predicate (CTX), and
    the stage count (T_SUMMARY) and ordinal (T_ACTIVE) as words."""
    stage = task.stages[stage_index]

    def head(noun: str) -> str | None:
        words = re.sub(r"\(.*?\)", " ", noun).replace("_", " ").split()
        return words[-1].lower() if words else None

    verb = stage.subgoal_operation.split()
    words = {
        "OP": verb[0].lower() if verb else None,
        "OBJ": head(stage.object_noun),
        "TGT": head(stage.target_noun),
        "CTX": PREDICATE_WORDS.get(stage.predicate),
        "T_SUMMARY": COUNT_WORDS[task.num_plans - 1],
        "T_ACTIVE": ORDINAL_WORDS[stage_index],
    }
    return [words[slot] for slot in slots]


def word_rms(embed, tokens, mask, scale: float, device):
    """Mean RMS [B] of a prompt's word-token embeddings as the prefix carries them (x `scale`):
    the valid tokens without the leading BOS and the closing newline."""
    import torch

    tokens = torch.as_tensor(tokens)
    words = torch.as_tensor(mask).clone().bool()
    words[:, 0] = False
    last = words.long().cumsum(1).argmax(1)  # the last valid position: the newline
    words[torch.arange(words.shape[0]), last] = False
    rms = (embed(tokens.to(device)).float() * scale).pow(2).mean(-1).sqrt()
    w = words.to(rms.device).float()
    return (rms * w).sum(1) / w.sum(1).clamp(min=1)


def make_prefix(model, recorder, *, tokens=None, where="pre_bos", scale="language", target_rms=None, own=0):
    """An embed_observation_prefix for one variant, which hands its position groups to
    `recorder` and leaves the per-group embedding RMS on the function (`.rms`).

    own > 0: the model's own prefix with its `own` module tokens (before BOS). Otherwise the
    stock prefix (images + language), with `tokens(observation)` [B, N, D] inserted before BOS
    (`where="pre_bos"`, after the images) or right after it (`"post_bos"`), each token rescaled
    to the language tokens' pooled RMS as the model does (`scale="language"`), to `target_rms`
    [B] (`"word"`), or left as given (`"raw"`); `tokens=None` inserts nothing.
    """
    import torch

    own_prefix = type(model).embed_observation_prefix

    def prefix(observation, images, img_masks, lang_tokens, lang_masks):
        n_lang = lang_tokens.shape[1]
        if own:
            embs, pad_masks, att_masks = own_prefix(model, observation, images, img_masks, lang_tokens, lang_masks)
            n_mod, at = own, embs.shape[1] - n_lang - own
        else:
            embs, pad_masks, att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
            at = n_img = embs.shape[1] - n_lang
            n_mod = 0
            if tokens is not None:
                t = tokens(observation).float()
                if scale != "raw":
                    if scale == "language":
                        lang = embs[:, n_img:].float()
                        valid = lang_masks.float()[..., None]
                        rms = torch.sqrt(
                            (lang.pow(2).mean(-1, keepdim=True) * valid).sum(1, keepdim=True)
                            / valid.sum(1, keepdim=True).clamp(min=1.0)
                        )
                    elif scale == "word":
                        rms = target_rms.to(t.device).float()[:, None, None]
                    else:
                        raise ValueError(f"unknown scale {scale!r}")
                    t = t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6) * rms
                if where == "post_bos":
                    at = n_img + 1
                elif where != "pre_bos":
                    raise ValueError(f"unknown placement {where!r}")
                n_mod = t.shape[1]
                b = embs.shape[0]
                embs = torch.cat([embs[:, :at], t.to(embs.dtype), embs[:, at:]], dim=1)
                pad_masks = torch.cat([pad_masks[:, :at], pad_masks.new_ones(b, n_mod), pad_masks[:, at:]], dim=1)
                att_masks = torch.cat([att_masks[:, :at], att_masks.new_zeros(b, n_mod), att_masks[:, at:]], dim=1)
        n_img = embs.shape[1] - n_lang - n_mod
        groups = torch.full(pad_masks.shape, GROUPS.index("language"), dtype=torch.long, device=pad_masks.device)
        groups[:, :n_img] = GROUPS.index("images")
        groups[:, at : at + n_mod] = GROUPS.index("module")
        groups = torch.where(pad_masks.bool(), groups, torch.full_like(groups, -1))
        rms_by_group = {}
        e = embs.float()
        for name in ("images", "module", "language"):
            sel = groups == GROUPS.index(name)
            if sel.any():
                rms_by_group[name] = float(e[sel].pow(2).mean(-1).sqrt().mean())
        prefix.rms = rms_by_group
        recorder.prefix = groups
        return embs, pad_masks, att_masks

    prefix.rms = None
    return prefix


def suite_summary(flows: dict[str, np.ndarray], tasks: np.ndarray) -> dict:
    """Mean flow per variant over all samples and over LIBERO-10 / LIBERO-Goal apart."""
    suites = {
        "all": np.ones(len(tasks), dtype=bool),
        "libero_10": tasks < SUITE_SIZE,
        "libero_goal": tasks >= SUITE_SIZE,
    }
    return {
        name: {
            suite: {"flow": float(f[sel].mean()), "samples": int(sel.sum())}
            for suite, sel in suites.items()
            if sel.any()
        }
        for name, f in flows.items()
    }


def load_model(cfg, args, device):
    import safetensors.torch
    import torch

    model_cfg = cfg.model
    object.__setattr__(model_cfg, "dtype", args.precision)
    torch.manual_seed(args.init_seed)
    model = model_cfg.create_pytorch().to(device)
    path = str(pathlib.Path(args.weights_dir) / "model.safetensors")
    prefixes = tuple(getattr(model, "NEW_PARAM_PREFIXES", ()))
    if args.init:
        if not prefixes:
            raise ValueError(f"--init needs a model with NEW_PARAM_PREFIXES; {args.config} has none")
        missing, unexpected = safetensors.torch.load_model(model, path, strict=False, device=str(device))
        stray = [k for k in missing if not k.startswith(prefixes)]
        if stray or unexpected:
            raise RuntimeError(
                f"{path} does not match outside {prefixes}: missing {stray[:5]}, unexpected {unexpected[:5]}"
            )
        logging.info(f"init: {len(missing)} tensors under {prefixes} fresh (seed {args.init_seed})")
    else:
        safetensors.torch.load_model(model, path, device=str(device))
    model.eval()
    return model


def main(args: Args) -> dict:
    import jax
    from lerobot.common.datasets import lerobot_dataset
    import torch

    from openpi import transforms as _transforms
    from openpi.glow import module_inputs as _mi
    from openpi.models import model as _model
    from openpi.models import tokenizer as _tokenizer
    from openpi.training import config as _config
    from openpi.training import data_loader as _data_loader

    cfg = _config.get_config(args.config)
    assets = _config.AssetsConfig(assets_dir=args.norm_stats_dir, asset_id=args.repo_id)
    dc = dataclasses.replace(cfg.data, assets=assets).create(cfg.assets_dirs, cfg.model)
    if dc.norm_stats is None:
        raise ValueError(f"no norm_stats under {args.norm_stats_dir}/{args.repo_id}")
    modules = [t for t in dc.model_transforms.inputs if isinstance(t, _mi.GlowModuleInputs)]
    module = modules[0] if modules else None

    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = load_model(cfg, args, device)
    num_layers = model.paligemma_with_expert.paligemma.config.text_config.num_hidden_layers

    result = {"args": dataclasses.asdict(args)}
    if module is not None:
        result["collapse"] = module_collapse(model, module, device)
        for name, s in result["collapse"]["slots"].items():
            a, g = s["all"], s["libero_goal"]
            logging.info(
                f"COLLAPSE {name:9s} all: cos {a['mean_cos']:.3f} spread {a['spread']:.3f} "
                f"closest {a['min_1_minus_cos']:.2e} | goal: cos {g['mean_cos']:.3f} spread {g['spread']:.3f} "
                f"closest {g['min_1_minus_cos']:.2e}"
            )

    episodes = _data_loader.resolve_train_episodes(args.episodes)
    meta = lerobot_dataset.LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    # The smallest 0..n-1 prefix holding the episodes (what the pinned LeRobot can load).
    raw = lerobot_dataset.LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        episodes=list(range(episodes[-1] + 1)),
        delta_timestamps={
            key: [t / meta.fps for t in range(cfg.model.action_horizon)] for key in dc.action_sequence_keys
        },
    )
    rows = _data_loader.episode_frame_indices(raw.episode_data_index, episodes)
    raw = _data_loader.TransformedDataset(raw, [_transforms.PromptFromLeRobotTask(meta.tasks)])
    pipeline = _transforms.compose(
        [
            *dc.repack_transforms.inputs,
            *dc.data_transforms.inputs,
            _transforms.Normalize(dc.norm_stats, use_quantiles=dc.use_quantile_norm),
            *dc.model_transforms.inputs,
        ]
    )
    task_tokenizer = _tokenizer.PaligemmaTokenizer(TASK_PROMPT_LEN)

    if module is None:
        if args.variants:
            raise ValueError(f"{args.config} has no module tokens: only `correct` applies")
        variants = ("correct",)
    else:
        variants = tuple(args.variants) or tuple(FLOW_VARIANTS)
        if "correct" not in variants:
            variants = ("correct", *variants)
    unknown = [v for v in variants if v not in FLOW_VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants {unknown}; choose from {list(FLOW_VARIANTS)}")
    needs_plan = "plan" in {FLOW_VARIANTS[v][0] for v in variants}
    if needs_plan:
        plan_cfg = _config.get_config(PLAN_TEXT_CONFIG)
        plan_tokenizer = _tokenizer.PaligemmaTokenizer(plan_cfg.model.max_token_len)
    slots = slot_names(model) if module is not None else []
    embed = model.paligemma_with_expert.embed_language_tokens
    embed_scale = math.sqrt(model.paligemma_with_expert.paligemma.config.text_config.hidden_size)

    rng = np.random.default_rng(args.seed)
    n = args.num_samples - args.num_samples % args.batch_size
    indices = rng.choice(rows, size=n, replace=False)

    def _to_device(x):
        t = torch.as_tensor(x)
        # Normalize can promote state to float64 (numpy 2 promotion); the model computes in float32.
        return t.to(device, torch.float32) if t.dtype == torch.float64 else t.to(device)

    recorders = {v: AttentionRecorder(num_layers) for v in variants}
    flows = {v: [] for v in variants}
    rms = {v: [] for v in variants}
    tasks = []
    t0 = time.monotonic()
    for b in range(n // args.batch_size):
        idx = indices[b * args.batch_size : (b + 1) * args.batch_size]
        frames = [raw[int(i)] for i in idx]
        samples = [pipeline(dict(f)) for f in frames]
        tasks += [int(np.asarray(f["task_index"])) for f in frames]
        batch = _flow_probe._collate(samples)  # noqa: SLF001
        prompts = [task_tokenizer.tokenize(f["prompt"]) for f in frames]
        task_tokens = np.stack([p[0] for p in prompts])
        task_mask = np.stack([p[1] for p in prompts])
        batches = {
            "empty": batch,
            "task": {**batch, "tokenized_prompt": task_tokens, "tokenized_prompt_mask": task_mask},
        }
        token_sources = {}
        target_rms = None
        if module is not None:
            stages = [
                (
                    module.pack.task(f["prompt"]),
                    module.pack.plan_index(int(np.asarray(f["episode_index"])), int(np.asarray(f["frame_index"]))),
                )
                for f in frames
            ]
            if needs_plan:
                plans = [
                    plan_tokenizer.tokenize(
                        module.pack.render_plan(
                            task.task, k, nouns=plan_cfg.data.plan_nouns, anchors=plan_cfg.data.plan_anchors
                        )
                    )
                    for task, k in stages
                ]
                batches["plan"] = {
                    **batch,
                    "tokenized_prompt": np.stack([p[0] for p in plans]),
                    "tokenized_prompt_mask": np.stack([p[1] for p in plans]),
                }
            head_ids = np.zeros((len(frames), len(slots)), dtype=np.int64)
            head_valid = np.zeros((len(frames), len(slots)), dtype=bool)
            for i, (task, k) in enumerate(stages):
                for j, word in enumerate(head_words(task, k, slots)):
                    if word:
                        head_ids[i, j] = module.tokenizer.encode_text(word)[0]
                        head_valid[i, j] = True
            head_ids_t = torch.as_tensor(head_ids, device=device)
            head_valid_t = torch.as_tensor(head_valid, device=device)
            with torch.no_grad():
                target_rms = word_rms(embed, task_tokens, task_mask, embed_scale, device)
            token_sources = {
                "module": model.module_tokens,
                # Real vocabulary embeddings at the scale the prefix gives language tokens.
                "head": lambda _obs, ids=head_ids_t, valid=head_valid_t: (
                    embed(ids).float() * embed_scale * valid[..., None]
                ),
            }

        torch.manual_seed(args.seed * 100_003 + b)
        actions = torch.as_tensor(batch["actions"]).to(device, torch.float32)
        noise = model.sample_noise(actions.shape, device)
        flow_time = model.sample_time(actions.shape[0], device)
        for name in variants:
            prompt, layout = FLOW_VARIANTS[name]
            if layout == "own":
                prefix = make_prefix(model, recorders[name], own=len(slots))
            elif layout == "stock":
                prefix = make_prefix(model, recorders[name])
            else:
                source, where, scale = layout
                prefix = make_prefix(
                    model,
                    recorders[name],
                    tokens=token_sources[source],
                    where=where,
                    scale=scale,
                    target_rms=target_rms,
                )
            obs = _model.Observation.from_dict(jax.tree.map(_to_device, batches[prompt]))
            model.embed_observation_prefix = prefix
            torch.manual_seed(args.seed * 100_003 + b)  # identical image augmentation
            with torch.no_grad(), capture_attention(recorders[name]):
                per_element = model.forward(obs, actions, noise=noise, time=flow_time)
            recorders[name].groups = None
            flows[name].append(per_element.float().mean(dim=(1, 2)).cpu().numpy())
            rms[name].append(prefix.rms)
        del model.embed_observation_prefix
        if b % 16 == 0:
            done = (b + 1) * args.batch_size
            means = {k: round(float(np.concatenate(v).mean()), 6) for k, v in flows.items()}
            logging.info(f"{done}/{n} samples ({done / (time.monotonic() - t0):.1f}/s) running flow means {means}")

    flows = {k: np.concatenate(v) for k, v in flows.items()}
    tasks = np.asarray(tasks)
    result["flow"] = _flow_probe.paired_summary(flows, tasks, "correct")
    result["flow_by_suite"] = suite_summary(flows, tasks)
    result["attention"] = {v: r.summary() for v, r in recorders.items()}
    result["embedding_rms"] = {
        v: {
            k: float(np.mean([r[k] for r in rs if k in r]))
            for k in ("images", "module", "language")
            if any(k in r for r in rs)
        }
        for v, rs in rms.items()
    }

    logging.info(f"{len(tasks)} samples; embedding RMS of `correct` {result['embedding_rms']['correct']}")
    for name, o in result["flow"]["overall"].items():
        by_suite = result["flow_by_suite"][name]
        line = f"FLOW {name:20s} {o['flow']:.5f}  x{o['ratio']:.3f}  diff {o['diff']:+.5f} +- {o['diff_se']:.5f}"
        line += " | " + " ".join(f"{suite} {v['flow']:.5f}" for suite, v in by_suite.items())
        if "no_module" in flows:
            line += f" | vs no_module x{o['flow'] / result['flow']['overall']['no_module']['flow']:.2f}"
        logging.info(line)
    for name, att in result["attention"].items():
        key = "module" if att["keys_per_sample"]["module"] > 0 else "language"
        a = att["by_query"]["actions"]
        per_layer = a["mass_per_layer"][key]
        top = int(np.argmax(per_layer))
        logging.info(
            f"ATTN {name:20s} actions -> {key} mass {a['mass_mean'][key]:.4f} (max layer {top}: {per_layer[top]:.4f}) "
            f"value share {att['actions_value_weighted_share_mean'][key]:.4f} | per token vs prefix "
            f"x{a['per_token_vs_prefix_mean'].get(key, float('nan')):.2f} | images -> {key} "
            f"{att['by_query']['images']['mass_mean'][key]:.4f} | rms {result['embedding_rms'][name]}"
        )

    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {**result, "per_sample_flow": {k: v.tolist() for k, v in flows.items()}, "indices": indices.tolist()}
        out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        logging.info(f"wrote {out}")
    return result


if __name__ == "__main__":
    import tyro

    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
