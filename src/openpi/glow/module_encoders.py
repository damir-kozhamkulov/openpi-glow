"""GLOW's Spatial and Temporal encoders: the planning module as prefix tokens.

Both read the arrays `openpi.glow.module_inputs.GlowModuleInputs` emits. Text pieces (atoms,
subgoals, nouns, runtime keys) are embedded by the backbone's own token embedding (passed in as
`embed`, so no vocabulary is added) and mean-pooled over their valid tokens; everything else is
learned here. Each encoder is a small pre-norm transformer over a few slots, projected to the
backbone width. The caller rescales the output to a target RMS and masks the padded tokens out
of the prefix (`token_mask`).

With `text_anchor`, each output token is its slot's text (a direction the backbone already reads
as language) plus the encoder's output as a residual whose last layer starts at zero, so a fresh
encoder emits exactly the anchors and the residual grows from there. The anchor is either the
slot text's mean-pooled token embedding (`anchor="mean"`) or the embedding of one head word
(`anchor="head"`: glow_ints' INT_HEAD ids, module_inputs.head_words). Mean pooling washes
different tasks' texts toward one shared direction; head words stay as distinct as words.

Spatial (what every stage does and where) - one group of 4 slots per stage, K_MAX groups in the
order glow_ints' INT_SPATIAL_SLOT gives (no position of its own: the order is the data's), then
a summary:
    OP   predicate embedding + pooled subgoal_operation      anchor: subgoal_operation | its verb
    OBJ  pooled object noun + Fourier MLP of the object anchor anchor: object noun | head noun
    TGT  pooled target noun + Fourier MLP of the target anchor anchor: target noun | head noun
         (a null embedding with neither)
    ATOM pooled atom                                          anchor: atom | the predicate word
    CTX  learned query                                        anchor: the shown stages' text |
                                                              the active stage's predicate word
Only the first INT_SHOWN groups are shown (all K stages, or with the "remaining" order the active
stage and the later ones); the rest are padding: masked inside the encoder and emitted as zero
tokens with token_mask False. `subgoal_text`, `nouns` and `atom` switch those texts off (their
slots keep the predicate embedding / the grounding / nothing), `grounding` the anchors' xyz.
Temporal (when: the chain of runtime keys in execution order and the active stage) - K_MAX
slots plus a summary query; a slot is pooled runtime_key + stage-position + status (done /
active / to do) + stage-count embeddings, and with `stage_clock` the active slot adds an
embedding of the frames since its stage began (log buckets, one for unknown); padding slots
masked. Output: [summary, active slot]; anchors: every runtime key of the plan and the active
stage's runtime key | the stage count and the active stage's ordinal as words.

A missing piece (no target, a text switched off) anchors to zero: a zero token, which the
backbone's attention gives a zero value.
"""

import math

import torch
from torch import nn

from openpi.glow import module_inputs as _mi

# OP, OBJ and TGT first: a checkpoint of the earlier one-group layout (OP, OBJ, TGT, CTX) keeps
# its learned slot types for them, and its CTX type becomes the ATOM slot's.
SPATIAL_SLOTS = ("OP", "OBJ", "TGT", "ATOM")
SPATIAL_TOKENS = _mi.K_MAX * len(SPATIAL_SLOTS) + 1
TEMPORAL_TOKENS = 2
ANCHORS = ("mean", "head")
# Stage-clock buckets: 0 = unknown, then [0, 1), [1, 4), [4, 8), ..., [128, inf) frames.
CLOCK_EDGES = (1, 4, 8, 16, 32, 64, 128)
# The text of each Spatial slot, and the head word that anchors it.
SLOT_TEXT = {"OP": "subgoal", "OBJ": "object", "TGT": "target", "ATOM": "atom"}
SLOT_HEAD = {"OP": "op", "OBJ": "object", "TGT": "target", "ATOM": "atom"}


def _pool(emb: torch.Tensor, mask: torch.Tensor, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean of segment `name`'s valid token embeddings [B, D] and whether it has any [B]."""
    start, stop = _mi.TEXT_SEGMENTS[name]
    m = mask[..., start:stop].to(emb.dtype)[..., None]
    count = m.sum(dim=-2)
    return (emb[..., start:stop, :] * m).sum(dim=-2) / count.clamp(min=1.0), count[..., 0] > 0


def _pool_many(emb: torch.Tensor, mask: torch.Tensor, names) -> torch.Tensor:
    """Mean embedding [B, D] of the valid tokens of several segments together."""
    sel = torch.zeros(mask.shape[-1], dtype=torch.bool, device=mask.device)
    for name in names:
        start, stop = _mi.TEXT_SEGMENTS[name]
        sel[start:stop] = True
    m = (mask & sel).to(emb.dtype)[..., None]
    return (emb * m).sum(dim=-2) / m.sum(dim=-2).clamp(min=1.0)


def _pool_stages(emb: torch.Tensor, mask: torch.Tensor, fields, stages: torch.Tensor) -> torch.Tensor:
    """Mean embedding [B, D] of the valid tokens of `fields` over the stages `stages` [B, K_MAX]
    marks."""
    seg = torch.zeros(_mi.K_MAX, mask.shape[-1], dtype=emb.dtype, device=mask.device)
    for j in range(_mi.K_MAX):
        for field in fields:
            start, stop = _mi.TEXT_SEGMENTS[f"{field}{j}"]
            seg[j, start:stop] = 1.0
    m = (mask.to(emb.dtype) * (stages.to(emb.dtype) @ seg))[..., None]
    return (emb * m).sum(dim=-2) / m.sum(dim=-2).clamp(min=1.0)


def head_embeddings(embed, ints: torch.Tensor) -> torch.Tensor:
    """[B, K_MAX, len(HEAD_FIELDS), D]: each stage's head words embedded, zero where none."""
    ids = ints[..., _mi.INT_HEAD].long().reshape(ints.shape[0], _mi.K_MAX, len(_mi.HEAD_FIELDS))
    return embed(ids).to(torch.float32) * (ids > 0)[..., None].to(torch.float32)


def clock_bucket(frames: torch.Tensor) -> torch.Tensor:
    """Stage-clock bucket per CLOCK_EDGES: 0 for unknown (negative) frames."""
    edges = torch.tensor(CLOCK_EDGES, device=frames.device)
    return torch.where(frames < 0, 0, 1 + torch.bucketize(frames, edges, right=True))


def _per_stage(emb, mask, field: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Pooled `field` of every stage [B, K_MAX, D] and whether each has any token [B, K_MAX]."""
    pooled, has = zip(*(_pool(emb, mask, f"{field}{j}") for j in range(_mi.K_MAX)), strict=True)
    return torch.stack(pooled, dim=-2), torch.stack(has, dim=-1)


def _zero_init(linear: nn.Linear) -> None:
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)


def _gather_groups(x: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """x [B, K_MAX, ...] reordered so that position i holds stage order[i]."""
    index = order.reshape(*order.shape, *([1] * (x.ndim - 2))).expand(-1, -1, *x.shape[2:])
    return torch.gather(x, 1, index)


class FourierXYZ(nn.Module):
    """xyz -> [xyz, sin(2^i pi xyz), cos(2^i pi xyz)] -> MLP."""

    def __init__(self, width: int, num_freqs: int = 6):
        super().__init__()
        self.register_buffer("freqs", (2.0 ** torch.arange(num_freqs)) * math.pi, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(3 + 6 * num_freqs, width), nn.GELU(), nn.Linear(width, width))

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        angles = xyz[..., None] * self.freqs
        feats = torch.cat([xyz, torch.sin(angles).flatten(-2), torch.cos(angles).flatten(-2)], dim=-1)
        return self.mlp(feats)


def _transformer(width: int, depth: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        width, nhead=8, dim_feedforward=4 * width, dropout=0.0, activation="gelu", batch_first=True, norm_first=True
    )
    return nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)


def stage_valid(ints: torch.Tensor) -> torch.Tensor:
    """[B, K_MAX]: whether stage position j is within the plan's K stages."""
    j = torch.arange(_mi.K_MAX, device=ints.device)
    return j < ints[..., _mi.INT_NUM_STAGES, None].long()


def shown_groups(ints: torch.Tensor) -> torch.Tensor:
    """[B, K_MAX]: whether Spatial group i is shown (i < INT_SHOWN)."""
    i = torch.arange(_mi.K_MAX, device=ints.device)
    return i < ints[..., _mi.INT_SHOWN, None].long()


def shown_stages(ints: torch.Tensor) -> torch.Tensor:
    """[B, K_MAX]: whether stage j (execution index) is in a shown Spatial group."""
    order = ints[..., _mi.INT_SPATIAL_SLOT].long()
    return torch.zeros_like(order, dtype=torch.bool).scatter(1, order, shown_groups(ints))


def spatial_token_mask(ints: torch.Tensor) -> torch.Tensor:
    """[B, SPATIAL_TOKENS]: the shown groups' slots, and the summary."""
    valid = shown_groups(ints).repeat_interleave(len(SPATIAL_SLOTS), dim=-1)
    return torch.cat([valid, valid.new_ones(valid.shape[0], 1)], dim=-1)


class SpatialEncoder(nn.Module):
    def __init__(
        self,
        text_width: int,
        out_width: int,
        width: int = 512,
        depth: int = 2,
        *,
        grounding: bool = True,
        subgoal_text: bool = True,
        nouns: bool = True,
        atom: bool = True,
        text_anchor: bool = False,
        anchor: str = "mean",
    ):
        super().__init__()
        if text_anchor and text_width != out_width:
            raise ValueError("SpatialEncoder: text_anchor needs the text and output widths to match")
        if anchor not in ANCHORS:
            raise ValueError(f"SpatialEncoder: anchor must be one of {ANCHORS}, got {anchor!r}")
        self.grounding = grounding
        self.subgoal_text = subgoal_text
        self.nouns = nouns
        self.atom = atom
        self.text_anchor = text_anchor
        self.anchor = anchor
        # The slots whose text the encoder reads.
        self.texts = {"OP": subgoal_text, "OBJ": nouns, "TGT": nouns, "ATOM": atom}
        self.text_proj = nn.Sequential(nn.LayerNorm(text_width), nn.Linear(text_width, width))
        self.predicate = nn.Embedding(len(_mi.PREDICATES) + 1, width)
        self.xyz = FourierXYZ(width)
        self.null_target = nn.Parameter(torch.zeros(width))
        self.ctx = nn.Parameter(torch.zeros(width))
        self.slot_type = nn.Parameter(torch.randn(len(SPATIAL_SLOTS), width) * 0.02)
        self.transformer = _transformer(width, depth)
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, out_width))
        if text_anchor:
            _zero_init(self.out[1])

    def anchors(self, embed, emb, text_mask, ints) -> torch.Tensor:
        """The slots' anchors [B, SPATIAL_TOKENS, D]: per group (in Spatial order) OP, OBJ, TGT,
        ATOM, then CTX (see the module docstring); zero for a slot whose text is switched off."""
        if self.anchor == "head":
            head = head_embeddings(embed, ints)
            per = torch.stack(
                [head[:, :, _mi.HEAD_FIELDS.index(SLOT_HEAD[slot])] for slot in SPATIAL_SLOTS], dim=-2
            )  # [B, K, 4, D]
            rows = torch.arange(ints.shape[0], device=ints.device)
            ctx = head[rows, ints[..., _mi.INT_STAGE].long(), _mi.HEAD_FIELDS.index("atom")]
        else:
            per = torch.stack([_per_stage(emb, text_mask, SLOT_TEXT[slot])[0] for slot in SPATIAL_SLOTS], dim=-2)
            fields = [SLOT_TEXT[slot] for slot in ("OP", "OBJ", "ATOM") if self.texts[slot]]
            fields += ["target"] if self.nouns else []
            ctx = _pool_stages(emb, text_mask, fields, shown_stages(ints))
        keep = torch.tensor([self.texts[slot] for slot in SPATIAL_SLOTS], dtype=per.dtype, device=per.device)
        per = per * keep[:, None]
        groups = _gather_groups(per, ints[..., _mi.INT_SPATIAL_SLOT].long())
        return torch.cat([groups.flatten(1, 2), ctx[:, None]], dim=1)

    def forward(self, embed, text, text_mask, ints, floats) -> torch.Tensor:
        emb = embed(text).to(torch.float32)
        atom, _ = _per_stage(emb, text_mask, "atom")
        subgoal, _ = _per_stage(emb, text_mask, "subgoal")
        obj, _ = _per_stage(emb, text_mask, "object")
        tgt, has_tgt_noun = _per_stage(emb, text_mask, "target")

        op = self.predicate(ints[..., _mi.INT_PREDICATE].long())
        if self.subgoal_text:
            op = op + self.text_proj(subgoal)
        atom = self.text_proj(atom) if self.atom else torch.zeros_like(op)
        obj = self.text_proj(obj) if self.nouns else torch.zeros_like(op)
        tgt = self.text_proj(tgt) if self.nouns else torch.zeros_like(op)
        has_tgt = has_tgt_noun if self.nouns else torch.zeros_like(has_tgt_noun)
        if self.grounding:
            f = floats.reshape(*floats.shape[:-1], _mi.K_MAX, _mi.STAGE_FLOATS)
            obj_xyz, obj_valid, tgt_xyz, tgt_valid = f[..., 0:3], f[..., 3:4], f[..., 4:7], f[..., 7:8]
            obj = obj + obj_valid * self.xyz(obj_xyz)
            tgt = tgt + tgt_valid * self.xyz(tgt_xyz)
            has_tgt = has_tgt | (tgt_valid[..., 0] > 0)
        tgt = torch.where(has_tgt[..., None], tgt, self.null_target.expand_as(tgt))

        groups = torch.stack([op, obj, tgt, atom], dim=-2) + self.slot_type  # [B, K, 4, W]
        groups = _gather_groups(groups, ints[..., _mi.INT_SPATIAL_SLOT].long())
        slots = torch.cat([groups.flatten(1, 2), self.ctx.expand(groups.shape[0], 1, -1)], dim=1)
        token_mask = spatial_token_mask(ints)
        out = self.out(self.transformer(slots, src_key_padding_mask=~token_mask))
        if self.text_anchor:
            out = out + self.anchors(embed, emb, text_mask, ints).to(out.dtype)
        return out * token_mask[..., None].to(out.dtype)


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        text_width: int,
        out_width: int,
        width: int = 512,
        depth: int = 2,
        *,
        text_anchor: bool = False,
        anchor: str = "mean",
        stage_clock: bool = False,
    ):
        super().__init__()
        if text_anchor and text_width != out_width:
            raise ValueError("TemporalEncoder: text_anchor needs the text and output widths to match")
        if anchor not in ANCHORS:
            raise ValueError(f"TemporalEncoder: anchor must be one of {ANCHORS}, got {anchor!r}")
        self.text_anchor = text_anchor
        self.anchor = anchor
        self.text_proj = nn.Sequential(nn.LayerNorm(text_width), nn.Linear(text_width, width))
        self.position = nn.Embedding(_mi.K_MAX, width)
        self.status = nn.Embedding(3, width)  # done, active, to do
        self.num_stages = nn.Embedding(_mi.K_MAX + 1, width)
        self.clock = nn.Embedding(len(CLOCK_EDGES) + 2, width) if stage_clock else None
        self.summary = nn.Parameter(torch.zeros(width))
        self.transformer = _transformer(width, depth)
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, out_width))
        if text_anchor:
            _zero_init(self.out[1])

    def forward(self, embed, text, text_mask, ints) -> torch.Tensor:
        emb = embed(text).to(torch.float32)
        keys, _ = _per_stage(emb, text_mask, "rk")
        stage = ints[..., _mi.INT_STAGE].long()
        num_stages = ints[..., _mi.INT_NUM_STAGES].long()
        j = torch.arange(_mi.K_MAX, device=text.device)
        status = torch.sign(j - stage[..., None]) + 1
        slots = (
            self.text_proj(keys) + self.position(j) + self.status(status) + self.num_stages(num_stages)[..., None, :]
        )
        if self.clock is not None:
            clock = self.clock(clock_bucket(ints[..., _mi.INT_STAGE_FRAMES].long()))
            slots = slots + (status == 1)[..., None].to(slots.dtype) * clock[:, None]
        slots = torch.cat([self.summary.expand(slots.shape[0], 1, -1), slots], dim=-2)
        padding = torch.cat([torch.zeros_like(stage[..., None], dtype=torch.bool), ~stage_valid(ints)], dim=-1)
        out = self.transformer(slots, src_key_padding_mask=padding)
        rows = torch.arange(out.shape[0], device=out.device)
        active = out[rows, 1 + stage]
        tokens = self.out(torch.stack([out[:, 0], active], dim=-2))
        if not self.text_anchor:
            return tokens
        if self.anchor == "head":
            # Anchors: the stage count and the active stage's ordinal, as words.
            head = head_embeddings(embed, ints)
            count = head[rows, num_stages - 1, _mi.HEAD_FIELDS.index("count")]
            ordinal = head[rows, stage, _mi.HEAD_FIELDS.index("ordinal")]
            return tokens + torch.stack([count, ordinal], dim=-2).to(tokens.dtype)
        # Anchors: every runtime key of the plan, and the active stage's.
        every = _pool_many(emb, text_mask, [f"rk{j}" for j in range(_mi.K_MAX)])
        return tokens + torch.stack([every, keys[rows, stage]], dim=-2).to(tokens.dtype)
