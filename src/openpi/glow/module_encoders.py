"""GLOW's Spatial and Temporal encoders: the planning module as prefix tokens.

Both read the arrays `openpi.glow.module_inputs.GlowModuleInputs` emits. Text pieces (atoms,
subgoals, nouns, runtime keys) are embedded by the backbone's own token embedding (passed in as
`embed`, so no vocabulary is added) and mean-pooled over their valid tokens; everything else is
learned here. Each encoder is a small pre-norm transformer over a few slots, projected to the
backbone width. The caller rescales the output to a target RMS and masks the padded tokens out
of the prefix (`token_mask`).

With `text_anchor`, each output token is its slot's text (the pooled token embedding: a direction
the backbone already reads as language) plus the encoder's output as a residual whose last layer
starts at zero, so a fresh encoder emits exactly the anchors and the residual grows from there.

Spatial (what every stage does and where) - one group of 4 slots per stage, K_MAX groups in the
order glow_ints' INT_SPATIAL_SLOT gives (a set: no position of its own), then a summary:
    OP   predicate embedding + pooled subgoal_operation            text anchor: subgoal_operation
    OBJ  pooled object noun + Fourier MLP of the object anchor      text anchor: object noun
    TGT  pooled target noun + Fourier MLP of the target anchor      text anchor: target noun
         (a null embedding with neither)
    ATOM pooled atom                                                text anchor: atom
    CTX  learned query                                              text anchor: all stages' text
Groups past the plan's K stages are padding: masked inside the encoder and emitted as zero
tokens with token_mask False.
Temporal (when: the chain of runtime keys in execution order and the active stage) - K_MAX
slots plus a summary query; a slot is pooled runtime_key + stage-position + status (done /
active / to do) + stage-count embeddings, padding slots masked. Output: [summary, active slot];
text anchors: every runtime key of the plan, the active stage's runtime key.

A missing piece (no target, the subgoal switched off) anchors to zero: a zero token, which the
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


def spatial_token_mask(ints: torch.Tensor) -> torch.Tensor:
    """[B, SPATIAL_TOKENS]: the valid groups' slots, and the summary."""
    valid = stage_valid(ints).repeat_interleave(len(SPATIAL_SLOTS), dim=-1)
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
        text_anchor: bool = False,
    ):
        super().__init__()
        if text_anchor and text_width != out_width:
            raise ValueError("SpatialEncoder: text_anchor needs the text and output widths to match")
        self.grounding = grounding
        self.subgoal_text = subgoal_text
        self.text_anchor = text_anchor
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

    def anchors(self, emb, text_mask, ints) -> torch.Tensor:
        """The slots' text anchors [B, SPATIAL_TOKENS, D]: per group (in Spatial order) subgoal,
        object noun, target noun, atom; then every valid stage's Spatial text together."""
        fields = ["subgoal", "object", "target", "atom"] if self.subgoal_text else ["object", "target", "atom"]
        pooled = {f: _per_stage(emb, text_mask, f)[0] for f in fields}
        if not self.subgoal_text:
            pooled["subgoal"] = torch.zeros_like(pooled["atom"])
        per = torch.stack([pooled[f] for f in ("subgoal", "object", "target", "atom")], dim=-2)  # [B, K, 4, D]
        groups = _gather_groups(per, ints[..., _mi.INT_SPATIAL_SLOT].long())
        ctx = _pool_many(emb, text_mask, [f"{f}{j}" for j in range(_mi.K_MAX) for f in fields])
        return torch.cat([groups.flatten(1, 2), ctx[:, None]], dim=1)

    def forward(self, embed, text, text_mask, ints, floats) -> torch.Tensor:
        emb = embed(text).to(torch.float32)
        atom, _ = _per_stage(emb, text_mask, "atom")
        subgoal, _ = _per_stage(emb, text_mask, "subgoal")
        obj, _ = _per_stage(emb, text_mask, "object")
        tgt, has_tgt_noun = _per_stage(emb, text_mask, "target")

        atom = self.text_proj(atom)
        op = self.predicate(ints[..., _mi.INT_PREDICATE].long())
        if self.subgoal_text:
            op = op + self.text_proj(subgoal)
        obj = self.text_proj(obj)
        tgt = self.text_proj(tgt)
        has_tgt = has_tgt_noun
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
            out = out + self.anchors(emb, text_mask, ints).to(out.dtype)
        return out * token_mask[..., None].to(out.dtype)


class TemporalEncoder(nn.Module):
    def __init__(self, text_width: int, out_width: int, width: int = 512, depth: int = 2, *, text_anchor: bool = False):
        super().__init__()
        if text_anchor and text_width != out_width:
            raise ValueError("TemporalEncoder: text_anchor needs the text and output widths to match")
        self.text_anchor = text_anchor
        self.text_proj = nn.Sequential(nn.LayerNorm(text_width), nn.Linear(text_width, width))
        self.position = nn.Embedding(_mi.K_MAX, width)
        self.status = nn.Embedding(3, width)  # done, active, to do
        self.num_stages = nn.Embedding(_mi.K_MAX + 1, width)
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
        slots = torch.cat([self.summary.expand(slots.shape[0], 1, -1), slots], dim=-2)
        padding = torch.cat([torch.zeros_like(stage[..., None], dtype=torch.bool), ~stage_valid(ints)], dim=-1)
        out = self.transformer(slots, src_key_padding_mask=padding)
        rows = torch.arange(out.shape[0], device=out.device)
        active = out[rows, 1 + stage]
        tokens = self.out(torch.stack([out[:, 0], active], dim=-2))
        if not self.text_anchor:
            return tokens
        # Anchors: every runtime key of the plan, and the active stage's.
        every = _pool_many(emb, text_mask, [f"rk{j}" for j in range(_mi.K_MAX)])
        return tokens + torch.stack([every, keys[rows, stage]], dim=-2).to(tokens.dtype)
