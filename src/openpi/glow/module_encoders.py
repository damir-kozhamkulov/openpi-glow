"""GLOW's Spatial and Temporal encoders: the planning module as prefix tokens.

Both read the arrays `openpi.glow.module_inputs.GlowModuleInputs` emits. Text pieces (subgoal,
nouns, runtime keys) are embedded by the backbone's own token embedding (passed in as `embed`,
so no vocabulary is added) and mean-pooled over their valid tokens; everything else is learned
here. Each encoder is a small pre-norm transformer over a few slots, projected to the backbone
width. The caller rescales the output to the language tokens' RMS.

Spatial (the active stage: what, to what, where) - 4 slots:
    OP   predicate embedding + pooled subgoal_operation
    OBJ  pooled object noun + Fourier MLP of the object anchor
    TGT  pooled target noun + Fourier MLP of the target anchor (a null embedding with neither)
    CTX  learned query
Temporal (where in the plan: the chain of runtime keys and the active stage) - K_MAX slots plus
a summary query; a slot is pooled runtime_key + stage-position + status (done / active / to do)
+ stage-count embeddings, padding slots masked. Output: [summary, active slot].
"""

import math

import torch
from torch import nn

from openpi.glow import module_inputs as _mi

SPATIAL_TOKENS = 4
TEMPORAL_TOKENS = 2


def _pool(embed, ids: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean of the valid tokens' embeddings [B, D] and whether the piece has any [B]."""
    emb = embed(ids).to(torch.float32)
    m = mask.to(emb.dtype)[..., None]
    count = m.sum(dim=-2)
    return (emb * m).sum(dim=-2) / count.clamp(min=1.0), count[..., 0] > 0


def _segment(text: torch.Tensor, mask: torch.Tensor, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    start, stop = _mi.TEXT_SEGMENTS[name]
    return text[..., start:stop], mask[..., start:stop]


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


class SpatialEncoder(nn.Module):
    def __init__(self, text_width: int, out_width: int, width: int = 512, depth: int = 2, *, grounding: bool = True,
                 subgoal_text: bool = True):
        super().__init__()
        self.grounding = grounding
        self.subgoal_text = subgoal_text
        self.text_proj = nn.Sequential(nn.LayerNorm(text_width), nn.Linear(text_width, width))
        self.predicate = nn.Embedding(len(_mi.PREDICATES) + 1, width)
        self.xyz = FourierXYZ(width)
        self.null_target = nn.Parameter(torch.zeros(width))
        self.ctx = nn.Parameter(torch.zeros(width))
        self.slot_type = nn.Parameter(torch.randn(SPATIAL_TOKENS, width) * 0.02)
        self.transformer = _transformer(width, depth)
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, out_width))

    def forward(self, embed, text, text_mask, ints, floats) -> torch.Tensor:
        subgoal, _ = _pool(embed, *_segment(text, text_mask, "subgoal"))
        obj, _ = _pool(embed, *_segment(text, text_mask, "object"))
        tgt, has_tgt_noun = _pool(embed, *_segment(text, text_mask, "target"))

        op = self.predicate(ints[..., _mi.INT_PREDICATE].long())
        if self.subgoal_text:
            op = op + self.text_proj(subgoal)
        obj = self.text_proj(obj)
        tgt = self.text_proj(tgt)
        has_tgt = has_tgt_noun
        if self.grounding:
            obj_valid = floats[..., _mi.FLOAT_OBJECT_VALID, None]
            tgt_valid = floats[..., _mi.FLOAT_TARGET_VALID, None]
            obj = obj + obj_valid * self.xyz(floats[..., _mi.FLOAT_OBJECT])
            tgt = tgt + tgt_valid * self.xyz(floats[..., _mi.FLOAT_TARGET])
            has_tgt = has_tgt | (tgt_valid[..., 0] > 0)
        tgt = torch.where(has_tgt[..., None], tgt, self.null_target.expand_as(tgt))

        slots = torch.stack([op, obj, tgt, self.ctx.expand_as(op)], dim=-2) + self.slot_type
        return self.out(self.transformer(slots))


class TemporalEncoder(nn.Module):
    def __init__(self, text_width: int, out_width: int, width: int = 512, depth: int = 2):
        super().__init__()
        self.text_proj = nn.Sequential(nn.LayerNorm(text_width), nn.Linear(text_width, width))
        self.position = nn.Embedding(_mi.K_MAX, width)
        self.status = nn.Embedding(3, width)  # done, active, to do
        self.num_stages = nn.Embedding(_mi.K_MAX + 1, width)
        self.summary = nn.Parameter(torch.zeros(width))
        self.transformer = _transformer(width, depth)
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, out_width))

    def forward(self, embed, text, text_mask, ints) -> torch.Tensor:
        keys = torch.stack(
            [_pool(embed, *_segment(text, text_mask, f"rk{j}"))[0] for j in range(_mi.K_MAX)], dim=-2
        )
        stage = ints[..., _mi.INT_STAGE].long()
        num_stages = ints[..., _mi.INT_NUM_STAGES].long()
        j = torch.arange(_mi.K_MAX, device=text.device)
        status = torch.sign(j - stage[..., None]) + 1
        slots = (
            self.text_proj(keys)
            + self.position(j)
            + self.status(status)
            + self.num_stages(num_stages)[..., None, :]
        )
        slots = torch.cat([self.summary.expand(slots.shape[0], 1, -1), slots], dim=-2)
        padding = torch.cat([torch.zeros_like(stage[..., None], dtype=torch.bool), j >= num_stages[..., None]], dim=-1)
        out = self.transformer(slots, src_key_padding_mask=padding)
        active = out[torch.arange(out.shape[0], device=out.device), 1 + stage]
        return self.out(torch.stack([out[:, 0], active], dim=-2))
