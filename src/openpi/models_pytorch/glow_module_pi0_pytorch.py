"""GLOW module model: pi05 whose goal comes from learned Spatial and Temporal encoders.

The planning module (openpi.glow.module_inputs) enters the prefix as encoder tokens, between the
images and the language, all bidirectional as in the stock prefix:

    [images][spatial x4][temporal x2][language: BOS "\n" - the task string is dropped][actions]

The action expert attends to the whole prefix, as stock; nothing else changes, so training and
inference are the stock forward and sample_actions with this prefix (`embed_observation_prefix`).
The encoders' outputs are rescaled to the RMS of the sample's language-token embeddings, so they
enter the backbone at the scale its input tokens have.

The encoders are new parameters: the fine-tuning init (pi05_libero) has none of them, so the
trainer loads it non-strictly for exactly the names under NEW_PARAM_PREFIXES, and trains them at
`config.encoder_lr_scale` times the backbone's learning rate.
"""

import torch

from openpi.glow import module_encoders as _enc
from openpi.models import gemma as _gemma
from openpi.models_pytorch import pi0_pytorch


class GlowModulePI0Pytorch(pi0_pytorch.PI0Pytorch):
    NEW_PARAM_PREFIXES = ("glow_spatial_encoder.", "glow_temporal_encoder.")

    def __init__(self, config):
        super().__init__(config)
        if not (config.module_spatial or config.module_temporal):
            raise ValueError("GlowModulePI0Pytorch needs the spatial or the temporal encoder")
        width = _gemma.get_config(config.paligemma_variant).width
        self.glow_spatial_encoder = (
            _enc.SpatialEncoder(
                width,
                width,
                config.module_width,
                config.module_depth,
                grounding=config.module_grounding,
                subgoal_text=config.module_subgoal_text,
            )
            if config.module_spatial
            else None
        )
        self.glow_temporal_encoder = (
            _enc.TemporalEncoder(width, width, config.module_width, config.module_depth)
            if config.module_temporal
            else None
        )

    def module_tokens(self, observation) -> torch.Tensor:
        """The encoders' tokens [B, N, width], before rescaling."""
        fields = (observation.glow_text, observation.glow_text_mask, observation.glow_ints, observation.glow_floats)
        if any(f is None for f in fields):
            raise ValueError(
                "GlowModulePI0Pytorch: the observation has no planning-module arrays; the data config must "
                "set module_inputs (GlowModuleInputs)"
            )
        text, text_mask, ints, floats = fields
        embed = self.paligemma_with_expert.embed_language_tokens
        tokens = []
        if self.glow_spatial_encoder is not None:
            tokens.append(self.glow_spatial_encoder(embed, text, text_mask, ints, floats.to(torch.float32)))
        if self.glow_temporal_encoder is not None:
            tokens.append(self.glow_temporal_encoder(embed, text, text_mask, ints))
        return torch.cat(tokens, dim=1)

    def embed_observation_prefix(self, observation, images, img_masks, lang_tokens, lang_masks):
        embs, pad_masks, att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        num_img = embs.shape[1] - lang_tokens.shape[1]

        lang = embs[:, num_img:].to(torch.float32)
        valid = lang_masks.to(torch.float32)[..., None]
        lang_rms = torch.sqrt((lang.pow(2).mean(-1, keepdim=True) * valid).sum(1, keepdim=True)
                              / valid.sum(1, keepdim=True).clamp(min=1.0))
        module = self.module_tokens(observation)
        module = module * torch.rsqrt(module.pow(2).mean(-1, keepdim=True) + 1e-6) * lang_rms.detach()
        module = module.to(embs.dtype)

        bsize, num_module = module.shape[:2]
        embs = torch.cat([embs[:, :num_img], module, embs[:, num_img:]], dim=1)
        pad_masks = torch.cat(
            [pad_masks[:, :num_img], pad_masks.new_ones(bsize, num_module), pad_masks[:, num_img:]], dim=1
        )
        att_masks = torch.cat(
            [att_masks[:, :num_img], att_masks.new_zeros(bsize, num_module), att_masks[:, num_img:]], dim=1
        )
        return embs, pad_masks, att_masks
