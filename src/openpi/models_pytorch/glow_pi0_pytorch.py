"""GLOW model: pi05 whose prompt carries the plan clause, plus the subtask cross-entropy head.

Two halves of pi05's own factorization pi(a, l | o, task) = pi(a | o, l) * pi(l | o, task):

- Conditioning: the plan clause is plain prompt text (openpi.glow.plan_prompt), so the flow
  pass is the stock `PI0Pytorch.forward`. The only change is that the subtask target block the
  data pipeline appends after the prompt (token_loss_mask) is removed from the language pad
  mask before the flow pass, so those positions are padding to it exactly as in the stock
  layout.
- Prediction: on the samples that carry a subtask target (the plan-dropped ones), a second,
  prefix-only pass runs the PaliGemma language model over images + prompt + target block with
  causal attention inside the block, and decodes the block through the tied token embedding
  (no new parameters). `subtask_ce_weight` (lambda) scales it; 0 skips the pass.
"""

import torch
import torch.nn.functional as F  # noqa: N812

from openpi.models_pytorch import pi0_pytorch
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def _select_rows(observation, selected):
    """The observation restricted to the `selected` batch rows."""

    def sel(x):
        return None if x is None else x[selected]

    return observation.replace(
        images={k: v[selected] for k, v in observation.images.items()},
        image_masks={k: v[selected] for k, v in observation.image_masks.items()},
        state=observation.state[selected],
        tokenized_prompt=sel(observation.tokenized_prompt),
        tokenized_prompt_mask=sel(observation.tokenized_prompt_mask),
        token_ar_mask=sel(observation.token_ar_mask),
        token_loss_mask=sel(observation.token_loss_mask),
    )


class GlowPI0Pytorch(pi0_pytorch.PI0Pytorch):
    def __init__(self, config):
        super().__init__(config)
        self.subtask_ce_weight = float(config.subtask_ce_weight)

    def _preprocess_observation(self, observation, *, train=True):
        """Stock preprocessing, with the subtask block masked out of the language pad mask."""
        obs = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        lang_masks = obs.tokenized_prompt_mask
        if obs.token_loss_mask is not None:
            lang_masks = lang_masks & ~obs.token_loss_mask.to(torch.bool)
        return (
            list(obs.images.values()),
            list(obs.image_masks.values()),
            obs.tokenized_prompt,
            lang_masks,
            obs.state,
        )

    def forward(self, observation, actions, noise=None, time=None):
        """Flow loss as the stock model, plus lambda * subtask CE when a target block is present.

        Returns the stock per-element flow loss when lambda == 0, so the trainer's stock path
        applies unchanged; otherwise a dict with the combined scalar loss and its terms.
        """
        flow = super().forward(observation, actions, noise=noise, time=time)
        if self.subtask_ce_weight == 0.0:
            return flow
        flow_loss = flow.mean()
        out = {"flow": flow_loss.detach(), "ce_samples": torch.zeros((), device=flow.device)}
        loss_mask = observation.token_loss_mask
        selected = loss_mask.to(torch.bool).any(dim=1) if loss_mask is not None else None
        if selected is not None and bool(selected.any()):
            ce = self.subtask_ce(observation, selected)
            out["subtask_ce"] = ce.detach()
            out["ce_samples"] = selected.sum().to(torch.float32)
            out["loss"] = flow_loss + self.subtask_ce_weight * ce
        else:
            out["loss"] = flow_loss
        return out

    def subtask_ce(self, observation, selected):
        """Mean token cross-entropy of the subtask block over the selected samples.

        Preprocessing runs again on the selected rows instead of sharing the flow pass's
        result, which is what inheriting the stock `forward` costs: the image augmentation
        is therefore redrawn, so the two passes see the same frames augmented differently.
        """
        obs = _preprocessing.preprocess_observation_pytorch(_select_rows(observation, selected), train=self.training)
        images = list(obs.images.values())
        img_masks = list(obs.image_masks.values())
        lang_tokens = obs.tokenized_prompt
        lang_masks = obs.tokenized_prompt_mask  # prompt AND block
        ar_mask = obs.token_ar_mask.to(torch.bool)
        loss_mask = obs.token_loss_mask.to(torch.bool)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        num_lang = lang_tokens.shape[1]
        num_img = prefix_att_masks.shape[1] - num_lang
        # images + prompt attend bidirectionally (0); every block token starts a new causal
        # segment (1), so the block sees the prompt and its own past only.
        att_masks = torch.cat([prefix_att_masks[:, :num_img], ar_mask], dim=1)
        att_2d_masks = pi0_pytorch.make_att_2d_masks(prefix_pad_masks, att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        hidden = self._language_forward(prefix_embs, att_2d_masks_4d, position_ids)
        lang_hidden = hidden[:, num_img:]

        # Token at block position p is predicted from the hidden state at p - 1 (the first
        # block token from the prompt's final "\n").
        source = lang_hidden[:, :-1]
        targets = lang_tokens[:, 1:]
        target_mask = loss_mask[:, 1:]
        hs = source[target_mask]
        tg = targets[target_mask].to(torch.long)
        # Decode through the tied input embedding (no new parameters). Cast the hidden states to
        # the weight dtype, never the weight to float32: under bfloat16 weights that cast puts a
        # 257152 x 2048 copy (2.11 GB) in the autograd graph. Cross-entropy itself runs in
        # float32 - a no-op under float32 weights - because log-softmax over 257152 classes in
        # bfloat16 is not accurate enough.
        embed_weight = self.paligemma_with_expert.paligemma.language_model.embed_tokens.weight
        logits = F.linear(hs.to(embed_weight.dtype), embed_weight)
        return F.cross_entropy(logits.to(torch.float32), tg)

    def _language_forward(self, embs, att_2d_masks_4d, position_ids):
        """PaliGemma language model over `embs`, one checkpointed decoder layer at a time."""
        lm = self.paligemma_with_expert.paligemma.language_model
        # Required, not incidental: only the eager path consumes the 4-D additive mask from
        # `_prepare_attention_masks_4d` verbatim, which is what makes a masked key contribute an
        # exact zero. Stock `sample_actions` sets the same flag on the same module.
        lm.config._attn_implementation = "eager"  # noqa: SLF001
        hidden = embs
        if lm.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            hidden = hidden.to(torch.bfloat16)
        position_embeddings = lm.rotary_emb(hidden, position_ids)

        def layer_fn(hidden_states, layer):
            return layer(
                hidden_states,
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                adarms_cond=None,
            )[0]

        for layer in lm.layers:
            hidden = self._apply_checkpoint(layer_fn, hidden, layer)
        hidden, _ = lm.norm(hidden, None)
        return hidden
