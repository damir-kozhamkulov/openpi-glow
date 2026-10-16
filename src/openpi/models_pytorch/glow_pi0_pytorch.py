"""GLOW unified model: pi05 that predicts its own plan state and acts on it, in one prefix.

pi05's factorization with the plan state carried across queries:
pi(a, l_t | o_t, task, l_{t-delta}) = pi(a | o_t, task, l_{t-delta}, l_t) * pi(l_t | o_t, task, l_{t-delta}).

Prefix layout (openpi.glow.plan_prompt builds the tokens):

    [images][task. Was: <plan t-delta> \n][Now: <plan t> eos][action expert: noisy actions]
     bidirectional (ar 0) ----------------- causal (ar 1) --- attends to the whole prefix

- Training: ONE forward over prefix and suffix. The plan block is teacher-forced ground truth;
  the action expert sees it, and its prefix outputs (which cannot see the suffix) give the plan
  cross-entropy through the tied token embedding, with no new parameters. `subtask_ce_weight`
  (lambda) scales it; 0 returns the flow loss alone, as the stock tensor.
- Inference: `tokenized_prompt` carries K candidate rows of max_token_len tokens side by side,
  [B, K * max_token_len] (Observation's shape checks admit no extra axis), each row = prompt +
  one plan block. One prefix pass over the B*K rows scores each block by its summed token
  log-likelihood, the argmax is the chosen plan, and its row's KV cache conditions the stock
  denoising loop. K = 1 is a given plan (an oracle or forced plan, or a one-stage task).
"""

import math

import torch
import torch.nn.functional as F  # noqa: N812

from openpi.models_pytorch import pi0_pytorch


class GlowPI0Pytorch(pi0_pytorch.PI0Pytorch):
    def __init__(self, config):
        super().__init__(config)
        self.subtask_ce_weight = float(config.subtask_ce_weight)

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, lang_ar_mask=None, image_repeats=1):
        """Stock `embed_prefix`, plus a language autoregressive mask and repeated image embeddings.

        `lang_ar_mask` (1 = starts a new causal segment) makes the plan block causal; None is the
        stock all-bidirectional prefix. `image_repeats` = K embeds each image once and repeats it
        for the K candidate rows of that sample (rows ordered sample-major, candidate-minor).
        """
        embs = []
        pad_masks = []
        att_masks = []
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            img_pad = img_mask[:, None].expand(img_emb.shape[:2])
            if image_repeats > 1:
                img_emb = img_emb.repeat_interleave(image_repeats, dim=0)
                img_pad = img_pad.repeat_interleave(image_repeats, dim=0)
            embs.append(img_emb)
            pad_masks.append(img_pad)
            att_masks.append(torch.zeros(img_pad.shape, dtype=torch.bool, device=img_pad.device))

        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            return lang_emb * math.sqrt(lang_emb.shape[-1])

        embs.append(self._apply_checkpoint(lang_embed_func, lang_tokens))
        pad_masks.append(lang_masks)
        if lang_ar_mask is None:
            att_masks.append(torch.zeros(lang_masks.shape, dtype=torch.bool, device=lang_masks.device))
        else:
            att_masks.append(lang_ar_mask.to(torch.bool))
        return torch.cat(embs, dim=1), torch.cat(pad_masks, dim=1), torch.cat(att_masks, dim=1)

    def forward(self, observation, actions, noise=None, time=None):
        """Flow loss with the plan block visible to the action expert, plus lambda * plan CE.

        Returns the stock per-element flow loss when lambda == 0, so the trainer's stock path
        applies unchanged; otherwise a dict with the combined scalar loss and its terms.
        """
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)
        if lang_tokens.shape[1] != self.config.max_token_len:
            raise ValueError("training samples carry one token row (the plan target), not candidate rows")
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, observation.token_ar_mask
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks_4d = self._prepare_attention_masks_4d(pi0_pytorch.make_att_2d_masks(pad_masks, att_masks))
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return prefix_out, suffix_out

        prefix_out, suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )
        suffix_out = suffix_out[:, -self.config.action_horizon :].to(dtype=torch.float32)
        v_t = self._apply_checkpoint(self.action_out_proj, suffix_out)
        flow = F.mse_loss(u_t, v_t, reduction="none")
        if self.subtask_ce_weight == 0.0:
            return flow

        flow_loss = flow.mean()
        out = {"flow": flow_loss.detach(), "ce_samples": torch.zeros((), device=flow.device)}
        loss_mask = observation.token_loss_mask
        if loss_mask is None or not bool(loss_mask.any()):
            out["loss"] = flow_loss
            return out
        num_lang = lang_tokens.shape[1]
        ce, token_acc, exact = self._plan_ce(prefix_out[:, -num_lang:], lang_tokens, loss_mask.to(torch.bool))
        out.update(
            loss=flow_loss + self.subtask_ce_weight * ce,
            plan_ce=ce.detach(),
            plan_token_acc=token_acc,
            plan_exact=exact,
            ce_samples=loss_mask.any(dim=1).sum().to(torch.float32),
        )
        return out

    def _block_logits(self, lang_hidden, lang_tokens, loss_mask):
        """Logits and targets of the plan-block tokens, flattened over the batch.

        The token at block position p is predicted from the hidden state at p - 1 (the first
        block token from the prompt's final "\\n"). Decoded through the tied input embedding.
        Cast the hidden states to the weight dtype, never the weight to float32: under bfloat16
        weights that cast puts a 257152 x 2048 copy (2.11 GB) in the autograd graph.
        """
        target_mask = loss_mask[:, 1:]
        hs = lang_hidden[:, :-1][target_mask]
        targets = lang_tokens[:, 1:][target_mask].to(torch.long)
        embed_weight = self.paligemma_with_expert.paligemma.language_model.embed_tokens.weight
        # Float32 log-softmax: over 257152 classes bfloat16 is not accurate enough.
        logits = F.linear(hs.to(embed_weight.dtype), embed_weight).to(torch.float32)
        return logits, targets, target_mask

    def _plan_ce(self, lang_hidden, lang_tokens, loss_mask):
        """Mean token cross-entropy over the plan blocks, plus teacher-forced token accuracy and
        the fraction of blocks whose every token is the argmax (detached diagnostics)."""
        logits, targets, target_mask = self._block_logits(lang_hidden, lang_tokens, loss_mask)
        ce = F.cross_entropy(logits, targets)
        with torch.no_grad():
            hit = logits.argmax(dim=-1) == targets
            per_row = torch.zeros(target_mask.shape, dtype=torch.bool, device=hit.device)
            per_row[target_mask] = ~hit
            rows = target_mask.any(dim=1)
            exact = (~per_row.any(dim=1))[rows].to(torch.float32).mean()
        return ce, hit.to(torch.float32).mean(), exact

    def _block_logp(self, lang_hidden, lang_tokens, loss_mask):
        """Summed log-likelihood of each row's plan block, shape [rows]."""
        logits, targets, target_mask = self._block_logits(lang_hidden, lang_tokens, loss_mask)
        token_logp = torch.log_softmax(logits, dim=-1).gather(-1, targets[:, None])[:, 0]
        row_of_token = target_mask.nonzero()[:, 0]
        return torch.zeros(lang_tokens.shape[0], device=token_logp.device).index_add(0, row_of_token, token_logp)

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10):
        """Actions conditioned on the best of K candidate plan blocks ([B, K * max_token_len]).

        Returns {"actions": [B, H, D], "plan_choice": [B] (the argmax candidate's position, which
        is its plan index when the candidates are the task's stages in order), "plan_logp": [B, K]
        summed block log-likelihoods}.
        """
        bsize = observation.state.shape[0]
        if noise is None:
            noise = self.sample_noise((bsize, self.config.action_horizon, self.config.action_dim), device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
        ar_mask, loss_mask = observation.token_ar_mask, observation.token_loss_mask
        if ar_mask is None or loss_mask is None:
            raise ValueError("GlowPI0Pytorch.sample_actions needs a plan block (token_ar_mask / token_loss_mask)")
        num_cand, rem = divmod(lang_tokens.shape[1], self.config.max_token_len)
        if rem or num_cand < 1:
            raise ValueError(
                f"token width {lang_tokens.shape[1]} is not a multiple of max_token_len {self.config.max_token_len}"
            )
        lang_tokens, lang_masks, ar_mask, loss_mask = (
            x.reshape(bsize * num_cand, self.config.max_token_len)
            for x in (lang_tokens, lang_masks, ar_mask, loss_mask)
        )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, ar_mask, image_repeats=num_cand
        )
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(
            pi0_pytorch.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        )
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        num_lang = lang_tokens.shape[1]
        logp = self._block_logp(prefix_out[:, -num_lang:], lang_tokens, loss_mask.to(torch.bool))
        logp = logp.reshape(bsize, num_cand)
        choice = logp.argmax(dim=1)
        rows = torch.arange(bsize, device=choice.device) * num_cand + choice
        if num_cand > 1:
            past_key_values.batch_select_indices(rows)
        prefix_pad_masks = prefix_pad_masks[rows]

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, time.expand(bsize))
            x_t = x_t + dt * v_t
            time += dt

        return {"actions": x_t, "plan_choice": choice, "plan_logp": logp}
