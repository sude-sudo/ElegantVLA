"""Action-expert refresh and within-denoising residual reuse."""

from __future__ import annotations

import torch

from elegantvla.openpi.core.types import ActionHeadAction


class OpenPIActionHead:
    def __init__(self, model):
        self.model = model
        self.expert = model.paligemma_with_expert.gemma_expert.model

    @staticmethod
    def refresh_steps(action, num_steps):
        action = ActionHeadAction.from_any(action)
        if num_steps < 1:
            raise ValueError("num_steps must be positive.")
        if action == ActionHeadAction.REFRESH:
            return tuple(range(num_steps))
        if action == ActionHeadAction.REUSE_MIDDLE:
            return tuple(sorted({0, num_steps - 1}))
        return (0,)

    def sample(self, state, pad_mask, kv, noise, num_steps, action):
        action = ActionHeadAction.from_any(action)
        refresh = self.refresh_steps(action, num_steps)
        cached_delta = None
        x_t = noise
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=noise.device)
        time = torch.tensor(1.0, dtype=torch.float32, device=noise.device)
        for step in range(num_steps):
            timestep = time.expand(noise.shape[0])
            if action == ActionHeadAction.REFRESH:
                velocity = self.model.denoise_step(state, pad_mask, kv, x_t, timestep)
            else:
                velocity, delta = self._denoise_with_cache(
                    state,
                    pad_mask,
                    kv,
                    x_t,
                    timestep,
                    None if step in refresh else cached_delta,
                )
                if step == 0:
                    cached_delta = delta
            x_t = x_t + dt * velocity
            time = time + dt
        depth = len(self.expert.layers)
        return x_t, {
            "action_head_action": int(action),
            "action_head_refresh_steps": len(refresh),
            "action_head_reuse_steps": num_steps - len(refresh),
            "action_head_computed_blocks": depth * len(refresh),
        }

    def _denoise_with_cache(self, state, pad_mask, kv, x_t, timestep, cached_delta):
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        hidden, suffix_mask, suffix_att, cond = self.model.embed_suffix(state, x_t, timestep)
        if self.expert.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            hidden = hidden.to(torch.bfloat16)
        if cached_delta is None:
            prefix_mask = pad_mask[:, None, :].expand(pad_mask.shape[0], suffix_mask.shape[1], -1)
            attention = self.model._prepare_attention_masks_4d(
                torch.cat([prefix_mask, make_att_2d_masks(suffix_mask, suffix_att)], dim=2)
            )
            position_ids = pad_mask.sum(dim=-1)[:, None] + suffix_mask.cumsum(dim=1) - 1
            cache_position = torch.arange(pad_mask.shape[1], pad_mask.shape[1] + hidden.shape[1], device=hidden.device)
            position_embeddings = self.expert.rotary_emb(hidden, position_ids)
            self.expert.config._attn_implementation = "eager"
            before = hidden
            for layer in self.expert.layers:
                hidden = layer(
                    hidden,
                    attention_mask=attention,
                    position_ids=position_ids,
                    past_key_value=kv,
                    use_cache=False,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    adarms_cond=cond,
                )[0]
            delta = (hidden - before).detach()
        else:
            delta = cached_delta
            hidden = hidden + delta
        # State/noise/time embeddings and the adaptive output norm stay current.
        normed = self.expert.norm(hidden, cond=cond)
        hidden = normed[0] if isinstance(normed, tuple) else normed
        hidden = hidden[:, -self.model.config.action_horizon :].float()
        return self.model.action_out_proj(hidden), delta
