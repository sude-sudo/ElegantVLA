"""OpenPI model operations behind the shared temporal-skip schedule."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers.cache_utils import DynamicCache

from elegantvla.openpi.core.action_head import OpenPIActionHead


@dataclass
class PrefixOutput:
    kv: Any
    pad_mask: torch.Tensor


@dataclass
class MiddleCache:
    delta: torch.Tensor
    kv: Any
    pad_mask: torch.Tensor


@dataclass
class PrefixProbe:
    hidden_states: torch.Tensor
    kv: Any
    pad_mask: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    cache_position: torch.Tensor
    position_embeddings: Any

    @property
    def gate_feature(self):
        return self.hidden_states[self.pad_mask.bool()]


class OpenPIPolicyTemporalSkipAdapter:
    def __init__(self, model, *, first_layers: int = 1, last_layers: int = 1):
        self.model = model
        self.language_model = model.paligemma_with_expert.paligemma.language_model
        self.depth = len(self.language_model.layers)
        if first_layers < 1 or last_layers < 1 or first_layers + last_layers > self.depth:
            raise ValueError("Boundary layers must be positive and fit the VLM depth.")
        self.first_layers = first_layers
        self.last_layers = last_layers
        self.action_head = OpenPIActionHead(model)

    def cache_key(self, observation, *, prompt_key=None):
        def values(tensor):
            return None if tensor is None else tuple(tensor.detach().cpu().reshape(-1).tolist())

        tokens = observation.tokenized_prompt
        return (
            prompt_key if prompt_key is not None else values(tokens),
            None if tokens is None else tuple(tokens.shape),
            values(observation.tokenized_prompt_mask),
            tuple(
                (key, tuple(value.shape), str(value.dtype), str(value.device))
                for key, value in observation.images.items()
            ),
            tuple((key, values(value)) for key, value in observation.image_masks.items()),
            str(observation.state.device),
            str(observation.state.dtype),
        )

    def prepare_inputs(self, observation, *, noise=None, num_steps: int = 10):
        if num_steps < 1:
            raise ValueError("num_steps must be positive.")
        state = observation.state
        if state.shape[0] != 1:
            raise ValueError("Each temporal-skip wrapper owns one episode; batch size must be 1.")
        shape = (state.shape[0], self.model.config.action_horizon, self.model.config.action_dim)
        if noise is None:
            noise = self.model.sample_noise(shape, state.device)
        if tuple(noise.shape) != shape:
            raise ValueError(f"Expected noise shape {shape}, got {tuple(noise.shape)}.")
        return {"observation": observation, "state": state, "noise": noise, "num_steps": num_steps}

    def get_first_block_gate_probe(self, prepared) -> PrefixProbe:
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        images, image_masks, tokens, token_masks, state = self.model._preprocess_observation(
            prepared["observation"], train=False
        )
        prepared["state"] = state
        embeddings, pad_mask, att_mask = self.model.embed_prefix(images, image_masks, tokens, token_masks)
        attention = self.model._prepare_attention_masks_4d(make_att_2d_masks(pad_mask, att_mask))
        position_ids = torch.cumsum(pad_mask, dim=1) - 1
        self.language_model.config._attn_implementation = "eager"
        hidden = embeddings
        if self.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            hidden = hidden.to(torch.bfloat16)
        probe = PrefixProbe(
            hidden_states=hidden,
            kv=DynamicCache(),
            pad_mask=pad_mask,
            attention_mask=attention,
            position_ids=position_ids,
            cache_position=torch.arange(hidden.shape[1], device=hidden.device),
            position_embeddings=self.language_model.rotary_emb(hidden, position_ids),
        )
        probe.hidden_states = self._run_layer(0, hidden, probe, probe.kv)
        return probe

    def _run_layer(self, index, hidden, probe, cache):
        return self.language_model.layers[index](
            hidden,
            attention_mask=probe.attention_mask,
            position_ids=probe.position_ids,
            past_key_value=cache,
            use_cache=True,
            cache_position=probe.cache_position,
            position_embeddings=probe.position_embeddings,
            adarms_cond=None,
        )[0]

    @staticmethod
    def _copy_kv(cache, indices):
        copied = DynamicCache()
        for index in indices:
            copied.update(cache.key_cache[index], cache.value_cache[index], index)
        return copied

    def continue_backbone_from_probe_with_middle_cache(self, probe):
        hidden = probe.hidden_states
        cache = self._copy_kv(probe.kv, [0])
        for index in range(1, self.first_layers):
            hidden = self._run_layer(index, hidden, probe, cache)
        middle_input = hidden
        for index in range(self.first_layers, self.depth - self.last_layers):
            hidden = self._run_layer(index, hidden, probe, cache)
        middle_delta = (hidden - middle_input).detach()
        for index in range(self.depth - self.last_layers, self.depth):
            hidden = self._run_layer(index, hidden, probe, cache)
        # The action expert consumes layer-wise prefix K/V, not the final norm output.
        output = PrefixOutput(cache, probe.pad_mask)
        anchor = MiddleCache(middle_delta, cache, probe.pad_mask.detach().clone())
        return output, anchor

    def continue_backbone_from_middle_cache(self, probe, middle_cache):
        compatible = (
            middle_cache is not None
            and middle_cache.delta.shape == probe.hidden_states.shape
            and middle_cache.delta.dtype == probe.hidden_states.dtype
            and middle_cache.delta.device == probe.hidden_states.device
            and len(middle_cache.kv.key_cache) == self.depth
            and torch.equal(middle_cache.pad_mask, probe.pad_mask)
        )
        if not compatible:
            output, anchor = self.continue_backbone_from_probe_with_middle_cache(probe)
            return output, anchor, False
        hidden = probe.hidden_states
        cache = self._copy_kv(probe.kv, [0])
        for index in range(1, self.first_layers):
            hidden = self._run_layer(index, hidden, probe, cache)
        hidden = hidden + middle_cache.delta
        for index in range(self.first_layers, self.depth - self.last_layers):
            cache.update(middle_cache.kv.key_cache[index], middle_cache.kv.value_cache[index], index)
        for index in range(self.depth - self.last_layers, self.depth):
            hidden = self._run_layer(index, hidden, probe, cache)
        return PrefixOutput(cache, probe.pad_mask), None, True

    def run_action_head(self, backbone_output, prepared, action):
        return self.action_head.sample(
            prepared["state"],
            backbone_output.pad_mask,
            backbone_output.kv,
            prepared["noise"],
            prepared["num_steps"],
            action,
        )
