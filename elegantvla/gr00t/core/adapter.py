"""GR00T input preparation, backbone recomputation and action-head adapter."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import tree
from transformers.feature_extraction_utils import BatchFeature

from gr00t.data.types import MessageType
from gr00t.policy.gr00t_policy import Gr00tPolicy

from elegantvla.gr00t.core.action_head import _ACTION_HEAD_REUSE_STRATEGY_ATTR


class Gr00tPolicyTemporalSkipAdapter:
    """External adapter that adds temporal-skip helper methods to a clean `Gr00tPolicy`."""

    def __init__(self, policy: Gr00tPolicy):
        self.policy = policy
        self.processor = policy.processor
        self.model = policy.model
        self._action_head = self.model.action_head
        self.collate_fn = policy.collate_fn
        self.modality_configs = policy.modality_configs
        self.embodiment_tag = policy.embodiment_tag
        self.static_inference_cache_enabled = True
        self._qwen_static_input_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    def configure_static_inference_cache(self, enabled: bool) -> None:
        self.static_inference_cache_enabled = bool(enabled)
        if not self.static_inference_cache_enabled:
            self._qwen_static_input_cache.clear()

        eagle_model = getattr(getattr(self.model, "backbone", None), "model", None)
        vision_model = getattr(eagle_model, "vision_model", None)
        vision_transformer = getattr(vision_model, "vision_model", None)
        embeddings = getattr(vision_transformer, "embeddings", None)
        encoder = getattr(vision_transformer, "encoder", None)
        rope_2d = getattr(encoder, "rope_2d", None)
        for module in (embeddings, rope_2d):
            if module is None:
                continue
            module._temporal_skip_static_layout_cache_enabled = bool(enabled)
            if not enabled:
                for cache_name in (
                    "_temporal_skip_static_layout_cache",
                    "_temporal_skip_freqs_layout_cache",
                ):
                    cache = getattr(module, cache_name, None)
                    if cache is not None:
                        cache.clear()

        text_processors = {
            id(processor): processor
            for processor in (
                getattr(self.collate_fn, "processor", None),
                getattr(getattr(self.model, "collator", None), "processor", None),
            )
            if processor is not None
        }
        for text_processor in text_processors.values():
            text_processor._temporal_skip_static_text_cache_enabled = bool(enabled)
            if not enabled:
                text_cache = getattr(
                    text_processor,
                    "_temporal_skip_static_text_cache",
                    None,
                )
                if text_cache is not None:
                    text_cache.clear()

    @staticmethod
    def _static_backbone_input_signature(
        collated_inputs: dict[str, Any],
    ) -> tuple[Any, ...] | None:
        input_ids = collated_inputs.get("input_ids")
        attention_mask = collated_inputs.get("attention_mask")
        if not isinstance(input_ids, torch.Tensor) or input_ids.device.type != "cpu":
            return None
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.device.type != "cpu":
            return None
        return (
            tuple(input_ids.shape),
            str(input_ids.dtype),
            input_ids.detach().contiguous().numpy().tobytes(),
            tuple(attention_mask.shape),
            str(attention_mask.dtype),
            attention_mask.detach().contiguous().numpy().tobytes(),
        )

    # ------------------------------------------------------------------
    # Processor/model input preparation
    # ------------------------------------------------------------------

    def prepare_inputs(
        self,
        observation: dict[str, Any],
        *,
        action_only: bool = False,
    ) -> dict[str, Any]:
        vla_step_data = self.policy._to_vla_step_data(self.policy._unbatch_observation(observation)[0])
        if action_only:
            embodiment_tag = vla_step_data.embodiment
            normalized_states, _ = self.processor.state_action_processor.apply(
                state=vla_step_data.states,
                action=vla_step_data.actions,
                embodiment_tag=embodiment_tag.value,
            )
            state_keys = self.processor.modality_configs[embodiment_tag.value]["state"].modality_keys
            state_tensor = torch.cat(
                [torch.from_numpy(normalized_states[key]) for key in state_keys],
                dim=-1,
            )
            state_tensor = F.pad(
                state_tensor,
                (0, self.processor.max_state_dim - state_tensor.shape[-1]),
            )
            collated_inputs = {
                "state": state_tensor.unsqueeze(0).to(torch.get_default_dtype()),
                "embodiment_id": torch.tensor(
                    [self.processor.embodiment_id_mapping[embodiment_tag.value]],
                    dtype=torch.long,
                ),
            }
        else:
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
            collated_inputs = self.collate_fn([self.processor(messages)])["inputs"]

        return {
            "collated_inputs": collated_inputs,
            "state": vla_step_data.states,
        }

    def decode_action(
        self,
        model_pred: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        normalized_action = model_pred["action_pred"].float()
        batched_states = {
            key: np.expand_dims(state[key], axis=0)
            for key in self.modality_configs["state"].modality_keys
        }
        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(),
            self.embodiment_tag,
            batched_states,
        )
        return {key: value.astype(np.float32) for key, value in unnormalized_action.items()}

    # ------------------------------------------------------------------
    # Backbone probing and cache construction
    # ------------------------------------------------------------------

    def _prepare_model_inputs(
        self,
        collated_inputs: dict[str, Any],
    ) -> tuple[BatchFeature, BatchFeature]:
        """Move shared backbone/action inputs once on the optimized cache path."""
        backbone = getattr(self.model, "backbone", None)
        action_head = getattr(self.model, "action_head", None)
        can_split_inputs = (
            self.static_inference_cache_enabled
            and "vlm_content" not in collated_inputs
            and backbone is not None
            and action_head is not None
            and hasattr(backbone, "prepare_input")
            and hasattr(action_head, "prepare_input")
        )
        if not can_split_inputs:
            return self.model.prepare_input(dict(collated_inputs))

        moved_tensors: dict[int, torch.Tensor] = {}

        def move_once(value: Any) -> Any:
            if not isinstance(value, torch.Tensor):
                return value
            identity = id(value)
            moved = moved_tensors.get(identity)
            if moved is None:
                moved = (
                    value.to(self.model.device, dtype=self.model.dtype)
                    if torch.is_floating_point(value)
                    else value.to(self.model.device)
                )
                moved_tensors[identity] = moved
            return moved

        device_inputs = tree.map_structure(move_once, dict(collated_inputs))
        return (
            backbone.prepare_input(device_inputs),
            action_head.prepare_input(device_inputs),
        )

    def _run_qwen_layer(
        self,
        layer: Any,
        hidden_states: torch.Tensor,
        probe_output: dict[str, Any],
    ) -> torch.Tensor:
        layer_outputs = layer(
            hidden_states,
            attention_mask=probe_output["causal_mask"],
            position_ids=probe_output["position_ids"],
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=probe_output["cache_position"],
            position_embeddings=probe_output["position_embeddings"],
        )
        return layer_outputs[0]

    @staticmethod
    def _middle_delta_cache(
        first_hidden_states: torch.Tensor,
        hidden_before_last: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "middle_delta": (hidden_before_last - first_hidden_states).detach(),
        }

    def _backbone_output_from_hidden_states(
        self,
        hidden_states: torch.Tensor,
        probe_output: dict[str, Any],
    ) -> BatchFeature:
        qwen_model = self.model.backbone.model.language_model.model
        hidden_states = qwen_model.norm(hidden_states)
        return BatchFeature(
            data={
                "backbone_features": hidden_states,
                "backbone_attention_mask": probe_output["backbone_attention_mask"],
                "image_mask": probe_output["image_mask"],
            }
        )

    def get_first_block_gate_probe(
        self,
        collated_inputs: dict[str, Any],
    ) -> tuple[dict[str, Any], Any]:
        static_signature = None
        if self.static_inference_cache_enabled:
            static_signature = self._static_backbone_input_signature(collated_inputs)
        backbone_inputs, action_inputs = self._prepare_model_inputs(collated_inputs)
        backbone = self.model.backbone
        backbone.set_frozen_modules_to_eval_mode()

        input_ids = backbone_inputs["input_ids"]
        attention_mask = backbone_inputs["attention_mask"]
        pixel_values = backbone_inputs["pixel_values"]
        cache_key = None
        cached_static_inputs = None
        if static_signature is not None:
            cache_key = (
                *static_signature,
                str(input_ids.device),
                str(self.model.dtype),
            )
            cached_static_inputs = self._qwen_static_input_cache.get(cache_key)

        if cached_static_inputs is None:
            input_embeds = backbone.model.language_model.get_input_embeddings()(input_ids)
            base_input_embeds = input_embeds.detach().clone()
            flat_input_ids = input_ids.reshape(-1)
            image_token_indices = (
                flat_input_ids == backbone.model.config.image_token_index
            ).nonzero(as_tuple=True)[0]
            backbone_attention_mask = attention_mask == 1
            image_mask = input_ids == backbone.model.config.image_token_index
        else:
            input_embeds = cached_static_inputs["base_input_embeds"].clone()
            image_token_indices = cached_static_inputs["image_token_indices"]
            backbone_attention_mask = cached_static_inputs["backbone_attention_mask"]
            image_mask = cached_static_inputs["image_mask"]

        vit_embeds = backbone.model.extract_feature(pixel_values)

        batch_size, seq_len, hidden_size = input_embeds.shape
        flat_input_embeds = input_embeds.reshape(batch_size * seq_len, hidden_size)
        num_image_tokens = min(image_token_indices.numel(), vit_embeds.shape[0])
        if num_image_tokens > 0:
            flat_input_embeds[image_token_indices[:num_image_tokens]] = vit_embeds[:num_image_tokens]

        input_embeds = flat_input_embeds.reshape(batch_size, seq_len, hidden_size)
        qwen_model = backbone.model.language_model.model
        if cached_static_inputs is None:
            cache_position = torch.arange(seq_len, device=input_embeds.device)
            position_ids = cache_position.unsqueeze(0)
            causal_mask = qwen_model._update_causal_mask(
                attention_mask,
                input_embeds,
                cache_position,
                None,
                False,
            )
            position_embeddings = qwen_model.rotary_emb(input_embeds, position_ids)
            if cache_key is not None:
                if len(self._qwen_static_input_cache) >= 8:
                    self._qwen_static_input_cache.pop(next(iter(self._qwen_static_input_cache)))
                self._qwen_static_input_cache[cache_key] = {
                    "base_input_embeds": base_input_embeds,
                    "image_token_indices": image_token_indices,
                    "cache_position": cache_position,
                    "position_ids": position_ids,
                    "causal_mask": causal_mask,
                    "position_embeddings": position_embeddings,
                    "backbone_attention_mask": backbone_attention_mask,
                    "image_mask": image_mask,
                }
        else:
            cache_position = cached_static_inputs["cache_position"]
            position_ids = cached_static_inputs["position_ids"]
            causal_mask = cached_static_inputs["causal_mask"]
            position_embeddings = cached_static_inputs["position_embeddings"]
        hidden_states = self._run_qwen_layer(
            qwen_model.layers[0],
            input_embeds,
            {
                "causal_mask": causal_mask,
                "position_ids": position_ids,
                "cache_position": cache_position,
                "position_embeddings": position_embeddings,
            },
        )
        probe_output = {
            "causal_mask": causal_mask,
            "position_ids": position_ids,
            "cache_position": cache_position,
            "position_embeddings": position_embeddings,
            "backbone_attention_mask": backbone_attention_mask,
            "image_mask": image_mask,
            "hidden_states": hidden_states,
        }
        return probe_output, action_inputs

    def continue_backbone_from_probe(self, probe_output: dict[str, Any]) -> BatchFeature:
        self.model.backbone.set_frozen_modules_to_eval_mode()
        qwen_model = self.model.backbone.model.language_model.model
        hidden_states = probe_output["hidden_states"]
        for layer in qwen_model.layers[1:]:
            hidden_states = self._run_qwen_layer(layer, hidden_states, probe_output)
        return self._backbone_output_from_hidden_states(hidden_states, probe_output)

    def continue_backbone_from_probe_with_middle_cache(
        self,
        probe_output: dict[str, Any],
    ) -> tuple[BatchFeature, dict[str, torch.Tensor]]:
        self.model.backbone.set_frozen_modules_to_eval_mode()
        qwen_model = self.model.backbone.model.language_model.model
        first_hidden_states = probe_output["hidden_states"]
        layers = list(qwen_model.layers)
        suffix_graph = getattr(qwen_model, "_temporal_skip_qwen_suffix_graph", None)
        if suffix_graph is not None and len(layers) > 1:
            hidden_before_last, hidden_states = suffix_graph(
                first_hidden_states,
                attention_mask=probe_output["causal_mask"],
                position_ids=probe_output["position_ids"],
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=probe_output["cache_position"],
                position_embeddings=probe_output["position_embeddings"],
            )
        else:
            hidden_before_last = first_hidden_states
            for layer in layers[1:-1]:
                hidden_before_last = self._run_qwen_layer(
                    layer,
                    hidden_before_last,
                    probe_output,
                )
            if len(layers) > 1:
                hidden_states = self._run_qwen_layer(
                    layers[-1],
                    hidden_before_last,
                    probe_output,
                )
            else:
                hidden_states = hidden_before_last
        middle_cache = {
            "middle_delta": (hidden_before_last - first_hidden_states).detach(),
        }
        middle_cache["post_first_delta"] = (hidden_states - first_hidden_states).detach()
        return self._backbone_output_from_hidden_states(hidden_states, probe_output), middle_cache

    def continue_backbone_from_middle_cache(
        self,
        probe_output: dict[str, Any],
        middle_cache: dict[str, torch.Tensor] | None,
    ) -> tuple[BatchFeature, dict[str, torch.Tensor] | None, bool]:
        first_hidden_states = probe_output["hidden_states"]
        middle_delta = None if middle_cache is None else middle_cache.get("middle_delta")
        if middle_delta is None or tuple(middle_delta.shape) != tuple(first_hidden_states.shape):
            backbone_output, refreshed_middle_cache = (
                self.continue_backbone_from_probe_with_middle_cache(probe_output)
            )
            return backbone_output, refreshed_middle_cache, False

        self.model.backbone.set_frozen_modules_to_eval_mode()
        qwen_model = self.model.backbone.model.language_model.model
        layers = list(qwen_model.layers)
        hidden_states = first_hidden_states + middle_delta.to(
            device=first_hidden_states.device,
            dtype=first_hidden_states.dtype,
        )
        if len(layers) > 1:
            hidden_states = self._run_qwen_layer(layers[-1], hidden_states, probe_output)
        return self._backbone_output_from_hidden_states(hidden_states, probe_output), None, True

    def continue_backbone_from_post_first_cache(
        self,
        probe_output: dict[str, Any],
        middle_cache: dict[str, torch.Tensor] | None,
    ) -> tuple[BatchFeature, dict[str, torch.Tensor] | None, bool]:
        first_hidden_states = probe_output["hidden_states"]
        post_first_delta = (
            None if middle_cache is None else middle_cache.get("post_first_delta")
        )
        if post_first_delta is None or tuple(post_first_delta.shape) != tuple(
            first_hidden_states.shape
        ):
            backbone_output, refreshed_middle_cache = (
                self.continue_backbone_from_probe_with_middle_cache(probe_output)
            )
            return backbone_output, refreshed_middle_cache, False

        hidden_states = first_hidden_states + post_first_delta.to(
            device=first_hidden_states.device,
            dtype=first_hidden_states.dtype,
        )
        return self._backbone_output_from_hidden_states(hidden_states, probe_output), None, True

    def continue_backbone_from_probe_with_layer_boundary_cache(
        self,
        probe_output: dict[str, Any],
    ) -> tuple[BatchFeature, dict[str, tuple[torch.Tensor, ...]]]:
        """Run the full backbone and retain every language-layer boundary."""
        self.model.backbone.set_frozen_modules_to_eval_mode()
        qwen_model = self.model.backbone.model.language_model.model
        hidden_states = probe_output["hidden_states"]
        layer_boundaries = [hidden_states.detach()]
        for layer in qwen_model.layers[1:]:
            hidden_states = self._run_qwen_layer(layer, hidden_states, probe_output)
            layer_boundaries.append(hidden_states.detach())
        cache = {"layer_boundaries": tuple(layer_boundaries)}
        return self._backbone_output_from_hidden_states(hidden_states, probe_output), cache

    def continue_backbone_from_layer_subset_cache(
        self,
        probe_output: dict[str, Any],
        layer_boundary_cache: dict[str, tuple[torch.Tensor, ...]] | None,
        recompute_layers: tuple[int, ...],
    ) -> tuple[BatchFeature, dict[str, tuple[torch.Tensor, ...]] | None, bool]:
        """Recompute selected layers and transport cached residuals between them."""
        self.model.backbone.set_frozen_modules_to_eval_mode()
        qwen_model = self.model.backbone.model.language_model.model
        layers = list(qwen_model.layers)
        selected_layers = tuple(sorted(set(int(index) for index in recompute_layers)))
        if not selected_layers or selected_layers[0] != 0:
            raise ValueError("Layer-subset reuse must include the already-computed probe layer 0.")
        if selected_layers[-1] >= len(layers):
            raise ValueError(
                f"Recompute layer {selected_layers[-1]} is outside a {len(layers)}-layer backbone."
            )

        first_hidden_states = probe_output["hidden_states"]
        boundaries = (
            None
            if layer_boundary_cache is None
            else layer_boundary_cache.get("layer_boundaries")
        )
        cache_is_valid = (
            boundaries is not None
            and len(boundaries) == len(layers)
            and all(tuple(value.shape) == tuple(first_hidden_states.shape) for value in boundaries)
        )
        if not cache_is_valid:
            backbone_output, refreshed_cache = (
                self.continue_backbone_from_probe_with_layer_boundary_cache(probe_output)
            )
            return backbone_output, refreshed_cache, False

        hidden_states = first_hidden_states
        previous_selected_layer = 0
        for layer_index in selected_layers[1:]:
            if layer_index > previous_selected_layer + 1:
                cached_delta = boundaries[layer_index - 1] - boundaries[previous_selected_layer]
                hidden_states = hidden_states + cached_delta.to(
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
            hidden_states = self._run_qwen_layer(
                layers[layer_index],
                hidden_states,
                probe_output,
            )
            previous_selected_layer = layer_index

        final_layer_index = len(layers) - 1
        if previous_selected_layer < final_layer_index:
            cached_delta = boundaries[final_layer_index] - boundaries[previous_selected_layer]
            hidden_states = hidden_states + cached_delta.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        return self._backbone_output_from_hidden_states(hidden_states, probe_output), None, True

    # ------------------------------------------------------------------
    # Action-head segmented execution
    # ------------------------------------------------------------------

    def process_backbone_output_for_action_head(
        self,
        backbone_output: BatchFeature,
    ) -> BatchFeature:
        return self._action_head.process_backbone_output(backbone_output)

    def run_action_head(
        self,
        *,
        backbone_cache: BatchFeature | None = None,
        backbone_output: BatchFeature | None = None,
        action_input: BatchFeature | None = None,
        collated_inputs: dict[str, Any] | None = None,
        reuse_strategy: str | None,
    ) -> tuple[BatchFeature, BatchFeature]:
        exact_backbone_reuse = backbone_cache is not None
        if backbone_cache is None:
            backbone_cache = self._action_head.process_backbone_output(backbone_output)
        backbone_cache[self._action_head._EXACT_BACKBONE_REUSE_KEY] = exact_backbone_reuse

        if action_input is None:
            state = collated_inputs["state"].to(self.model.device, dtype=self.model.dtype)
            embodiment_id = collated_inputs["embodiment_id"].to(self.model.device)
        else:
            state = action_input.state
            embodiment_id = action_input.embodiment_id

        self._action_head.set_frozen_modules_to_eval_mode()
        state_features = self._action_head.state_encoder(
            state,
            embodiment_id,
        )
        if reuse_strategy is not None:
            reuse_strategy = str(reuse_strategy).strip().lower()
        use_compiled_cache = getattr(self._action_head, "_compiledcached_enabled", False)
        action_graph_proxy = self._action_head.model
        can_dispatch_action_graph = hasattr(
            action_graph_proxy,
            "set_temporal_skip_force_eager",
        )
        inference_steps_override = getattr(self.policy, "num_inference_steps", None)
        original_inference_steps = None
        if inference_steps_override is not None:
            original_inference_steps = self._action_head.num_inference_timesteps
            self._action_head.num_inference_timesteps = int(inference_steps_override)
        if use_compiled_cache and reuse_strategy is not None:
            setattr(
                self._action_head,
                _ACTION_HEAD_REUSE_STRATEGY_ATTR,
                reuse_strategy,
            )
        if can_dispatch_action_graph:
            # A full action-head refresh is graph-replayable. Inner denoising
            # reuse must instead reach the compiled residual-cache path.
            action_graph_proxy.set_temporal_skip_force_eager(
                use_compiled_cache and reuse_strategy is not None
            )
        try:
            model_pred = self._action_head.get_action_with_features(
                backbone_features=backbone_cache.backbone_features,
                state_features=state_features,
                embodiment_id=embodiment_id,
                backbone_output=backbone_cache,
            )
        finally:
            if original_inference_steps is not None:
                self._action_head.num_inference_timesteps = original_inference_steps
            if can_dispatch_action_graph:
                action_graph_proxy.set_temporal_skip_force_eager(False)
            if use_compiled_cache and reuse_strategy is not None:
                delattr(self._action_head, _ACTION_HEAD_REUSE_STRATEGY_ATTR)
        return model_pred, backbone_cache


__all__ = ["Gr00tPolicyTemporalSkipAdapter"]
