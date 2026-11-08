"""Public OpenPI policy interface and lifecycle."""

from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

from elegantvla.openpi.legacy._helpers import (
    _load_prefix_vlm_action_deviation_gate,
    _load_prefix_vlm_adaptive_scheduler,
    _prefix_action_deviation_info,
)
from elegantvla.openpi.legacy.config import PrefixVLMCacheConfig
from elegantvla.openpi.legacy.diagnostics import CacheDiagnostics
from elegantvla.openpi.legacy.rule_policy import ThresholdScheduler
from elegantvla.openpi.legacy.runtime import PrefixCacheRuntime


BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(ThresholdScheduler, PrefixCacheRuntime, CacheDiagnostics, BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        prefix_vlm_cache: PrefixVLMCacheConfig | None = None,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
            prefix_vlm_cache: Optional cross-environment cache for the VLM/prefix prefill.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._prefix_vlm_cache_config = prefix_vlm_cache or PrefixVLMCacheConfig()
        self._prefix_vlm_cached_residuals = None
        self._prefix_vlm_cached_sublayer_deltas = None
        self._prefix_vlm_cached_kv = None
        self._prefix_vlm_cached_prefix_pad_masks = None
        self._prefix_vlm_cached_first_layer = None
        self._prefix_vlm_cache_bank = []
        self._prefix_vlm_cached_state_signature = None
        self._prefix_vlm_previous_state_signature = None
        self._prefix_vlm_motion_state_buffer = []
        self._prefix_vlm_stable_state_deltas = 0
        self._prefix_vlm_signature = None
        self._prefix_vlm_prompt_key = None
        self._prefix_vlm_accumulated_image_l2 = 0.0
        self._prefix_vlm_previous_raw_images = None
        self._prefix_vlm_current_prompt_text = ""
        self._prefix_vlm_queries = 0
        self._prefix_vlm_episode_queries = 0
        self._prefix_vlm_episode_hits = 0
        self._prefix_vlm_continuous_cached = 0
        self._prefix_vlm_skip_v4_no_image_remaining = 0
        self._prefix_vlm_active_compute_layer_mask = None
        self._prefix_vlm_active_no_image_budget = None
        self._prefix_vlm_active_motion_tier = "disabled"
        self._prefix_vlm_active_motion_signal = np.nan
        self._prefix_vlm_active_motion_cka_threshold = np.nan
        self._prefix_vlm_active_motion_cached_state_delta_threshold = np.nan
        self._prefix_vlm_active_motion_prompt_guard = ""
        self._prefix_vlm_last_sublayer_reused = 0
        self._prefix_vlm_last_sublayer_total = 0
        self._prefix_vlm_last_sublayer_attn_mask = ""
        self._prefix_vlm_last_sublayer_mlp_mask = ""
        self._prefix_vlm_last_sublayer_cka_gate_accept = True
        self._prefix_vlm_last_sublayer_cka_gate_threshold = float(
            self._prefix_vlm_cache_config.sublayer_cache_min_first_layer_cka
        )
        self._prefix_vlm_last_cross_layer_residual_reused = 0
        self._prefix_vlm_last_cross_layer_residual_total = 0
        self._prefix_vlm_last_cross_layer_residual_mask = ""
        self._prefix_vlm_similarity_features = []
        self._prefix_vlm_layer_similarity_features = []
        self._prefix_vlm_adaptive_scheduler = _load_prefix_vlm_adaptive_scheduler(
            self._prefix_vlm_cache_config.adaptive_scheduler_model_path
        )
        self._prefix_vlm_action_deviation_gate = _load_prefix_vlm_action_deviation_gate(
            self._prefix_vlm_cache_config.action_deviation_gate_model_path
        )
        self._pytorch_compiled_prefix_vlm_hit = {}
        self._pytorch_compiled_intra_action_suffix_fns = {}
        self._action_suffix_cached_residuals = None
        self._action_suffix_cached_num_steps = None

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng if rng is not None else jax.random.key(0)
            if self._prefix_vlm_cache_config.intra_action_cache_enabled:
                self._sample_actions_with_intra_action_cache = nnx_utils.module_jit(
                    model.sample_actions_with_intra_action_cache
                )
            if self._prefix_vlm_cache_config.enabled:
                self._sample_actions_with_prefix_refresh_cache = nnx_utils.module_jit(
                    model.sample_actions_with_prefix_refresh_cache
                )
                self._sample_actions_with_cached_prefix_middle = nnx_utils.module_jit(
                    model.sample_actions_with_cached_prefix_middle
                )
                self._prefill_prefix_cache_with_residuals = nnx_utils.module_jit(
                    model.prefill_prefix_cache_with_residuals
                )
                self._prefill_prefix_cache_with_cached_middle = nnx_utils.module_jit(
                    model.prefill_prefix_cache_with_cached_middle
                )
                self._sample_actions_with_prefix_cache = nnx_utils.module_jit(model.sample_actions_with_prefix_cache)

    @override
    def reset(self) -> None:
        self._prefix_vlm_cached_residuals = None
        self._prefix_vlm_cached_sublayer_deltas = None
        self._prefix_vlm_cached_kv = None
        self._prefix_vlm_cached_prefix_pad_masks = None
        self._prefix_vlm_cached_first_layer = None
        self._prefix_vlm_cache_bank = []
        self._prefix_vlm_cached_state_signature = None
        self._prefix_vlm_previous_state_signature = None
        self._prefix_vlm_motion_state_buffer = []
        self._prefix_vlm_stable_state_deltas = 0
        self._prefix_vlm_signature = None
        self._prefix_vlm_prompt_key = None
        self._prefix_vlm_accumulated_image_l2 = 0.0
        self._prefix_vlm_previous_raw_images = None
        self._prefix_vlm_current_prompt_text = ""
        self._prefix_vlm_queries = 0
        self._prefix_vlm_episode_queries = 0
        self._prefix_vlm_episode_hits = 0
        self._prefix_vlm_continuous_cached = 0
        self._prefix_vlm_skip_v4_no_image_remaining = 0
        self._prefix_vlm_active_compute_layer_mask = None
        self._prefix_vlm_active_no_image_budget = None
        self._prefix_vlm_active_motion_tier = "disabled"
        self._prefix_vlm_active_motion_signal = np.nan
        self._prefix_vlm_active_motion_cka_threshold = np.nan
        self._prefix_vlm_active_motion_cached_state_delta_threshold = np.nan
        self._prefix_vlm_active_motion_prompt_guard = ""
        self._prefix_vlm_last_sublayer_reused = 0
        self._prefix_vlm_last_sublayer_total = 0
        self._prefix_vlm_last_sublayer_attn_mask = ""
        self._prefix_vlm_last_sublayer_mlp_mask = ""
        self._prefix_vlm_similarity_features = []
        self._prefix_vlm_layer_similarity_features = []
        self._action_suffix_cached_residuals = None
        self._action_suffix_cached_num_steps = None

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        raw_inputs = jax.tree.map(lambda x: x, obs)
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        prefix_cache_inputs = inputs if self._use_prefix_vlm_cache else None
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        cache_info: dict[str, Any] = {}
        if self._use_prefix_vlm_cache:
            actions, cache_info = self._infer_with_prefix_vlm_cache(
                sample_rng_or_pytorch_device,
                observation,
                sample_kwargs,
                prefix_cache_inputs,
                raw_inputs,
            )
        else:
            actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        action_deviation_candidate_actions = cache_info.pop("_prefix_vlm_action_deviation_candidate_actions", None)
        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        if action_deviation_candidate_actions is not None:
            candidate_outputs = {
                "state": inputs["state"],
                "actions": action_deviation_candidate_actions,
            }
            if self._is_pytorch_model:
                candidate_outputs = jax.tree.map(
                    lambda x: np.asarray(x[0, ...].detach().cpu()),
                    candidate_outputs,
                )
            else:
                candidate_outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), candidate_outputs)
            candidate_outputs = self._output_transform(candidate_outputs)
            cache_info.update(
                _prefix_action_deviation_info(
                    outputs.get("actions"),
                    candidate_outputs.get("actions"),
                    self._prefix_vlm_cache_config,
                )
            )
        elif cache_info.get("prefix_vlm_cache_action_deviation_label_enabled"):
            cache_info.setdefault("prefix_vlm_cache_action_deviation_evaluated", False)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        outputs["policy_timing"].update(cache_info)
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def _use_prefix_vlm_cache(self) -> bool:
        return self._prefix_vlm_cache_config.enabled or (
            self._prefix_vlm_cache_config.intra_action_cache_enabled
        )


from elegantvla.openpi.core.recorder import PolicyRecorder
