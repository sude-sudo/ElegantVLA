"""OpenPI transforms and public API; scheduling belongs to the wrapper."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
import torch
from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import nnx_utils
from openpi_client import base_policy as _base_policy

from elegantvla.openpi.core.config import PrefixVLMCacheConfig, TemporalSkipConfig
from elegantvla.openpi.core.recorder import PolicyRecorder as PolicyRecorder

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model,
        *,
        rng=None,
        transforms: Sequence = (),
        output_transforms: Sequence = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        prefix_vlm_cache: PrefixVLMCacheConfig | None = None,
        temporal_skip: TemporalSkipConfig | None = None,
        schedule_policy=None,
    ):
        self._legacy_policy = None
        self._temporal_skip = temporal_skip or TemporalSkipConfig()
        legacy_enabled = prefix_vlm_cache is not None and (
            prefix_vlm_cache.enabled or prefix_vlm_cache.intra_action_cache_enabled
        )
        if legacy_enabled:
            if self._temporal_skip.enabled or schedule_policy is not None:
                raise ValueError("Choose temporal_skip or legacy prefix_vlm_cache, not both.")
            from elegantvla.openpi.legacy.policy import Policy as LegacyPolicy

            self._legacy_policy = LegacyPolicy(
                model,
                rng=rng,
                transforms=transforms,
                output_transforms=output_transforms,
                sample_kwargs=sample_kwargs,
                metadata=metadata,
                pytorch_device=pytorch_device,
                is_pytorch=is_pytorch,
                prefix_vlm_cache=prefix_vlm_cache,
            )
            self._model = self._legacy_policy._model
            return
        if self._temporal_skip.enabled and not is_pytorch:
            raise ValueError("The GR00T-style OpenPI backend currently requires a PyTorch checkpoint.")
        if schedule_policy is not None and not self._temporal_skip.enabled:
            raise ValueError("Enable temporal_skip before supplying a schedule policy.")
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = dict(sample_kwargs or {})
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self.wrapper = None
        if is_pytorch:
            self._model = model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = self._model.sample_actions
            if self._temporal_skip.enabled:
                from elegantvla.openpi.core.wrapper import TemporalSkipPolicyWrapper

                self.wrapper = TemporalSkipPolicyWrapper(
                    self._model, config=self._temporal_skip, schedule_policy=schedule_policy
                )
        else:
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng if rng is not None else jax.random.key(0)

    def reset(self):
        if self._legacy_policy is not None:
            return self._legacy_policy.reset()
        if self.wrapper is not None:
            self.wrapper.reset()

    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        if self._legacy_policy is not None:
            return self._legacy_policy.infer(obs, noise=noise)
        inputs = self._input_transform(jax.tree.map(lambda value: value, obs))
        if self._is_pytorch_model:
            inputs = jax.tree.map(
                lambda value: torch.from_numpy(np.array(value)).to(self._pytorch_device)[None, ...], inputs
            )
            device_or_rng = self._pytorch_device
        else:
            inputs = jax.tree.map(lambda value: jnp.asarray(value)[np.newaxis, ...], inputs)
            self._rng, device_or_rng = jax.random.split(self._rng)
        kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)
            kwargs["noise"] = noise[None, ...] if noise.ndim == 2 else noise
        observation = _model.Observation.from_dict(inputs)
        start = time.monotonic()
        info = {}
        if self.wrapper is None:
            actions = self._sample_actions(device_or_rng, observation, **kwargs)
        else:
            actions, info = self.wrapper.infer(
                observation,
                motion_observation=self._motion_observation(obs),
                prompt_key=obs.get("prompt"),
                **kwargs,
            )
        outputs = {"state": inputs["state"], "actions": actions}
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda value: np.asarray(value[0].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda value: np.asarray(value[0]), outputs)
        elapsed = time.monotonic() - start
        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {"infer_ms": elapsed * 1000, **info}
        return outputs

    def _motion_observation(self, obs):
        if "scheduler_motion" in obs:
            return obs["scheduler_motion"]
        key = self._temporal_skip.motion_state_key
        if key is None or key not in obs:
            return {}
        state = np.asarray(obs[key], dtype=np.float32).reshape(-1)
        config = self._temporal_skip
        indices = (*config.translation_indices, *config.rotation_indices, config.gripper_index)
        if not indices or max(indices) >= state.size:
            return {}
        return {
            "state": {
                "eef_pose": state[list(config.translation_indices + config.rotation_indices)],
                "gripper": state[config.gripper_index],
            }
        }

    @property
    def metadata(self):
        return self._legacy_policy.metadata if self._legacy_policy is not None else self._metadata
