"""OpenPI prefix recomputation, residual reuse and action sampling."""

from typing import Any

import numpy as np
import torch

from openpi.models import model as _model
from openpi.shared import array_typing as at

from elegantvla.openpi.legacy._helpers import (
    _prefix_image_signature,
    _prefix_mask_to_spec,
    _prefix_observation_info_to_cache_info,
    _prefix_prompt_key,
    _prefix_raw_prompt,
    _prefix_state_info_to_cache_info,
    _prefix_state_signature,
    _torch_gated_residual,
    _torch_linear_cka,
    _torch_relative_l2,
)


class PrefixCacheRuntime:
    """Internal Policy methods; Policy owns the model and cache state."""

    def _infer_with_prefix_vlm_cache(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        sample_kwargs: dict[str, Any],
        inputs: dict,
        raw_inputs: dict | None = None,
    ):
        self._prefix_vlm_current_prompt_text = _prefix_raw_prompt(raw_inputs)
        if self._is_pytorch_model:
            return self._infer_with_pytorch_prefix_vlm_cache(observation, sample_kwargs, inputs, raw_inputs)

        if not self._prefix_vlm_cache_config.enabled and self._prefix_vlm_cache_config.intra_action_cache_enabled:
            actions = self._sample_actions_with_intra_action_cache(
                rng,
                observation,
                **sample_kwargs,
                fn_compute_blocks=int(self._prefix_vlm_cache_config.intra_action_cache_fn_compute_blocks),
                bn_compute_blocks=int(self._prefix_vlm_cache_config.intra_action_cache_bn_compute_blocks),
                refresh_interval=int(self._prefix_vlm_cache_config.intra_action_cache_refresh_interval),
            )
            num_steps = int(sample_kwargs.get("num_steps", 10))
            return actions, self._jax_intra_action_cache_info(num_steps)

        state_signature = _prefix_state_signature(raw_inputs, inputs)
        bypass_cache_path, bypass_reason, bypass_state_info = self._prefix_vlm_cache_path_bypass_info(state_signature)
        if bypass_cache_path:
            actions = self._sample_actions(rng, observation, **sample_kwargs)
            self._update_prefix_vlm_state_phase(state_signature)
            self._update_prefix_vlm_raw_observation(raw_inputs)
            self._prefix_vlm_queries += 1
            self._prefix_vlm_episode_queries += 1
            self._prefix_vlm_continuous_cached = 0
            self._prefix_vlm_accumulated_image_l2 = 0.0
            depth = int(self._model.llm_depth)
            return actions, {
                "prefix_vlm_cache_enabled": True,
                "prefix_vlm_cache_hit": False,
                "prefix_vlm_cache_reason": bypass_reason,
                "prefix_vlm_cache_adaptive_bypass": True,
                "prefix_vlm_cache_computed_blocks": depth,
                "prefix_vlm_cache_skipped_blocks": 0,
                "prefix_vlm_cache_image_signature_l2": np.inf,
                "prefix_vlm_cache_image_signature_accum_l2": 0.0,
                **_prefix_state_info_to_cache_info(bypass_state_info),
                "prefix_vlm_cache_query": self._prefix_vlm_queries,
                "prefix_vlm_cache_episode_query": self._prefix_vlm_episode_queries,
                "prefix_vlm_cache_episode_hits": self._prefix_vlm_episode_hits,
                "prefix_vlm_cache_continuous_hits": self._prefix_vlm_continuous_cached,
            }

        signature = _prefix_image_signature(inputs) if self._prefix_vlm_requires_image_signature() else None
        prompt_key = _prefix_prompt_key(inputs)
        observation_info = self._prefix_vlm_observation_gate_info(raw_inputs)
        use_cache, cache_reason, signature_l2, state_info = self._prefix_vlm_cache_decision(
            signature,
            prompt_key,
            state_signature,
            observation_info,
        )

        if use_cache:
            prefix_compute_mask = self._prefix_vlm_compute_mask()
            actions, prefix_mask, kv_cache, prefix_residuals = self._sample_actions_with_cached_prefix_middle(
                rng,
                observation,
                self._prefix_vlm_cached_residuals,
                self._prefix_vlm_cached_kv,
                prefix_compute_mask,
                **sample_kwargs,
            )
        else:
            prefix_compute_mask = None
            actions, prefix_mask, kv_cache, prefix_residuals = self._sample_actions_with_prefix_refresh_cache(
                rng,
                observation,
                **sample_kwargs,
            )

        self._prefix_vlm_cached_residuals = prefix_residuals
        self._prefix_vlm_cached_kv = kv_cache
        self._prefix_vlm_cached_state_signature = state_signature
        self._prefix_vlm_signature = signature
        self._prefix_vlm_prompt_key = prompt_key
        self._update_prefix_vlm_state_phase(state_signature)
        self._update_prefix_vlm_raw_observation(raw_inputs)
        self._prefix_vlm_queries += 1
        self._prefix_vlm_episode_queries += 1
        if use_cache:
            self._prefix_vlm_continuous_cached += 1
            self._prefix_vlm_episode_hits += 1
            self._prefix_vlm_accumulated_image_l2 += float(signature_l2)
        else:
            self._prefix_vlm_continuous_cached = 0
            self._prefix_vlm_accumulated_image_l2 = 0.0

        depth = int(prefix_residuals.shape[0])
        computed_blocks = depth if prefix_compute_mask is None else int(np.asarray(prefix_compute_mask).sum())
        return actions, {
            "prefix_vlm_cache_enabled": True,
            "prefix_vlm_cache_hit": bool(use_cache),
            "prefix_vlm_cache_reason": cache_reason,
            "prefix_vlm_cache_computed_blocks": computed_blocks,
            "prefix_vlm_cache_skipped_blocks": depth - computed_blocks,
            "prefix_vlm_cache_image_signature_l2": float(signature_l2),
            "prefix_vlm_cache_image_signature_accum_l2": float(self._prefix_vlm_accumulated_image_l2),
            **_prefix_state_info_to_cache_info(state_info),
            **_prefix_observation_info_to_cache_info(observation_info),
            "prefix_vlm_cache_query": self._prefix_vlm_queries,
            "prefix_vlm_cache_episode_query": self._prefix_vlm_episode_queries,
            "prefix_vlm_cache_episode_hits": self._prefix_vlm_episode_hits,
            "prefix_vlm_cache_continuous_hits": self._prefix_vlm_continuous_cached,
        }

    def _infer_with_pytorch_prefix_vlm_cache(
        self,
        observation: _model.Observation,
        sample_kwargs: dict[str, Any],
        inputs: dict,
        raw_inputs: dict | None = None,
    ):
        prefix_cache_enabled = bool(self._prefix_vlm_cache_config.enabled)
        observe_only = bool(prefix_cache_enabled and self._prefix_vlm_cache_config.observe_only)
        state_signature = _prefix_state_signature(raw_inputs, inputs)
        prompt_key = _prefix_prompt_key(inputs) if prefix_cache_enabled else ""
        state = observation.state
        noise = sample_kwargs.pop("noise", None)
        if noise is None:
            noise = self._model.sample_noise(
                (state.shape[0], self._model.config.action_horizon, self._model.config.action_dim),
                self._pytorch_device,
            )
        num_steps = int(sample_kwargs.pop("num_steps", 10))
        if sample_kwargs:
            raise ValueError(f"Unsupported PyTorch prefix cache sample kwargs: {sorted(sample_kwargs)}")

        first_layer_l2 = np.inf
        first_layer_cka = -np.inf
        first_layer_feature = None
        prefix_sublayer_deltas = None
        selected_cache_entry = None
        selected_bank_cka = -np.inf
        action_cache_info: dict[str, Any] = {}
        action_deviation_gate_accept = False
        action_deviation_gate_reason = "not_evaluated"
        action_deviation_candidate_actions = None
        action_deviation_candidate_reason = "disabled"
        action_deviation_candidate_compute_layer_mask = ""
        action_deviation_candidate_shared_noise = False
        action_deviation_candidate_suffix_cache_isolated = False
        prompt_conservative_tier = ""
        action_deviation_candidate_mode = str(self._prefix_vlm_cache_config.action_deviation_candidate_mode).lower()
        if action_deviation_candidate_mode not in {"accepted", "pre_guard", "available"}:
            action_deviation_candidate_mode = "accepted"
        action_deviation_probe_enabled = bool(self._pytorch_prefix_probe_enabled())
        no_image_gate_blocked = bool(
            self._prefix_vlm_action_deviation_gate is not None
            and self._prefix_vlm_cache_config.action_deviation_gate_disable_no_image
        )
        if prefix_cache_enabled and not observe_only and not no_image_gate_blocked:
            no_image_hit, no_image_reason, no_image_state_info = self._pytorch_skip_v4_no_image_decision(
                prompt_key,
                state_signature,
            )
        else:
            no_image_hit = False
            no_image_reason = "action_deviation_gate" if no_image_gate_blocked else "disabled"
            no_image_state_info = {}
        if no_image_hit:
            prefix_pad_masks = self._prefix_vlm_cached_prefix_pad_masks
            kv_cache = self._prefix_vlm_cached_kv
            if prefix_pad_masks is None or kv_cache is None:
                raise RuntimeError("skip_v4 no-image cache hit without cached prefix masks/KV")
            actions, action_cache_info = self._pytorch_run_suffix_with_cache(
                state,
                prefix_pad_masks,
                kv_cache,
                noise,
                num_steps,
                allow_cross_env_action_cache=False,
                state_info=no_image_state_info,
            )
            depth = len(kv_cache.key_cache)
            self._prefix_vlm_queries += 1
            self._prefix_vlm_episode_queries += 1
            self._update_prefix_vlm_state_phase(state_signature)
            self._prefix_vlm_continuous_cached += 1
            self._prefix_vlm_episode_hits += 1
            self._prefix_vlm_skip_v4_no_image_remaining = max(
                self._prefix_vlm_skip_v4_no_image_remaining - 1,
                0,
            )
            return actions, {
                "prefix_vlm_cache_enabled": True,
                "prefix_vlm_cache_hit": True,
                "prefix_vlm_cache_reason": no_image_reason,
                "prefix_vlm_cache_computed_blocks": 0,
                "prefix_vlm_cache_skipped_blocks": depth,
                **self._prefix_vlm_layer_cache_info(depth, 0),
                "prefix_vlm_cache_image_signature_l2": np.inf,
                "prefix_vlm_cache_image_signature_accum_l2": float(self._prefix_vlm_accumulated_image_l2),
                "prefix_vlm_cache_first_layer_l2": float(first_layer_l2),
                "prefix_vlm_cache_first_layer_cka": float(first_layer_cka),
                "prefix_vlm_cache_first_layer_metric": self._prefix_vlm_cache_config.first_layer_metric,
                **_prefix_state_info_to_cache_info(no_image_state_info),
                "prefix_vlm_cache_query": self._prefix_vlm_queries,
                "prefix_vlm_cache_episode_query": self._prefix_vlm_episode_queries,
                "prefix_vlm_cache_episode_hits": self._prefix_vlm_episode_hits,
                "prefix_vlm_cache_continuous_hits": self._prefix_vlm_continuous_cached,
                "prefix_vlm_cache_pytorch_compile_hit_path": bool(self._prefix_vlm_cache_config.pytorch_compile_hit_path),
                "prefix_vlm_cache_skip_v4_no_image": True,
                "prefix_vlm_cache_skip_v4_no_image_decision_reason": no_image_reason,
                "prefix_vlm_cache_skip_v4_no_image_remaining": int(self._prefix_vlm_skip_v4_no_image_remaining),
                **self._prefix_vlm_motion_tier_cache_info(),
                **self._prefix_vlm_sublayer_cache_info(),
                **self._prefix_vlm_cross_layer_residual_cache_info(),
                **self._prefix_vlm_cache_bank_info(False, -np.inf),
                **action_cache_info,
            }

        shadow_use_cache = False
        shadow_cache_reason = "disabled"
        shadow_computed_blocks = 18
        previous_cached_residuals = self._prefix_vlm_cached_residuals
        previous_cached_sublayer_deltas = self._prefix_vlm_cached_sublayer_deltas
        previous_cached_kv = self._prefix_vlm_cached_kv
        signature = _prefix_image_signature(inputs) if prefix_cache_enabled else np.zeros((1,), dtype=np.float32)
        images, img_masks, lang_tokens, lang_masks, state = self._model._preprocess_observation(  # noqa: SLF001
            observation, train=False
        )
        prefix_embs, prefix_pad_masks, prefix_att_4d, prefix_position_ids = self._pytorch_prefix_inputs(
            images, img_masks, lang_tokens, lang_masks
        )
        if prefix_cache_enabled:
            use_cache, cache_reason, signature_l2, state_info = self._prefix_vlm_cache_decision(
                signature,
                prompt_key,
                state_signature,
            )
        else:
            use_cache, cache_reason, signature_l2, state_info = (
                False,
                "disabled",
                np.inf,
                self._prefix_vlm_state_guard_info(state_signature),
            )
        pre_guard_use_cache = bool(use_cache)
        pre_guard_cache_reason = cache_reason
        available_label_candidate = bool(
            previous_cached_residuals is not None
            and previous_cached_kv is not None
            and prompt_key == self._prefix_vlm_prompt_key
        )
        force_label_probe = bool(
            observe_only
            and self._prefix_vlm_cache_config.action_deviation_label_enabled
            and (
                (action_deviation_candidate_mode == "pre_guard" and pre_guard_use_cache)
                or (action_deviation_candidate_mode == "available" and available_label_candidate)
            )
        )
        if prefix_cache_enabled and (use_cache or force_label_probe) and self._pytorch_prefix_probe_enabled():
            first_layer_feature = self._pytorch_first_layer_feature(prefix_embs, prefix_att_4d, prefix_position_ids)
            if self._prefix_vlm_cached_first_layer is None:
                if use_cache:
                    use_cache, cache_reason = False, "first_layer_empty"
            else:
                first_layer_l2 = _torch_relative_l2(first_layer_feature, self._prefix_vlm_cached_first_layer)
                first_layer_cka = _torch_linear_cka(
                    first_layer_feature,
                    self._prefix_vlm_cached_first_layer,
                    prefix_pad_masks,
                )
                bank_entry, bank_l2, bank_cka = self._prefix_vlm_select_cache_bank_entry(
                    first_layer_feature,
                    prefix_pad_masks,
                    prompt_key,
                )
                if bank_entry is not None and bank_cka > first_layer_cka:
                    selected_cache_entry = bank_entry
                    selected_bank_cka = float(bank_cka)
                    first_layer_l2 = float(bank_l2)
                    first_layer_cka = float(bank_cka)
                prompt_conservative_cache = self._prefix_vlm_prompt_conservative_cache_match()
                if use_cache and prompt_conservative_cache:
                    conservative_cka = float(
                        self._prefix_vlm_cache_config.prompt_conservative_first_layer_cka_threshold
                    )
                    if conservative_cka >= 0 and (
                        not np.isfinite(first_layer_cka) or float(first_layer_cka) < conservative_cka
                    ):
                        use_cache = False
                        cache_reason = f"prompt_conservative_first_layer_cka_{conservative_cka:g}"
                    else:
                        conservative_mask = self._prefix_vlm_cache_config.prompt_conservative_compute_layer_mask
                        prompt_conservative_tier = "prompt_conservative"
                        conservative_high_mask = (
                            self._prefix_vlm_cache_config.prompt_conservative_high_compute_layer_mask
                        )
                        conservative_high_cka = float(
                            self._prefix_vlm_cache_config.prompt_conservative_high_first_layer_cka_threshold
                        )
                        if (
                            conservative_high_mask
                            and conservative_high_cka >= 0
                            and np.isfinite(first_layer_cka)
                            and float(first_layer_cka) >= conservative_high_cka
                        ):
                            conservative_mask = conservative_high_mask
                            prompt_conservative_tier = "prompt_conservative_high"
                        elif conservative_high_mask and conservative_high_cka >= 0:
                            prompt_conservative_tier = "prompt_conservative_low"
                        if conservative_mask:
                            self._prefix_vlm_active_compute_layer_mask = conservative_mask
                if use_cache and self._prefix_vlm_action_deviation_gate is not None:
                    action_deviation_gate_accept, action_deviation_gate_reason = (
                        self._prefix_vlm_action_deviation_gate_decision(
                            signature_l2,
                            state_info,
                            first_layer_l2=float(first_layer_l2),
                            first_layer_cka=float(first_layer_cka),
                            bank_selected_cka=float(selected_bank_cka),
                        )
                    )
                    use_cache, cache_reason = bool(action_deviation_gate_accept), action_deviation_gate_reason
                    if use_cache and prompt_conservative_cache:
                        cache_reason = f"{cache_reason}_{prompt_conservative_tier or 'prompt_conservative'}"
                elif use_cache and self._pytorch_first_layer_guard_enabled():
                    metric = self._prefix_vlm_cache_config.first_layer_metric.lower()
                    if metric == "l2":
                        if first_layer_l2 > self._prefix_vlm_cache_config.first_layer_l2_threshold:
                            use_cache, cache_reason = False, "first_layer"
                    elif metric == "cka":
                        if first_layer_cka < self._prefix_vlm_cache_config.first_layer_cka_threshold:
                            use_cache, cache_reason = False, "first_layer_cka"
                    else:
                        raise ValueError(f"Unsupported prefix VLM first-layer metric: {metric!r}")
                    if use_cache and prompt_conservative_cache:
                        cache_reason = f"{cache_reason}_{prompt_conservative_tier or 'prompt_conservative'}"
                if (
                    use_cache
                    and self._prefix_vlm_action_deviation_gate is None
                    and self._prefix_vlm_motion_tier_enabled()
                ):
                    tier_accept, tier_reason = self._prefix_vlm_motion_tier_decision(state_info, first_layer_cka)
                    use_cache, cache_reason = bool(tier_accept), tier_reason

        if observe_only:
            shadow_use_cache = bool(use_cache)
            shadow_cache_reason = cache_reason
            action_deviation_policy_shadow_hit = bool(shadow_use_cache)
            if shadow_use_cache and self._prefix_vlm_cached_residuals is not None:
                shadow_depth = int(self._prefix_vlm_cached_residuals.shape[0])
                shadow_computed_blocks = sum(self._prefix_vlm_compute_mask_list(shadow_depth))
            label_candidate = bool(shadow_use_cache)
            if (
                not label_candidate
                and self._prefix_vlm_cache_config.action_deviation_label_enabled
                and action_deviation_candidate_mode == "pre_guard"
            ):
                label_candidate = bool(pre_guard_use_cache)
                if label_candidate:
                    action_deviation_candidate_reason = f"pre_guard_policy_rejected_{shadow_cache_reason}"
            if (
                not label_candidate
                and self._prefix_vlm_cache_config.action_deviation_label_enabled
                and action_deviation_candidate_mode == "available"
            ):
                label_candidate = bool(available_label_candidate)
                if label_candidate:
                    action_deviation_candidate_reason = f"available_policy_rejected_{shadow_cache_reason}"
            if label_candidate and self._prefix_vlm_cache_config.action_deviation_label_enabled:
                candidate_cached_residuals = (
                    selected_cache_entry["residuals"]
                    if selected_cache_entry is not None
                    else previous_cached_residuals
                )
                candidate_cached_sublayers = (
                    selected_cache_entry["sublayer_deltas"]
                    if selected_cache_entry is not None
                    else previous_cached_sublayer_deltas
                )
                candidate_cached_kv = (
                    selected_cache_entry["kv"] if selected_cache_entry is not None else previous_cached_kv
                )
                if candidate_cached_residuals is not None and candidate_cached_kv is not None:
                    candidate_depth = int(candidate_cached_residuals.shape[0])
                    action_deviation_candidate_compute_layer_mask = _prefix_mask_to_spec(
                        self._prefix_vlm_compute_mask_list(candidate_depth)
                    )
                    action_deviation_candidate_actions = self._pytorch_prefix_action_deviation_candidate(
                        prefix_embs,
                        prefix_att_4d,
                        prefix_position_ids,
                        prefix_pad_masks,
                        state,
                        noise,
                        num_steps,
                        candidate_cached_residuals,
                        candidate_cached_kv,
                        cached_sublayer_deltas=candidate_cached_sublayers,
                        first_layer_cka=float(first_layer_cka),
                        state_info=state_info,
                    )
                    action_deviation_candidate_shared_noise = True
                    action_deviation_candidate_suffix_cache_isolated = True
                    if action_deviation_candidate_reason == "disabled":
                        action_deviation_candidate_reason = "shadow_hit"
                else:
                    action_deviation_candidate_reason = "missing_cached_state"
            elif self._prefix_vlm_cache_config.action_deviation_label_enabled:
                action_deviation_candidate_reason = "shadow_miss"
            use_cache = False
            cache_reason = f"observe_only_{shadow_cache_reason}"
        else:
            action_deviation_policy_shadow_hit = False

        if use_cache:
            hit_cached_residuals = (
                selected_cache_entry["residuals"] if selected_cache_entry is not None else self._prefix_vlm_cached_residuals
            )
            hit_cached_sublayers = (
                selected_cache_entry["sublayer_deltas"]
                if selected_cache_entry is not None
                else self._prefix_vlm_cached_sublayer_deltas
            )
            hit_cached_kv = selected_cache_entry["kv"] if selected_cache_entry is not None else self._prefix_vlm_cached_kv
            if (
                self._prefix_vlm_cache_config.action_suffix_cache_enabled
                or self._prefix_vlm_cache_config.intra_action_cache_enabled
                or self._prefix_vlm_sublayer_cache_enabled()
                or self._prefix_vlm_cache_bank_enabled()
            ):
                kv_cache = self._pytorch_coarse_prefix_forward(
                    prefix_embs,
                    prefix_att_4d,
                    prefix_position_ids,
                    hit_cached_residuals,
                    hit_cached_kv,
                    cached_sublayer_deltas=hit_cached_sublayers,
                    first_layer_cka=float(first_layer_cka),
                )
                actions, action_cache_info = self._pytorch_run_suffix_with_cache(
                    state,
                    prefix_pad_masks,
                    kv_cache,
                    noise,
                    num_steps,
                    allow_cross_env_action_cache=True,
                    state_info=state_info,
                )
            else:
                actions = self._pytorch_compiled_prefix_cache_hit(
                    images,
                    img_masks,
                    lang_tokens,
                    lang_masks,
                    state,
                    noise,
                    hit_cached_residuals,
                    hit_cached_kv,
                    num_steps,
                )
            depth = int(hit_cached_residuals.shape[0])
            computed_blocks = sum(self._prefix_vlm_compute_mask_list(depth))
            self._prefix_vlm_skip_v4_no_image_remaining = self._pytorch_skip_v4_budget_after_probe(
                first_layer_cka
            )
        else:
            self._prefix_vlm_skip_v4_no_image_remaining = 0
            if prefix_cache_enabled:
                kv_cache, prefix_residuals, first_layer_feature, prefix_sublayer_deltas = (
                    self._pytorch_native_prefix_forward(
                        prefix_embs,
                        prefix_att_4d,
                        prefix_position_ids,
                        collect_residuals=True,
                    )
                )
            else:
                kv_cache = self._pytorch_native_prefix_forward(
                    prefix_embs,
                    prefix_att_4d,
                    prefix_position_ids,
                    collect_residuals=False,
                )
                prefix_residuals = None
                first_layer_feature = None
                prefix_sublayer_deltas = None
            actions, action_cache_info = self._pytorch_run_suffix_with_cache(
                state,
                prefix_pad_masks,
                kv_cache,
                noise,
                num_steps,
                allow_cross_env_action_cache=False,
                state_info=state_info,
            )
            if prefix_cache_enabled:
                if not (observe_only and shadow_use_cache):
                    self._prefix_vlm_cached_residuals = prefix_residuals
                    self._prefix_vlm_cached_sublayer_deltas = prefix_sublayer_deltas
                    self._prefix_vlm_cached_kv = kv_cache
                    self._prefix_vlm_cached_prefix_pad_masks = prefix_pad_masks.detach()
                    self._prefix_vlm_cached_first_layer = first_layer_feature
                    self._prefix_vlm_cached_state_signature = state_signature
                    self._prefix_vlm_signature = signature
                    self._prefix_vlm_prompt_key = prompt_key
                    self._prefix_vlm_add_cache_bank_entry(
                        residuals=prefix_residuals,
                        sublayer_deltas=prefix_sublayer_deltas,
                        kv_cache=kv_cache,
                        prefix_pad_masks=prefix_pad_masks.detach(),
                        first_layer=first_layer_feature,
                        state_signature=state_signature,
                        signature=signature,
                        prompt_key=prompt_key,
                    )
                depth = int(prefix_residuals.shape[0])
            else:
                depth = len(self._model.paligemma_with_expert.paligemma.language_model.layers)
            computed_blocks = depth

        similarity_info = self._prefix_vlm_similarity_observation_info(first_layer_feature, prefix_pad_masks)
        layer_similarity_info = self._prefix_vlm_layer_similarity_observation_info(
            prefix_residuals if prefix_cache_enabled and not use_cache else None,
            previous_cached_residuals,
            kv_cache if not use_cache else None,
            previous_cached_kv,
            prefix_pad_masks,
        )
        layer_pairwise_similarity_info = self._prefix_vlm_layer_pairwise_similarity_observation_info(
            prefix_residuals if prefix_cache_enabled and not use_cache else None,
            prefix_pad_masks,
        )
        adjacent_layer_similarity_info = self._prefix_vlm_adjacent_layer_similarity_observation_info(
            prefix_residuals if prefix_cache_enabled and not use_cache else None,
            prefix_pad_masks,
        )
        lagged_adjacent_layer_similarity_info = self._prefix_vlm_lagged_adjacent_layer_similarity_observation_info(
            prefix_residuals if prefix_cache_enabled and not use_cache else None,
            previous_cached_residuals,
            prefix_pad_masks,
        )
        sublayer_similarity_info = self._prefix_vlm_sublayer_similarity_observation_info(
            prefix_sublayer_deltas if prefix_cache_enabled and not use_cache else None,
            previous_cached_sublayer_deltas,
            prefix_pad_masks,
        )

        if observe_only and not shadow_use_cache:
            shadow_computed_blocks = depth
        shadow_reused_blocks = depth - shadow_computed_blocks if observe_only else 0
        shadow_compute_layer_mask = (
            _prefix_mask_to_spec(self._prefix_vlm_compute_mask_list(depth))
            if observe_only and shadow_use_cache
            else ""
        )

        self._prefix_vlm_queries += 1
        self._prefix_vlm_episode_queries += 1
        if prefix_cache_enabled or self._prefix_vlm_action_scheduler_needs_motion():
            self._update_prefix_vlm_state_phase(state_signature)
        cache_state_hit = bool(shadow_use_cache if observe_only else use_cache)
        if cache_state_hit:
            self._prefix_vlm_continuous_cached += 1
            self._prefix_vlm_episode_hits += 1
            self._prefix_vlm_accumulated_image_l2 += float(signature_l2)
        else:
            self._prefix_vlm_continuous_cached = 0
            self._prefix_vlm_accumulated_image_l2 = 0.0

        return actions, {
            "prefix_vlm_cache_enabled": True,
            "prefix_vlm_cache_hit": bool(use_cache),
            "prefix_vlm_cache_reason": cache_reason,
            "prefix_vlm_cache_computed_blocks": computed_blocks,
            "prefix_vlm_cache_skipped_blocks": depth - computed_blocks,
            **self._prefix_vlm_layer_cache_info(depth, computed_blocks),
            "prefix_vlm_cache_image_signature_l2": float(signature_l2),
            "prefix_vlm_cache_image_signature_accum_l2": float(self._prefix_vlm_accumulated_image_l2),
            "prefix_vlm_cache_first_layer_l2": float(first_layer_l2),
            "prefix_vlm_cache_first_layer_cka": float(first_layer_cka),
            "prefix_vlm_cache_first_layer_metric": self._prefix_vlm_cache_config.first_layer_metric,
            **_prefix_state_info_to_cache_info(state_info),
            "prefix_vlm_cache_query": self._prefix_vlm_queries,
            "prefix_vlm_cache_episode_query": self._prefix_vlm_episode_queries,
            "prefix_vlm_cache_episode_hits": self._prefix_vlm_episode_hits,
            "prefix_vlm_cache_continuous_hits": self._prefix_vlm_continuous_cached,
            "prefix_vlm_cache_pytorch_compile_hit_path": bool(self._prefix_vlm_cache_config.pytorch_compile_hit_path),
            "prefix_vlm_cache_skip_v4_no_image": False,
            "prefix_vlm_cache_skip_v4_no_image_decision_reason": no_image_reason,
            "prefix_vlm_cache_skip_v4_no_image_remaining": int(self._prefix_vlm_skip_v4_no_image_remaining),
            "prefix_vlm_cache_observe_only": bool(observe_only),
            "prefix_vlm_cache_shadow_hit": bool(shadow_use_cache) if observe_only else False,
            "prefix_vlm_cache_shadow_reason": shadow_cache_reason if observe_only else "disabled",
            "prefix_vlm_cache_shadow_computed_blocks": int(shadow_computed_blocks) if observe_only else depth,
            "prefix_vlm_cache_shadow_skipped_blocks": int(shadow_reused_blocks) if observe_only else 0,
            "prefix_vlm_cache_shadow_reused_blocks": int(shadow_reused_blocks) if observe_only else 0,
            "prefix_vlm_cache_shadow_reused_block_fraction": float(shadow_reused_blocks / depth)
            if observe_only and depth
            else 0.0,
            "prefix_vlm_cache_shadow_compute_layer_mask": shadow_compute_layer_mask,
            "prefix_vlm_cache_action_deviation_label_enabled": bool(
                self._prefix_vlm_cache_config.action_deviation_label_enabled
            ),
            "prefix_vlm_cache_action_deviation_candidate_mode": action_deviation_candidate_mode,
            "prefix_vlm_cache_action_deviation_candidate": bool(action_deviation_candidate_actions is not None),
            "prefix_vlm_cache_action_deviation_candidate_reason": action_deviation_candidate_reason,
            "prefix_vlm_cache_action_deviation_candidate_compute_layer_mask": (
                action_deviation_candidate_compute_layer_mask
            ),
            "prefix_vlm_cache_action_deviation_probe_enabled": bool(action_deviation_probe_enabled),
            "prefix_vlm_cache_action_deviation_shared_noise": bool(action_deviation_candidate_shared_noise),
            "prefix_vlm_cache_action_deviation_candidate_suffix_cache_isolated": bool(
                action_deviation_candidate_suffix_cache_isolated
            ),
            "prefix_vlm_cache_action_deviation_policy_shadow_hit": bool(action_deviation_policy_shadow_hit),
            "prefix_vlm_cache_action_deviation_pre_guard_hit": bool(pre_guard_use_cache),
            "prefix_vlm_cache_action_deviation_pre_guard_reason": pre_guard_cache_reason,
            "prefix_vlm_cache_action_deviation_gate_enabled": bool(
                self._prefix_vlm_action_deviation_gate is not None
            ),
            "prefix_vlm_cache_action_deviation_gate_replace_legacy_guards": bool(
                self._prefix_vlm_action_deviation_gate is not None
                and self._prefix_vlm_cache_config.action_deviation_gate_replace_legacy_guards
            ),
            "prefix_vlm_cache_action_deviation_gate_accept": bool(action_deviation_gate_accept),
            "prefix_vlm_cache_action_deviation_gate_reason": action_deviation_gate_reason,
            **(
                {"_prefix_vlm_action_deviation_candidate_actions": action_deviation_candidate_actions}
                if action_deviation_candidate_actions is not None
                else {}
            ),
            **similarity_info,
            **layer_similarity_info,
            **layer_pairwise_similarity_info,
            **adjacent_layer_similarity_info,
            **lagged_adjacent_layer_similarity_info,
            **sublayer_similarity_info,
            **self._prefix_vlm_motion_tier_cache_info(),
            **self._prefix_vlm_sublayer_cache_info(),
            **self._prefix_vlm_cross_layer_residual_cache_info(),
            **self._prefix_vlm_cache_bank_info(selected_cache_entry is not None, selected_bank_cka),
            **action_cache_info,
        }

    def _pytorch_run_suffix_with_cache(
        self,
        state,
        prefix_pad_masks,
        kv_cache,
        noise,
        num_steps: int,
        *,
        allow_cross_env_action_cache: bool,
        state_info: dict[str, Any] | None = None,
    ):
        if self._prefix_vlm_cache_config.action_suffix_cache_enabled:
            return self._pytorch_sample_suffix_with_action_cache(
                state,
                prefix_pad_masks,
                kv_cache,
                noise,
                num_steps,
                allow_cache=allow_cross_env_action_cache,
            )
        if self._prefix_vlm_cache_config.intra_action_cache_enabled:
            plan = self._prefix_vlm_intra_action_cache_plan(
                state_info,
                prefix_cache_hit=allow_cross_env_action_cache,
            )
            scheduler_info = plan["info"]
            if not plan["enabled"]:
                actions = self._pytorch_sample_suffix(state, prefix_pad_masks, kv_cache, noise, num_steps)
                return actions, scheduler_info
            if self._prefix_vlm_cache_config.intra_action_cache_compile:
                actions = self._pytorch_compiled_intra_action_suffix(
                    state,
                    prefix_pad_masks,
                    kv_cache,
                    noise,
                    num_steps,
                    fn_blocks=plan["fn_blocks"],
                    bn_blocks=plan["bn_blocks"],
                    schedule=plan["schedule"],
                    refresh_interval=plan["refresh_interval"],
                )
                info = self._pytorch_intra_action_cache_info(
                    num_steps,
                    fn_blocks=plan["fn_blocks"],
                    bn_blocks=plan["bn_blocks"],
                    schedule=plan["schedule"],
                    refresh_interval=plan["refresh_interval"],
                )
                return actions, {**scheduler_info, **info}
            actions, info = self._pytorch_sample_suffix_with_intra_action_cache(
                state,
                prefix_pad_masks,
                kv_cache,
                noise,
                num_steps,
                fn_blocks=plan["fn_blocks"],
                bn_blocks=plan["bn_blocks"],
                schedule=plan["schedule"],
                refresh_interval=plan["refresh_interval"],
            )
            return actions, {**scheduler_info, **info}
        return self._pytorch_sample_suffix(state, prefix_pad_masks, kv_cache, noise, num_steps), {}

    def _pytorch_prefix_action_deviation_candidate(
        self,
        prefix_embs,
        prefix_att_4d,
        prefix_position_ids,
        prefix_pad_masks,
        state,
        noise,
        num_steps: int,
        cached_residuals,
        cached_kv,
        *,
        cached_sublayer_deltas=None,
        first_layer_cka: float = -np.inf,
        state_info: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        saved_action_suffix_cached_residuals = self._action_suffix_cached_residuals
        saved_action_suffix_cached_num_steps = self._action_suffix_cached_num_steps
        try:
            with torch.no_grad():
                candidate_kv = self._pytorch_coarse_prefix_forward(
                    prefix_embs,
                    prefix_att_4d,
                    prefix_position_ids,
                    cached_residuals,
                    cached_kv,
                    cached_sublayer_deltas=cached_sublayer_deltas,
                    first_layer_cka=first_layer_cka,
                )
                candidate_actions, _candidate_action_info = self._pytorch_run_suffix_with_cache(
                    state,
                    prefix_pad_masks,
                    candidate_kv,
                    noise,
                    num_steps,
                    allow_cross_env_action_cache=False,
                    state_info=state_info,
                )
                return candidate_actions.detach()
        finally:
            self._action_suffix_cached_residuals = saved_action_suffix_cached_residuals
            self._action_suffix_cached_num_steps = saved_action_suffix_cached_num_steps

    def _prefix_vlm_add_cache_bank_entry(
        self,
        *,
        residuals,
        sublayer_deltas,
        kv_cache,
        prefix_pad_masks,
        first_layer,
        state_signature,
        signature,
        prompt_key: str,
    ) -> None:
        if not self._prefix_vlm_cache_bank_enabled():
            return
        if residuals is None or kv_cache is None or prefix_pad_masks is None or first_layer is None:
            return
        entry = {
            "residuals": residuals,
            "sublayer_deltas": sublayer_deltas,
            "kv": kv_cache,
            "prefix_pad_masks": prefix_pad_masks,
            "first_layer": first_layer,
            "state_signature": None if state_signature is None else np.asarray(state_signature, dtype=np.float32).copy(),
            "signature": None if signature is None else np.asarray(signature, dtype=np.float32).copy(),
            "prompt_key": str(prompt_key),
        }
        self._prefix_vlm_cache_bank.append(entry)
        max_size = max(1, int(self._prefix_vlm_cache_config.cache_bank_size))
        if len(self._prefix_vlm_cache_bank) > max_size:
            self._prefix_vlm_cache_bank = self._prefix_vlm_cache_bank[-max_size:]

    def _prefix_vlm_select_cache_bank_entry(
        self,
        first_layer_feature,
        prefix_pad_masks,
        prompt_key: str,
    ) -> tuple[dict[str, Any] | None, float, float]:
        if not self._prefix_vlm_cache_bank_enabled() or first_layer_feature is None:
            return None, np.inf, -np.inf
        best_entry = None
        best_cka = -np.inf
        best_l2 = np.inf
        for entry in self._prefix_vlm_cache_bank:
            if entry.get("prompt_key") != prompt_key:
                continue
            candidate = entry.get("first_layer")
            if candidate is None or tuple(candidate.shape) != tuple(first_layer_feature.shape):
                continue
            cka = _torch_linear_cka(first_layer_feature, candidate, prefix_pad_masks)
            if not np.isfinite(cka) or cka <= best_cka:
                continue
            best_entry = entry
            best_cka = float(cka)
            best_l2 = float(_torch_relative_l2(first_layer_feature, candidate))
        return best_entry, best_l2, best_cka

    def _pytorch_prefix_inputs(self, images, img_masks, lang_tokens, lang_masks):
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        prefix_embs, prefix_pad_masks, prefix_att_masks = self._model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_4d = self._model._prepare_attention_masks_4d(prefix_att_2d_masks)  # noqa: SLF001
        return prefix_embs, prefix_pad_masks, prefix_att_4d, prefix_position_ids

    def _pytorch_native_prefix_forward(
        self,
        prefix_embs,
        prefix_att_4d,
        prefix_position_ids,
        *,
        collect_residuals: bool = False,
    ):
        language_model = self._model.paligemma_with_expert.paligemma.language_model
        language_model.config._attn_implementation = "eager"  # noqa: SLF001
        residuals = []
        handles = []
        first_layer_feature = None
        sublayer_deltas = None
        if collect_residuals:
            depth = len(language_model.layers)
            residuals = [None] * depth
            collect_sublayers = bool(
                self._prefix_vlm_cache_config.observe_sublayer_similarity
                or self._prefix_vlm_sublayer_cache_enabled()
            )
            attn_deltas = [None] * depth
            mlp_deltas = [None] * depth

            def make_hook(layer_idx):
                def hook(_module, args, output):
                    nonlocal first_layer_feature
                    if layer_idx == 0:
                        first_layer_feature = output[0].detach()
                    residuals[layer_idx] = (output[0].detach() - args[0].detach()).to(dtype=output[0].dtype)

                return hook

            def make_attn_hook(layer_idx):
                def hook(_module, _args, output):
                    attn_output = output[0] if isinstance(output, tuple) else output
                    attn_deltas[layer_idx] = attn_output.detach()

                return hook

            def make_mlp_hook(layer_idx):
                def hook(_module, _args, output):
                    mlp_deltas[layer_idx] = output.detach()

                return hook

            for layer_idx, layer in enumerate(language_model.layers):
                handles.append(layer.register_forward_hook(make_hook(layer_idx)))
                if collect_sublayers:
                    handles.append(layer.self_attn.register_forward_hook(make_attn_hook(layer_idx)))
                    handles.append(layer.mlp.register_forward_hook(make_mlp_hook(layer_idx)))
        try:
            _, cache = self._model.paligemma_with_expert.forward(
                attention_mask=prefix_att_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
        finally:
            for handle in handles:
                handle.remove()
        if collect_residuals:
            if bool(
                self._prefix_vlm_cache_config.observe_sublayer_similarity
                or self._prefix_vlm_sublayer_cache_enabled()
            ):
                sublayer_deltas = {
                    "attn": torch.stack(attn_deltas),
                    "mlp": torch.stack(mlp_deltas),
                }
            return cache, torch.stack(residuals), first_layer_feature, sublayer_deltas
        return cache

    def _pytorch_first_layer_feature(self, prefix_embs, prefix_att_4d, prefix_position_ids):
        from transformers.cache_utils import DynamicCache

        language_model = self._model.paligemma_with_expert.paligemma.language_model
        language_model.config._attn_implementation = "eager"  # noqa: SLF001
        hidden_states = prefix_embs
        if language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.bfloat16)
        cache_position = torch.arange(prefix_embs.shape[1], device=prefix_embs.device)
        position_embeddings = language_model.rotary_emb(hidden_states, prefix_position_ids)
        return language_model.layers[0](
            hidden_states,
            attention_mask=prefix_att_4d,
            position_ids=prefix_position_ids,
            past_key_value=DynamicCache(),
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            adarms_cond=None,
        )[0].detach()

    def _pytorch_coarse_prefix_forward(
        self,
        prefix_embs,
        prefix_att_4d,
        prefix_position_ids,
        cached_residuals,
        cached_cache,
        *,
        cached_sublayer_deltas=None,
        first_layer_cka: float = -np.inf,
    ):
        from transformers.cache_utils import DynamicCache

        language_model = self._model.paligemma_with_expert.paligemma.language_model
        language_model.config._attn_implementation = "eager"  # noqa: SLF001
        depth = len(language_model.layers)
        compute_mask = self._prefix_vlm_compute_mask_list(depth)
        cached_sublayers = cached_sublayer_deltas if cached_sublayer_deltas is not None else self._prefix_vlm_cached_sublayer_deltas
        sublayer_cache_ready = (
            self._prefix_vlm_sublayer_cache_enabled()
            and isinstance(cached_sublayers, dict)
            and cached_sublayers.get("attn") is not None
            and cached_sublayers.get("mlp") is not None
        )
        attn_reuse_mask = (
            self._prefix_vlm_sublayer_mask_list(
                depth,
                self._prefix_vlm_cache_config.sublayer_cache_attn_layer_mask,
            )
            if sublayer_cache_ready
            else [False] * depth
        )
        mlp_reuse_mask = (
            self._prefix_vlm_sublayer_mask_list(
                depth,
                self._prefix_vlm_cache_config.sublayer_cache_mlp_layer_mask,
            )
            if sublayer_cache_ready
            else [False] * depth
        )
        sublayer_cka_threshold = float(self._prefix_vlm_cache_config.sublayer_cache_min_first_layer_cka)
        sublayer_cka_gate_accept = (
            sublayer_cka_threshold < 0
            or (np.isfinite(first_layer_cka) and float(first_layer_cka) >= sublayer_cka_threshold)
        )
        if not sublayer_cka_gate_accept:
            attn_reuse_mask = [False] * depth
            mlp_reuse_mask = [False] * depth
        cross_layer_residual_reuse_enabled = self._prefix_vlm_cross_layer_residual_cache_enabled()
        cross_layer_residual_reuse_mask = (
            self._prefix_vlm_sublayer_mask_list(
                depth,
                self._prefix_vlm_cache_config.cross_layer_residual_layer_mask,
            )
            if cross_layer_residual_reuse_enabled
            else [False] * depth
        )
        reused_sublayers = 0
        cross_layer_residual_reused = 0
        cross_layer_residual_total = 0
        self._prefix_vlm_last_sublayer_reused = 0
        self._prefix_vlm_last_sublayer_total = depth * 2 if self._prefix_vlm_sublayer_cache_enabled() else 0
        self._prefix_vlm_last_sublayer_attn_mask = ""
        self._prefix_vlm_last_sublayer_mlp_mask = ""
        self._prefix_vlm_last_sublayer_cka_gate_accept = bool(sublayer_cka_gate_accept)
        self._prefix_vlm_last_sublayer_cka_gate_threshold = float(sublayer_cka_threshold)
        self._prefix_vlm_last_cross_layer_residual_reused = 0
        self._prefix_vlm_last_cross_layer_residual_total = 0
        self._prefix_vlm_last_cross_layer_residual_mask = ""
        hidden_states = prefix_embs
        if language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.bfloat16)

        cache_position = torch.arange(prefix_embs.shape[1], device=prefix_embs.device)
        position_embeddings = language_model.rotary_emb(hidden_states, prefix_position_ids)
        new_cache = DynamicCache()
        last_layer_residual = None

        for layer_idx, should_compute in enumerate(compute_mask):
            layer_input = hidden_states
            if should_compute:
                reuse_attn = bool(attn_reuse_mask[layer_idx])
                reuse_mlp = bool(mlp_reuse_mask[layer_idx])
                if reuse_attn or reuse_mlp:
                    hidden_states = self._pytorch_prefix_layer_with_sublayer_cache(
                        language_model.layers[layer_idx],
                        layer_idx,
                        hidden_states,
                        prefix_att_4d,
                        prefix_position_ids,
                        new_cache,
                        cached_cache,
                        cached_sublayers,
                        cache_position,
                        position_embeddings,
                        reuse_attn=reuse_attn,
                        reuse_mlp=reuse_mlp,
                    )
                    reused_sublayers += int(reuse_attn) + int(reuse_mlp)
                else:
                    hidden_states = language_model.layers[layer_idx](
                        hidden_states,
                        attention_mask=prefix_att_4d,
                        position_ids=prefix_position_ids,
                        past_key_value=new_cache,
                        use_cache=True,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        adarms_cond=None,
                    )[0]
                last_layer_residual = hidden_states - layer_input
            else:
                residual = cached_residuals[layer_idx].to(hidden_states.dtype)
                if cross_layer_residual_reuse_enabled and cross_layer_residual_reuse_mask[layer_idx]:
                    cross_layer_residual_total += 1
                    if last_layer_residual is not None and last_layer_residual.shape == hidden_states.shape:
                        residual = last_layer_residual.to(hidden_states.dtype)
                        cross_layer_residual_reused += 1
                hidden_states = hidden_states + residual
                last_layer_residual = residual
                new_cache.key_cache.append(cached_cache.key_cache[layer_idx])
                new_cache.value_cache.append(cached_cache.value_cache[layer_idx])

        self._prefix_vlm_last_sublayer_reused = int(reused_sublayers)
        if reused_sublayers:
            self._prefix_vlm_last_sublayer_attn_mask = _prefix_mask_to_spec(attn_reuse_mask)
            self._prefix_vlm_last_sublayer_mlp_mask = _prefix_mask_to_spec(mlp_reuse_mask)
        self._prefix_vlm_last_cross_layer_residual_reused = int(cross_layer_residual_reused)
        self._prefix_vlm_last_cross_layer_residual_total = int(cross_layer_residual_total)
        if cross_layer_residual_reuse_enabled:
            self._prefix_vlm_last_cross_layer_residual_mask = _prefix_mask_to_spec(cross_layer_residual_reuse_mask)
        language_model.norm(hidden_states, None)
        return new_cache

    def _pytorch_prefix_layer_with_sublayer_cache(
        self,
        layer,
        layer_idx: int,
        hidden_states,
        prefix_att_4d,
        prefix_position_ids,
        new_cache,
        cached_cache,
        cached_sublayers,
        cache_position,
        position_embeddings,
        *,
        reuse_attn: bool,
        reuse_mlp: bool,
    ):
        residual = hidden_states
        normed, gate = layer.input_layernorm(hidden_states, None)
        if reuse_attn:
            attn_delta = cached_sublayers["attn"][layer_idx].to(normed.dtype)
            new_cache.key_cache.append(cached_cache.key_cache[layer_idx])
            new_cache.value_cache.append(cached_cache.value_cache[layer_idx])
        else:
            attn_delta = layer.self_attn(
                hidden_states=normed,
                attention_mask=prefix_att_4d,
                position_ids=prefix_position_ids,
                past_key_value=new_cache,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )[0]
        hidden_states = _torch_gated_residual(residual, attn_delta, gate)

        residual = hidden_states
        normed, gate = layer.post_attention_layernorm(hidden_states, None)
        if reuse_mlp:
            mlp_delta = cached_sublayers["mlp"][layer_idx].to(normed.dtype)
        else:
            mlp_delta = layer.mlp(normed)
        return _torch_gated_residual(residual, mlp_delta, gate)

    def _pytorch_sample_suffix(self, state, prefix_pad_masks, kv_cache, noise, num_steps: int):
        x_t = noise
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=x_t.device)
        for step in range(num_steps):
            timestep = torch.full((x_t.shape[0],), 1.0 - step / num_steps, dtype=torch.float32, device=x_t.device)
            v_t = self._model.denoise_step(state, prefix_pad_masks, kv_cache, x_t, timestep)
            x_t = x_t + dt * v_t
        return x_t

    def _pytorch_sample_suffix_with_action_cache(
        self,
        state,
        prefix_pad_masks,
        kv_cache,
        noise,
        num_steps: int,
        *,
        allow_cache: bool,
    ):
        cached_steps = self._action_suffix_cached_residuals
        can_use_cache = (
            allow_cache
            and cached_steps is not None
            and self._action_suffix_cached_num_steps == num_steps
            and int(cached_steps.shape[0]) == num_steps
        )
        x_t = noise
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=x_t.device)
        next_step_residuals = []
        hit_steps = 0
        computed_blocks = None
        skipped_blocks = None
        cache_reason = "hit" if can_use_cache else ("disabled" if not allow_cache else "empty")

        for step in range(num_steps):
            timestep = torch.full((x_t.shape[0],), 1.0 - step / num_steps, dtype=torch.float32, device=x_t.device)
            cached_step = cached_steps[step] if can_use_cache else None
            v_t, step_residuals, used_cache, depth, step_computed_blocks = (
                self._pytorch_denoise_step_with_action_residuals(
                    state,
                    prefix_pad_masks,
                    kv_cache,
                    x_t,
                    timestep,
                    cached_step,
                )
            )
            x_t = x_t + dt * v_t
            next_step_residuals.append(step_residuals)
            if used_cache:
                hit_steps += 1
            computed_blocks = step_computed_blocks
            skipped_blocks = depth - step_computed_blocks

        self._action_suffix_cached_residuals = torch.stack(next_step_residuals).detach()
        self._action_suffix_cached_num_steps = num_steps

        return x_t, {
            "prefix_vlm_cache_action_suffix_enabled": True,
            "prefix_vlm_cache_action_suffix_hit_steps": int(hit_steps),
            "prefix_vlm_cache_action_suffix_total_steps": int(num_steps),
            "prefix_vlm_cache_action_suffix_reason": cache_reason,
            "prefix_vlm_cache_action_suffix_computed_blocks": int(computed_blocks or 0),
            "prefix_vlm_cache_action_suffix_skipped_blocks": int(skipped_blocks or 0),
        }

    def _pytorch_sample_suffix_with_intra_action_cache(
        self,
        state,
        prefix_pad_masks,
        kv_cache,
        noise,
        num_steps: int,
        *,
        fn_blocks: int | None = None,
        bn_blocks: int | None = None,
        schedule: str | None = None,
        refresh_interval: int | None = None,
    ):
        x_t = noise
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=x_t.device)
        cached_residuals = None
        schedule = self._prefix_vlm_intra_action_schedule(schedule)
        refresh_interval = max(
            1,
            int(
                self._prefix_vlm_cache_config.intra_action_cache_refresh_interval
                if refresh_interval is None
                else refresh_interval
            ),
        )
        fn_blocks = int(
            self._prefix_vlm_cache_config.intra_action_cache_fn_compute_blocks if fn_blocks is None else fn_blocks
        )
        bn_blocks = int(
            self._prefix_vlm_cache_config.intra_action_cache_bn_compute_blocks if bn_blocks is None else bn_blocks
        )
        cached_steps = 0
        refresh_steps = 0
        computed_blocks_total = 0
        depth = None
        refresh_step_set = set(self._prefix_vlm_intra_action_refresh_steps(schedule, num_steps, refresh_interval))
        anchor_schedule = schedule in {"first_last", "every3"}

        for step in range(num_steps):
            timestep = torch.full((x_t.shape[0],), 1.0 - step / num_steps, dtype=torch.float32, device=x_t.device)
            if anchor_schedule:
                should_refresh = cached_residuals is None or step in refresh_step_set
            else:
                should_refresh = cached_residuals is None or step % refresh_interval == 0
            v_t, residuals, used_cache, step_depth, computed_blocks = (
                self._pytorch_denoise_step_with_action_residuals(
                    state,
                    prefix_pad_masks,
                    kv_cache,
                    x_t,
                    timestep,
                    None if should_refresh else cached_residuals,
                    fn_blocks=fn_blocks,
                    bn_blocks=bn_blocks,
                )
            )
            x_t = x_t + dt * v_t
            if should_refresh or not anchor_schedule:
                cached_residuals = residuals.detach()
            cached_steps += int(used_cache)
            refresh_steps += int(not used_cache)
            computed_blocks_total += int(computed_blocks)
            depth = int(step_depth)

        depth = int(depth or 0)
        full_blocks = depth * num_steps
        return x_t, {
            "prefix_vlm_cache_intra_action_enabled": True,
            "prefix_vlm_cache_intra_action_cached_steps": int(cached_steps),
            "prefix_vlm_cache_intra_action_refresh_steps": int(refresh_steps),
            "prefix_vlm_cache_intra_action_total_steps": int(num_steps),
            "prefix_vlm_cache_intra_action_fn_blocks": int(fn_blocks),
            "prefix_vlm_cache_intra_action_bn_blocks": int(bn_blocks),
            "prefix_vlm_cache_intra_action_schedule": schedule,
            "prefix_vlm_cache_intra_action_refresh_interval": int(refresh_interval),
            "prefix_vlm_cache_intra_action_computed_blocks": int(computed_blocks_total),
            "prefix_vlm_cache_intra_action_skipped_blocks": int(full_blocks - computed_blocks_total),
            "prefix_vlm_cache_intra_action_computed_block_fraction": float(
                computed_blocks_total / max(1, full_blocks)
            ),
        }

    def _pytorch_compiled_intra_action_suffix(
        self,
        state,
        prefix_pad_masks,
        kv_cache,
        noise,
        num_steps: int,
        *,
        fn_blocks: int | None = None,
        bn_blocks: int | None = None,
        schedule: str | None = None,
        refresh_interval: int | None = None,
    ):
        depth = self._prefix_vlm_action_depth()
        schedule = self._prefix_vlm_intra_action_schedule(schedule)
        fn_blocks = min(
            max(
                int(
                    self._prefix_vlm_cache_config.intra_action_cache_fn_compute_blocks
                    if fn_blocks is None
                    else fn_blocks
                ),
                0,
            ),
            depth,
        )
        bn_blocks = min(
            max(
                int(
                    self._prefix_vlm_cache_config.intra_action_cache_bn_compute_blocks
                    if bn_blocks is None
                    else bn_blocks
                ),
                0,
            ),
            depth,
        )
        refresh_interval = max(
            1,
            int(
                self._prefix_vlm_cache_config.intra_action_cache_refresh_interval
                if refresh_interval is None
                else refresh_interval
            ),
        )
        key = (
            int(fn_blocks),
            int(bn_blocks),
            schedule,
            int(refresh_interval),
            int(num_steps),
            bool(self._prefix_vlm_cache_config.intra_action_cache_compile),
            self._prefix_vlm_cache_config.intra_action_cache_compile_mode,
        )
        fn = self._pytorch_compiled_intra_action_suffix_fns.get(key)
        if fn is None:
            fn = self._make_pytorch_intra_action_suffix_fn(
                num_steps,
                fn_blocks=fn_blocks,
                bn_blocks=bn_blocks,
                schedule=schedule,
                refresh_interval=refresh_interval,
            )
            if self._prefix_vlm_cache_config.intra_action_cache_compile:
                fn = torch.compile(fn, mode=self._prefix_vlm_cache_config.intra_action_cache_compile_mode)
            self._pytorch_compiled_intra_action_suffix_fns[key] = fn
        return fn(state, prefix_pad_masks, noise, *kv_cache.key_cache, *kv_cache.value_cache)

    def _make_pytorch_intra_action_suffix_fn(
        self,
        num_steps: int,
        *,
        fn_blocks: int,
        bn_blocks: int,
        schedule: str,
        refresh_interval: int,
    ):
        from transformers.cache_utils import DynamicCache

        depth = len(self._model.paligemma_with_expert.paligemma.language_model.layers)

        def intra_action_suffix_fn(state, prefix_pad_masks, noise, *kv_tensors):
            kv_cache = DynamicCache()
            kv_cache.key_cache = list(kv_tensors[:depth])
            kv_cache.value_cache = list(kv_tensors[depth:])
            return self._pytorch_sample_suffix_with_intra_action_cache(
                state,
                prefix_pad_masks,
                kv_cache,
                noise,
                num_steps,
                fn_blocks=fn_blocks,
                bn_blocks=bn_blocks,
                schedule=schedule,
                refresh_interval=refresh_interval,
            )[0]

        return intra_action_suffix_fn

    def _pytorch_denoise_step_with_action_residuals(
        self,
        state,
        prefix_pad_masks,
        kv_cache,
        x_t,
        timestep,
        cached_step_residuals,
        *,
        fn_blocks: int | None = None,
        bn_blocks: int | None = None,
    ):
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self._model.embed_suffix(state, x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        full_att_4d = self._model._prepare_attention_masks_4d(full_att_2d_masks)  # noqa: SLF001

        expert_model = self._model.paligemma_with_expert.gemma_expert.model
        depth = len(expert_model.layers)
        cached_ok = (
            cached_step_residuals is not None
            and cached_step_residuals.shape[0] == depth
            and cached_step_residuals.shape[1:] == suffix_embs.shape
        )

        if cached_ok:
            suffix_out, residuals, computed_blocks = self._pytorch_action_head_cached_forward(
                suffix_embs,
                full_att_4d,
                position_ids,
                kv_cache,
                adarms_cond,
                cached_step_residuals,
                fn_blocks=fn_blocks,
                bn_blocks=bn_blocks,
            )
            used_cache = True
        else:
            suffix_out, residuals = self._pytorch_action_head_full_forward(
                suffix_embs,
                full_att_4d,
                position_ids,
                kv_cache,
                adarms_cond,
            )
            computed_blocks = depth
            used_cache = False

        suffix_out = suffix_out[:, -self._model.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self._model.action_out_proj(suffix_out), residuals, used_cache, depth, computed_blocks

    def _pytorch_action_head_full_forward(
        self,
        suffix_embs,
        full_att_4d,
        position_ids,
        kv_cache,
        adarms_cond,
    ):
        expert_model = self._model.paligemma_with_expert.gemma_expert.model
        expert_model.config._attn_implementation = "eager"  # noqa: SLF001
        residuals = [None] * len(expert_model.layers)
        handles = []

        def make_hook(layer_idx):
            def hook(_module, args, output):
                residuals[layer_idx] = (output[0].detach() - args[0].detach()).to(dtype=output[0].dtype)

            return hook

        for layer_idx, layer in enumerate(expert_model.layers):
            handles.append(layer.register_forward_hook(make_hook(layer_idx)))
        try:
            outputs_embeds, _ = self._model.paligemma_with_expert.forward(
                attention_mask=full_att_4d,
                position_ids=position_ids,
                past_key_values=kv_cache,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
        finally:
            for handle in handles:
                handle.remove()
        return outputs_embeds[1], torch.stack(residuals)

    def _pytorch_action_head_cached_forward(
        self,
        suffix_embs,
        full_att_4d,
        position_ids,
        kv_cache,
        adarms_cond,
        cached_residuals,
        *,
        fn_blocks: int | None = None,
        bn_blocks: int | None = None,
    ):
        expert_model = self._model.paligemma_with_expert.gemma_expert.model
        expert_model.config._attn_implementation = "eager"  # noqa: SLF001
        depth = len(expert_model.layers)
        fn_blocks = (
            self._prefix_vlm_cache_config.action_suffix_fn_compute_blocks if fn_blocks is None else fn_blocks
        )
        bn_blocks = (
            self._prefix_vlm_cache_config.action_suffix_bn_compute_blocks if bn_blocks is None else bn_blocks
        )
        fn_blocks = min(max(fn_blocks, 0), depth)
        bn_blocks = min(max(bn_blocks, 0), depth)
        middle_start = fn_blocks
        middle_end = depth - bn_blocks
        hidden_states = suffix_embs
        if expert_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.bfloat16)
        prefix_len = kv_cache.key_cache[0].shape[-2] if kv_cache.key_cache else 0
        cache_position = torch.arange(prefix_len, prefix_len + suffix_embs.shape[1], device=suffix_embs.device)
        position_embeddings = expert_model.rotary_emb(hidden_states, position_ids)
        residuals = []

        for layer_idx, layer in enumerate(expert_model.layers):
            if layer_idx < middle_start or layer_idx >= middle_end:
                before = hidden_states
                hidden_states = layer(
                    hidden_states,
                    attention_mask=full_att_4d,
                    position_ids=position_ids,
                    past_key_value=kv_cache,
                    use_cache=False,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    adarms_cond=adarms_cond,
                )[0]
                residual = (hidden_states.detach() - before.detach()).to(dtype=hidden_states.dtype)
            else:
                residual = cached_residuals[layer_idx].to(device=hidden_states.device, dtype=hidden_states.dtype)
                hidden_states = hidden_states + residual
            residuals.append(residual.detach())

        norm_out = expert_model.norm(hidden_states, cond=adarms_cond)
        hidden_states = norm_out[0] if isinstance(norm_out, tuple) else norm_out
        computed_blocks = fn_blocks + bn_blocks
        return hidden_states, torch.stack(residuals), computed_blocks

    def _pytorch_compiled_prefix_cache_hit(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise,
        cached_residuals,
        cached_kv,
        num_steps: int,
    ):
        key = (
            int(self._prefix_vlm_cache_config.fn_compute_blocks),
            int(self._prefix_vlm_cache_config.bn_compute_blocks),
            self._prefix_vlm_cache_config.compute_layer_mask or "",
            int(num_steps),
            bool(self._prefix_vlm_cache_config.pytorch_compile_hit_path),
        )
        fn = self._pytorch_compiled_prefix_vlm_hit.get(key)
        if fn is None:
            fn = self._make_pytorch_prefix_cache_hit_fn(num_steps)
            if self._prefix_vlm_cache_config.pytorch_compile_hit_path:
                fn = torch.compile(fn, mode=self._prefix_vlm_cache_config.pytorch_compile_mode)
            self._pytorch_compiled_prefix_vlm_hit[key] = fn
        return fn(
            images[0],
            images[1],
            images[2],
            img_masks[0],
            img_masks[1],
            img_masks[2],
            lang_tokens,
            lang_masks,
            state,
            noise,
            cached_residuals,
            *cached_kv.key_cache,
            *cached_kv.value_cache,
        )

    def _make_pytorch_prefix_cache_hit_fn(self, num_steps: int):
        from transformers.cache_utils import DynamicCache

        depth = len(self._model.paligemma_with_expert.paligemma.language_model.layers)

        def cache_hit_fn(
            image_0,
            image_1,
            image_2,
            mask_0,
            mask_1,
            mask_2,
            lang_tokens,
            lang_masks,
            state,
            noise,
            cached_residuals,
            *kv_tensors,
        ):
            cached_cache = DynamicCache()
            cached_cache.key_cache = list(kv_tensors[:depth])
            cached_cache.value_cache = list(kv_tensors[depth:])
            prefix_embs, prefix_pad_masks, prefix_att_4d, prefix_position_ids = self._pytorch_prefix_inputs(
                [image_0, image_1, image_2],
                [mask_0, mask_1, mask_2],
                lang_tokens,
                lang_masks,
            )
            coarse_cache = self._pytorch_coarse_prefix_forward(
                prefix_embs,
                prefix_att_4d,
                prefix_position_ids,
                cached_residuals,
                cached_cache,
            )
            return self._pytorch_sample_suffix(state, prefix_pad_masks, coarse_cache, noise, num_steps)

        return cache_hit_fn
