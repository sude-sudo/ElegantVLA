"""Threshold decisions, motion guards and reuse budgets."""

from typing import Any

import jax.numpy as jnp
import numpy as np

from elegantvla.openpi.legacy._helpers import (
    _prefix_compute_layer_mask,
    _prefix_prompt_matches,
    _prefix_raw_images,
    _relative_l2,
    _state_delta_components,
    _state_translation_delta,
    _threshold_condition_accepts,
)


class ThresholdScheduler:
    """Internal Policy methods; Policy owns the model and cache state."""

    def _prefix_vlm_cache_decision(
        self,
        signature: np.ndarray | None,
        prompt_key: str,
        state_signature: np.ndarray | None = None,
        observation_info: dict[str, Any] | None = None,
    ):
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
        state_info = self._prefix_vlm_state_guard_info(state_signature)
        observation_info = observation_info or {"enabled": False, "accept": True, "reason": "disabled"}
        action_deviation_gate_active = bool(
            self._prefix_vlm_action_deviation_gate is not None
            and self._prefix_vlm_cache_config.action_deviation_gate_replace_legacy_guards
        )
        if self._prefix_vlm_cached_residuals is None or self._prefix_vlm_cached_kv is None:
            return False, "empty", np.inf, state_info
        if self._prefix_vlm_queries < self._prefix_vlm_cache_config.max_warmup_queries:
            return False, "warmup", np.inf, state_info
        if self._prefix_vlm_episode_queries < self._prefix_vlm_cache_config.min_episode_queries_before_cache:
            return False, "episode_warmup", np.inf, state_info
        if _prefix_prompt_matches(
            self._prefix_vlm_current_prompt_text,
            self._prefix_vlm_cache_config.prompt_disable_cache_substrings,
        ):
            return False, "prompt_disabled", np.inf, state_info
        prompt_conservative_cache = self._prefix_vlm_prompt_conservative_cache_match()
        prompt_conservative_max_continuous = int(
            self._prefix_vlm_cache_config.prompt_conservative_max_continuous_cached_queries
        )
        if (
            prompt_conservative_cache
            and prompt_conservative_max_continuous > 0
            and self._prefix_vlm_continuous_cached >= prompt_conservative_max_continuous
        ):
            return False, "prompt_conservative_continuous_hit_budget", np.inf, state_info
        signature_l2 = 0.0
        if signature is not None:
            signature_l2 = _relative_l2(signature, self._prefix_vlm_signature)
        max_cache_episode_query = int(self._prefix_vlm_cache_config.max_cache_episode_query)
        prompt_phase_guards = (
            (
                self._prefix_vlm_cache_config.prompt_max_cache_episode_query_substrings,
                self._prefix_vlm_cache_config.prompt_max_cache_episode_query,
            ),
            (
                self._prefix_vlm_cache_config.prompt2_max_cache_episode_query_substrings,
                self._prefix_vlm_cache_config.prompt2_max_cache_episode_query,
            ),
            (
                self._prefix_vlm_cache_config.prompt3_max_cache_episode_query_substrings,
                self._prefix_vlm_cache_config.prompt3_max_cache_episode_query,
            ),
            (
                self._prefix_vlm_cache_config.prompt4_max_cache_episode_query_substrings,
                self._prefix_vlm_cache_config.prompt4_max_cache_episode_query,
            ),
        )
        for prompt_substrings, prompt_query_limit in prompt_phase_guards:
            prompt_query_limit = int(prompt_query_limit)
            if prompt_query_limit <= 0 or not _prefix_prompt_matches(
                self._prefix_vlm_current_prompt_text,
                prompt_substrings,
            ):
                continue
            if max_cache_episode_query > 0:
                max_cache_episode_query = min(max_cache_episode_query, prompt_query_limit)
            else:
                max_cache_episode_query = prompt_query_limit
        if (
            not action_deviation_gate_active
            and max_cache_episode_query > 0
            and self._prefix_vlm_episode_queries + 1 > max_cache_episode_query
        ):
            return False, "episode_phase", np.inf, state_info
        max_episode_hit_ratio = float(self._prefix_vlm_cache_config.max_episode_hit_ratio)
        prompt_max_episode_hit_ratio = float(self._prefix_vlm_cache_config.prompt_max_episode_hit_ratio)
        if prompt_max_episode_hit_ratio > 0 and _prefix_prompt_matches(
            self._prefix_vlm_current_prompt_text,
            self._prefix_vlm_cache_config.prompt_max_episode_hit_ratio_substrings,
        ):
            max_episode_hit_ratio = min(max_episode_hit_ratio, prompt_max_episode_hit_ratio)
        if not action_deviation_gate_active and max_episode_hit_ratio < 1.0:
            next_episode_queries = self._prefix_vlm_episode_queries + 1
            next_episode_hits = self._prefix_vlm_episode_hits + 1
            if next_episode_hits / max(1, next_episode_queries) > max_episode_hit_ratio:
                return False, "hit_budget", np.inf, state_info
        prompt_conservative_max_hit_ratio = float(
            self._prefix_vlm_cache_config.prompt_conservative_max_episode_hit_ratio
        )
        if prompt_conservative_cache and 0.0 < prompt_conservative_max_hit_ratio < 1.0:
            next_episode_queries = self._prefix_vlm_episode_queries + 1
            next_episode_hits = self._prefix_vlm_episode_hits + 1
            if next_episode_hits / max(1, next_episode_queries) > prompt_conservative_max_hit_ratio:
                return False, "prompt_conservative_hit_budget", np.inf, state_info
        if prompt_key != self._prefix_vlm_prompt_key:
            return False, "prompt", np.inf, state_info
        if (
            not action_deviation_gate_active
            and self._prefix_vlm_continuous_cached >= self._prefix_vlm_cache_config.max_continuous_cached_queries
        ):
            return False, "continuous_cap", np.inf, state_info
        if not action_deviation_gate_active and state_info["enabled"] and not state_info["accept"]:
            return False, str(state_info["reason"]), np.inf, state_info
        if not action_deviation_gate_active and observation_info["enabled"] and not observation_info["accept"]:
            return False, str(observation_info["reason"]), np.inf, state_info

        if not action_deviation_gate_active and self._prefix_vlm_adaptive_scheduler is not None:
            accept, scheduler_reason, compute_layer_mask = self._prefix_vlm_adaptive_scheduler_decision(
                signature_l2,
                state_info,
            )
            if not accept:
                return False, scheduler_reason, signature_l2, state_info
            self._prefix_vlm_active_compute_layer_mask = compute_layer_mask
            return True, scheduler_reason, signature_l2, state_info

        if (
            not action_deviation_gate_active
            and signature_l2 > self._prefix_vlm_cache_config.image_signature_l2_threshold
        ):
            return False, "image", signature_l2, state_info
        accum_threshold = float(self._prefix_vlm_cache_config.image_signature_accum_l2_threshold)
        if not action_deviation_gate_active and accum_threshold >= 0:
            projected_accum_l2 = self._prefix_vlm_accumulated_image_l2 + float(signature_l2)
            if projected_accum_l2 > accum_threshold:
                return False, "image_accum", signature_l2, state_info
        reason = "action_deviation_gate_candidate" if action_deviation_gate_active else "hit"
        return True, reason, signature_l2, state_info

    def _prefix_vlm_observation_gate_info(self, raw_inputs: dict | None) -> dict[str, Any]:
        info: dict[str, Any] = {
            "enabled": bool(self._prefix_vlm_cache_config.adaptive_observation_gate_enabled),
            "accept": True,
            "reason": "disabled",
            "base_image_mean_abs": np.inf,
            "wrist_image_rel_l2": np.inf,
        }
        if not info["enabled"]:
            return info
        current = _prefix_raw_images(raw_inputs)
        previous = self._prefix_vlm_previous_raw_images
        if current is None or previous is None:
            info.update({"accept": False, "reason": "observation_warmup"})
            return info

        base_mean_abs = float(np.mean(np.abs(current["base"] - previous["base"])))
        wrist_rel_l2 = _relative_l2(current["wrist"], previous["wrist"])
        info["base_image_mean_abs"] = base_mean_abs
        info["wrist_image_rel_l2"] = wrist_rel_l2

        base_threshold = float(self._prefix_vlm_cache_config.observation_base_image_mean_abs_max)
        if base_threshold >= 0 and base_mean_abs > base_threshold:
            info.update({"accept": False, "reason": "observation_base_image"})
            return info
        wrist_threshold = float(self._prefix_vlm_cache_config.observation_wrist_image_rel_l2_min)
        if wrist_threshold >= 0 and wrist_rel_l2 < wrist_threshold:
            info.update({"accept": False, "reason": "observation_wrist_image"})
            return info
        info.update({"accept": True, "reason": "observation_accept"})
        return info

    def _update_prefix_vlm_raw_observation(self, raw_inputs: dict | None) -> None:
        current = _prefix_raw_images(raw_inputs)
        self._prefix_vlm_previous_raw_images = current

    def _prefix_vlm_requires_image_signature(self) -> bool:
        if float(self._prefix_vlm_cache_config.image_signature_l2_threshold) < 1.0e8:
            return True
        if float(self._prefix_vlm_cache_config.image_signature_accum_l2_threshold) >= 0:
            return True
        scheduler = self._prefix_vlm_adaptive_scheduler
        if scheduler is not None:
            for schedule in scheduler.get("schedules", []):
                if "image_l2_prev" in schedule.get("thresholds", {}):
                    return True
        return False

    def _prefix_vlm_cache_path_bypass_info(self, state_signature: np.ndarray | None) -> tuple[bool, str, dict[str, Any]]:
        state_info = self._prefix_vlm_state_guard_info(state_signature)
        if not self._prefix_vlm_cache_config.adaptive_bypass_after_cache_window:
            return False, "none", state_info
        max_cache_episode_query = int(self._prefix_vlm_cache_config.max_cache_episode_query)
        next_query = self._prefix_vlm_episode_queries + 1
        if max_cache_episode_query > 0 and next_query > max_cache_episode_query:
            return True, "adaptive_bypass_phase", state_info
        return False, "none", state_info

    def _prefix_vlm_adaptive_scheduler_decision(
        self,
        signature_l2: float,
        state_info: dict[str, Any],
    ) -> tuple[bool, str, str | None]:
        scheduler = self._prefix_vlm_adaptive_scheduler
        if scheduler is None:
            return True, "hit", None

        features = {
            "episode_local_query_idx": float(self._prefix_vlm_episode_queries + 1),
            "episode_query_idx": float(self._prefix_vlm_episode_queries + 1),
            "image_l2_prev": float(signature_l2),
            "state_delta": float(state_info.get("state_delta", np.inf)),
            "state_translation_delta": float(state_info.get("state_translation_delta", np.inf)),
            "state_rotation_delta": float(state_info.get("state_rotation_delta", np.inf)),
            "state_gripper_delta": float(state_info.get("state_gripper_delta", np.inf)),
            "cached_state_delta": float(state_info.get("cached_state_delta", np.inf)),
        }
        for schedule in scheduler.get("schedules", []):
            thresholds = schedule.get("thresholds", {})
            accepted = True
            for feature_name, threshold in thresholds.items():
                value = features.get(feature_name, np.inf)
                if not np.isfinite(value) or value > float(threshold):
                    accepted = False
                    break
            if accepted:
                name = str(schedule.get("name", "scheduler"))
                return True, f"hit_{name}", schedule.get("mask")
        return False, "scheduler_reject", None

    def _prefix_vlm_action_deviation_gate_decision(
        self,
        signature_l2: float,
        state_info: dict[str, Any],
        *,
        first_layer_l2: float,
        first_layer_cka: float,
        bank_selected_cka: float,
    ) -> tuple[bool, str]:
        gate = self._prefix_vlm_action_deviation_gate
        if gate is None:
            return True, "action_deviation_gate_disabled"
        features = {
            "prefix_vlm_cache_first_layer_cka": float(first_layer_cka),
            "prefix_vlm_cache_first_layer_l2": float(first_layer_l2),
            "prefix_vlm_cache_image_signature_l2": float(signature_l2),
            "prefix_vlm_cache_image_signature_accum_l2": float(self._prefix_vlm_accumulated_image_l2),
            "prefix_vlm_cache_state_delta": float(state_info.get("state_delta", np.inf)),
            "prefix_vlm_cache_state_translation_delta": float(state_info.get("state_translation_delta", np.inf)),
            "prefix_vlm_cache_state_translation_window_delta": float(
                state_info.get("state_translation_window_delta", np.inf)
            ),
            "prefix_vlm_cache_state_rotation_delta": float(state_info.get("state_rotation_delta", np.inf)),
            "prefix_vlm_cache_state_gripper_delta": float(state_info.get("state_gripper_delta", np.inf)),
            "prefix_vlm_cache_cached_state_delta": float(state_info.get("cached_state_delta", np.inf)),
            "prefix_vlm_cache_stable_state_deltas": float(state_info.get("stable_count", 0)),
            "prefix_vlm_cache_next_stable_state_deltas": float(state_info.get("next_stable_count", 0)),
            "prefix_vlm_cache_episode_query": float(self._prefix_vlm_episode_queries + 1),
            "prefix_vlm_cache_continuous_hits": float(self._prefix_vlm_continuous_cached),
            "prefix_vlm_cache_motion_signal": float(self._prefix_vlm_motion_signal(state_info)),
            "prefix_vlm_cache_bank_selected_cka": float(bank_selected_cka),
        }
        features.update(
            {
                "first_layer_cka": features["prefix_vlm_cache_first_layer_cka"],
                "first_layer_l2": features["prefix_vlm_cache_first_layer_l2"],
                "image_signature_l2": features["prefix_vlm_cache_image_signature_l2"],
                "state_delta": features["prefix_vlm_cache_state_delta"],
                "state_translation_delta": features["prefix_vlm_cache_state_translation_delta"],
                "state_rotation_delta": features["prefix_vlm_cache_state_rotation_delta"],
                "state_gripper_delta": features["prefix_vlm_cache_state_gripper_delta"],
                "cached_state_delta": features["prefix_vlm_cache_cached_state_delta"],
                "episode_query": features["prefix_vlm_cache_episode_query"],
                "continuous_hits": features["prefix_vlm_cache_continuous_hits"],
                "motion_signal": features["prefix_vlm_cache_motion_signal"],
                "bank_selected_cka": features["prefix_vlm_cache_bank_selected_cka"],
            }
        )
        for condition in gate.get("conditions", []):
            feature_name = str(condition.get("feature", ""))
            value = features.get(feature_name, np.inf)
            threshold = float(condition.get("threshold", np.inf))
            op = str(condition.get("op", "<="))
            if not np.isfinite(value):
                return False, f"action_deviation_gate_missing_{feature_name}"
            if not _threshold_condition_accepts(float(value), op, threshold):
                return False, f"action_deviation_gate_{feature_name}_{op}_{threshold:g}"
        name = str(gate.get("name", "action_deviation_gate"))
        return True, f"action_deviation_gate_{name}"

    def _prefix_vlm_state_guard_info(self, state_signature: np.ndarray | None) -> dict[str, Any]:
        guard_enabled = bool(self._prefix_vlm_cache_config.state_velocity_guard_enabled)
        motion_needed = self._prefix_vlm_action_scheduler_needs_motion() or self._prefix_vlm_motion_tier_enabled()
        info: dict[str, Any] = {
            "enabled": guard_enabled,
            "motion_enabled": motion_needed,
            "accept": True,
            "stable_now": True,
            "reason": "disabled" if not guard_enabled else "state_stable",
            "stable_count": int(self._prefix_vlm_stable_state_deltas),
            "next_stable_count": int(self._prefix_vlm_stable_state_deltas),
            "state_delta": np.inf,
            "state_translation_delta": np.inf,
            "state_translation_window_delta": np.inf,
            "state_rotation_delta": np.inf,
            "state_gripper_delta": np.inf,
            "cached_state_delta": np.inf,
        }
        if not guard_enabled and not motion_needed:
            return info
        if state_signature is None:
            info.update({"stable_now": False, "reason": "state_missing"})
            if guard_enabled:
                info["accept"] = False
            return info

        if self._prefix_vlm_previous_state_signature is None:
            info.update({"stable_now": False, "reason": "state_warmup"})
            if guard_enabled:
                info["accept"] = False
            return info

        deltas = _state_delta_components(state_signature, self._prefix_vlm_previous_state_signature)
        info.update(deltas)
        info["state_translation_window_delta"] = self._prefix_vlm_translation_window_delta(state_signature)
        stable_now = True

        if self._prefix_vlm_cached_state_signature is not None:
            info["cached_state_delta"] = _state_delta_components(
                state_signature,
                self._prefix_vlm_cached_state_signature,
            )["state_delta"]

        if guard_enabled:
            checks = [
                ("state_delta_threshold", "state_delta", "state_velocity"),
                ("state_translation_delta_threshold", "state_translation_delta", "state_translation"),
                ("state_rotation_delta_threshold", "state_rotation_delta", "state_rotation"),
                ("state_gripper_delta_threshold", "state_gripper_delta", "state_gripper"),
            ]
            for config_name, value_name, reason in checks:
                threshold = float(getattr(self._prefix_vlm_cache_config, config_name))
                if threshold >= 0 and float(info[value_name]) > threshold:
                    stable_now = False
                    info.update({"accept": False, "stable_now": False, "reason": reason})
                    break

            cached_delta_threshold = float(self._prefix_vlm_cache_config.cached_state_delta_threshold)
            if stable_now and cached_delta_threshold >= 0 and self._prefix_vlm_cached_state_signature is not None:
                if float(info["cached_state_delta"]) > cached_delta_threshold:
                    stable_now = False
                    info.update({"accept": False, "stable_now": False, "reason": "cached_state"})

            next_stable_count = self._prefix_vlm_stable_state_deltas + 1 if stable_now else 0
            info["next_stable_count"] = int(next_stable_count)
            info["stable_now"] = bool(stable_now)
            min_stable = int(max(1, self._prefix_vlm_cache_config.min_stable_state_deltas_for_cache))
            if stable_now and next_stable_count < min_stable:
                info.update({"accept": False, "reason": "state_stable_warmup"})
            elif stable_now:
                info.update({"accept": True, "reason": "state_stable"})
        else:
            info.update({"accept": True, "stable_now": True, "reason": "state_motion_only"})
        return info

    def _update_prefix_vlm_state_phase(self, state_signature: np.ndarray | None) -> None:
        if state_signature is None:
            self._prefix_vlm_previous_state_signature = None
            self._prefix_vlm_motion_state_buffer = []
            self._prefix_vlm_stable_state_deltas = 0
            return
        if self._prefix_vlm_cache_config.state_velocity_guard_enabled:
            if self._prefix_vlm_previous_state_signature is None:
                self._prefix_vlm_stable_state_deltas = 0
            else:
                info = self._prefix_vlm_state_guard_info(state_signature)
                self._prefix_vlm_stable_state_deltas = int(info["next_stable_count"]) if info["stable_now"] else 0
        self._prefix_vlm_previous_state_signature = state_signature
        self._prefix_vlm_update_motion_state_buffer(state_signature)

    def _prefix_vlm_update_motion_state_buffer(self, state_signature: np.ndarray | None) -> None:
        if state_signature is None:
            self._prefix_vlm_motion_state_buffer = []
            return
        window = max(1, int(self._prefix_vlm_cache_config.threshold_scheduler_motion_buffer_size))
        state = np.ravel(np.asarray(state_signature, dtype=np.float32)).copy()
        self._prefix_vlm_motion_state_buffer.append(state)
        if len(self._prefix_vlm_motion_state_buffer) > window:
            self._prefix_vlm_motion_state_buffer = self._prefix_vlm_motion_state_buffer[-window:]

    def _prefix_vlm_translation_window_delta(self, state_signature: np.ndarray | None) -> float:
        if state_signature is None:
            return np.inf
        current = np.ravel(np.asarray(state_signature, dtype=np.float32))
        window = max(2, int(self._prefix_vlm_cache_config.threshold_scheduler_motion_buffer_size))
        states = [np.ravel(np.asarray(state, dtype=np.float32)) for state in self._prefix_vlm_motion_state_buffer]
        states.append(current)
        states = states[-window:]
        if len(states) < 2:
            return np.inf
        deltas = []
        for prev, cur in zip(states[:-1], states[1:]):
            if prev.shape != cur.shape:
                continue
            deltas.append(_state_translation_delta(cur, prev))
        if not deltas:
            return np.inf
        return float(np.mean(deltas))

    def _prefix_vlm_prompt_conservative_cache_match(self) -> bool:
        return _prefix_prompt_matches(
            self._prefix_vlm_current_prompt_text,
            self._prefix_vlm_cache_config.prompt_conservative_cache_substrings,
        )

    def _prefix_vlm_skip_v4_no_image_prompt_blocked(self) -> bool:
        return _prefix_prompt_matches(
            self._prefix_vlm_current_prompt_text,
            self._prefix_vlm_cache_config.skip_v4_no_image_disable_prompt_substrings,
        )

    def _prefix_vlm_compute_mask(self):
        depth = int(self._prefix_vlm_cached_residuals.shape[0])
        return jnp.asarray(self._prefix_vlm_compute_mask_list(depth), dtype=jnp.bool_)

    def _prefix_vlm_compute_mask_list(self, depth: int) -> list[bool]:
        return _prefix_compute_layer_mask(
            depth,
            self._prefix_vlm_active_compute_layer_mask or self._prefix_vlm_cache_config.compute_layer_mask,
            self._prefix_vlm_cache_config.fn_compute_blocks,
            self._prefix_vlm_cache_config.bn_compute_blocks,
            self._prefix_vlm_cache_config.compute_layer_strategy,
            self._prefix_vlm_cache_config.compute_layer_count,
        )

    def _pytorch_skip_v4_no_image_decision(
        self,
        prompt_key: str,
        state_signature: np.ndarray | None,
    ) -> tuple[bool, str, dict[str, Any]]:
        state_info = self._prefix_vlm_state_guard_info(state_signature)
        if self._pytorch_skip_v4_max_budget() <= 0:
            return False, "skip_v4_disabled", state_info
        if self._prefix_vlm_skip_v4_no_image_remaining <= 0:
            return False, "skip_v4_budget_empty", state_info
        if self._prefix_vlm_skip_v4_no_image_prompt_blocked():
            self._prefix_vlm_skip_v4_no_image_remaining = 0
            return False, "skip_v4_prompt_no_image_disabled", state_info
        cache_ready = (
            self._prefix_vlm_cached_kv is not None
            and self._prefix_vlm_cached_prefix_pad_masks is not None
            and self._prefix_vlm_prompt_key is not None
        )
        if not cache_ready:
            self._prefix_vlm_skip_v4_no_image_remaining = 0
            return False, "skip_v4_cache_empty", state_info
        if prompt_key != self._prefix_vlm_prompt_key:
            self._prefix_vlm_skip_v4_no_image_remaining = 0
            return False, "skip_v4_prompt", state_info
        max_episode_hit_ratio = float(self._prefix_vlm_cache_config.max_episode_hit_ratio)
        if max_episode_hit_ratio < 1.0:
            next_episode_queries = self._prefix_vlm_episode_queries + 1
            next_episode_hits = self._prefix_vlm_episode_hits + 1
            if next_episode_hits / max(1, next_episode_queries) > max_episode_hit_ratio:
                self._prefix_vlm_skip_v4_no_image_remaining = 0
                return False, "skip_v4_hit_budget", state_info
        max_continuous = int(self._prefix_vlm_cache_config.max_continuous_cached_queries)
        if max_continuous >= 0 and self._prefix_vlm_continuous_cached >= max_continuous:
            self._prefix_vlm_skip_v4_no_image_remaining = 0
            return False, "skip_v4_continuous_cap", state_info
        if state_info["enabled"] and not state_info["accept"]:
            self._prefix_vlm_skip_v4_no_image_remaining = 0
            return False, f"skip_v4_{state_info['reason']}", state_info
        cached_delta_threshold = float(self._prefix_vlm_cache_config.cached_state_delta_threshold)
        cached_delta = float(state_info.get("cached_state_delta", np.inf))
        if cached_delta_threshold >= 0 and (
            not np.isfinite(cached_delta) or cached_delta > cached_delta_threshold
        ):
            self._prefix_vlm_skip_v4_no_image_remaining = 0
            return False, "skip_v4_cached_state", state_info
        if self._prefix_vlm_motion_tier_enabled():
            min_motion = float(self._prefix_vlm_cache_config.motion_tier_no_image_min_speed)
            motion_signal = self._prefix_vlm_motion_signal(state_info)
            if min_motion >= 0 and (not np.isfinite(motion_signal) or motion_signal < min_motion):
                self._prefix_vlm_skip_v4_no_image_remaining = 0
                return False, "skip_v4_motion_tier_slow", state_info
            cached_state_delta_threshold = float(
                self._prefix_vlm_cache_config.motion_tier_no_image_cached_state_delta_threshold
            )
            cached_state_delta = float(state_info.get("cached_state_delta", np.inf))
            if cached_state_delta_threshold >= 0 and (
                not np.isfinite(cached_state_delta) or cached_state_delta > cached_state_delta_threshold
            ):
                self._prefix_vlm_skip_v4_no_image_remaining = 0
                return False, "skip_v4_motion_tier_cached_state", state_info
        return True, "skip_v4_no_image", state_info

    def _pytorch_skip_v4_max_budget(self) -> int:
        if self._prefix_vlm_motion_tier_enabled():
            return max(
                0,
                int(self._prefix_vlm_cache_config.motion_tier_slow_no_image_budget),
                int(self._prefix_vlm_cache_config.motion_tier_medium_no_image_budget),
                int(self._prefix_vlm_cache_config.motion_tier_fast_no_image_budget),
            )
        if self._prefix_vlm_cache_config.threshold_scheduler_enabled:
            return max(0, int(self._prefix_vlm_cache_config.threshold_scheduler_max_no_image_budget))
        return max(0, int(self._prefix_vlm_cache_config.skip_v4_no_image_budget_after_hit))

    def _pytorch_skip_v4_budget_after_probe(self, first_layer_cka: float) -> int:
        if self._prefix_vlm_skip_v4_no_image_prompt_blocked():
            return 0
        if self._prefix_vlm_active_no_image_budget is not None:
            return max(0, int(self._prefix_vlm_active_no_image_budget))
        if not self._prefix_vlm_cache_config.threshold_scheduler_enabled:
            return max(0, int(self._prefix_vlm_cache_config.skip_v4_no_image_budget_after_hit))
        max_budget = self._pytorch_skip_v4_max_budget()
        if max_budget <= 0 or not np.isfinite(float(first_layer_cka)):
            return 0
        tiers = [
            (3, float(self._prefix_vlm_cache_config.threshold_scheduler_free3_cka_threshold)),
            (2, float(self._prefix_vlm_cache_config.threshold_scheduler_free2_cka_threshold)),
            (1, float(self._prefix_vlm_cache_config.threshold_scheduler_free1_cka_threshold)),
        ]
        any_enabled = False
        for budget, threshold in tiers:
            if threshold < 0:
                continue
            any_enabled = True
            if float(first_layer_cka) >= threshold:
                return min(int(budget), max_budget)
        if not any_enabled:
            return min(max(0, int(self._prefix_vlm_cache_config.skip_v4_no_image_budget_after_hit)), max_budget)
        return 0

    def _prefix_vlm_motion_tier_enabled(self) -> bool:
        return bool(self._prefix_vlm_cache_config.motion_tier_cache_enabled)

    def _pytorch_prefix_probe_enabled(self) -> bool:
        return (
            self._pytorch_first_layer_guard_enabled()
            or self._prefix_vlm_motion_tier_enabled()
            or self._prefix_vlm_action_deviation_gate is not None
            or (
                self._prefix_vlm_cache_config.observe_only
                and self._prefix_vlm_cache_config.action_deviation_label_enabled
            )
        )

    def _prefix_vlm_motion_tier_decision(
        self,
        state_info: dict[str, Any],
        first_layer_cka: float,
    ) -> tuple[bool, str]:
        motion_signal = self._prefix_vlm_motion_signal(state_info)
        slow_speed = float(self._prefix_vlm_cache_config.motion_tier_slow_speed_threshold)
        fast_speed = float(self._prefix_vlm_cache_config.motion_tier_fast_speed_threshold)
        very_fast_speed = float(self._prefix_vlm_cache_config.motion_tier_very_fast_speed_threshold)
        next_episode_query = int(self._prefix_vlm_episode_queries) + 1
        early_fast_max_query = int(self._prefix_vlm_cache_config.motion_tier_early_fast_max_episode_query)
        early_fast_speed = float(self._prefix_vlm_cache_config.motion_tier_early_fast_speed_threshold)
        if early_fast_max_query > 0 and next_episode_query <= early_fast_max_query and early_fast_speed >= 0:
            fast_speed = early_fast_speed if fast_speed < 0 else min(fast_speed, early_fast_speed)

        if not np.isfinite(motion_signal):
            self._prefix_vlm_active_motion_tier = "missing"
            self._prefix_vlm_active_motion_signal = float(motion_signal)
            self._prefix_vlm_active_motion_cka_threshold = np.nan
            self._prefix_vlm_active_motion_cached_state_delta_threshold = np.nan
            self._prefix_vlm_active_motion_prompt_guard = ""
            return False, "motion_tier_missing"

        self._prefix_vlm_active_motion_prompt_guard = ""
        very_fast_max_query = int(self._prefix_vlm_cache_config.motion_tier_very_fast_max_episode_query)
        very_fast_phase_accept = very_fast_max_query <= 0 or next_episode_query <= very_fast_max_query
        if very_fast_speed >= 0 and motion_signal >= very_fast_speed and very_fast_phase_accept:
            tier = "very_fast"
            mask = self._prefix_vlm_cache_config.motion_tier_very_fast_compute_layer_mask
            cka_threshold = float(self._prefix_vlm_cache_config.motion_tier_very_fast_cka_threshold)
            cached_state_delta_threshold = float(
                self._prefix_vlm_cache_config.motion_tier_very_fast_cached_state_delta_threshold
            )
            no_image_budget = 0
            no_image_cka_threshold = -1.0
            fallback_mask = (
                self._prefix_vlm_cache_config.motion_tier_very_fast_fallback_compute_layer_mask
                or self._prefix_vlm_cache_config.motion_tier_fast_compute_layer_mask
            )
            fallback_cka_threshold = float(
                self._prefix_vlm_cache_config.motion_tier_very_fast_fallback_cka_threshold
            )
            if fallback_cka_threshold < 0:
                fallback_cka_threshold = float(self._prefix_vlm_cache_config.motion_tier_fast_cka_threshold)
        elif fast_speed >= 0 and motion_signal >= fast_speed:
            tier = "fast"
            mask = self._prefix_vlm_cache_config.motion_tier_fast_compute_layer_mask
            cka_threshold = float(self._prefix_vlm_cache_config.motion_tier_fast_cka_threshold)
            cached_state_delta_threshold = float(
                self._prefix_vlm_cache_config.motion_tier_fast_cached_state_delta_threshold
            )
            no_image_budget = int(self._prefix_vlm_cache_config.motion_tier_fast_no_image_budget)
            no_image_cka_threshold = float(self._prefix_vlm_cache_config.motion_tier_fast_no_image_cka_threshold)
            fallback_mask = self._prefix_vlm_cache_config.motion_tier_fast_fallback_compute_layer_mask
            fallback_cka_threshold = float(self._prefix_vlm_cache_config.motion_tier_fast_fallback_cka_threshold)
            aggressive_max_query = int(self._prefix_vlm_cache_config.motion_tier_fast_aggressive_max_episode_query)
            prompt_text = self._prefix_vlm_current_prompt_text
            aggressive_disabled = _prefix_prompt_matches(
                prompt_text,
                self._prefix_vlm_cache_config.motion_tier_fast_aggressive_disable_prompt_substrings,
            )
            aggressive_conservative = _prefix_prompt_matches(
                prompt_text,
                self._prefix_vlm_cache_config.motion_tier_fast_aggressive_conservative_prompt_substrings,
            )
            if aggressive_disabled:
                self._prefix_vlm_active_motion_prompt_guard = "fast_aggressive_disabled_prompt"
                aggressive_max_query = 0
            elif aggressive_conservative:
                self._prefix_vlm_active_motion_prompt_guard = "fast_aggressive_conservative_prompt"
                conservative_max_query = int(
                    self._prefix_vlm_cache_config.motion_tier_fast_aggressive_conservative_max_episode_query
                )
                if conservative_max_query > 0:
                    aggressive_max_query = conservative_max_query
            if aggressive_max_query > 0 and next_episode_query <= aggressive_max_query:
                aggressive_mask = self._prefix_vlm_cache_config.motion_tier_fast_aggressive_compute_layer_mask
                if aggressive_conservative:
                    aggressive_mask = (
                        self._prefix_vlm_cache_config.motion_tier_fast_aggressive_conservative_compute_layer_mask
                        or aggressive_mask
                    )
                if aggressive_mask:
                    tier = "fast_aggressive"
                    mask = aggressive_mask
                aggressive_cka = float(self._prefix_vlm_cache_config.motion_tier_fast_aggressive_cka_threshold)
                if aggressive_conservative:
                    conservative_cka = float(
                        self._prefix_vlm_cache_config.motion_tier_fast_aggressive_conservative_cka_threshold
                    )
                    if conservative_cka >= 0:
                        aggressive_cka = conservative_cka
                if aggressive_cka >= 0:
                    cka_threshold = aggressive_cka
                aggressive_cached_state = float(
                    self._prefix_vlm_cache_config.motion_tier_fast_aggressive_cached_state_delta_threshold
                )
                if aggressive_conservative:
                    conservative_cached_state = float(
                        self._prefix_vlm_cache_config
                        .motion_tier_fast_aggressive_conservative_cached_state_delta_threshold
                    )
                    if conservative_cached_state >= 0:
                        aggressive_cached_state = conservative_cached_state
                if aggressive_cached_state >= 0:
                    cached_state_delta_threshold = aggressive_cached_state
                aggressive_fallback_mask = (
                    self._prefix_vlm_cache_config.motion_tier_fast_aggressive_fallback_compute_layer_mask
                )
                if aggressive_conservative:
                    aggressive_fallback_mask = (
                        self._prefix_vlm_cache_config
                        .motion_tier_fast_aggressive_conservative_fallback_compute_layer_mask
                        or aggressive_fallback_mask
                    )
                if aggressive_fallback_mask:
                    fallback_mask = aggressive_fallback_mask
                aggressive_fallback_cka = float(
                    self._prefix_vlm_cache_config.motion_tier_fast_aggressive_fallback_cka_threshold
                )
                if aggressive_conservative:
                    conservative_fallback_cka = float(
                        self._prefix_vlm_cache_config
                        .motion_tier_fast_aggressive_conservative_fallback_cka_threshold
                    )
                    if conservative_fallback_cka >= 0:
                        aggressive_fallback_cka = conservative_fallback_cka
                if aggressive_fallback_cka >= 0:
                    fallback_cka_threshold = aggressive_fallback_cka
        elif slow_speed >= 0 and motion_signal < slow_speed:
            tier = "slow"
            mask = self._prefix_vlm_cache_config.motion_tier_slow_compute_layer_mask
            cka_threshold = float(self._prefix_vlm_cache_config.motion_tier_slow_cka_threshold)
            cached_state_delta_threshold = float(
                self._prefix_vlm_cache_config.motion_tier_slow_cached_state_delta_threshold
            )
            no_image_budget = int(self._prefix_vlm_cache_config.motion_tier_slow_no_image_budget)
            no_image_cka_threshold = float(self._prefix_vlm_cache_config.motion_tier_slow_no_image_cka_threshold)
            fallback_mask = None
            fallback_cka_threshold = -1.0
        else:
            tier = "medium"
            mask = self._prefix_vlm_cache_config.motion_tier_medium_compute_layer_mask
            cka_threshold = float(self._prefix_vlm_cache_config.motion_tier_medium_cka_threshold)
            cached_state_delta_threshold = float(
                self._prefix_vlm_cache_config.motion_tier_medium_cached_state_delta_threshold
            )
            no_image_budget = int(self._prefix_vlm_cache_config.motion_tier_medium_no_image_budget)
            no_image_cka_threshold = float(self._prefix_vlm_cache_config.motion_tier_medium_no_image_cka_threshold)
            fallback_mask = None
            fallback_cka_threshold = -1.0

        self._prefix_vlm_active_motion_tier = tier
        self._prefix_vlm_active_motion_signal = float(motion_signal)
        self._prefix_vlm_active_motion_cka_threshold = float(cka_threshold)
        self._prefix_vlm_active_motion_cached_state_delta_threshold = float(cached_state_delta_threshold)
        if no_image_cka_threshold >= 0 and (
            not np.isfinite(first_layer_cka) or float(first_layer_cka) < no_image_cka_threshold
        ):
            no_image_budget = 0
        self._prefix_vlm_active_no_image_budget = max(0, no_image_budget)

        cached_state_delta = float(state_info.get("cached_state_delta", np.inf))
        if (
            cached_state_delta_threshold >= 0
            and (not np.isfinite(cached_state_delta) or cached_state_delta > cached_state_delta_threshold)
        ):
            self._prefix_vlm_active_compute_layer_mask = None
            self._prefix_vlm_active_no_image_budget = 0
            return False, f"motion_tier_{tier}_cached_state"

        if cka_threshold >= 0 and (not np.isfinite(first_layer_cka) or float(first_layer_cka) < cka_threshold):
            if (
                tier in {"fast", "fast_aggressive", "very_fast"}
                and fallback_mask
                and fallback_cka_threshold >= 0
                and np.isfinite(first_layer_cka)
                and float(first_layer_cka) >= fallback_cka_threshold
            ):
                tier = "fast_aggressive_fallback" if tier == "fast_aggressive" else "fast_fallback"
                self._prefix_vlm_active_motion_tier = tier
                self._prefix_vlm_active_motion_cka_threshold = float(fallback_cka_threshold)
                self._prefix_vlm_active_compute_layer_mask = fallback_mask
                self._prefix_vlm_active_no_image_budget = 0
                return True, f"motion_tier_{tier}"
            self._prefix_vlm_active_compute_layer_mask = None
            self._prefix_vlm_active_no_image_budget = 0
            return False, f"motion_tier_{tier}_cka"

        if np.isfinite(first_layer_cka):
            high_cka_mask = None
            high_cka_threshold = -1.0
            if tier == "medium":
                high_cka_mask = self._prefix_vlm_cache_config.motion_tier_medium_high_cka_compute_layer_mask
                high_cka_threshold = float(self._prefix_vlm_cache_config.motion_tier_medium_high_cka_threshold)
            elif tier == "fast":
                high_cka_mask = self._prefix_vlm_cache_config.motion_tier_fast_high_cka_compute_layer_mask
                high_cka_threshold = float(self._prefix_vlm_cache_config.motion_tier_fast_high_cka_threshold)
            if high_cka_mask and high_cka_threshold >= 0 and float(first_layer_cka) >= high_cka_threshold:
                tier = f"{tier}_high_cka"
                self._prefix_vlm_active_motion_tier = tier
                self._prefix_vlm_active_motion_cka_threshold = float(high_cka_threshold)
                mask = high_cka_mask

        self._prefix_vlm_active_compute_layer_mask = mask
        return True, f"motion_tier_{tier}"

    def _prefix_vlm_sublayer_cache_enabled(self) -> bool:
        return bool(self._prefix_vlm_cache_config.sublayer_cache_enabled)

    def _prefix_vlm_sublayer_mask_list(self, depth: int, spec: str | None) -> list[bool]:
        if not spec or not str(spec).strip():
            return [False] * depth
        return _prefix_compute_layer_mask(depth, str(spec), 0, 0, "first_last", 0)

    def _prefix_vlm_cross_layer_residual_cache_enabled(self) -> bool:
        return bool(self._prefix_vlm_cache_config.cross_layer_residual_cache_enabled)

    def _prefix_vlm_cache_bank_enabled(self) -> bool:
        return bool(self._prefix_vlm_cache_config.cache_bank_enabled)

    def _prefix_vlm_action_scheduler_needs_motion(self) -> bool:
        return bool(self._prefix_vlm_cache_config.threshold_scheduler_action_head_enabled)

    def _prefix_vlm_motion_signal(self, state_info: dict[str, Any] | None) -> float:
        state_info = state_info or {}
        trans = float(state_info.get("state_translation_window_delta", np.inf))
        if not np.isfinite(trans):
            trans = float(state_info.get("state_translation_delta", np.inf))
        return trans if np.isfinite(trans) else np.nan

    def _prefix_vlm_action_depth(self) -> int:
        if self._is_pytorch_model:
            return len(self._model.paligemma_with_expert.gemma_expert.model.layers)
        return int(self._model.llm_depth)

    def _prefix_vlm_intra_action_schedule(self, schedule: str | None = None) -> str:
        schedule = str(
            self._prefix_vlm_cache_config.intra_action_cache_schedule if schedule is None else schedule
        )
        schedule = schedule.strip().lower().replace("-", "_")
        aliases = {
            "prev_step": "refresh_interval",
            "previous_step": "refresh_interval",
            "refresh": "refresh_interval",
            "firstlast": "first_last",
            "every_3": "every3",
        }
        schedule = aliases.get(schedule, schedule)
        if schedule not in {"refresh_interval", "first_last", "every3"}:
            raise ValueError(f"Unsupported intra-action cache schedule: {schedule!r}")
        return schedule

    def _prefix_vlm_intra_action_refresh_steps(
        self,
        schedule: str,
        num_steps: int,
        refresh_interval: int,
    ) -> tuple[int, ...]:
        if num_steps <= 0:
            return ()
        if schedule == "first_last":
            return tuple(sorted({0, num_steps - 1}))
        steps = list(range(0, num_steps, max(1, int(refresh_interval))))
        if schedule == "every3" and steps[-1] != num_steps - 1:
            steps.append(num_steps - 1)
        return tuple(steps)

    def _prefix_vlm_intra_action_cache_plan(
        self,
        state_info: dict[str, Any] | None,
        *,
        prefix_cache_hit: bool = True,
    ) -> dict[str, Any]:
        depth = self._prefix_vlm_action_depth()
        schedule = self._prefix_vlm_intra_action_schedule()
        refresh_interval = max(1, int(self._prefix_vlm_cache_config.intra_action_cache_refresh_interval))
        fn_blocks = min(max(int(self._prefix_vlm_cache_config.intra_action_cache_fn_compute_blocks), 0), depth)
        bn_blocks = min(max(int(self._prefix_vlm_cache_config.intra_action_cache_bn_compute_blocks), 0), depth)
        signal = self._prefix_vlm_motion_signal(state_info) if self._prefix_vlm_action_scheduler_needs_motion() else np.nan
        action = "static"
        enabled = True
        if self._prefix_vlm_cache_config.intra_action_cache_requires_prefix_hit and not prefix_cache_hit:
            enabled = False
            action = "prefix_refresh"
        if self._prefix_vlm_action_scheduler_needs_motion():
            mode = str(self._prefix_vlm_cache_config.threshold_scheduler_action_head_selection_mode or "motion_speed_inner")
            mode = mode.strip().lower()
            speed_threshold = float(self._prefix_vlm_cache_config.threshold_scheduler_ah_speed_threshold)
            secondary_threshold = float(self._prefix_vlm_cache_config.threshold_scheduler_ah_secondary_threshold)
            if not enabled:
                pass
            elif not np.isfinite(signal) or signal < speed_threshold:
                enabled = False
                action = "refresh"
            elif mode.endswith("_inner") and secondary_threshold >= 0 and signal >= secondary_threshold:
                fn_blocks = min(
                    max(int(self._prefix_vlm_cache_config.threshold_scheduler_ah_reuse_all_fn_compute_blocks), 0),
                    depth,
                )
                bn_blocks = min(
                    max(int(self._prefix_vlm_cache_config.threshold_scheduler_ah_reuse_all_bn_compute_blocks), 0),
                    depth,
                )
                tier_refresh_interval = int(
                    self._prefix_vlm_cache_config.threshold_scheduler_ah_reuse_all_refresh_interval
                )
                if tier_refresh_interval > 0:
                    refresh_interval = max(1, tier_refresh_interval)
                action = "reuse_all_but_first"
            else:
                tier_fn_blocks = int(self._prefix_vlm_cache_config.threshold_scheduler_ah_reuse_middle_fn_compute_blocks)
                tier_bn_blocks = int(self._prefix_vlm_cache_config.threshold_scheduler_ah_reuse_middle_bn_compute_blocks)
                if tier_fn_blocks >= 0:
                    fn_blocks = min(max(tier_fn_blocks, 0), depth)
                if tier_bn_blocks >= 0:
                    bn_blocks = min(max(tier_bn_blocks, 0), depth)
                tier_refresh_interval = int(
                    self._prefix_vlm_cache_config.threshold_scheduler_ah_reuse_middle_refresh_interval
                )
                if tier_refresh_interval > 0:
                    refresh_interval = max(1, tier_refresh_interval)
                action = "reuse_middle"
        info = {
            "prefix_vlm_cache_threshold_scheduler_enabled": bool(self._prefix_vlm_cache_config.threshold_scheduler_enabled),
            "prefix_vlm_cache_threshold_scheduler_action_head_enabled": bool(
                self._prefix_vlm_cache_config.threshold_scheduler_action_head_enabled
            ),
            "prefix_vlm_cache_threshold_scheduler_ah_action": action,
            "prefix_vlm_cache_threshold_scheduler_ah_motion_signal": float(signal),
            "prefix_vlm_cache_threshold_scheduler_ah_fn_blocks": int(fn_blocks),
            "prefix_vlm_cache_threshold_scheduler_ah_bn_blocks": int(bn_blocks),
            "prefix_vlm_cache_threshold_scheduler_ah_schedule": schedule,
            "prefix_vlm_cache_threshold_scheduler_ah_refresh_interval": int(refresh_interval),
        }
        return {
            "enabled": enabled,
            "fn_blocks": int(fn_blocks),
            "bn_blocks": int(bn_blocks),
            "schedule": schedule,
            "refresh_interval": int(refresh_interval),
            "info": info,
        }

    def _pytorch_first_layer_guard_enabled(self) -> bool:
        metric = self._prefix_vlm_cache_config.first_layer_metric.lower()
        if metric == "l2":
            return self._prefix_vlm_cache_config.first_layer_l2_threshold >= 0
        if metric == "cka":
            return self._prefix_vlm_cache_config.first_layer_cka_threshold >= 0
        raise ValueError(f"Unsupported prefix VLM first-layer metric: {metric!r}")
