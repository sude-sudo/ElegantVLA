"""Cache metrics and optional trajectory diagnostics."""

from typing import Any

import numpy as np
import torch

from elegantvla.openpi.legacy._helpers import (
    _kv_cache_has_layer,
    _prefix_mask_to_spec,
    _torch_linear_cka,
    _torch_relative_l2,
)


class CacheDiagnostics:
    """Internal Policy methods; Policy owns the model and cache state."""

    def _jax_intra_action_cache_info(self, num_steps: int) -> dict[str, Any]:
        depth = int(self._model.llm_depth)
        refresh_interval = max(1, int(self._prefix_vlm_cache_config.intra_action_cache_refresh_interval))
        fn_blocks = min(max(int(self._prefix_vlm_cache_config.intra_action_cache_fn_compute_blocks), 0), depth)
        bn_blocks = min(max(int(self._prefix_vlm_cache_config.intra_action_cache_bn_compute_blocks), 0), depth)
        cached_block_count = min(fn_blocks + bn_blocks, depth)
        refresh_steps = sum(1 for step in range(num_steps) if step == 0 or step % refresh_interval == 0)
        cached_steps = num_steps - refresh_steps
        computed_blocks_total = refresh_steps * depth + cached_steps * cached_block_count
        full_blocks = depth * num_steps
        return {
            "prefix_vlm_cache_enabled": False,
            "prefix_vlm_cache_intra_action_enabled": True,
            "prefix_vlm_cache_intra_action_cached_steps": int(cached_steps),
            "prefix_vlm_cache_intra_action_refresh_steps": int(refresh_steps),
            "prefix_vlm_cache_intra_action_total_steps": int(num_steps),
            "prefix_vlm_cache_intra_action_fn_blocks": int(fn_blocks),
            "prefix_vlm_cache_intra_action_bn_blocks": int(bn_blocks),
            "prefix_vlm_cache_intra_action_refresh_interval": int(refresh_interval),
            "prefix_vlm_cache_intra_action_computed_blocks": int(computed_blocks_total),
            "prefix_vlm_cache_intra_action_skipped_blocks": int(full_blocks - computed_blocks_total),
            "prefix_vlm_cache_intra_action_computed_block_fraction": float(
                computed_blocks_total / max(1, full_blocks)
            ),
        }

    def _prefix_vlm_layer_cache_info(self, depth: int, computed_blocks: int) -> dict[str, Any]:
        reused_blocks = max(0, int(depth) - int(computed_blocks))
        if computed_blocks == depth:
            mask_spec = "full"
        elif computed_blocks == 0:
            mask_spec = "none"
        else:
            mask_spec = _prefix_mask_to_spec(self._prefix_vlm_compute_mask_list(depth))
        return {
            "prefix_vlm_cache_compute_layer_mask": mask_spec,
            "prefix_vlm_cache_computed_block_fraction": float(computed_blocks / max(1, depth)),
            "prefix_vlm_cache_reused_blocks": int(reused_blocks),
            "prefix_vlm_cache_reused_block_fraction": float(reused_blocks / max(1, depth)),
        }

    def _prefix_vlm_similarity_observation_info(
        self,
        first_layer_feature: torch.Tensor | None,
        prefix_pad_masks: torch.Tensor | None,
    ) -> dict[str, Any]:
        if not bool(self._prefix_vlm_cache_config.observe_similarity_matrix):
            return {}
        if first_layer_feature is None:
            return {
                "prefix_vlm_cache_pairwise_enabled": True,
                "prefix_vlm_cache_pairwise_query_index": len(self._prefix_vlm_similarity_features),
                "prefix_vlm_cache_pairwise_cka_row": [],
                "prefix_vlm_cache_pairwise_truncated": False,
            }

        max_queries = max(1, int(self._prefix_vlm_cache_config.observe_similarity_max_queries))
        query_index = len(self._prefix_vlm_similarity_features)
        truncated = query_index >= max_queries
        row: list[float] = []
        if not truncated:
            for previous in self._prefix_vlm_similarity_features:
                row.append(_torch_linear_cka(first_layer_feature, previous, prefix_pad_masks))
            row.append(1.0)
            self._prefix_vlm_similarity_features.append(first_layer_feature.detach().to("cpu"))
        return {
            "prefix_vlm_cache_pairwise_enabled": True,
            "prefix_vlm_cache_pairwise_query_index": int(query_index),
            "prefix_vlm_cache_pairwise_cka_row": row,
            "prefix_vlm_cache_pairwise_truncated": bool(truncated),
        }

    def _prefix_vlm_layer_similarity_observation_info(
        self,
        current_residuals: torch.Tensor | None,
        previous_residuals: torch.Tensor | None,
        current_kv: Any | None,
        previous_kv: Any | None,
        prefix_pad_masks: torch.Tensor | None,
    ) -> dict[str, Any]:
        if not bool(self._prefix_vlm_cache_config.observe_layer_similarity):
            return {}
        info: dict[str, Any] = {
            "prefix_vlm_cache_layerwise_enabled": True,
            "prefix_vlm_cache_layerwise_residual_cka": [],
            "prefix_vlm_cache_layerwise_residual_l2": [],
            "prefix_vlm_cache_layerwise_key_l2": [],
            "prefix_vlm_cache_layerwise_value_l2": [],
            "prefix_vlm_cache_layerwise_reuse_mask": "",
            "prefix_vlm_cache_layerwise_reuse_blocks": 0,
        }
        if current_residuals is None or previous_residuals is None:
            return info
        if current_residuals.shape != previous_residuals.shape:
            return info

        residual_cka: list[float] = []
        residual_l2: list[float] = []
        key_l2: list[float] = []
        value_l2: list[float] = []
        threshold = float(self._prefix_vlm_cache_config.observe_layer_cka_threshold)
        reuse_mask: list[bool] = []
        depth = int(current_residuals.shape[0])
        for layer_idx in range(depth):
            current_layer = current_residuals[layer_idx]
            previous_layer = previous_residuals[layer_idx]
            cka = _torch_linear_cka(current_layer, previous_layer, prefix_pad_masks)
            residual_cka.append(float(cka))
            residual_l2.append(float(_torch_relative_l2(current_layer, previous_layer)))
            reuse_mask.append(bool(np.isfinite(cka) and cka >= threshold))

            key_l2.append(
                float(_torch_relative_l2(current_kv.key_cache[layer_idx], previous_kv.key_cache[layer_idx]))
                if _kv_cache_has_layer(current_kv, previous_kv, layer_idx)
                else float("inf")
            )
            value_l2.append(
                float(_torch_relative_l2(current_kv.value_cache[layer_idx], previous_kv.value_cache[layer_idx]))
                if _kv_cache_has_layer(current_kv, previous_kv, layer_idx)
                else float("inf")
            )

        info.update(
            {
                "prefix_vlm_cache_layerwise_residual_cka": residual_cka,
                "prefix_vlm_cache_layerwise_residual_l2": residual_l2,
                "prefix_vlm_cache_layerwise_key_l2": key_l2,
                "prefix_vlm_cache_layerwise_value_l2": value_l2,
                "prefix_vlm_cache_layerwise_reuse_mask": _prefix_mask_to_spec(reuse_mask),
                "prefix_vlm_cache_layerwise_reuse_blocks": int(sum(reuse_mask)),
            }
        )
        return info

    def _prefix_vlm_layer_pairwise_similarity_observation_info(
        self,
        current_residuals: torch.Tensor | None,
        prefix_pad_masks: torch.Tensor | None,
    ) -> dict[str, Any]:
        if not bool(self._prefix_vlm_cache_config.observe_layer_similarity_matrix):
            return {}
        if current_residuals is None:
            return {
                "prefix_vlm_cache_layer_pairwise_enabled": True,
                "prefix_vlm_cache_layer_pairwise_query_index": len(self._prefix_vlm_layer_similarity_features),
                "prefix_vlm_cache_layer_pairwise_cka_rows": [],
                "prefix_vlm_cache_layer_pairwise_truncated": False,
            }

        max_queries = max(1, int(self._prefix_vlm_cache_config.observe_layer_similarity_matrix_max_queries))
        query_index = len(self._prefix_vlm_layer_similarity_features)
        truncated = query_index >= max_queries
        rows: list[list[float]] = []
        if not truncated:
            depth = int(current_residuals.shape[0])
            for layer_idx in range(depth):
                current_layer = current_residuals[layer_idx]
                row = [
                    _torch_linear_cka(current_layer, previous[layer_idx], prefix_pad_masks)
                    for previous in self._prefix_vlm_layer_similarity_features
                    if int(previous.shape[0]) == depth
                ]
                row.append(1.0)
                rows.append(row)
            self._prefix_vlm_layer_similarity_features.append(current_residuals.detach().to("cpu"))
        return {
            "prefix_vlm_cache_layer_pairwise_enabled": True,
            "prefix_vlm_cache_layer_pairwise_query_index": int(query_index),
            "prefix_vlm_cache_layer_pairwise_cka_rows": rows,
            "prefix_vlm_cache_layer_pairwise_truncated": bool(truncated),
        }

    def _prefix_vlm_adjacent_layer_similarity_observation_info(
        self,
        current_residuals: torch.Tensor | None,
        prefix_pad_masks: torch.Tensor | None,
    ) -> dict[str, Any]:
        if not bool(self._prefix_vlm_cache_config.observe_adjacent_layer_similarity):
            return {}
        info: dict[str, Any] = {
            "prefix_vlm_cache_adjacent_layer_enabled": True,
            "prefix_vlm_cache_adjacent_layer_residual_cka": [],
            "prefix_vlm_cache_adjacent_layer_residual_l2": [],
        }
        if current_residuals is None:
            return info

        residual_cka: list[float | None] = [None]
        residual_l2: list[float | None] = [None]
        depth = int(current_residuals.shape[0])
        for layer_idx in range(1, depth):
            current_layer = current_residuals[layer_idx]
            previous_layer = current_residuals[layer_idx - 1]
            residual_cka.append(float(_torch_linear_cka(current_layer, previous_layer, prefix_pad_masks)))
            residual_l2.append(float(_torch_relative_l2(current_layer, previous_layer)))
        info.update(
            {
                "prefix_vlm_cache_adjacent_layer_residual_cka": residual_cka,
                "prefix_vlm_cache_adjacent_layer_residual_l2": residual_l2,
            }
        )
        return info

    def _prefix_vlm_lagged_adjacent_layer_similarity_observation_info(
        self,
        current_residuals: torch.Tensor | None,
        previous_residuals: torch.Tensor | None,
        prefix_pad_masks: torch.Tensor | None,
    ) -> dict[str, Any]:
        if not bool(self._prefix_vlm_cache_config.observe_lagged_adjacent_layer_similarity):
            return {}
        info: dict[str, Any] = {
            "prefix_vlm_cache_lagged_adjacent_layer_enabled": True,
            "prefix_vlm_cache_lagged_adjacent_layer_residual_cka": [],
            "prefix_vlm_cache_lagged_adjacent_layer_residual_l2": [],
        }
        if current_residuals is None or previous_residuals is None:
            return info
        if current_residuals.shape != previous_residuals.shape:
            return info

        residual_cka: list[float | None] = [None]
        residual_l2: list[float | None] = [None]
        depth = int(current_residuals.shape[0])
        for layer_idx in range(1, depth):
            current_layer = current_residuals[layer_idx]
            previous_layer = previous_residuals[layer_idx - 1]
            residual_cka.append(float(_torch_linear_cka(current_layer, previous_layer, prefix_pad_masks)))
            residual_l2.append(float(_torch_relative_l2(current_layer, previous_layer)))
        info.update(
            {
                "prefix_vlm_cache_lagged_adjacent_layer_residual_cka": residual_cka,
                "prefix_vlm_cache_lagged_adjacent_layer_residual_l2": residual_l2,
            }
        )
        return info

    def _prefix_vlm_sublayer_similarity_observation_info(
        self,
        current_sublayer_deltas: dict[str, torch.Tensor] | None,
        previous_sublayer_deltas: dict[str, torch.Tensor] | None,
        prefix_pad_masks: torch.Tensor | None,
    ) -> dict[str, Any]:
        if not bool(self._prefix_vlm_cache_config.observe_sublayer_similarity):
            return {}
        info: dict[str, Any] = {
            "prefix_vlm_cache_sublayer_enabled": True,
            "prefix_vlm_cache_sublayer_attn_delta_cka": [],
            "prefix_vlm_cache_sublayer_attn_delta_l2": [],
            "prefix_vlm_cache_sublayer_mlp_delta_cka": [],
            "prefix_vlm_cache_sublayer_mlp_delta_l2": [],
        }
        if current_sublayer_deltas is None or previous_sublayer_deltas is None:
            return info

        for label, output_prefix in (("attn", "attn"), ("mlp", "mlp")):
            current = current_sublayer_deltas.get(label)
            previous = previous_sublayer_deltas.get(label)
            if current is None or previous is None or current.shape != previous.shape:
                continue
            cka_values: list[float] = []
            l2_values: list[float] = []
            for layer_idx in range(int(current.shape[0])):
                current_layer = current[layer_idx]
                previous_layer = previous[layer_idx]
                cka_values.append(float(_torch_linear_cka(current_layer, previous_layer, prefix_pad_masks)))
                l2_values.append(float(_torch_relative_l2(current_layer, previous_layer)))
            info[f"prefix_vlm_cache_sublayer_{output_prefix}_delta_cka"] = cka_values
            info[f"prefix_vlm_cache_sublayer_{output_prefix}_delta_l2"] = l2_values
        return info

    def _prefix_vlm_motion_tier_cache_info(self) -> dict[str, Any]:
        return {
            "prefix_vlm_cache_motion_tier_enabled": bool(self._prefix_vlm_motion_tier_enabled()),
            "prefix_vlm_cache_motion_tier": str(self._prefix_vlm_active_motion_tier),
            "prefix_vlm_cache_motion_signal": float(self._prefix_vlm_active_motion_signal),
            "prefix_vlm_cache_motion_tier_cka_threshold": float(self._prefix_vlm_active_motion_cka_threshold),
            "prefix_vlm_cache_motion_tier_cached_state_delta_threshold": float(
                self._prefix_vlm_active_motion_cached_state_delta_threshold
            ),
            "prefix_vlm_cache_motion_tier_prompt_guard": str(self._prefix_vlm_active_motion_prompt_guard),
            "prefix_vlm_cache_motion_tier_no_image_budget": int(self._prefix_vlm_active_no_image_budget or 0),
            "prefix_vlm_cache_motion_tier_no_image_min_speed": float(
                self._prefix_vlm_cache_config.motion_tier_no_image_min_speed
            ),
            "prefix_vlm_cache_motion_tier_no_image_cached_state_delta_threshold": float(
                self._prefix_vlm_cache_config.motion_tier_no_image_cached_state_delta_threshold
            ),
        }

    def _prefix_vlm_sublayer_cache_info(self) -> dict[str, Any]:
        total = int(self._prefix_vlm_last_sublayer_total)
        reused = int(self._prefix_vlm_last_sublayer_reused)
        return {
            "prefix_vlm_cache_sublayer_cache_enabled": bool(self._prefix_vlm_sublayer_cache_enabled()),
            "prefix_vlm_cache_sublayer_reused": reused,
            "prefix_vlm_cache_sublayer_total": total,
            "prefix_vlm_cache_sublayer_reused_fraction": float(reused / max(1, total)),
            "prefix_vlm_cache_sublayer_attn_mask": str(self._prefix_vlm_last_sublayer_attn_mask),
            "prefix_vlm_cache_sublayer_mlp_mask": str(self._prefix_vlm_last_sublayer_mlp_mask),
            "prefix_vlm_cache_sublayer_cka_gate_accept": bool(self._prefix_vlm_last_sublayer_cka_gate_accept),
            "prefix_vlm_cache_sublayer_cka_gate_threshold": float(
                self._prefix_vlm_last_sublayer_cka_gate_threshold
            ),
        }

    def _prefix_vlm_cross_layer_residual_cache_info(self) -> dict[str, Any]:
        total = int(self._prefix_vlm_last_cross_layer_residual_total)
        reused = int(self._prefix_vlm_last_cross_layer_residual_reused)
        return {
            "prefix_vlm_cache_cross_layer_residual_cache_enabled": bool(
                self._prefix_vlm_cross_layer_residual_cache_enabled()
            ),
            "prefix_vlm_cache_cross_layer_residual_reused": reused,
            "prefix_vlm_cache_cross_layer_residual_total": total,
            "prefix_vlm_cache_cross_layer_residual_reused_fraction": float(reused / max(1, total)),
            "prefix_vlm_cache_cross_layer_residual_mask": str(self._prefix_vlm_last_cross_layer_residual_mask),
        }

    def _prefix_vlm_cache_bank_info(self, selected: bool, selected_cka: float) -> dict[str, Any]:
        return {
            "prefix_vlm_cache_bank_enabled": bool(self._prefix_vlm_cache_bank_enabled()),
            "prefix_vlm_cache_bank_size": int(len(self._prefix_vlm_cache_bank)),
            "prefix_vlm_cache_bank_selected": bool(selected),
            "prefix_vlm_cache_bank_selected_cka": float(selected_cka),
        }

    def _pytorch_intra_action_cache_info(
        self,
        num_steps: int,
        *,
        fn_blocks: int | None = None,
        bn_blocks: int | None = None,
        schedule: str | None = None,
        refresh_interval: int | None = None,
    ) -> dict[str, Any]:
        depth = self._prefix_vlm_action_depth()
        schedule = self._prefix_vlm_intra_action_schedule(schedule)
        refresh_interval = max(
            1,
            int(
                self._prefix_vlm_cache_config.intra_action_cache_refresh_interval
                if refresh_interval is None
                else refresh_interval
            ),
        )
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
        cached_block_count = min(fn_blocks + bn_blocks, depth)
        refresh_steps = len(self._prefix_vlm_intra_action_refresh_steps(schedule, num_steps, refresh_interval))
        cached_steps = num_steps - refresh_steps
        computed_blocks_total = refresh_steps * depth + cached_steps * cached_block_count
        full_blocks = depth * num_steps
        return {
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
