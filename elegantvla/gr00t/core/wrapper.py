"""Policy wrapper that owns temporal skip scheduling and runtime control."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from gr00t.policy.policy import PolicyWrapper

from elegantvla.gr00t.core.action_head import _enable_compiled_action_head_cache
from elegantvla.gr00t.core.adapter import Gr00tPolicyTemporalSkipAdapter
from elegantvla.gr00t.core.types import (
    ActionHeadAction,
    BackboneAction,
    ScheduleAction,
    ScheduleObservation,
    ScheduleState,
)
from elegantvla.scheduling.controller import SkipController


# ---------------------------------------------------------------------------
# Public wrapper: start reading here
# ---------------------------------------------------------------------------
class TemporalSkipPolicyWrapper(PolicyWrapper):
    """Wrap a clean policy and add temporal skip orchestration."""

    def __init__(
        self,
        policy: Any,
        *,
        adapter: Gr00tPolicyTemporalSkipAdapter | None = None,
        enabled: bool = True,
        max_reuse_steps: int | None = 5,
        speed_smoothing_window: int = 4,
        compile_action_head: bool = True,
        compile_mode: str = "max-autotune",
        static_inference_cache: bool = True,
        strict: bool | None = None,
    ):
        wrapper_strict = policy.strict if strict is None and hasattr(policy, "strict") else True
        if strict is not None:
            wrapper_strict = strict
        super().__init__(policy, strict=wrapper_strict)
        self.adapter = adapter or Gr00tPolicyTemporalSkipAdapter(policy)
        self.ctrl = SkipController(
            enabled=enabled,
            max_reuse_steps=max_reuse_steps,
            speed_smoothing_window=speed_smoothing_window,
        )
        self._last_feature_similarity = 0.0
        self.compile_action_head = bool(compile_action_head)
        self.compile_mode = str(compile_mode)
        self.static_inference_cache = bool(static_inference_cache)
        self.adapter.configure_static_inference_cache(self.static_inference_cache)
        action_dit_graph_enabled = bool(
            getattr(self.policy, "action_dit_cuda_graph", False)
        )
        if action_dit_graph_enabled and self.compile_action_head:
            # The adapter dispatches REFRESH to the CUDA Graph and inner
            # denoising reuse to the compiled residual-cache path.
            print(
                "[TemporalSkip] Action DiT CUDA Graph 已启用，"
                "安装 refresh/reuse 双路径"
            )
        if (
            self.compile_action_head
            and getattr(torch, "compile", None) is not None
        ):
            action_head = self.policy.model.action_head
            if (
                getattr(action_head, "model", None) is not None
                and not getattr(action_head, "_compiledcached_enabled", False)
            ):
                _enable_compiled_action_head_cache(
                    self.policy,
                    compile_mode=self.compile_mode,
                )
                if torch.cuda.is_available():
                    torch.backends.cudnn.benchmark = True

    def check_observation(self, observation: dict[str, Any]) -> None:
        self.policy.check_observation(observation)

    def check_action(self, action: dict[str, Any]) -> None:
        self.policy.check_action(action)

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        self.ctrl.reset()
        self._last_feature_similarity = 0.0
        info = self.policy.reset(options)
        return dict(info) if isinstance(info, dict) else {}

    # ------------------------------------------------------------------
    # Env-facing schedule API
    # ------------------------------------------------------------------
    def build_schedule_state(
        self,
        observation: dict[str, Any],
    ) -> ScheduleState:
        schedule_observation, execution_ctx = self._prepare_schedule_observation(observation)
        return ScheduleState(
            schedule_obs=schedule_observation.as_gym_observation(),
            backbone_action_mask=self.ctrl.build_backbone_action_mask().astype(
                np.float32,
                copy=False,
            ),
            execution_ctx=execution_ctx,
        )

    def execute_schedule_action(
        self,
        observation: dict[str, Any],
        schedule_action: ScheduleAction | tuple[int, int] | list[int] | np.ndarray,
        schedule_state: ScheduleState,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._execute_schedule_action_with_ctx(
            observation,
            schedule_action,
            execution_ctx=schedule_state.execution_ctx,
        )

    # ------------------------------------------------------------------
    # BasePolicy-facing compatibility API
    # ------------------------------------------------------------------
    def prepare_schedule_step(
        self,
        observation: dict[str, Any],
    ) -> tuple[ScheduleObservation, dict[str, Any] | None]:
        """Build schedule observation plus optional execution context for BasePolicy callers."""
        return self._prepare_schedule_observation(observation)

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Legacy BasePolicy bridge used by callers that still drive temporal skip via get_action()."""
        options = dict(options or {})
        ts_opts = dict(options.get("temporal_skip") or {})
        enabled = bool(ts_opts.get("enabled", self.ctrl.enabled))
        raw_backbone_action = options.get("backbone_action")
        action_head_action = ts_opts.get("action_head_action")

        if not enabled:
            action, info = self.policy._get_action(observation, options)
            self.ctrl.record_motion(observation)
            return action, dict(info or {})

        if raw_backbone_action is None or action_head_action is None:
            raise ValueError(
                "Temporal-skip legacy _get_action requires `backbone_action` and "
                "`temporal_skip.action_head_action`. New code should use "
                "`build_schedule_state()` and `execute_schedule_action()`."
            )

        schedule_action = ScheduleAction(
            backbone_action=BackboneAction.from_any(raw_backbone_action),
            action_head_action=ActionHeadAction.from_any(action_head_action),
        )
        return self._execute_schedule_action_with_ctx(
            observation,
            schedule_action,
            execution_ctx=ts_opts.get("_precomputed_step"),
        )

    # ------------------------------------------------------------------
    # Shared internals
    # ------------------------------------------------------------------
    def _prepare_schedule_observation(
        self,
        observation: dict[str, Any],
    ) -> tuple[ScheduleObservation, dict[str, Any] | None]:
        execution_ctx = None
        if self.ctrl._no_image_budget > 0:
            schedule_observation = self.ctrl.build_schedule_observation(
                observation,
                feature_similarity=self._last_feature_similarity,
            )
        else:
            execution_ctx = self._prepare_first_block_precompute(observation)
            schedule_observation = self.ctrl.build_schedule_observation(
                observation,
                current_gate_feature=execution_ctx["probe_output"]["hidden_states"],
            )
        self._last_feature_similarity = float(schedule_observation.feature_similarity)
        return schedule_observation, execution_ctx

    def _execute_schedule_action_with_ctx(
        self,
        observation: dict[str, Any],
        schedule_action: ScheduleAction | tuple[int, int] | list[int] | np.ndarray,
        *,
        execution_ctx: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        resolved_schedule_action = ScheduleAction.from_any(schedule_action)
        return self._run_skip_step(
            observation,
            backbone_action=resolved_schedule_action.backbone_action,
            action_head_action=resolved_schedule_action.action_head_action,
            precomputed=execution_ctx,
        )

    def _run_skip_step(
        self,
        observation: dict[str, Any],
        *,
        backbone_action: BackboneAction,
        action_head_action: ActionHeadAction,
        precomputed: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        resolved_backbone_action = self.ctrl.resolve_backbone_action(backbone_action)
        use_backbone_cache = resolved_backbone_action != BackboneAction.REFRESH
        image_free_skip = use_backbone_cache and self.ctrl._no_image_budget > 0
        action_head_reuse_strategy = action_head_action.reuse_strategy
        effective_use_backbone_cache = use_backbone_cache
        used_middle_cache = False
        if precomputed is None:
            if image_free_skip:
                prepared = self.adapter.prepare_inputs(observation, action_only=True)
                probe_output = None
                action_input = None
            else:
                precomputed = self._prepare_first_block_precompute(observation)
                prepared = precomputed["prepared"]
                probe_output = precomputed["probe_output"]
                action_input = precomputed["action_input"]
        else:
            prepared = precomputed["prepared"]
            probe_output = precomputed["probe_output"]
            action_input = precomputed["action_input"]

        with torch.inference_mode():
            if use_backbone_cache:
                if image_free_skip or probe_output is None:
                    backbone_cache = self.ctrl._backbone_cache
                    backbone_output = None
                else:
                    backbone_output, refreshed_middle_cache, used_middle_cache = (
                        self.adapter.continue_backbone_from_middle_cache(
                            probe_output,
                            self.ctrl._backbone_middle_cache,
                        )
                    )
                    if refreshed_middle_cache is not None:
                        self.ctrl._backbone_middle_cache = refreshed_middle_cache
                        effective_use_backbone_cache = False
                    backbone_cache = None
            else:
                if self.static_inference_cache:
                    backbone_output, refreshed_middle_cache = (
                        self.adapter.continue_backbone_from_probe_with_middle_cache(probe_output)
                    )
                    self.ctrl._backbone_middle_cache = refreshed_middle_cache
                else:
                    backbone_output = self.adapter.continue_backbone_from_probe(probe_output)
                    self.ctrl._backbone_middle_cache = None
                backbone_cache = None

            model_pred, cache_to_use = self.adapter.run_action_head(
                backbone_cache=backbone_cache,
                backbone_output=backbone_output,
                action_input=action_input,
                collated_inputs=prepared["collated_inputs"] if image_free_skip else None,
                reuse_strategy=action_head_reuse_strategy,
            )
            if (not effective_use_backbone_cache) or (use_backbone_cache and not image_free_skip):
                self.ctrl._backbone_cache = cache_to_use

        action = self.adapter.decode_action(model_pred, prepared["state"])
        self.ctrl.record_motion(observation)
        gate_feature = None if probe_output is None else probe_output["hidden_states"]
        step_info = self.ctrl.record_step(
            use_backbone_cache=effective_use_backbone_cache,
            image_free_skip=image_free_skip,
            gate_feature=gate_feature,
            backbone_action=(
                resolved_backbone_action
                if effective_use_backbone_cache
                else BackboneAction.REFRESH
            ),
        )
        step_info.update(
            {
                "use_backbone_cache": bool(effective_use_backbone_cache),
                "image_free_skip": bool(image_free_skip),
                "backbone_middle_cache_used": bool(used_middle_cache),
            }
        )
        return action, step_info

    def _prepare_first_block_precompute(
        self,
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the shared first-block probe payload used by both scheduling and execution."""
        prepared = self.adapter.prepare_inputs(observation)
        with torch.inference_mode():
            probe_output, action_input = self.adapter.get_first_block_gate_probe(
                prepared["collated_inputs"]
            )
        return {
            "prepared": prepared,
            "probe_output": probe_output,
            "action_input": action_input,
        }
