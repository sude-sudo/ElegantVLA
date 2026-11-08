"""Episode-local cache lifecycle, CKA and robot-motion observations."""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np
import torch

from elegantvla.scheduling.types import BackboneAction, ScheduleObservation


class SkipController:
    def __init__(
        self,
        *,
        enabled: bool = True,
        max_reuse_steps: int | None = 5,
        speed_smoothing_window: int = 4,
    ):
        self.enabled = bool(enabled)
        self.max_reuse_steps = max_reuse_steps
        self.speed_smoothing_window = max(int(speed_smoothing_window), 1)
        self._backbone_cache: Any | None = None
        self._backbone_middle_cache: Any | None = None
        self._consecutive_reuse_steps: int = 0
        self._anchor_matrix: torch.Tensor | None = None
        self._no_image_budget: int = 0
        window = max(2, self.speed_smoothing_window + 1)
        self._gripper_motion_history: deque[float] = deque(maxlen=window)
        self._eef_position_history: deque[np.ndarray] = deque(maxlen=window)
        self._eef_orientation_history: deque[np.ndarray] = deque(maxlen=window)

    def reset(self) -> None:
        self._backbone_cache = None
        self._backbone_middle_cache = None
        self._consecutive_reuse_steps = 0
        self._anchor_matrix = None
        self._no_image_budget = 0
        self._gripper_motion_history.clear()
        self._eef_position_history.clear()
        self._eef_orientation_history.clear()

    def record_motion(self, observation: dict[str, Any]) -> None:
        gripper_history, position_history, orientation_history = self._preview_motion_histories(
            observation
        )
        self._gripper_motion_history = deque(
            gripper_history, maxlen=self._gripper_motion_history.maxlen
        )
        self._eef_position_history = deque(
            position_history, maxlen=self._eef_position_history.maxlen
        )
        self._eef_orientation_history = deque(
            orientation_history, maxlen=self._eef_orientation_history.maxlen
        )

    def build_backbone_action_mask(self) -> np.ndarray:
        max_reuse_steps = self.max_reuse_steps
        can_reuse = (
            self.enabled
            and self._backbone_cache is not None
            and self._backbone_middle_cache is not None
            and (max_reuse_steps is None or self._consecutive_reuse_steps < int(max_reuse_steps))
        )
        if self._no_image_budget > 0:
            forced = np.zeros(len(BackboneAction), dtype=bool)
            if can_reuse:
                forced[int(BackboneAction.REUSE)] = True
            else:
                forced[int(BackboneAction.REFRESH)] = True
            return forced
        if self._anchor_matrix is None or not can_reuse:
            return np.asarray([True, False, False, False, False], dtype=bool)
        return np.ones(len(BackboneAction), dtype=bool)

    def record_step(
        self,
        *,
        use_backbone_cache: bool,
        image_free_skip: bool,
        gate_feature: torch.Tensor | None,
        backbone_action: BackboneAction,
    ) -> dict[str, Any]:
        if use_backbone_cache:
            self._consecutive_reuse_steps += 1
        else:
            self._consecutive_reuse_steps = 0
        if not use_backbone_cache:
            self._anchor_matrix = self._reshape_feature_matrix(gate_feature)
        if image_free_skip:
            self._no_image_budget = max(self._no_image_budget - 1, 0)
        else:
            self._no_image_budget = backbone_action.budget
        return {"backbone_action": int(backbone_action)}

    def resolve_backbone_action(
        self,
        backbone_action: int | BackboneAction,
    ) -> BackboneAction:
        resolved_action = BackboneAction.from_any(backbone_action)
        mask = self.build_backbone_action_mask()
        if bool(mask[int(resolved_action)]):
            return resolved_action
        valid_actions = np.flatnonzero(mask)
        return BackboneAction(int(valid_actions[0])) if valid_actions.size else BackboneAction.REFRESH

    def build_schedule_observation(
        self,
        observation: dict[str, Any],
        current_gate_feature: torch.Tensor | None = None,
        feature_similarity: float = 0.0,
    ) -> ScheduleObservation:
        gripper_history, position_history, orientation_history = self._preview_motion_histories(
            observation
        )
        gripper = self._compute_gripper_speed(
            gripper_history, smoothing_window=self.speed_smoothing_window
        ) or 0.0
        trans = self._compute_eef_translation_speed(
            position_history, smoothing_window=self.speed_smoothing_window
        ) or 0.0
        rot = self._compute_eef_rotation_speed(
            orientation_history, smoothing_window=self.speed_smoothing_window
        ) or 0.0
        if current_gate_feature is not None:
            current_matrix = self._reshape_feature_matrix(current_gate_feature)
            feature_similarity = (
                0.0
                if self._anchor_matrix is None
                else self._feature_cka_similarity(self._anchor_matrix, current_matrix)
            )
        return ScheduleObservation(
            feature_similarity=float(feature_similarity),
            gripper_speed=gripper,
            eef_translation_speed=trans,
            eef_rotation_speed=rot,
        )

    def _preview_motion_histories(
        self,
        observation: dict[str, Any],
    ) -> tuple[list[float], list[np.ndarray], list[np.ndarray]]:
        g_hist = list(self._gripper_motion_history)
        p_hist = list(self._eef_position_history)
        o_hist = list(self._eef_orientation_history)
        gripper = self._extract_gripper_position(observation)
        if gripper is not None and math.isfinite(float(gripper)):
            g_hist.append(float(gripper))
        else:
            g_hist = []
        pos, orient = self._extract_eef_pose(observation)
        resolved_pos = None
        if pos is not None:
            arr = np.asarray(pos, dtype=np.float32).reshape(-1)
            if arr.size == 3 and np.all(np.isfinite(arr)):
                resolved_pos = arr.copy()
        if resolved_pos is None:
            p_hist = []
        else:
            p_hist.append(resolved_pos)
        resolved_orient = None
        if orient is not None:
            arr = np.asarray(orient, dtype=np.float32).reshape(-1)
            if arr.size in (3, 4) and np.all(np.isfinite(arr)):
                resolved_orient = arr.copy()
        if resolved_orient is None:
            o_hist = []
        else:
            o_hist.append(resolved_orient)
        return g_hist, p_hist, o_hist

    @staticmethod
    def _lookup_state_value(observation: dict[str, Any], key: str) -> Any | None:
        nested_state = observation.get("state")
        if isinstance(nested_state, dict) and key in nested_state:
            return nested_state[key]
        flat_key = f"state.{key}"
        if flat_key in observation:
            return observation[flat_key]
        return observation.get(key)

    @staticmethod
    def _as_float_array(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        try:
            array = np.asarray(value, dtype=np.float32).reshape(-1)
        except Exception:
            return None
        if array.size <= 0 or not np.all(np.isfinite(array)):
            return None
        return array

    @classmethod
    def _extract_gripper_position(cls, observation: dict[str, Any] | None) -> float | None:
        if not isinstance(observation, dict):
            return None

        nested_state = observation.get("state")
        if isinstance(nested_state, dict):
            for key in ("gripper", "left_gripper", "right_gripper"):
                arr = cls._as_float_array(nested_state.get(key))
                scalar = None if arr is None else float(arr[0])
                if scalar is not None:
                    return scalar
            for key, value in nested_state.items():
                if "gripper" in str(key).lower():
                    arr = cls._as_float_array(value)
                    scalar = None if arr is None else float(arr[0])
                    if scalar is not None:
                        return scalar

        for key in ("gripper", "left_gripper", "right_gripper"):
            arr = cls._as_float_array(cls._lookup_state_value(observation, key))
            scalar = None if arr is None else float(arr[0])
            if scalar is not None:
                return scalar

        for key, value in observation.items():
            key_str = str(key).lower()
            if key_str.startswith("state.") and "gripper" in key_str:
                arr = cls._as_float_array(value)
                scalar = None if arr is None else float(arr[0])
                if scalar is not None:
                    return scalar
        return None

    @classmethod
    def _extract_eef_pose(
        cls,
        observation: dict[str, Any] | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not isinstance(observation, dict):
            return None, None

        position_scalars = []
        for key in ("x", "y", "z"):
            arr = cls._as_float_array(cls._lookup_state_value(observation, key))
            position_scalars.append(None if arr is None else float(arr[0]))
        position = None
        if all(value is not None for value in position_scalars):
            position = np.asarray(position_scalars, dtype=np.float32)

        orientation_scalars = []
        for key in ("rx", "ry", "rz", "rw"):
            arr = cls._as_float_array(cls._lookup_state_value(observation, key))
            orientation_scalars.append(None if arr is None else float(arr[0]))
        if any(value is None for value in orientation_scalars):
            orientation_scalars = []
            for key in ("roll", "pitch", "yaw"):
                arr = cls._as_float_array(cls._lookup_state_value(observation, key))
                orientation_scalars.append(None if arr is None else float(arr[0]))
        orientation = None
        if all(value is not None for value in orientation_scalars):
            orientation = np.asarray(orientation_scalars, dtype=np.float32)

        if position is not None and orientation is not None:
            return position, orientation

        for key in ("robot_eef_pose", "eef_pose", "end_effector_pose"):
            pose = cls._as_float_array(cls._lookup_state_value(observation, key))
            if pose is None:
                continue
            if position is None and pose.size >= 3:
                position = pose[:3]
            if orientation is None:
                if pose.size >= 7:
                    orientation = pose[3:7]
                elif pose.size >= 6:
                    orientation = pose[3:6]
            if position is not None and orientation is not None:
                break

        return position, orientation

    @classmethod
    def _compute_eef_translation_speed(
        cls,
        position_history: list[np.ndarray | None],
        smoothing_window: int = 4,
    ) -> float | None:
        w = int(smoothing_window)
        positions: list[np.ndarray] = []
        for value in position_history:
            array = cls._as_float_array(value)
            if array is None or array.size != 3:
                continue
            positions.append(array)
        if len(positions) < w + 1:
            return None
        return float(np.linalg.norm(positions[-1] - positions[-(w + 1)]) / float(w))

    @classmethod
    def _compute_eef_rotation_speed(
        cls,
        orientation_history: list[np.ndarray | None],
        smoothing_window: int = 4,
    ) -> float | None:
        w = int(smoothing_window)
        orientations: list[np.ndarray] = []
        dim: int | None = None
        for value in orientation_history:
            array = cls._as_float_array(value)
            if array is None or array.size not in (3, 4):
                continue
            if dim is None:
                dim = int(array.size)
            if array.size == dim:
                orientations.append(array)
        if len(orientations) < w + 1:
            return None

        start, end = orientations[-(w + 1)], orientations[-1]
        if dim == 4:
            start_norm = float(np.linalg.norm(start))
            end_norm = float(np.linalg.norm(end))
            if start_norm <= 1e-8 or end_norm <= 1e-8:
                return None
            start = start / start_norm
            end = end / end_norm
            return float((2.0 * np.arccos(np.clip(abs(np.dot(start, end)), -1.0, 1.0))) / float(w))

        return float(np.linalg.norm(end - start) / float(w))

    @staticmethod
    def _compute_gripper_speed(
        gripper_position_history: list[float | None],
        smoothing_window: int = 4,
    ) -> float | None:
        w = int(smoothing_window)
        positions: list[float] = []
        for value in gripper_position_history:
            if value is None:
                continue
            scalar = float(value)
            if np.isfinite(scalar):
                positions.append(scalar)
        if len(positions) < w + 1:
            return None
        return float(abs(positions[-1] - positions[-(w + 1)]) / float(w))

    @staticmethod
    def _reshape_feature_matrix(feature: torch.Tensor | None) -> torch.Tensor | None:
        if feature is None:
            return None
        matrix = feature.detach().float()
        if matrix.ndim == 0:
            return matrix.reshape(1, 1)
        if matrix.ndim == 1:
            return matrix.reshape(1, -1)
        return matrix.reshape(-1, matrix.shape[-1])

    @classmethod
    def _feature_cka_similarity(
        cls,
        anchor_feature: torch.Tensor,
        current_feature: torch.Tensor,
    ) -> float:
        anchor = cls._reshape_feature_matrix(anchor_feature)
        current = cls._reshape_feature_matrix(current_feature)
        if anchor is None or current is None:
            return 0.0

        if anchor.shape != current.shape:
            pooled_anchor = anchor.mean(dim=0).reshape(-1)
            pooled_current = current.mean(dim=0).reshape(-1)
            common_dim = min(pooled_anchor.numel(), pooled_current.numel())
            if common_dim <= 0:
                return 0.0
            pooled_anchor = pooled_anchor[:common_dim]
            pooled_current = pooled_current[:common_dim]
            a_norm = torch.linalg.norm(pooled_anchor)
            c_norm = torch.linalg.norm(pooled_current)
            if a_norm <= 0.0 or c_norm <= 0.0:
                return 0.0
            sim = torch.dot(pooled_anchor, pooled_current) / (a_norm * c_norm)
            return float(sim.clamp(-1.0, 1.0).item())

        anchor_centered = anchor - anchor.mean(dim=0, keepdim=True)
        current_centered = current - current.mean(dim=0, keepdim=True)
        if anchor_centered.shape[0] <= anchor_centered.shape[1]:
            anchor_gram = anchor_centered @ anchor_centered.T
            current_gram = current_centered @ current_centered.T
            hsic = torch.sum(anchor_gram * current_gram)
            denom = torch.linalg.norm(anchor_gram) * torch.linalg.norm(current_gram)
        else:
            cross_cov = anchor_centered.T @ current_centered
            hsic = torch.sum(cross_cov * cross_cov)
            anchor_cov = anchor_centered.T @ anchor_centered
            current_cov = current_centered.T @ current_centered
            denom = torch.linalg.norm(anchor_cov) * torch.linalg.norm(current_cov)
        if denom <= 0.0:
            return 0.0
        return float((hsic / denom).clamp(min=0.0, max=1.0).item())
