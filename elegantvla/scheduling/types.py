"""Runtime-facing schedule types for temporal skip."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np


class ActionHeadAction(IntEnum):
    """Discrete action-head caching actions for RL scheduling."""

    REFRESH = 0
    REUSE_MIDDLE = 1
    REUSE_ALL_BUT_FIRST = 2

    @classmethod
    def from_any(cls, value: int | "ActionHeadAction") -> "ActionHeadAction":
        if isinstance(value, cls):
            return value
        return cls(int(value))

    @property
    def reuse_strategy(self) -> str | None:
        if self == ActionHeadAction.REFRESH:
            return None
        if self == ActionHeadAction.REUSE_MIDDLE:
            return "inner_middle_steps"
        return "inner_all_but_first_step"


class BackboneAction(IntEnum):
    """Discrete backbone-cache actions available at each scheduler decision point."""

    REFRESH = 0
    REUSE = 1
    # Current step still executes as reuse; the suffix only sets future no-image budget.
    REUSE_THEN_NO_IMAGE_1 = 2
    REUSE_THEN_NO_IMAGE_2 = 3
    REUSE_THEN_NO_IMAGE_3 = 4

    @classmethod
    def from_any(cls, value: int | "BackboneAction") -> "BackboneAction":
        if isinstance(value, cls):
            return value
        return cls(int(value))

    @classmethod
    def from_budget(cls, budget: int) -> "BackboneAction":
        clamped = max(int(budget), 0)
        if clamped <= 0:
            return cls.REUSE
        if clamped == 1:
            return cls.REUSE_THEN_NO_IMAGE_1
        if clamped == 2:
            return cls.REUSE_THEN_NO_IMAGE_2
        return cls.REUSE_THEN_NO_IMAGE_3

    @property
    def budget(self) -> int:
        if self in (BackboneAction.REFRESH, BackboneAction.REUSE):
            return 0
        if self == BackboneAction.REUSE_THEN_NO_IMAGE_1:
            return 1
        if self == BackboneAction.REUSE_THEN_NO_IMAGE_2:
            return 2
        return 3


@dataclass(frozen=True)
class ScheduleAction:
    """Full scheduler action with backbone and action-head decisions."""

    backbone_action: BackboneAction
    action_head_action: ActionHeadAction

    @classmethod
    def from_any(
        cls,
        value: "ScheduleAction" | tuple[int, int] | list[int] | np.ndarray,
    ) -> "ScheduleAction":
        if isinstance(value, cls):
            return value
        if isinstance(value, np.ndarray):
            flat_value = value.reshape(-1).tolist()
        elif isinstance(value, (tuple, list)):
            flat_value = list(value)
        else:
            raise TypeError(f"Unsupported schedule action value: {type(value)!r}")
        if len(flat_value) != 2:
            raise ValueError(
                f"Schedule action must contain exactly 2 elements, got {len(flat_value)}."
            )
        return cls(
            backbone_action=BackboneAction.from_any(flat_value[0]),
            action_head_action=ActionHeadAction.from_any(flat_value[1]),
        )

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [int(self.backbone_action), int(self.action_head_action)],
            dtype=np.int64,
        )

    def as_tuple(self) -> tuple[int, int]:
        return int(self.backbone_action), int(self.action_head_action)


@dataclass
class ScheduleState:
    """Wrapper-owned state for one scheduler decision point."""

    schedule_obs: dict[str, np.ndarray]
    backbone_action_mask: np.ndarray
    execution_ctx: Any | None = None


@dataclass(frozen=True)
class ScheduleObservation:
    """Observation for the RL scheduler."""

    feature_similarity: float
    gripper_speed: float
    eef_translation_speed: float
    eef_rotation_speed: float
    step_progress: float = 0.0

    def as_gym_observation(self) -> dict[str, np.ndarray]:
        return {
            "feature_similarity": np.asarray([self.feature_similarity], dtype=np.float32),
            "gripper_speed": np.asarray([self.gripper_speed], dtype=np.float32),
            "eef_translation_speed": np.asarray([self.eef_translation_speed], dtype=np.float32),
            "eef_rotation_speed": np.asarray([self.eef_rotation_speed], dtype=np.float32),
            "step_progress": np.asarray(
                [np.clip(self.step_progress, 0.0, 1.0)],
                dtype=np.float32,
            ),
        }

    @staticmethod
    def space():
        from gymnasium import spaces

        return spaces.Dict({
            "feature_similarity": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "gripper_speed": spaces.Box(0.0, np.inf, shape=(1,), dtype=np.float32),
            "eef_translation_speed": spaces.Box(0.0, np.inf, shape=(1,), dtype=np.float32),
            "eef_rotation_speed": spaces.Box(0.0, np.inf, shape=(1,), dtype=np.float32),
            "step_progress": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        })
