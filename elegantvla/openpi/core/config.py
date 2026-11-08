"""Small release configuration; historical settings are kept separately."""

from dataclasses import dataclass

from elegantvla.openpi.legacy.config import PrefixVLMCacheConfig


@dataclass(frozen=True)
class TemporalSkipConfig:
    enabled: bool = False
    first_layers: int = 1
    last_layers: int = 1
    max_reuse_steps: int | None = 5
    speed_smoothing_window: int = 4
    gate_threshold: float = 0.95
    free1_similarity_threshold: float | None = 0.98
    free2_similarity_threshold: float | None = 0.99
    free3_similarity_threshold: float | None = 0.995
    action_head_selection_mode: str = "motion_speed_inner"
    ah_speed_threshold: float = 0.005
    ah_secondary_threshold: float | None = 0.05
    action_head_eef_rotation_speed_weight: float = 1.0
    action_head_gripper_speed_weight: float = 1.0
    # Progress denominator in policy calls, not simulator physics steps.
    max_episode_steps: int | None = None
    # Default raw LIBERO state: xyz, axis-angle, two gripper positions.
    motion_state_key: str | None = "observation/state"
    translation_indices: tuple[int, int, int] = (0, 1, 2)
    rotation_indices: tuple[int, int, int] = (3, 4, 5)
    gripper_index: int = 6

    def __post_init__(self):
        if self.first_layers < 1 or self.last_layers < 1:
            raise ValueError("Both boundary layer counts must be positive.")
        if self.max_reuse_steps is not None and self.max_reuse_steps < 0:
            raise ValueError("max_reuse_steps must be nonnegative or None.")
        if self.speed_smoothing_window < 1:
            raise ValueError("speed_smoothing_window must be positive.")
        if self.max_episode_steps is not None and self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive or None.")
        if len(self.translation_indices) != 3 or len(self.rotation_indices) != 3:
            raise ValueError("Translation and rotation need three state indices each.")
        if min((*self.translation_indices, *self.rotation_indices, self.gripper_index)) < 0:
            raise ValueError("Motion state indices must be nonnegative.")
        thresholds = [self.gate_threshold] + [
            value
            for value in (
                self.free1_similarity_threshold,
                self.free2_similarity_threshold,
                self.free3_similarity_threshold,
            )
            if value is not None
        ]
        if any(not 0.0 <= value <= 1.0 for value in thresholds) or thresholds != sorted(thresholds):
            raise ValueError("CKA thresholds must be ordered within [0, 1].")


__all__ = ["TemporalSkipConfig", "PrefixVLMCacheConfig"]
