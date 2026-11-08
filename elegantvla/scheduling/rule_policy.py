"""Rule-based schedule decisions that share the RL observation/action interface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from elegantvla.scheduling.types import ActionHeadAction, BackboneAction, ScheduleAction


@dataclass
class RuleSchedulePolicy:
    """Simple rule policy for scheduler actions."""

    gate_threshold: float = 0.95
    free1_similarity_threshold: float | None = 0.98
    free2_similarity_threshold: float | None = 0.99
    free3_similarity_threshold: float | None = 0.995
    action_head_selection_mode: str = "motion_speed_inner"
    ah_speed_threshold: float = 0.005
    ah_secondary_threshold: float | None = 0.05
    action_head_eef_rotation_speed_weight: float = 1.0
    action_head_gripper_speed_weight: float = 1.0

    def select_action(
        self,
        observation: dict[str, np.ndarray],
        backbone_action_mask: np.ndarray | None = None,
    ) -> ScheduleAction:
        mode = "motion_speed_inner" if self.action_head_selection_mode is None else str(
            self.action_head_selection_mode
        ).strip().lower()
        if mode not in {
            "gripper_speed",
            "gripper_speed_inner",
            "motion_speed",
            "motion_speed_inner",
        }:
            mode = "motion_speed_inner"

        if backbone_action_mask is None:
            mask = np.ones(len(BackboneAction), dtype=bool)
        else:
            mask = np.asarray(backbone_action_mask, dtype=np.float32).reshape(-1)
            if mask.size != len(BackboneAction):
                mask = np.ones(len(BackboneAction), dtype=bool)
            else:
                mask = mask > 0.5

        feature_similarity = float(
            np.asarray(observation.get("feature_similarity", 0.0), dtype=np.float32).reshape(-1)[0]
        )
        backbone_action = BackboneAction.REFRESH
        for action, threshold in (
            (BackboneAction.REUSE_THEN_NO_IMAGE_3, self.free3_similarity_threshold),
            (BackboneAction.REUSE_THEN_NO_IMAGE_2, self.free2_similarity_threshold),
            (BackboneAction.REUSE_THEN_NO_IMAGE_1, self.free1_similarity_threshold),
            (BackboneAction.REUSE, self.gate_threshold),
            (BackboneAction.REFRESH, -np.inf),
        ):
            if threshold is None or not bool(mask[int(action)]):
                continue
            if feature_similarity >= float(threshold):
                backbone_action = action
                break
        if not bool(mask[int(backbone_action)]):
            for action in BackboneAction:
                if bool(mask[int(action)]):
                    backbone_action = action
                    break

        gripper_speed = float(
            np.asarray(observation.get("gripper_speed", 0.0), dtype=np.float32).reshape(-1)[0]
        )
        eef_trans = float(
            np.asarray(observation.get("eef_translation_speed", 0.0), dtype=np.float32).reshape(-1)[0]
        )
        eef_rot = float(
            np.asarray(observation.get("eef_rotation_speed", 0.0), dtype=np.float32).reshape(-1)[0]
        )

        if mode.startswith("gripper_speed"):
            signal = gripper_speed if np.isfinite(gripper_speed) else None
        else:
            candidates: list[float] = []
            if np.isfinite(eef_trans):
                candidates.append(float(eef_trans))
            if np.isfinite(eef_rot):
                candidates.append(float(self.action_head_eef_rotation_speed_weight) * float(eef_rot))
            if np.isfinite(gripper_speed):
                candidates.append(
                    float(self.action_head_gripper_speed_weight) * float(gripper_speed)
                )
            signal = max(candidates) if candidates else None

        if signal is None or float(signal) < float(self.ah_speed_threshold):
            action_head_action = ActionHeadAction.REFRESH
        elif mode.endswith("_inner"):
            action_head_action = (
                ActionHeadAction.REUSE_ALL_BUT_FIRST
                if self.ah_secondary_threshold is not None
                and float(signal) >= float(self.ah_secondary_threshold)
                else ActionHeadAction.REUSE_MIDDLE
            )
        else:
            action_head_action = ActionHeadAction.REUSE_ALL_BUT_FIRST

        return ScheduleAction(
            backbone_action=backbone_action,
            action_head_action=action_head_action,
        )
