"""RL-backed schedule policy wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from elegantvla.scheduling.types import ActionHeadAction, BackboneAction, ScheduleAction


@dataclass
class RLSchedulePolicy:
    """Thin adapter around an SB3-style model."""

    model: Any
    deterministic: bool = True
    max_backbone_action_level: int | None = None
    max_action_head_action_level: int | None = None

    @staticmethod
    def _validate_action_level(
        value: int | None,
        *,
        action_count: int,
        name: str,
    ) -> int | None:
        if value is None:
            return None
        level = int(value)
        if level < 0 or level >= action_count:
            raise ValueError(
                f"{name} must be between 0 and {action_count - 1}, got {level}."
            )
        return level

    def _build_action_masks(
        self,
        backbone_action_mask: np.ndarray | None,
    ) -> np.ndarray:
        if backbone_action_mask is None:
            backbone_mask = np.ones(len(BackboneAction), dtype=bool)
        else:
            backbone_mask = np.asarray(backbone_action_mask, dtype=bool).reshape(-1)
            if backbone_mask.size != len(BackboneAction):
                backbone_mask = np.ones(len(BackboneAction), dtype=bool)

        max_backbone_level = self._validate_action_level(
            self.max_backbone_action_level,
            action_count=len(BackboneAction),
            name="max_backbone_action_level",
        )
        if max_backbone_level is not None:
            backbone_mask = backbone_mask.copy()
            backbone_mask[max_backbone_level + 1 :] = False

        action_head_mask = np.ones(len(ActionHeadAction), dtype=bool)
        max_action_head_level = self._validate_action_level(
            self.max_action_head_action_level,
            action_count=len(ActionHeadAction),
            name="max_action_head_action_level",
        )
        if max_action_head_level is not None:
            action_head_mask[max_action_head_level + 1 :] = False

        return np.concatenate(
            [backbone_mask, action_head_mask],
            axis=0,
        )

    def _predict_action(
        self,
        observation: dict[str, np.ndarray],
        backbone_action_mask: np.ndarray | None = None,
    ) -> tuple[ScheduleAction, np.ndarray, np.ndarray, np.ndarray]:
        action_masks = self._build_action_masks(backbone_action_mask)
        effective_backbone_mask = action_masks[: len(BackboneAction)]
        obs_tensor, _ = self.model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(
                obs_tensor,
                action_masks=action_masks.reshape(1, -1),
            )
            action_tensor = distribution.get_actions(deterministic=self.deterministic)
            backbone_probs = distribution.distributions[0].probs[0].detach().cpu().numpy()
            action_head_probs = distribution.distributions[1].probs[0].detach().cpu().numpy()
            action = action_tensor.detach().cpu().numpy().astype(np.int64).reshape(-1)
        return (
            ScheduleAction(
                backbone_action=BackboneAction.from_any(action[0]),
                action_head_action=ActionHeadAction.from_any(action[1]),
            ),
            effective_backbone_mask,
            backbone_probs.astype(np.float32, copy=False),
            action_head_probs.astype(np.float32, copy=False),
        )

    def select_action(
        self,
        observation: dict[str, np.ndarray],
        backbone_action_mask: np.ndarray | None = None,
    ) -> ScheduleAction:
        action, _, _, _ = self._predict_action(
            observation,
            backbone_action_mask=backbone_action_mask,
        )
        return action
