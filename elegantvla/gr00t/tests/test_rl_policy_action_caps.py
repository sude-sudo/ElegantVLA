import unittest

import numpy as np

from elegantvla.gr00t.core.rl_policy import RLSchedulePolicy


class ActionCapTest(unittest.TestCase):
    def test_action_caps_mask_only_more_aggressive_levels(self) -> None:
        policy = RLSchedulePolicy(
            model=None,
            max_backbone_action_level=2,
            max_action_head_action_level=1,
        )

        mask = policy._build_action_masks(None)

        np.testing.assert_array_equal(
            mask,
            np.asarray([True, True, True, False, False, True, True, False]),
        )

    def test_action_caps_preserve_runtime_backbone_mask(self) -> None:
        policy = RLSchedulePolicy(
            model=None,
            max_backbone_action_level=1,
            max_action_head_action_level=0,
        )

        mask = policy._build_action_masks(
            np.asarray([True, False, True, True, True], dtype=bool)
        )

        np.testing.assert_array_equal(
            mask,
            np.asarray([True, False, False, False, False, True, False, False]),
        )

    def test_invalid_action_caps_are_rejected(self) -> None:
        cases = [
            ("max_backbone_action_level", -1),
            ("max_backbone_action_level", 5),
            ("max_action_head_action_level", -1),
            ("max_action_head_action_level", 3),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                policy = RLSchedulePolicy(model=None, **{field: value})
                with self.assertRaises(ValueError):
                    policy._build_action_masks(None)


if __name__ == "__main__":
    unittest.main()
