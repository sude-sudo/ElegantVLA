"""Configuration and upstream forwarding checks without model dependencies."""

import dataclasses
import importlib.util
from pathlib import Path
import types
import unittest
from unittest.mock import patch

from elegantvla.openpi.core.config import PrefixVLMCacheConfig, TemporalSkipConfig


class OpenPIConfigTest(unittest.TestCase):
    def test_release_defaults_and_validation(self):
        config = TemporalSkipConfig(enabled=True)
        self.assertEqual((config.first_layers, config.last_layers), (1, 1))
        self.assertEqual(config.max_reuse_steps, 5)
        self.assertEqual(TemporalSkipConfig(**dataclasses.asdict(config)), config)
        for settings in (
            {"first_layers": 0},
            {"last_layers": 0},
            {"speed_smoothing_window": 0},
            {"max_reuse_steps": -1},
            {"gate_threshold": 1.01},
            {"free1_similarity_threshold": 0.2},
            {"max_episode_steps": 0},
            {"gripper_index": -1},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                TemporalSkipConfig(**settings)

    def test_saved_settings_round_trip(self):
        settings = {
            "enabled": True,
            "skip_v4_no_image_budget_after_hit": 2,
            "skip_v4_no_image_disable_prompt_substrings": "drawer",
            "threshold_scheduler_enabled": True,
            "threshold_scheduler_free1_cka_threshold": 0.997,
        }
        config = PrefixVLMCacheConfig(**settings)
        restored = PrefixVLMCacheConfig(**dataclasses.asdict(config))
        self.assertEqual(restored, config)
        for name, value in settings.items():
            self.assertEqual(getattr(restored, name), value)

    def test_config_remains_immutable(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            PrefixVLMCacheConfig().enabled = True

    def test_upstream_api_forwards_the_same_classes(self):
        implementation = types.ModuleType("elegantvla.openpi.core.policy")
        for name in ("BasePolicy", "Policy", "PolicyRecorder"):
            setattr(implementation, name, type(name, (), {}))
        path = Path(__file__).resolve().parents[1] / "integration/src/openpi/policies/policy.py"
        spec = importlib.util.spec_from_file_location("_test_openpi_bridge", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {implementation.__name__: implementation}):
            spec.loader.exec_module(module)
        self.assertIs(module.PrefixVLMCacheConfig, PrefixVLMCacheConfig)
        self.assertIs(module.TemporalSkipConfig, TemporalSkipConfig)
        for name in ("BasePolicy", "Policy", "PolicyRecorder"):
            self.assertIs(getattr(module, name), getattr(implementation, name))


if __name__ == "__main__":
    unittest.main()
