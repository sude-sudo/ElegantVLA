"""CPU forwards with tiny, real OpenPI Gemma layers; no checkpoint download."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import numpy as np
    import torch
    from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks
    from transformers import GemmaConfig
except ImportError as exc:
    raise unittest.SkipTest(f"Requires the OpenPI PyTorch environment: {exc}") from exc

from elegantvla.openpi.core.action_head import OpenPIActionHead
from elegantvla.openpi.core.adapter import OpenPIPolicyTemporalSkipAdapter
from elegantvla.openpi.core.config import TemporalSkipConfig
from elegantvla.openpi.core.rule_policy import RuleSchedulePolicy
from elegantvla.openpi.core.types import (
    ActionHeadAction,
    BackboneAction,
    ScheduleAction,
)
from elegantvla.openpi.core.wrapper import TemporalSkipPolicyWrapper


def load_gemma():
    # Load the pinned OpenPI replacement without changing site-packages.
    path = Path(__file__).resolve().parents[3] / (
        "third_party/openpi/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py"
    )
    name = "transformers.models.gemma.elegantvla_test_gemma"
    spec = importlib.util.spec_from_file_location(name, path.resolve())
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TinyPair(torch.nn.Module):
    forward = PaliGemmaWithExpertModel.forward

    def __init__(self, gemma, adaptive):
        super().__init__()

        def config(use_adarms):
            return GemmaConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=6,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                use_adarms=use_adarms,
                adarms_cond_dim=16 if use_adarms else None,
                _attn_implementation="eager",
            )

        self.prefix_model = gemma.GemmaModel(config(False)).float()
        self.expert_model = gemma.GemmaModel(config(adaptive)).float()
        self.paligemma = SimpleNamespace(language_model=self.prefix_model)
        self.gemma_expert = SimpleNamespace(model=self.expert_model)


class TinyPi(torch.nn.Module):
    sample_actions = PI0Pytorch.sample_actions
    denoise_step = PI0Pytorch.denoise_step
    _prepare_attention_masks_4d = PI0Pytorch._prepare_attention_masks_4d

    def __init__(self, gemma, adaptive=False):
        super().__init__()
        self.config = SimpleNamespace(action_horizon=3, action_dim=4)
        self.paligemma_with_expert = TinyPair(gemma, adaptive)
        self.action_in_proj = torch.nn.Linear(4, 16)
        self.state_proj = torch.nn.Linear(8, 16)
        self.action_out_proj = torch.nn.Linear(16, 4)
        self.time_proj = torch.nn.Linear(1, 16)
        self.adaptive = adaptive
        self.prefix_calls = []
        self.vision_calls = 0
        self.expert_calls = []
        for index, layer in enumerate(self.paligemma_with_expert.prefix_model.layers):
            layer.register_forward_hook(lambda _m, _a, _o, i=index: self.prefix_calls.append(i))
        for index, layer in enumerate(self.paligemma_with_expert.expert_model.layers):
            layer.register_forward_hook(lambda _m, _a, _o, i=index: self.expert_calls.append(i))

    def _preprocess_observation(self, observation, *, train=False):
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def embed_prefix(self, images, image_masks, tokens, token_masks):
        self.vision_calls += 1
        image = images[0]
        return image.reshape(image.shape[0], -1, 16), token_masks, torch.zeros_like(token_masks)

    def embed_suffix(self, state, actions, timestep):
        cond = self.time_proj(timestep[:, None])
        action_tokens = self.action_in_proj(actions) + cond[:, None]
        tokens = action_tokens if self.adaptive else torch.cat([self.state_proj(state)[:, None], action_tokens], dim=1)
        mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        att = torch.zeros_like(mask)
        att[:, 0] = True
        if not self.adaptive:
            att[:, 1] = True
        return tokens, mask, att, cond if self.adaptive else None

    def sample_noise(self, shape, device):
        return torch.randn(shape, device=device)

    def native_prefix(self, observation):
        images, image_masks, tokens, token_masks, _ = self._preprocess_observation(observation)
        hidden, pad, att = self.embed_prefix(images, image_masks, tokens, token_masks)
        _, kv = self.paligemma_with_expert.forward(
            attention_mask=self._prepare_attention_masks_4d(make_att_2d_masks(pad, att)),
            position_ids=pad.cumsum(dim=1) - 1,
            inputs_embeds=[hidden, None],
            use_cache=True,
        )
        return kv


class TemporalSkipTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gemma = load_gemma()
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)
        self.model = TinyPi(self.gemma).eval()
        self.obs = SimpleNamespace(
            images={"base": torch.randn(1, 5, 16)},
            image_masks={"base": torch.tensor([True])},
            tokenized_prompt=torch.arange(5)[None],
            tokenized_prompt_mask=torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.bool),
            state=torch.randn(1, 8),
        )
        self.noise = torch.randn(1, 3, 4)

    def run_step(self, wrapper, backbone, action_head=ActionHeadAction.REFRESH, num_steps=3):
        state = wrapper.build_schedule_state(self.obs, noise=self.noise, num_steps=num_steps)
        return wrapper.execute_schedule_action(self.obs, ScheduleAction(backbone, action_head), state)

    @torch.inference_mode()
    def test_full_refresh_matches_native_prefix_and_actions(self):
        for adaptive in (False, True):
            self.model = TinyPi(self.gemma, adaptive=adaptive).eval()
            expected_kv = self.model.native_prefix(self.obs)
            expected = self.model.sample_actions("cpu", self.obs, noise=self.noise, num_steps=3)
            adapter = OpenPIPolicyTemporalSkipAdapter(self.model)
            self.model.prefix_calls.clear()
            prepared = adapter.prepare_inputs(self.obs, noise=self.noise, num_steps=3)
            probe = adapter.get_first_block_gate_probe(prepared)
            output, _ = adapter.continue_backbone_from_probe_with_middle_cache(probe)
            self.assertEqual(self.model.prefix_calls, list(range(6)))
            for actual, reference in zip(
                output.kv.key_cache + output.kv.value_cache,
                expected_kv.key_cache + expected_kv.value_cache,
                strict=True,
            ):
                torch.testing.assert_close(actual, reference)
            actual, _ = adapter.run_action_head(output, prepared, ActionHeadAction.REFRESH)
            torch.testing.assert_close(actual, expected)

    @torch.inference_mode()
    def test_boundary_recompute_keeps_anchor_and_middle_kv(self):
        for first, last in ((1, 1), (2, 2)):
            wrapper = TemporalSkipPolicyWrapper(
                self.model, config=TemporalSkipConfig(enabled=True, first_layers=first, last_layers=last)
            )
            self.run_step(wrapper, BackboneAction.REFRESH)
            anchor = wrapper.ctrl._backbone_middle_cache
            old_keys = [key.clone() for key in anchor.kv.key_cache]
            self.obs.images["base"] = self.obs.images["base"] + torch.randn_like(self.obs.images["base"]) * 0.1
            self.model.prefix_calls.clear()
            _, info = self.run_step(wrapper, BackboneAction.REUSE)
            self.assertEqual(self.model.prefix_calls, list(range(first)) + list(range(6 - last, 6)))
            self.assertEqual(info["path"], "boundary")
            self.assertIs(wrapper.ctrl._backbone_middle_cache, anchor)
            output = wrapper.ctrl._backbone_cache
            for index in range(first, 6 - last):
                self.assertIs(output.kv.key_cache[index], anchor.kv.key_cache[index])
            for index in (0, 5):
                self.assertFalse(torch.equal(output.kv.key_cache[index], old_keys[index]))
            for before, after in zip(old_keys, anchor.kv.key_cache, strict=True):
                torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_image_free_budget_and_reuse_cap(self):
        wrapper = TemporalSkipPolicyWrapper(self.model, config=TemporalSkipConfig(enabled=True, max_reuse_steps=3))
        self.run_step(wrapper, BackboneAction.REFRESH)
        self.run_step(wrapper, BackboneAction.REUSE_THEN_NO_IMAGE_3)
        output = wrapper.ctrl._backbone_cache
        vision_calls = self.model.vision_calls
        for _ in range(2):
            self.model.prefix_calls.clear()
            self.obs.state = self.obs.state + 0.01
            _, info = self.run_step(wrapper, BackboneAction.REUSE)
            self.assertEqual(info["path"], "image_free")
            self.assertEqual(self.model.prefix_calls, [])
            self.assertEqual(self.model.vision_calls, vision_calls)
            self.assertIs(wrapper.ctrl._backbone_cache, output)
        _, info = self.run_step(wrapper, BackboneAction.REUSE)
        self.assertEqual(info["path"], "refresh")
        self.assertEqual(info["no_image_budget"], 0)

    @torch.inference_mode()
    def test_bfloat16_prefix_and_cache(self):
        PaliGemmaWithExpertModel.to_bfloat16_for_selected_params(self.model.paligemma_with_expert, "bfloat16")
        expected = self.model.sample_actions("cpu", self.obs, noise=self.noise, num_steps=3)
        wrapper = TemporalSkipPolicyWrapper(self.model)
        actual, _ = self.run_step(wrapper, BackboneAction.REFRESH)
        torch.testing.assert_close(actual, expected)
        _, info = self.run_step(wrapper, BackboneAction.REUSE)
        self.assertEqual(info["path"], "boundary")
        self.assertEqual(wrapper.ctrl._backbone_middle_cache.delta.dtype, torch.bfloat16)
        self.assertTrue(all(key.dtype == torch.bfloat16 for key in wrapper.ctrl._backbone_cache.kv.key_cache))

    def test_reset_prompt_and_mask_invalidate(self):
        wrapper = TemporalSkipPolicyWrapper(self.model)
        for change in ("prompt", "mask", "reset"):
            self.run_step(wrapper, BackboneAction.REFRESH)
            self.run_step(wrapper, BackboneAction.REUSE_THEN_NO_IMAGE_3)
            if change == "prompt":
                self.obs.tokenized_prompt = self.obs.tokenized_prompt + 1
            elif change == "mask":
                self.obs.tokenized_prompt_mask[0, -1] = True
            else:
                wrapper.reset()
            _, info = self.run_step(wrapper, BackboneAction.REUSE)
            self.assertEqual(info["path"], "refresh")

    def test_bad_middle_cache_falls_back_and_state_is_single_use(self):
        wrapper = TemporalSkipPolicyWrapper(self.model)
        self.run_step(wrapper, BackboneAction.REFRESH)
        wrapper.ctrl._backbone_middle_cache.delta = torch.zeros(1, 1, 16)
        state = wrapper.build_schedule_state(self.obs, noise=self.noise)
        action = ScheduleAction(BackboneAction.REUSE, ActionHeadAction.REFRESH)
        _, info = wrapper.execute_schedule_action(self.obs, action, state)
        self.assertEqual(info["path"], "refresh")
        with self.assertRaises(ValueError):
            wrapper.execute_schedule_action(self.obs, action, state)

    def test_failed_action_does_not_commit_cache(self):
        wrapper = TemporalSkipPolicyWrapper(self.model)
        self.run_step(wrapper, BackboneAction.REFRESH)
        previous = wrapper.ctrl._backbone_cache
        with patch.object(wrapper.adapter, "run_action_head", side_effect=RuntimeError("injected")):
            with self.assertRaises(RuntimeError):
                self.run_step(wrapper, BackboneAction.REUSE)
        self.assertIs(wrapper.ctrl._backbone_cache, previous)
        self.assertEqual(wrapper.ctrl._consecutive_reuse_steps, 0)

    @torch.inference_mode()
    def test_action_head_modes_and_current_state(self):
        for adaptive in (False, True):
            model = TinyPi(self.gemma, adaptive=adaptive).eval()
            adapter = OpenPIPolicyTemporalSkipAdapter(model)
            prepared = adapter.prepare_inputs(self.obs, noise=self.noise, num_steps=1)
            output, _ = adapter.continue_backbone_from_probe_with_middle_cache(
                adapter.get_first_block_gate_probe(prepared)
            )
            expected, _ = adapter.run_action_head(output, prepared, ActionHeadAction.REFRESH)
            for mode in ActionHeadAction:
                actual, _ = adapter.run_action_head(output, prepared, mode)
                torch.testing.assert_close(actual, expected)
            prepared["num_steps"] = 4
            for mode, count in (
                (ActionHeadAction.REFRESH, 4),
                (ActionHeadAction.REUSE_MIDDLE, 2),
                (ActionHeadAction.REUSE_ALL_BUT_FIRST, 1),
            ):
                model.expert_calls.clear()
                actual, info = adapter.run_action_head(output, prepared, mode)
                self.assertEqual(len(model.expert_calls), 6 * count)
                self.assertEqual(info["action_head_refresh_steps"], count)
                self.assertTrue(torch.isfinite(actual).all())
            if not adaptive:
                prepared["state"] = prepared["state"] + 1
                changed, _ = adapter.run_action_head(output, prepared, ActionHeadAction.REUSE_ALL_BUT_FIRST)
                self.assertFalse(torch.equal(changed, actual))

    def test_shared_schedule_identity_and_thresholds(self):
        from elegantvla.gr00t.core.rule_policy import RuleSchedulePolicy as Gr00tRule
        from elegantvla.gr00t.core.types import ScheduleAction as Gr00tAction

        self.assertIs(RuleSchedulePolicy, Gr00tRule)
        self.assertIs(ScheduleAction, Gr00tAction)
        policy = RuleSchedulePolicy()
        for similarity, expected in ((0.9, 0), (0.96, 1), (0.985, 2), (0.992, 3), (0.999, 4)):
            for speed, head in ((0.001, 0), (0.01, 1), (0.1, 2)):
                action = policy.select_action({"feature_similarity": similarity, "eef_translation_speed": speed})
                self.assertEqual(action.as_tuple(), (expected, head))
        self.assertEqual(OpenPIActionHead.refresh_steps(1, 1), (0,))

    def test_public_policy_wires_release_and_legacy_separately(self):
        from elegantvla.openpi.core.config import PrefixVLMCacheConfig
        from elegantvla.openpi.core.policy import Policy

        data = {
            "image": {key: value[0].numpy()[None] for key, value in self.obs.images.items()},
            "image_mask": {key: value[0].numpy() for key, value in self.obs.image_masks.items()},
            "state": self.obs.state[0].numpy(),
            "tokenized_prompt": self.obs.tokenized_prompt[0].numpy(),
            "tokenized_prompt_mask": self.obs.tokenized_prompt_mask[0].numpy(),
        }
        model = self.model
        noise = self.noise[0].numpy()
        baseline = Policy(model, is_pytorch=True, sample_kwargs={"num_steps": 3})
        expected = baseline.infer(data, noise=noise)["actions"]
        policy = Policy(
            model, is_pytorch=True, sample_kwargs={"num_steps": 3}, temporal_skip=TemporalSkipConfig(enabled=True)
        )
        actual = policy.infer(data, noise=noise)
        np.testing.assert_allclose(actual["actions"], expected, atol=1e-6)
        self.assertEqual(actual["policy_timing"]["path"], "refresh")
        paths = [policy.infer(data, noise=noise)["policy_timing"]["path"] for _ in range(3)]
        self.assertEqual(paths, ["boundary", "image_free", "image_free"])
        policy.reset()
        self.assertEqual(policy.infer(data, noise=noise)["policy_timing"]["path"], "refresh")
        with self.assertRaises(ValueError):
            Policy(
                model,
                is_pytorch=True,
                temporal_skip=TemporalSkipConfig(enabled=True),
                prefix_vlm_cache=PrefixVLMCacheConfig(enabled=True),
            )
        with self.assertRaises(ValueError):
            Policy(model, is_pytorch=False, temporal_skip=TemporalSkipConfig(enabled=True))
        with patch("elegantvla.openpi.legacy.policy.Policy") as legacy:
            compatible = Policy(model, is_pytorch=True, prefix_vlm_cache=PrefixVLMCacheConfig(enabled=True))
            compatible.infer(data, noise=noise)
            legacy.return_value.infer.assert_called_once()
        motion_policy = Policy(model, is_pytorch=True, temporal_skip=TemporalSkipConfig())
        motion = motion_policy._motion_observation({"observation/state": np.arange(8)})
        np.testing.assert_array_equal(motion["state"]["eef_pose"], np.arange(6))
        self.assertEqual(motion["state"]["gripper"], 6)

    def test_serving_cli_forwards_the_release_config(self):
        import openpi.policies
        import tyro

        from elegantvla.openpi.core import policy as implementation

        root = Path(__file__).resolve().parents[3]
        factory_spec = importlib.util.spec_from_file_location(
            "_elegantvla_factory_test", root / "elegantvla/openpi/integration/src/openpi/policies/policy_config.py"
        )
        factory = importlib.util.module_from_spec(factory_spec)
        # Exercise the installed import topology without copying into third_party.
        with (
            patch.object(openpi.policies, "policy", implementation, create=True),
            patch.dict(sys.modules, {"openpi.policies.policy": implementation}),
        ):
            factory_spec.loader.exec_module(factory)
            with (
                patch.object(openpi.policies, "policy_config", factory, create=True),
                patch.dict(sys.modules, {"openpi.policies.policy_config": factory}),
            ):
                from elegantvla.openpi.entrypoints import serve

                args = tyro.cli(
                    serve.Args,
                    args=[
                        "--temporal-skip.enabled",
                        "policy:checkpoint",
                        "--policy.config",
                        "pi05_libero",
                        "--policy.dir",
                        "/unused/test-checkpoint",
                    ],
                )
                self.assertTrue(args.temporal_skip.enabled)
                with (
                    patch.object(serve._policy_config, "create_trained_policy") as create,
                    patch.object(serve._config, "get_config"),
                ):
                    serve.create_policy(args)
                    self.assertIs(create.call_args.kwargs["temporal_skip"], args.temporal_skip)


if __name__ == "__main__":
    unittest.main()
