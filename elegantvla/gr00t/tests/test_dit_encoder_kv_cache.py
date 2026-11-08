import unittest
import types

import torch

from gr00t.model.modules.dit import AlternateVLDiT


class TestDiTEncoderKVCache(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = AlternateVLDiT(
            num_attention_heads=2,
            attention_head_dim=4,
            output_dim=8,
            num_layers=4,
            dropout=0.0,
            attention_bias=True,
            activation_fn="gelu-approximate",
            norm_type="ada_norm",
            final_dropout=False,
            positional_embeddings=None,
            interleave_self_attention=True,
            cross_attention_dim=10,
            attend_text_every_n_blocks=2,
        ).eval()
        self.hidden_states = torch.randn(1, 5, 8)
        self.encoder_hidden_states = torch.randn(1, 7, 10)
        self.timestep = torch.tensor([250])
        self.image_mask = torch.tensor(
            [[False, False, True, True, True, False, False]],
            dtype=torch.bool,
        )
        self.backbone_attention_mask = torch.ones(1, 7, dtype=torch.bool)

    def _forward(self, encoder_kv_cache=None):
        return self.model(
            hidden_states=self.hidden_states,
            encoder_hidden_states=self.encoder_hidden_states,
            timestep=self.timestep,
            image_mask=self.image_mask,
            backbone_attention_mask=self.backbone_attention_mask,
            encoder_kv_cache=encoder_kv_cache,
        )

    def test_projected_kv_matches_normal_cross_attention(self):
        with torch.inference_mode():
            expected = self._forward()
            encoder_kv_cache = self.model.build_encoder_kv_cache(
                self.encoder_hidden_states,
                image_mask=self.image_mask,
                backbone_attention_mask=self.backbone_attention_mask,
            )
            actual = self._forward(encoder_kv_cache)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_repeated_forward_does_not_repeat_encoder_projection(self):
        projection_calls = 0

        def count_projection(*_args):
            nonlocal projection_calls
            projection_calls += 1

        hooks = []
        for block_index, block in enumerate(self.model.transformer_blocks):
            if block_index % 2 == 0:
                hooks.append(block.attn1.to_k.register_forward_hook(count_projection))
                hooks.append(block.attn1.to_v.register_forward_hook(count_projection))
        try:
            with torch.inference_mode():
                encoder_kv_cache = self.model.build_encoder_kv_cache(
                    self.encoder_hidden_states,
                    image_mask=self.image_mask,
                    backbone_attention_mask=self.backbone_attention_mask,
                )
                calls_after_build = projection_calls
                self._forward(encoder_kv_cache)
                self._forward(encoder_kv_cache)
        finally:
            for hook in hooks:
                hook.remove()

        self.assertEqual(calls_after_build, 4)
        self.assertEqual(projection_calls, calls_after_build)

    def test_compact_cache_only_projects_visible_tokens(self):
        with torch.inference_mode():
            encoder_kv_cache = self.model.build_encoder_kv_cache(
                self.encoder_hidden_states,
                image_mask=self.image_mask,
                backbone_attention_mask=self.backbone_attention_mask,
            )

        non_image_key, _, non_image_compact = encoder_kv_cache[0]
        image_key, _, image_compact = encoder_kv_cache[2]
        self.assertTrue(non_image_compact)
        self.assertTrue(image_compact)
        self.assertEqual(non_image_key.shape[-2], 4)
        self.assertEqual(image_key.shape[-2], 3)

    def test_compact_forward_matches_legacy_masked_attention(self):
        compact_impl = self.model._compact_encoder_hidden_states
        self.model._compact_encoder_hidden_states = types.MethodType(
            lambda _self, *_args, **_kwargs: None,
            self.model,
        )
        try:
            with torch.inference_mode():
                expected = self._forward()
        finally:
            self.model._compact_encoder_hidden_states = compact_impl

        with torch.inference_mode():
            actual = self._forward()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
