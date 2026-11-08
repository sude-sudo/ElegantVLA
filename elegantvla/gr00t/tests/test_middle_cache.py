from types import SimpleNamespace

import torch

from elegantvla.gr00t.core.adapter import Gr00tPolicyTemporalSkipAdapter
from elegantvla.gr00t.core.wrapper import SkipController


class _AddLayer:
    def __init__(self, value: float) -> None:
        self.value = value
        self.calls = 0

    def __call__(self, hidden_states: torch.Tensor, **_: object) -> tuple[torch.Tensor]:
        self.calls += 1
        return (hidden_states + self.value,)


class _Backbone:
    def __init__(self) -> None:
        qwen = SimpleNamespace(
            layers=[_AddLayer(1.0), _AddLayer(2.0), _AddLayer(3.0), _AddLayer(4.0)],
            norm=lambda tensor: tensor,
        )
        self.model = SimpleNamespace(language_model=SimpleNamespace(model=qwen))

    def set_frozen_modules_to_eval_mode(self) -> None:
        return None


class _CountingEmbedding:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 1, 3)


class _CountingQwen:
    def __init__(self) -> None:
        self.layers = [_AddLayer(1.0)]
        self.mask_calls = 0
        self.rotary_calls = 0

    def _update_causal_mask(self, *_: object) -> None:
        self.mask_calls += 1
        return None

    def rotary_emb(self, hidden_states: torch.Tensor, position_ids: torch.Tensor):
        self.rotary_calls += 1
        return hidden_states.new_zeros(1), position_ids.new_zeros(1)


class _ProbeBackbone:
    def __init__(self) -> None:
        self.embedding = _CountingEmbedding()
        self.qwen = _CountingQwen()
        language_model = SimpleNamespace(
            get_input_embeddings=lambda: self.embedding,
            model=self.qwen,
        )
        self.model = SimpleNamespace(
            language_model=language_model,
            config=SimpleNamespace(image_token_index=99),
            extract_feature=lambda pixel_values: pixel_values,
        )
        self.dtype = torch.float32

    def set_frozen_modules_to_eval_mode(self) -> None:
        return None


class _ProbeModel:
    def __init__(self) -> None:
        self.backbone = _ProbeBackbone()
        self.dtype = torch.float32

    def prepare_input(self, inputs: dict[str, torch.Tensor]):
        return inputs, SimpleNamespace()


class _InputConsumer:
    def __init__(self) -> None:
        self.calls = 0
        self.inputs = None

    def prepare_input(self, inputs: dict[str, torch.Tensor]):
        self.calls += 1
        self.inputs = inputs
        return inputs


class _SplitInputModel:
    def __init__(self) -> None:
        self.backbone = _InputConsumer()
        self.action_head = _InputConsumer()
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.fallback_calls = 0

    def prepare_input(self, inputs: dict[str, torch.Tensor]):
        self.fallback_calls += 1
        return inputs, inputs


def _adapter() -> Gr00tPolicyTemporalSkipAdapter:
    adapter = object.__new__(Gr00tPolicyTemporalSkipAdapter)
    adapter.model = SimpleNamespace(backbone=_Backbone())
    return adapter


def _probe(hidden_states: torch.Tensor) -> dict[str, object]:
    return {
        "hidden_states": hidden_states,
        "causal_mask": None,
        "position_ids": None,
        "cache_position": None,
        "position_embeddings": None,
        "backbone_attention_mask": torch.ones(1, 1),
        "image_mask": torch.ones(1, 1, dtype=torch.bool),
    }


def test_middle_cache_recomputes_first_and_last_layers() -> None:
    adapter = _adapter()
    refresh_output, middle_cache = adapter.continue_backbone_from_probe_with_middle_cache(
        _probe(torch.ones(1, 2, 3))
    )

    torch.testing.assert_close(refresh_output["backbone_features"], torch.full((1, 2, 3), 10.0))
    torch.testing.assert_close(middle_cache["middle_delta"], torch.full((1, 2, 3), 5.0))
    torch.testing.assert_close(middle_cache["post_first_delta"], torch.full((1, 2, 3), 9.0))

    reuse_output, refreshed_cache, used_middle_cache = adapter.continue_backbone_from_middle_cache(
        _probe(torch.full((1, 2, 3), 10.0)),
        middle_cache,
    )

    assert used_middle_cache is True
    assert refreshed_cache is None
    torch.testing.assert_close(reuse_output["backbone_features"], torch.full((1, 2, 3), 19.0))


def test_post_first_cache_recomputes_only_first_layer() -> None:
    adapter = _adapter()
    _, cache = adapter.continue_backbone_from_probe_with_middle_cache(
        _probe(torch.ones(1, 2, 3))
    )
    layers = adapter.model.backbone.model.language_model.model.layers
    calls_before_reuse = [layer.calls for layer in layers]

    output, refreshed_cache, used_post_first_cache = (
        adapter.continue_backbone_from_post_first_cache(
            _probe(torch.full((1, 2, 3), 10.0)),
            cache,
        )
    )

    assert used_post_first_cache is True
    assert refreshed_cache is None
    assert [layer.calls for layer in layers] == calls_before_reuse
    torch.testing.assert_close(output["backbone_features"], torch.full((1, 2, 3), 19.0))


def test_layer_subset_cache_runs_only_selected_layers() -> None:
    adapter = _adapter()
    _, cache = adapter.continue_backbone_from_probe_with_layer_boundary_cache(
        _probe(torch.ones(1, 2, 3))
    )
    layers = adapter.model.backbone.model.language_model.model.layers
    calls_before_reuse = [layer.calls for layer in layers]

    output, refreshed_cache, cache_used = adapter.continue_backbone_from_layer_subset_cache(
        _probe(torch.full((1, 2, 3), 10.0)),
        cache,
        (0, 2),
    )

    assert cache_used is True
    assert refreshed_cache is None
    assert [layer.calls for layer in layers] == [
        calls_before_reuse[0],
        calls_before_reuse[1],
        calls_before_reuse[2] + 1,
        calls_before_reuse[3],
    ]
    torch.testing.assert_close(output["backbone_features"], torch.full((1, 2, 3), 19.0))


def test_full_compute_continuation_does_not_build_middle_cache() -> None:
    adapter = _adapter()

    output = adapter.continue_backbone_from_probe(_probe(torch.ones(1, 2, 3)))

    torch.testing.assert_close(output["backbone_features"], torch.full((1, 2, 3), 10.0))


def test_middle_cache_shape_mismatch_falls_back_to_refresh() -> None:
    adapter = _adapter()
    output, refreshed_cache, used_middle_cache = adapter.continue_backbone_from_middle_cache(
        _probe(torch.ones(1, 2, 3)),
        {"middle_delta": torch.ones(1, 1, 3)},
    )

    assert used_middle_cache is False
    assert refreshed_cache is not None
    torch.testing.assert_close(output["backbone_features"], torch.full((1, 2, 3), 10.0))


def test_dual_cka_matches_primal_definition() -> None:
    generator = torch.Generator().manual_seed(7)
    anchor = torch.randn(2, 51, 128, generator=generator)
    current = torch.randn(2, 51, 128, generator=generator)
    anchor_matrix = anchor.reshape(-1, anchor.shape[-1]).float()
    current_matrix = current.reshape(-1, current.shape[-1]).float()
    anchor_centered = anchor_matrix - anchor_matrix.mean(dim=0, keepdim=True)
    current_centered = current_matrix - current_matrix.mean(dim=0, keepdim=True)
    cross_cov = anchor_centered.T @ current_centered
    expected = torch.sum(cross_cov * cross_cov) / (
        torch.linalg.norm(anchor_centered.T @ anchor_centered)
        * torch.linalg.norm(current_centered.T @ current_centered)
    )

    actual = SkipController._feature_cka_similarity(anchor, current)

    torch.testing.assert_close(torch.tensor(actual), expected, rtol=2e-5, atol=2e-6)


def test_static_probe_cache_reuses_only_text_side_scaffolding() -> None:
    adapter = object.__new__(Gr00tPolicyTemporalSkipAdapter)
    adapter.model = _ProbeModel()
    adapter.static_inference_cache_enabled = True
    adapter._qwen_static_input_cache = {}
    common_inputs = {
        "input_ids": torch.tensor([[7, 99]], dtype=torch.long),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
    }

    first_probe, _ = adapter.get_first_block_gate_probe(
        {**common_inputs, "pixel_values": torch.tensor([[1.0, 2.0, 3.0]])}
    )
    second_probe, _ = adapter.get_first_block_gate_probe(
        {**common_inputs, "pixel_values": torch.tensor([[4.0, 5.0, 6.0]])}
    )

    backbone = adapter.model.backbone
    assert backbone.embedding.calls == 1
    assert backbone.qwen.mask_calls == 1
    assert backbone.qwen.rotary_calls == 1
    torch.testing.assert_close(first_probe["hidden_states"][0, 0], torch.full((3,), 8.0))
    torch.testing.assert_close(first_probe["hidden_states"][0, 1], torch.tensor([2.0, 3.0, 4.0]))
    torch.testing.assert_close(second_probe["hidden_states"][0, 1], torch.tensor([5.0, 6.0, 7.0]))


def test_static_cache_path_moves_shared_model_inputs_once() -> None:
    adapter = object.__new__(Gr00tPolicyTemporalSkipAdapter)
    adapter.model = _SplitInputModel()
    adapter.static_inference_cache_enabled = True
    inputs = {
        "state": torch.ones(1, 4),
        "input_ids": torch.ones(1, 2, dtype=torch.long),
    }

    backbone_inputs, action_inputs = adapter._prepare_model_inputs(inputs)

    assert adapter.model.fallback_calls == 0
    assert adapter.model.backbone.calls == 1
    assert adapter.model.action_head.calls == 1
    assert backbone_inputs["state"] is action_inputs["state"]
    assert backbone_inputs["input_ids"] is action_inputs["input_ids"]


def test_full_compute_path_keeps_original_input_preparation() -> None:
    adapter = object.__new__(Gr00tPolicyTemporalSkipAdapter)
    adapter.model = _SplitInputModel()
    adapter.static_inference_cache_enabled = False

    adapter._prepare_model_inputs({"state": torch.ones(1, 4)})

    assert adapter.model.fallback_calls == 1
    assert adapter.model.backbone.calls == 0
    assert adapter.model.action_head.calls == 0


if __name__ == "__main__":
    test_middle_cache_recomputes_first_and_last_layers()
    test_full_compute_continuation_does_not_build_middle_cache()
    test_middle_cache_shape_mismatch_falls_back_to_refresh()
    test_dual_cka_matches_primal_definition()
    test_static_probe_cache_reuses_only_text_side_scaffolding()
    test_static_cache_path_moves_shared_model_inputs_once()
    test_full_compute_path_keeps_original_input_preparation()
