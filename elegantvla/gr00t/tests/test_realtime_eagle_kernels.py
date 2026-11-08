"""CPU-safe regression tests for the optional RealtimeGR00T runtime."""

import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F


def _load_runtime_module():
    module_path = (
        Path(__file__).parents[1]
        / "integration"
        / "gr00t"
        / "model"
        / "modules"
        / "realtime_eagle_kernels.py"
    )
    spec = importlib.util.spec_from_file_location(
        "realtime_eagle_kernels_test_module", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpu_operator_fallbacks_match_pytorch() -> None:
    runtime = _load_runtime_module()
    hidden_states = torch.randn(2, 3, 4)
    weight = torch.randn(4)
    expected_norm = weight * (
        hidden_states
        * torch.rsqrt(hidden_states.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
    )
    assert torch.allclose(
        runtime.triton_rms_norm(hidden_states, weight, 1e-5), expected_norm
    )

    gate = torch.randn(2, 3, 5)
    up = torch.randn(2, 3, 5)
    assert torch.allclose(runtime.triton_silu_mul(gate, up), F.silu(gate) * up)


def test_cuda_graph_proxy_falls_back_when_cuda_is_unavailable() -> None:
    runtime = _load_runtime_module()

    class Toy(torch.nn.Module):
        def forward(self, x: torch.Tensor, scale: int = 1) -> torch.Tensor:
            return x * scale

    proxy = runtime.QwenDecoderCudaGraphProxy(Toy().eval()).eval()
    result = proxy(torch.ones(2), scale=3)
    assert torch.equal(result, torch.full((2,), 3.0))


def test_boundary_graph_proxy_falls_back_when_cuda_is_unavailable() -> None:
    runtime = _load_runtime_module()

    class ToyLayer(torch.nn.Module):
        def forward(self, hidden_states, scale=1):
            return (hidden_states * scale,)

    proxy = runtime.QwenLayerCudaGraphProxy(ToyLayer().eval(), layer_index=0).eval()
    result = proxy(torch.ones(2), scale=4)
    assert torch.equal(result[0], torch.full((2,), 4.0))


def test_suffix_graph_proxy_returns_middle_and_final_taps_on_cpu() -> None:
    runtime = _load_runtime_module()

    class ToyLayer(torch.nn.Module):
        def __init__(self, increment: float) -> None:
            super().__init__()
            self.increment = increment

        def forward(self, hidden_states, **kwargs):
            return (hidden_states + self.increment,)

    proxy = runtime.QwenSuffixCudaGraphProxy(
        [ToyLayer(1.0), ToyLayer(2.0), ToyLayer(3.0)]
    )
    middle, final = proxy(torch.zeros(2))
    assert torch.equal(middle, torch.full((2,), 3.0))
    assert torch.equal(final, torch.full((2,), 6.0))


def test_action_graph_proxy_can_force_temporal_eager_fallback() -> None:
    runtime = _load_runtime_module()

    class Toy(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + 1

    proxy = runtime.ActionDitCudaGraphProxy(Toy().eval()).eval()
    proxy.set_temporal_skip_force_eager(True)
    result = proxy(torch.ones(2))
    assert torch.equal(result, torch.full((2,), 2.0))


def test_action_graph_proxy_keeps_projected_and_cached_kv_variants() -> None:
    runtime = _load_runtime_module()
    proxy = runtime.ActionDitCudaGraphProxy(torch.nn.Identity())

    project_graph = object()
    project_output = object()
    proxy._graph_mode = False
    proxy._graph = project_graph
    proxy._static_output = project_output

    proxy._activate_graph_mode(True)
    assert proxy._graph is None
    cached_graph = object()
    cached_output = object()
    proxy._graph = cached_graph
    proxy._static_output = cached_output

    proxy._activate_graph_mode(False)
    assert proxy._graph is project_graph
    assert proxy._static_output is project_output

    proxy._activate_graph_mode(True)
    assert proxy._graph is cached_graph
    assert proxy._static_output is cached_output


def test_action_graph_proxy_can_reset_variants() -> None:
    runtime = _load_runtime_module()
    proxy = runtime.ActionDitCudaGraphProxy(torch.nn.Identity())
    proxy._graph_mode = False
    proxy._graph = object()
    proxy._graph_variants[True] = {"graph": object()}
    proxy._layout_signature = (("image_mask", "old"),)

    proxy._reset_graph_variants()

    assert proxy._graph is None
    assert proxy._graph_mode is None
    assert proxy._graph_variants == {}
