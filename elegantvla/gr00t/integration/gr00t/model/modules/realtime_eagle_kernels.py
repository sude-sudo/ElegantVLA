"""RealtimeVLA-style inference kernels for Eagle's Qwen3 and Siglip2 towers.

These kernels are inference-only. They intentionally keep a conservative
fallback path so loading a checkpoint remains portable to CPU and systems
without Triton.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _layer_norm_stats_kernel(
        input_ptr,
        mean_ptr,
        rstd_ptr,
        n_rows,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Compute row-wise LayerNorm statistics once for the following GEMM."""
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        values = tl.load(
            input_ptr + row * n_cols + offsets,
            mask=offsets < n_cols,
            other=0.0,
        ).to(tl.float32)
        mean = tl.sum(values, axis=0) / n_cols
        centered = tl.where(offsets < n_cols, values - mean, 0.0)
        variance = tl.sum(centered * centered, axis=0) / n_cols
        tl.store(mean_ptr + row, mean)
        tl.store(rstd_ptr + row, tl.rsqrt(variance + eps))

    @triton.jit
    def _layer_norm_linear_kernel(
        input_ptr,
        weight_ptr,
        output_ptr,
        mean_ptr,
        rstd_ptr,
        col_sum_ptr,
        bias_ptr,
        M,
        N,
        K,
        HAS_GELU: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """RealtimeVLA v1's folded LayerNorm -> GEMM (+ optional GELU)."""
        pid = tl.program_id(0)
        num_m = tl.cdiv(M, BLOCK_M)
        num_n = tl.cdiv(N, BLOCK_N)
        group_m: tl.constexpr = 8
        group_id = pid // (group_m * num_n)
        first_m = group_id * group_m
        group_size = min(num_m - first_m, group_m)
        pid_m = first_m + (pid % group_size)
        pid_n = (pid % (group_m * num_n)) // group_size

        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        k_offsets = tl.arange(0, BLOCK_K)
        row_mask = rows < M
        col_mask = cols < N
        a_ptrs = input_ptr + rows[:, None] * K + k_offsets[None, :]
        b_ptrs = weight_ptr + k_offsets[:, None] * N + cols[None, :]
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            k_mask = k_start + k_offsets < K
            a = tl.load(
                a_ptrs,
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=k_mask[:, None] & col_mask[None, :],
                other=0.0,
            )
            accumulator = tl.dot(a, b, accumulator)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * N

        mean = tl.load(mean_ptr + rows, mask=row_mask, other=0.0)
        rstd = tl.load(rstd_ptr + rows, mask=row_mask, other=1.0)
        col_sum = tl.load(col_sum_ptr + cols, mask=col_mask, other=0.0)
        bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0)
        accumulator = (
            rstd[:, None]
            * (accumulator - mean[:, None] * col_sum[None, :])
            + bias[None, :]
        )
        if HAS_GELU:
            # tanh-GELU written as 2*sigmoid(2x)-1, matching RealtimeVLA v1.
            x_sq = accumulator * accumulator
            inner = 1.5957691216057308 * accumulator * (1.0 + 0.044715 * x_sq)
            accumulator = accumulator * tl.sigmoid(inner)
        output_ptrs = output_ptr + rows[:, None] * N + cols[None, :]
        tl.store(
            output_ptrs,
            accumulator.to(tl.bfloat16),
            mask=row_mask[:, None] & col_mask[None, :],
        )

    @triton.jit
    def _linear_bias_residual_kernel(
        input_ptr,
        weight_ptr,
        bias_ptr,
        residual_ptr,
        output_ptr,
        M,
        N,
        K,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """GEMM + bias + residual without two extra BF16 pointwise launches."""
        pid = tl.program_id(0)
        num_m = tl.cdiv(M, BLOCK_M)
        num_n = tl.cdiv(N, BLOCK_N)
        group_m: tl.constexpr = 8
        group_id = pid // (group_m * num_n)
        first_m = group_id * group_m
        group_size = min(num_m - first_m, group_m)
        pid_m = first_m + (pid % group_size)
        pid_n = (pid % (group_m * num_n)) // group_size
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = rows < M
        col_mask = cols < N
        matrix_mask = row_mask[:, None] & col_mask[None, :]
        k_offsets = tl.arange(0, BLOCK_K)
        a_ptrs = input_ptr + rows[:, None] * K + k_offsets[None, :]
        b_ptrs = weight_ptr + k_offsets[:, None] * N + cols[None, :]
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            k_mask = k_start + k_offsets < K
            a = tl.load(
                a_ptrs,
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=k_mask[:, None] & col_mask[None, :],
                other=0.0,
            )
            accumulator = tl.dot(a, b, accumulator)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * N
        bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0)
        residual = tl.load(
            residual_ptr + rows[:, None] * N + cols[None, :],
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += bias[None, :] + residual
        tl.store(
            output_ptr + rows[:, None] * N + cols[None, :],
            accumulator.to(tl.bfloat16),
            mask=matrix_mask,
        )

    @triton.jit
    def _rms_norm_kernel(
        x_ptr,
        weight_ptr,
        output_ptr,
        row_stride: tl.constexpr,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row * row_stride + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        variance = tl.sum(x * x, axis=0) / n_cols
        # Match Qwen3RMSNorm's rounding order: normalize in FP32, cast the
        # normalized activation to BF16, then multiply by the BF16 weight.
        normalized = (x * tl.rsqrt(variance + eps)).to(tl.bfloat16)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
        tl.store(
            output_ptr + row * row_stride + offsets,
            normalized * weight,
            mask=mask,
        )

    @triton.jit
    def _silu_mul_kernel(
        gate_ptr,
        up_ptr,
        output_ptr,
        n_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        block = tl.program_id(0)
        offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        # Match PyTorch BF16 SiLU followed by BF16 multiplication: SiLU is
        # rounded to BF16 before multiplying by the up branch.
        silu = (gate * tl.sigmoid(gate)).to(tl.bfloat16)
        tl.store(output_ptr + offsets, silu * up, mask=mask)

    @triton.jit
    def _packed_swiglu_kernel(
        gate_up_ptr,
        output_ptr,
        n_cols: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        cols = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        row_start = row * (2 * n_cols)
        gate = tl.load(gate_up_ptr + row_start + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        up = tl.load(
            gate_up_ptr + row_start + n_cols + cols, mask=mask, other=0.0
        ).to(tl.float32)
        silu = gate * tl.sigmoid(gate)
        tl.store(output_ptr + row * n_cols + cols, silu * up, mask=mask)


def triton_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fused BF16/FP16 RMSNorm with FP32 accumulation."""
    if (
        not _TRITON_AVAILABLE
        or not hidden_states.is_cuda
        or hidden_states.dtype != torch.bfloat16
        or not hidden_states.is_contiguous()
    ):
        input_dtype = hidden_states.dtype
        values = hidden_states.to(torch.float32)
        variance = values.pow(2).mean(-1, keepdim=True)
        return weight * (values * torch.rsqrt(variance + eps)).to(input_dtype)

    n_cols = hidden_states.shape[-1]
    if n_cols > 65536:
        raise ValueError(f"RMSNorm hidden dimension is too large for one Triton block: {n_cols}")
    output = torch.empty_like(hidden_states)
    n_rows = hidden_states.numel() // n_cols
    block_size = triton.next_power_of_2(n_cols)
    num_warps = 8 if block_size >= 2048 else 4
    _rms_norm_kernel[(n_rows,)](
        hidden_states,
        weight,
        output,
        hidden_states.stride(-2) if hidden_states.ndim > 1 else n_cols,
        n_cols,
        eps,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output


def triton_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fuse SiLU(gate) * up into one elementwise Triton kernel."""
    if (
        not _TRITON_AVAILABLE
        or not gate.is_cuda
        or not gate.is_contiguous()
        or not up.is_contiguous()
    ):
        return F.silu(gate) * up
    output = torch.empty_like(gate)
    n_elements = gate.numel()
    block_size = 256
    _silu_mul_kernel[(triton.cdiv(n_elements, block_size),)](
        gate,
        up,
        output,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return output


def triton_packed_swiglu(
    gate_up: torch.Tensor, intermediate_size: int
) -> torch.Tensor:
    """Consume contiguous [..., gate||up] output without materializing slices."""
    if not _TRITON_AVAILABLE or not gate_up.is_cuda or not gate_up.is_contiguous():
        gate, up = gate_up.split(intermediate_size, dim=-1)
        return F.silu(gate) * up
    output_shape = (*gate_up.shape[:-1], intermediate_size)
    output = torch.empty(output_shape, dtype=gate_up.dtype, device=gate_up.device)
    n_rows = gate_up.numel() // (2 * intermediate_size)
    block_size = 256
    _packed_swiglu_kernel[
        (n_rows, triton.cdiv(intermediate_size, block_size))
    ](
        gate_up,
        output,
        intermediate_size,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return output


def fold_layer_norm_linear(
    layer_norm: nn.LayerNorm,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fold LayerNorm's affine transform into a following Linear.

    The returned weight uses GEMM's [K, N] layout, while the bias and column
    sums stay FP32 for the correction applied after the raw input GEMM.
    """
    weight_kn = linear_weight.detach().float().T.contiguous()
    norm_weight = layer_norm.weight.detach().float()
    norm_bias = layer_norm.bias.detach().float()
    fused_fp32 = norm_weight[:, None] * weight_kn
    fused_bias = norm_bias @ weight_kn + linear_bias.detach().float()
    return (
        fused_fp32.to(dtype=linear_weight.dtype).contiguous(),
        fused_bias.contiguous(),
        fused_fp32.sum(dim=0).contiguous(),
    )


def triton_layer_norm_linear(
    hidden_states: torch.Tensor,
    weight_kn: torch.Tensor,
    bias: torch.Tensor,
    col_sum: torch.Tensor,
    eps: float,
    *,
    gelu: bool = False,
) -> torch.Tensor:
    """Run folded LayerNorm -> Linear and optionally tanh-GELU."""
    if (
        not _TRITON_AVAILABLE
        or not hidden_states.is_cuda
        or hidden_states.dtype != torch.bfloat16
        or not hidden_states.is_contiguous()
    ):
        raise RuntimeError("folded LayerNorm Linear requires contiguous CUDA BF16 input")
    shape = hidden_states.shape
    k = shape[-1]
    x = hidden_states.view(-1, k)
    m = x.shape[0]
    n = weight_kn.shape[1]
    output = torch.empty((m, n), dtype=x.dtype, device=x.device)
    mean = torch.empty(m, dtype=torch.float32, device=x.device)
    rstd = torch.empty_like(mean)
    stats_block = triton.next_power_of_2(k)
    _layer_norm_stats_kernel[(m,)](
        x,
        mean,
        rstd,
        m,
        k,
        eps,
        BLOCK_SIZE=stats_block,
        num_warps=8,
    )
    block_m, block_n, block_k = 64, 128, 32
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    _layer_norm_linear_kernel[grid](
        x,
        weight_kn,
        output,
        mean,
        rstd,
        col_sum,
        bias,
        m,
        n,
        k,
        HAS_GELU=gelu,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_stages=3,
        num_warps=4,
    )
    return output.view(*shape[:-1], n)


def triton_linear_bias_residual(
    hidden_states: torch.Tensor,
    weight_kn: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    """Run Linear + bias + residual in one RealtimeVLA-style Triton kernel."""
    if (
        not _TRITON_AVAILABLE
        or not hidden_states.is_cuda
        or not hidden_states.is_contiguous()
        or not residual.is_contiguous()
    ):
        return F.linear(hidden_states, weight_kn.T, bias) + residual
    shape = hidden_states.shape
    k = shape[-1]
    x = hidden_states.view(-1, k)
    m = x.shape[0]
    n = weight_kn.shape[1]
    output = torch.empty((m, n), dtype=x.dtype, device=x.device)
    block_m, block_n, block_k = 64, 64, 32
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    _linear_bias_residual_kernel[grid](
        x,
        weight_kn,
        bias,
        residual.view(m, n),
        output,
        m,
        n,
        k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_stages=4,
        num_warps=4,
    )
    return output.view(*shape[:-1], n)


class TritonRMSNorm(nn.Module):
    def __init__(self, original: nn.Module):
        super().__init__()
        self.weight = original.weight
        self.variance_epsilon = float(original.variance_epsilon)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return triton_rms_norm(hidden_states, self.weight, self.variance_epsilon)


class FusedGateUpQwen3MLP(nn.Module):
    """Pack gate/up weights into one GEMM, then use fused Triton SwiGLU."""

    def __init__(self, original: nn.Module):
        super().__init__()
        self.hidden_size = int(original.hidden_size)
        self.intermediate_size = int(original.intermediate_size)
        packed = torch.cat(
            (original.gate_proj.weight.detach(), original.up_proj.weight.detach()),
            dim=0,
        ).contiguous()
        self.gate_up_weight = nn.Parameter(packed, requires_grad=False)
        self.down_proj = original.down_proj

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = F.linear(hidden_states, self.gate_up_weight)
        return self.down_proj(triton_packed_swiglu(gate_up, self.intermediate_size))


class TritonSwiGLUQwen3MLP(nn.Module):
    """Keep the original GEMMs and fuse only the BF16 SwiGLU pointwise path."""

    def __init__(self, original: nn.Module):
        super().__init__()
        self.hidden_size = int(original.hidden_size)
        self.intermediate_size = int(original.intermediate_size)
        self.gate_proj = original.gate_proj
        self.up_proj = original.up_proj
        self.down_proj = original.down_proj

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        return self.down_proj(triton_silu_mul(gate, up))


class FusedSiglip2Fc2ResidualLayer(nn.Module):
    """Siglip2 layer with RealtimeVLA's profitable FC2+bias+residual epilogue.

    On RTX 4090 at Eagle's real M=1152 shape, replacing the other cuBLAS
    GEMMs with RealtimeVLA's hand-written GEMM is slower. This wrapper leaves
    attention and FC1 untouched and fuses only the measured winning segment.
    """

    def __init__(self, original: nn.Module):
        super().__init__()
        self.embed_dim = int(original.embed_dim)
        self.layer_norm1 = original.layer_norm1
        self.self_attn = original.self_attn
        self.layer_norm2 = original.layer_norm2
        self.fc1 = original.mlp.fc1
        self.activation_fn = original.mlp.activation_fn
        self.fc2_weight_kn = nn.Parameter(
            original.mlp.fc2.weight.detach().T.contiguous(), requires_grad=False
        )
        self.fc2_bias = original.mlp.fc2.bias

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        rope_freqs_cis: torch.Tensor | None = None,
        win_meta_list: list[dict] | None = None,
        windows_attn: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        residual = hidden_states
        attention_output, attention_weights = self.self_attn(
            hidden_states=self.layer_norm1(hidden_states),
            output_attentions=output_attentions,
            rope_freqs_cis=rope_freqs_cis,
            win_meta_list=win_meta_list,
            windows_attn=windows_attn,
        )
        hidden_states = residual + attention_output

        residual = hidden_states
        mlp_hidden = self.activation_fn(self.fc1(self.layer_norm2(hidden_states)))
        hidden_states = triton_linear_bias_residual(
            mlp_hidden,
            self.fc2_weight_kn,
            self.fc2_bias,
            residual,
        )
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attention_weights,)
        return outputs


class StaticQwenAttentionProxy(nn.Module):
    """Deployment-only Qwen attention without the packed-position probe.

    Eagle passes already-computed RoPE tensors separately. For the local
    batch=1, non-packed deployment sequence, forwarding ``position_ids`` to
    Transformers FlashAttention only triggers a GPU scalar control-flow probe
    on every layer; it does not affect attention math.
    """

    def __init__(self, original: nn.Module):
        super().__init__()
        self.original = original

    def forward(self, *args, **kwargs):
        hidden_states = kwargs.get("hidden_states", args[0] if args else None)
        if hidden_states is not None and hidden_states.shape[0] == 1:
            kwargs.pop("position_ids", None)
        return self.original(*args, **kwargs)


class _DelegatingCudaGraphProxy(nn.Module):
    """Forward non-graph model attributes to the wrapped module."""

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError as original_error:
            try:
                modules = object.__getattribute__(self, "_modules")
                original = modules.get("original")
            except AttributeError:
                original = None
            if original is not None:
                try:
                    return getattr(original, name)
                except AttributeError:
                    pass
            raise original_error


class QwenDecoderCudaGraphProxy(_DelegatingCudaGraphProxy):
    """Capture the fixed-shape 16-layer Qwen decoder as one CUDA Graph."""

    def __init__(self, original: nn.Module):
        super().__init__()
        self.original = original
        self._graph = None
        self._static_args = None
        self._static_kwargs = None
        self._static_output = None
        self._drop_all_ones_attention_mask = False

    @property
    def layers(self):
        return self.original.layers

    @property
    def norm(self):
        return self.original.norm

    @property
    def config(self):
        return self.original.config

    def get_input_embeddings(self):
        """Preserve the PreTrainedModel interface used by QwenForCausalLM."""
        return self.original.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.original.set_input_embeddings(value)

    @staticmethod
    def _allocate(value):
        if isinstance(value, torch.Tensor):
            return torch.empty_like(value)
        if isinstance(value, dict):
            return {key: QwenDecoderCudaGraphProxy._allocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [QwenDecoderCudaGraphProxy._allocate(item) for item in value]
        if isinstance(value, tuple):
            return tuple(QwenDecoderCudaGraphProxy._allocate(item) for item in value)
        return value

    @staticmethod
    def _copy(destination, source):
        if isinstance(source, torch.Tensor):
            if destination.shape != source.shape or destination.dtype != source.dtype:
                raise ValueError("CUDA Graph input shape/dtype changed")
            destination.copy_(source)
        elif isinstance(source, dict):
            if destination.keys() != source.keys():
                raise ValueError("CUDA Graph input dictionary keys changed")
            for key in source:
                QwenDecoderCudaGraphProxy._copy(destination[key], source[key])
        elif isinstance(source, (list, tuple)):
            if len(destination) != len(source):
                raise ValueError("CUDA Graph input sequence length changed")
            for destination_item, source_item in zip(destination, source):
                QwenDecoderCudaGraphProxy._copy(destination_item, source_item)
        elif destination != source:
            raise ValueError("CUDA Graph non-tensor metadata changed")

    def _prepare_graph_inputs(self, args, kwargs):
        """Remove an all-ones padding mask before CUDA Graph capture.

        Qwen's FlashAttention path treats an all-ones mask as ``None`` but
        Transformers checks that fact with Python tensor membership. That
        check is legal in eager mode and illegal while a CUDA Graph is being
        captured, so decide it once on the first fixed-shape input.
        """
        attention_mask = kwargs.get("attention_mask")
        if (
            isinstance(attention_mask, torch.Tensor)
            and attention_mask.numel() > 0
            and bool(torch.all(attention_mask == 1).item())
        ):
            kwargs = dict(kwargs)
            kwargs["attention_mask"] = None
            self._drop_all_ones_attention_mask = True
        return args, kwargs

    def _initialize(self, args, kwargs):
        args, kwargs = self._prepare_graph_inputs(args, kwargs)
        self._static_args = self._allocate(args)
        self._static_kwargs = self._allocate(kwargs)
        self._copy(self._static_args, args)
        self._copy(self._static_kwargs, kwargs)
        for _ in range(3):
            self._static_output = self.original(*self._static_args, **self._static_kwargs)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            graph.capture_begin()
            self._static_output = self.original(*self._static_args, **self._static_kwargs)
            graph.capture_end()
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()
        self._graph = graph
        print("[QwenDecoder] CUDA Graph 捕获完成")

    def forward(self, *args, **kwargs):
        # This proxy is hot-installed after the parent model has already been
        # put in eval mode, so its own freshly-created ``training`` flag would
        # otherwise remain True. The wrapped decoder is the source of truth.
        if self.original.training or not torch.cuda.is_available():
            return self.original(*args, **kwargs)
        if self._graph is None:
            self._initialize(args, kwargs)
        else:
            if self._drop_all_ones_attention_mask:
                kwargs = dict(kwargs)
                kwargs["attention_mask"] = None
            self._copy(self._static_args, args)
            self._copy(self._static_kwargs, kwargs)
            self._graph.replay()
        return self._static_output


class QwenLayerCudaGraphProxy(_DelegatingCudaGraphProxy):
    """Capture one fixed-shape Qwen layer for temporal head/tail recompute.

    The temporal-skip adapter intentionally bypasses ``QwenModel.forward`` so
    it can inject a cached middle-layer delta. A graph around the whole
    decoder is therefore not used on that path. Capturing only the boundary
    layers preserves the cache boundary while removing Python/kernel-launch
    overhead from the layers that are still recomputed.
    """

    def __init__(self, original: nn.Module, *, layer_index: int):
        super().__init__()
        self.original = original
        self.layer_index = int(layer_index)
        self._graph = None
        self._static_args = None
        self._static_kwargs = None
        self._static_output = None
        self._static_kwargs_source_signature = None

    @staticmethod
    def _source_signature(value):
        """Describe fixed graph inputs without reading their tensor contents.

        The temporal adapter reuses the same mask/position/RoPE tensors for a
        fixed prompt and image layout. Tracking their identities lets the hot
        replay path avoid recursively copying those tensors on every frame,
        while the version field still catches an in-place metadata update.
        """
        if isinstance(value, torch.Tensor):
            try:
                version = value._version
            except RuntimeError:
                version = None
            return (
                "tensor",
                id(value),
                tuple(value.shape),
                value.dtype,
                value.device.type,
                value.device.index,
                version,
            )
        if isinstance(value, dict):
            return (
                "dict",
                tuple(
                    (key, QwenLayerCudaGraphProxy._source_signature(item))
                    for key, item in value.items()
                ),
            )
        if isinstance(value, list):
            return (
                "list",
                tuple(QwenLayerCudaGraphProxy._source_signature(item) for item in value),
            )
        if isinstance(value, tuple):
            return (
                "tuple",
                tuple(QwenLayerCudaGraphProxy._source_signature(item) for item in value),
            )
        return ("value", type(value), value)

    @staticmethod
    def _copy_dynamic_hidden_state(destination, source):
        if not isinstance(source, torch.Tensor):
            raise ValueError("CUDA Graph layer input must start with hidden_states")
        if (
            destination.shape != source.shape
            or destination.dtype != source.dtype
            or destination.device != source.device
        ):
            raise ValueError("CUDA Graph layer hidden_states shape/dtype/device changed")
        destination.copy_(source)

    def _initialize(self, args, kwargs):
        self._static_args = QwenDecoderCudaGraphProxy._allocate(args)
        self._static_kwargs = QwenDecoderCudaGraphProxy._allocate(kwargs)
        QwenDecoderCudaGraphProxy._copy(self._static_args, args)
        QwenDecoderCudaGraphProxy._copy(self._static_kwargs, kwargs)
        self._static_kwargs_source_signature = self._source_signature(kwargs)

        for _ in range(3):
            self._static_output = self.original(
                *self._static_args,
                **self._static_kwargs,
            )
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            graph.capture_begin()
            self._static_output = self.original(
                *self._static_args,
                **self._static_kwargs,
            )
            graph.capture_end()
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()
        self._graph = graph
        print(f"[QwenLayer] CUDA Graph 捕获完成 layer={self.layer_index}")

    def forward(self, *args, **kwargs):
        if self.original.training or not torch.cuda.is_available():
            return self.original(*args, **kwargs)
        if self._graph is None:
            self._initialize(args, kwargs)
        else:
            # The first positional input is the only per-frame value on the
            # fixed-shape temporal path. Masks, positions, and RoPE tensors
            # are cached by the adapter and can remain in graph-owned memory.
            # If that invariant changes, copy the metadata once so the graph
            # remains correct without paying the cost on every replay.
            if len(args) == len(self._static_args) and args:
                self._copy_dynamic_hidden_state(self._static_args[0], args[0])
                for index in range(1, len(args)):
                    QwenDecoderCudaGraphProxy._copy(
                        self._static_args[index],
                        args[index],
                    )
            else:
                QwenDecoderCudaGraphProxy._copy(self._static_args, args)

            source_signature = self._source_signature(kwargs)
            if source_signature != self._static_kwargs_source_signature:
                QwenDecoderCudaGraphProxy._copy(self._static_kwargs, kwargs)
                self._static_kwargs_source_signature = source_signature
            self._graph.replay()
        return self._static_output


class QwenSuffixCudaGraphProxy:
    """Capture layers after the probe layer as one graph with two taps.

    A temporal refresh already computed layer 0 to produce the scheduler
    probe. The remaining layers used to run through Python one layer at a
    time, making the periodic cache refresh much slower than the ordinary
    full-decoder graph. This proxy captures that suffix once and returns both
    the pre-final hidden state (for the middle delta) and the final hidden
    state (for the action head).
    """

    def __init__(self, layers):
        self._layers = tuple(
            getattr(layer, "original", layer)
            for layer in layers
        )
        self._graph = None
        self._static_hidden = None
        self._static_kwargs = None
        self._static_output = None
        self._static_kwargs_source_signature = None

    def _run_suffix(self):
        hidden_states = self._static_hidden
        hidden_before_last = hidden_states
        last_index = len(self._layers) - 1
        for index, layer in enumerate(self._layers):
            hidden_states = layer(
                hidden_states,
                **self._static_kwargs,
            )[0]
            if index == last_index - 1:
                hidden_before_last = hidden_states
        return hidden_before_last, hidden_states

    def _initialize(self, hidden_states, kwargs):
        self._static_hidden = torch.empty_like(hidden_states)
        self._static_kwargs = QwenDecoderCudaGraphProxy._allocate(kwargs)
        QwenLayerCudaGraphProxy._copy_dynamic_hidden_state(
            self._static_hidden,
            hidden_states,
        )
        QwenDecoderCudaGraphProxy._copy(self._static_kwargs, kwargs)
        self._static_kwargs_source_signature = QwenLayerCudaGraphProxy._source_signature(
            kwargs
        )

        for _ in range(3):
            self._static_output = self._run_suffix()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            graph.capture_begin()
            self._static_output = self._run_suffix()
            graph.capture_end()
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()
        self._graph = graph
        print(f"[QwenSuffix] CUDA Graph 捕获完成 layers={len(self._layers)}")

    def __call__(self, hidden_states, **kwargs):
        if (
            not torch.cuda.is_available()
            or any(layer.training for layer in self._layers)
        ):
            current = hidden_states
            hidden_before_last = current
            last_index = len(self._layers) - 1
            for index, layer in enumerate(self._layers):
                current = layer(current, **kwargs)[0]
                if index == last_index - 1:
                    hidden_before_last = current
            return hidden_before_last, current

        if self._graph is None:
            self._initialize(hidden_states, kwargs)
        else:
            QwenLayerCudaGraphProxy._copy_dynamic_hidden_state(
                self._static_hidden,
                hidden_states,
            )
            source_signature = QwenLayerCudaGraphProxy._source_signature(kwargs)
            if source_signature != self._static_kwargs_source_signature:
                QwenDecoderCudaGraphProxy._copy(self._static_kwargs, kwargs)
                self._static_kwargs_source_signature = source_signature
            self._graph.replay()
        return self._static_output


class SiglipEncoderCudaGraphProxy(_DelegatingCudaGraphProxy):
    """Capture the fixed-shape 27-layer Siglip2 encoder as one CUDA Graph.

    Window construction and reverse mapping stay outside the graph. This
    boundary captures the expensive, tensor-only encoder while leaving the
    vendored model's Python metadata preparation untouched.
    """

    def __init__(self, original: nn.Module):
        super().__init__()
        self.original = original
        self._graph = None
        self._static_args = None
        self._static_kwargs = None
        self._static_output = None

    @property
    def layers(self):
        return self.original.layers

    @property
    def config(self):
        return self.original.config

    def _initialize(self, args, kwargs):
        self._static_args = QwenDecoderCudaGraphProxy._allocate(args)
        self._static_kwargs = QwenDecoderCudaGraphProxy._allocate(kwargs)
        QwenDecoderCudaGraphProxy._copy(self._static_args, args)
        QwenDecoderCudaGraphProxy._copy(self._static_kwargs, kwargs)

        # The first warm call also materializes the Siglip2 RoPE cache.
        for _ in range(3):
            self._static_output = self.original(
                *self._static_args, **self._static_kwargs
            )
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            graph.capture_begin()
            self._static_output = self.original(
                *self._static_args, **self._static_kwargs
            )
            graph.capture_end()
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()
        self._graph = graph
        print("[Siglip2Encoder] CUDA Graph 捕获完成")

    def forward(self, *args, **kwargs):
        if self.original.training or not torch.cuda.is_available():
            return self.original(*args, **kwargs)
        if self._graph is None:
            self._initialize(args, kwargs)
        else:
            QwenDecoderCudaGraphProxy._copy(self._static_args, args)
            QwenDecoderCudaGraphProxy._copy(self._static_kwargs, kwargs)
            self._graph.replay()
        return self._static_output


class ActionDitCudaGraphProxy(_DelegatingCudaGraphProxy):
    """Capture the fixed-shape deterministic DiT forward.

    Diffusion noise generation remains outside this module, so every policy
    call still receives fresh noise. Only the tensor-only denoiser is replayed.
    """

    def __init__(self, original: nn.Module):
        super().__init__()
        self.original = original
        self._graph = None
        self._static_args = None
        self._static_kwargs = None
        self._static_output = None
        # Temporal-skip's denoising cache needs the eager/compiled path for
        # selected inner steps. The normal action-head refresh path continues
        # to use this graph.
        self._temporal_skip_force_eager = False
        self._static_condition_signatures = {}
        self._static_condition_sources = {}
        self._graph_mode = None
        self._graph_variants = {}
        self._layout_signature = None

    _STATIC_CONDITION_KEYS = {
        "encoder_hidden_states",
        "encoder_kv_cache",
        "image_mask",
        "backbone_attention_mask",
    }

    def set_temporal_skip_force_eager(self, enabled: bool) -> None:
        self._temporal_skip_force_eager = bool(enabled)

    def _snapshot_graph_state(self):
        return {
            "graph": self._graph,
            "static_args": self._static_args,
            "static_kwargs": self._static_kwargs,
            "static_output": self._static_output,
            "condition_signatures": self._static_condition_signatures,
            "condition_sources": self._static_condition_sources,
        }

    def _restore_graph_state(self, state) -> None:
        self._graph = state["graph"]
        self._static_args = state["static_args"]
        self._static_kwargs = state["static_kwargs"]
        self._static_output = state["static_output"]
        self._static_condition_signatures = state["condition_signatures"]
        self._static_condition_sources = state["condition_sources"]

    def _activate_graph_mode(self, encoder_kv_cache_enabled: bool) -> None:
        mode = bool(encoder_kv_cache_enabled)
        if self._graph_mode == mode:
            return
        if self._graph_mode is not None and self._graph is not None:
            self._graph_variants[self._graph_mode] = self._snapshot_graph_state()

        state = self._graph_variants.get(mode)
        if state is None:
            self._graph = None
            self._static_args = None
            self._static_kwargs = None
            self._static_output = None
            self._static_condition_signatures = {}
            self._static_condition_sources = {}
        else:
            self._restore_graph_state(state)
        self._graph_mode = mode

    def _reset_graph_variants(self) -> None:
        self._graph = None
        self._static_args = None
        self._static_kwargs = None
        self._static_output = None
        self._static_condition_signatures = {}
        self._static_condition_sources = {}
        self._graph_mode = None
        self._graph_variants = {}

    @staticmethod
    def _input_layout_signature(kwargs):
        return tuple(
            (
                key,
                QwenLayerCudaGraphProxy._source_signature(kwargs.get(key)),
            )
            for key in ("image_mask", "backbone_attention_mask")
        )

    @staticmethod
    def _condition_signature(key, value):
        if key == "encoder_kv_cache":
            # The cache builder returns a fresh immutable tuple on rebuild and
            # never mutates tensors in a live tuple. Its identity is therefore
            # a sufficient generation token and avoids walking 32 tensors on
            # every denoising step.
            return ("encoder_kv_cache", id(value))
        return QwenLayerCudaGraphProxy._source_signature(value)

    def _initialize(self, args, kwargs):
        self._static_args = QwenDecoderCudaGraphProxy._allocate(args)
        self._static_kwargs = QwenDecoderCudaGraphProxy._allocate(kwargs)
        QwenDecoderCudaGraphProxy._copy(self._static_args, args)
        QwenDecoderCudaGraphProxy._copy(self._static_kwargs, kwargs)
        self._static_condition_signatures = {
            key: self._condition_signature(key, kwargs[key])
            for key in self._STATIC_CONDITION_KEYS
            if key in kwargs
        }
        self._static_condition_sources = {
            key: kwargs[key]
            for key in self._STATIC_CONDITION_KEYS
            if key in kwargs
        }
        for _ in range(3):
            self._static_output = self.original(
                *self._static_args, **self._static_kwargs
            )
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            graph.capture_begin()
            self._static_output = self.original(
                *self._static_args, **self._static_kwargs
            )
            graph.capture_end()
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()
        self._graph = graph
        mode_name = "encoder-kv" if self._graph_mode else "project-kv"
        print(f"[ActionDiT] CUDA Graph 捕获完成 ({mode_name})")

    def _copy_replay_kwargs(self, kwargs):
        if self._static_kwargs.keys() != kwargs.keys():
            raise ValueError("CUDA Graph input dictionary keys changed")
        for key, source in kwargs.items():
            if key not in self._STATIC_CONDITION_KEYS:
                QwenDecoderCudaGraphProxy._copy(self._static_kwargs[key], source)
                continue

            if (
                key == "encoder_kv_cache"
                and self._static_condition_sources.get(key) is source
            ):
                continue

            source_signature = self._condition_signature(key, source)
            if self._static_condition_signatures.get(key) == source_signature:
                continue
            QwenDecoderCudaGraphProxy._copy(self._static_kwargs[key], source)
            self._static_condition_signatures[key] = source_signature
            self._static_condition_sources[key] = source

    def forward(self, *args, **kwargs):
        if (
            self._temporal_skip_force_eager
            or self.original.training
            or not torch.cuda.is_available()
        ):
            return self.original(*args, **kwargs)
        layout_signature = self._input_layout_signature(kwargs)
        if self._layout_signature is not None and layout_signature != self._layout_signature:
            self._reset_graph_variants()
        self._layout_signature = layout_signature
        self._activate_graph_mode(kwargs.get("encoder_kv_cache") is not None)
        if self._graph is None:
            self._initialize(args, kwargs)
        else:
            QwenDecoderCudaGraphProxy._copy(self._static_args, args)
            self._copy_replay_kwargs(kwargs)
            self._graph.replay()
        return self._static_output


class _PackedQKVCore:
    """Inference-only shared state for three projection slice modules."""

    def __init__(self, q_proj: nn.Linear, k_proj: nn.Linear, v_proj: nn.Linear):
        weights = (
            q_proj.weight.detach(),
            k_proj.weight.detach(),
            v_proj.weight.detach(),
        )
        self.weight = torch.cat(weights, dim=0).contiguous()
        if q_proj.bias is None:
            self.bias = None
        else:
            self.bias = torch.cat(
                (q_proj.bias.detach(), k_proj.bias.detach(), v_proj.bias.detach()),
                dim=0,
            ).contiguous()
        self.widths = tuple(int(weight.shape[0]) for weight in weights)
        self.offsets = (0, self.widths[0], self.widths[0] + self.widths[1])
        self._input_id = None
        self._output = None

    def get_slice(self, hidden_states: torch.Tensor, index: int) -> torch.Tensor:
        if index == 0 or self._output is None or self._input_id != id(hidden_states):
            self._output = F.linear(hidden_states, self.weight, self.bias)
            self._input_id = id(hidden_states)
        start = self.offsets[index]
        output = self._output[..., start : start + self.widths[index]]
        if index == 2:
            self._output = None
            self._input_id = None
        return output


class _PackedQKVSlice(nn.Module):
    def __init__(self, core: _PackedQKVCore, index: int):
        super().__init__()
        # Packed weights are created after model.to() and shared by Q/K/V.
        object.__setattr__(self, "_core", core)
        self.index = index

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._core.get_slice(hidden_states, self.index)


def _install_packed_qkv(attention: nn.Module) -> None:
    core = _PackedQKVCore(attention.q_proj, attention.k_proj, attention.v_proj)
    attention.q_proj = _PackedQKVSlice(core, 0)
    attention.k_proj = _PackedQKVSlice(core, 1)
    attention.v_proj = _PackedQKVSlice(core, 2)


def install_siglip_fc2_residual(model: nn.Module) -> int:
    """Hot-install only the profitable Siglip2 FC2 epilogue fusion."""
    vision_layers = model.backbone.model.vision_model.vision_model.encoder.layers
    count = 0
    for index, layer in enumerate(vision_layers):
        if isinstance(layer, FusedSiglip2Fc2ResidualLayer):
            continue
        vision_layers[index] = FusedSiglip2Fc2ResidualLayer(layer)
        count += 1
    return count


def install_static_qwen_attention(model: nn.Module) -> int:
    """Remove the repeated packed-sequence probe from batch-1 Qwen attention."""
    decoder = model.backbone.model.language_model.model
    count = 0
    for layer in decoder.layers:
        if isinstance(layer.self_attn, StaticQwenAttentionProxy):
            continue
        layer.self_attn = StaticQwenAttentionProxy(layer.self_attn)
        count += 1
    return count


def install_qwen_cuda_graph(model: nn.Module) -> int:
    """Wrap the Qwen decoder in a fixed-buffer CUDA Graph runtime."""
    language_model = model.backbone.model.language_model
    if isinstance(language_model.model, QwenDecoderCudaGraphProxy):
        return 0
    original = language_model.model
    proxy = QwenDecoderCudaGraphProxy(original)
    proxy.train(original.training)
    language_model.model = proxy
    return 1


def install_qwen_boundary_cuda_graphs(model: nn.Module) -> int:
    """Capture the first and last Qwen layers for temporal head/tail reuse."""
    decoder = model.backbone.model.language_model.model
    if isinstance(decoder, QwenDecoderCudaGraphProxy):
        # A whole-decoder graph and boundary graphs cannot be nested safely.
        return 0
    layers = decoder.layers
    if not layers:
        return 0
    boundary_indices = {0, len(layers) - 1}
    installed = 0
    for index in sorted(boundary_indices):
        if isinstance(layers[index], QwenLayerCudaGraphProxy):
            continue
        original = layers[index]
        proxy = QwenLayerCudaGraphProxy(original, layer_index=index)
        proxy.train(original.training)
        layers[index] = proxy
        installed += 1
    if len(layers) > 1 and not hasattr(decoder, "_temporal_skip_qwen_suffix_graph"):
        decoder._temporal_skip_qwen_suffix_graph = QwenSuffixCudaGraphProxy(
            layers[1:]
        )
    return installed


def install_siglip_encoder_cuda_graph(model: nn.Module) -> int:
    """Wrap only the fixed-shape Siglip2 encoder in a CUDA Graph runtime."""
    vision_transformer = model.backbone.model.vision_model.vision_model
    if isinstance(vision_transformer.encoder, SiglipEncoderCudaGraphProxy):
        return 0
    original = vision_transformer.encoder
    proxy = SiglipEncoderCudaGraphProxy(original)
    proxy.train(original.training)
    vision_transformer.encoder = proxy
    return 1


def install_action_dit_cuda_graph(model: nn.Module) -> int:
    """Wrap the action head's deterministic DiT forward in a CUDA Graph."""
    action_head = model.action_head
    if isinstance(action_head.model, ActionDitCudaGraphProxy):
        return 0
    original = action_head.model
    proxy = ActionDitCudaGraphProxy(original)
    proxy.train(original.training)
    action_head.model = proxy
    return 1


def install_realtime_eagle_kernels(
    model: nn.Module,
    *,
    fuse_gate_up: bool = False,
    fuse_qkv: bool = False,
    fuse_siglip_fc2_residual: bool = False,
    static_qwen_attention: bool = False,
    cuda_graph_qwen: bool = False,
    cuda_graph_vision: bool = False,
) -> dict[str, int | bool]:
    """Install optimized inference modules into a loaded GR00T model."""
    if not _TRITON_AVAILABLE:
        return {
            "triton_available": False,
            "rms_norms": 0,
            "mlps": 0,
            "qkv_projections": 0,
            "siglip_fc2_residuals": 0,
            "static_qwen_attentions": 0,
            "qwen_cuda_graphs": 0,
            "vision_cuda_graphs": 0,
            "fused_gate_up": False,
        }

    decoder = model.backbone.model.language_model.model
    rms_norm_count = 0
    mlp_count = 0
    qkv_count = 0
    siglip_fc2_residual_count = 0
    static_qwen_attention_count = 0
    qwen_cuda_graph_count = 0
    vision_cuda_graph_count = 0
    for layer in decoder.layers:
        layer.input_layernorm = TritonRMSNorm(layer.input_layernorm)
        layer.post_attention_layernorm = TritonRMSNorm(layer.post_attention_layernorm)
        # Keep the 128-wide per-head q/k norms on the Transformers kernel.
        # Restricting custom reduction to the 2048-wide residual path gives a
        # better BF16 speed/error tradeoff.
        if fuse_gate_up:
            layer.mlp = FusedGateUpQwen3MLP(layer.mlp)
            mlp_count += 1
        if fuse_qkv:
            _install_packed_qkv(layer.self_attn)
            qkv_count += 1
        rms_norm_count += 2
    decoder.norm = TritonRMSNorm(decoder.norm)
    rms_norm_count += 1

    vision_layers = model.backbone.model.vision_model.vision_model.encoder.layers
    for layer in vision_layers:
        if fuse_qkv:
            _install_packed_qkv(layer.self_attn)
            qkv_count += 1
    if fuse_siglip_fc2_residual:
        siglip_fc2_residual_count = install_siglip_fc2_residual(model)
    if static_qwen_attention:
        static_qwen_attention_count = install_static_qwen_attention(model)
    if cuda_graph_qwen:
        qwen_cuda_graph_count = install_qwen_cuda_graph(model)
    if cuda_graph_vision:
        vision_cuda_graph_count = install_siglip_encoder_cuda_graph(model)
    return {
        "triton_available": True,
        "rms_norms": rms_norm_count,
        "mlps": mlp_count,
        "qkv_projections": qkv_count,
        "siglip_fc2_residuals": siglip_fc2_residual_count,
        "static_qwen_attentions": static_qwen_attention_count,
        "qwen_cuda_graphs": qwen_cuda_graph_count,
        "vision_cuda_graphs": vision_cuda_graph_count,
        "fused_gate_up": fuse_gate_up,
    }
