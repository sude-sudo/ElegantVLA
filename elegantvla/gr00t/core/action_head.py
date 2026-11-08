"""Compiled GR00T action-head refresh and residual reuse."""

from __future__ import annotations

from contextlib import nullcontext
import types
from typing import Any

import torch
import torch.nn.functional as F
import torch._inductor.config as _inductor_config

from gr00t.model.modules.dit import BasicTransformerBlock

try:
    from gr00t.model.modules.dit import _sdpa_context
except ImportError:
    def _sdpa_context():
        return nullcontext()


_ACTION_HEAD_REUSE_STRATEGY_ATTR = "_temporal_skip_reuse_strategy"


def _run_transformer_block_with_cache(
    block: BasicTransformerBlock,
    hidden_states: torch.Tensor,
    *,
    encoder_hidden_states: torch.Tensor | None = None,
    encoder_attention_mask: torch.Tensor | None = None,
    temb: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if block.norm_type == "ada_norm":
        norm_hidden_states = block.norm1(hidden_states, temb)
    else:
        norm_hidden_states = block.norm1(hidden_states)
    if block.pos_embed is not None:
        norm_hidden_states = block.pos_embed(norm_hidden_states)
    with _sdpa_context():
        attn_output = block.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask if encoder_hidden_states is not None else None,
        )
    if block.final_dropout:
        attn_output = block.final_dropout(attn_output)
    hidden_states = attn_output + hidden_states
    if hidden_states.ndim == 4:
        hidden_states = hidden_states.squeeze(1)
    norm_hidden_states = block.norm3(hidden_states)
    ff_output = block.ff(norm_hidden_states)
    hidden_states = ff_output + hidden_states
    if hidden_states.ndim == 4:
        hidden_states = hidden_states.squeeze(1)
    return hidden_states, attn_output, ff_output


def _enable_compiled_action_head_cache(
    policy: Any,
    *,
    compile_mode: str = "max-autotune",
) -> None:
    action_head = policy.model.action_head
    model = action_head.model
    if getattr(action_head, "_compiledcached_enabled", False):
        return
    # When Action-DiT CUDA Graph is installed, compile the underlying model
    # and let ActionDitCudaGraphProxy dispatch refresh vs. temporal reuse.
    # Patching the proxy itself would make graph capture and cache replay
    # compete for the same forward method.
    if hasattr(model, "set_temporal_skip_force_eager") and hasattr(model, "original"):
        model = model.original
    original_forward = model.forward
    active = False
    reuse_strategy: str | None = None
    step_index = -1
    cached_reuse_delta: torch.Tensor | None = None
    shared_attention_processors = {}
    for block in model.transformer_blocks:
        attn_module = block.attn1
        processor = attn_module.processor
        if isinstance(processor, torch.nn.Module):
            if any(True for _ in processor.parameters(recurse=True)):
                continue
            if any(True for _ in processor.buffers(recurse=True)):
                continue
        processor_key = (type(processor), getattr(block, "cross_attention_dim", None))
        shared_processor = shared_attention_processors.get(processor_key)
        if shared_processor is None:
            shared_attention_processors[processor_key] = processor
        elif processor is not shared_processor:
            attn_module.set_processor(shared_processor)

    def static_reuse_forward(
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        reuse_delta: torch.Tensor,
    ) -> torch.Tensor:
        temb = model.timestep_encoder(timestep)
        hidden_states = hidden_states.contiguous() + reuse_delta
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        shift, scale = model.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden_states = model.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        return model.proj_out_2(hidden_states)

    def refresh_capture_forward(
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        image_mask: torch.Tensor,
        backbone_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        temb = model.timestep_encoder(timestep)
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        image_attention_mask = image_mask & backbone_attention_mask
        non_image_attention_mask = (~image_mask) & backbone_attention_mask
        reuse_delta = hidden_states.new_zeros(hidden_states.shape)

        for block_index, block in enumerate(model.transformer_blocks):
            if block_index % 2 == 1:
                block_encoder_hidden_states = None
                block_encoder_attention_mask = None
            else:
                block_encoder_hidden_states = encoder_hidden_states
                block_encoder_attention_mask = (
                    non_image_attention_mask
                    if block_index % (2 * model.attend_text_every_n_blocks) == 0
                    else image_attention_mask
                )
            hidden_states, attn_output, ff_output = _run_transformer_block_with_cache(
                block,
                hidden_states,
                encoder_hidden_states=block_encoder_hidden_states,
                encoder_attention_mask=block_encoder_attention_mask,
                temb=temb,
            )
            reuse_delta = reuse_delta + attn_output + ff_output

        shift, scale = model.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden_states = model.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        return model.proj_out_2(hidden_states), reuse_delta

    compiled_reuse_forward = torch.compile(
        static_reuse_forward,
        mode=compile_mode,
        fullgraph=True,
    )
    compiled_refresh_forward = torch.compile(
        refresh_capture_forward,
        mode=compile_mode,
        fullgraph=False,
    )
    action_head._compiledcached_enabled = True

    def forward_with_compile_cache(self, *args, **kwargs):
        nonlocal step_index, cached_reuse_delta
        if not active:
            return original_forward(*args, **kwargs)

        # A REFRESH action is handled by the outer CUDA-Graph proxy. Keep its
        # capture/replay path free of the compiled cache bookkeeping.
        if reuse_strategy is None:
            return original_forward(*args, **kwargs)

        hidden_states = kwargs["hidden_states"]
        encoder_hidden_states = kwargs["encoder_hidden_states"]
        timestep = kwargs["timestep"]
        image_mask = kwargs["image_mask"]
        backbone_attention_mask = kwargs["backbone_attention_mask"]
        step_index += 1
        should_refresh = (
            reuse_strategy is None
            or step_index == 0
            or (
                reuse_strategy == "inner_middle_steps"
                and step_index == action_head.num_inference_timesteps - 1
            )
            or cached_reuse_delta is None
        )
        with _inductor_config.patch(
            {
                "fx_graph_cache": False,
                "fx_graph_remote_cache": False,
            }
        ):
            if should_refresh:
                output, reuse_delta = compiled_refresh_forward(
                    hidden_states,
                    encoder_hidden_states,
                    timestep,
                    image_mask,
                    backbone_attention_mask,
                )
                if step_index == 0:
                    cached_reuse_delta = (
                        reuse_delta.clone()
                        if reuse_strategy is None
                        else reuse_delta.detach()
                    )
            else:
                output = compiled_reuse_forward(hidden_states, timestep, cached_reuse_delta)

        return output

    original_get_action_with_features = action_head.get_action_with_features

    def get_action_with_compile_cache(self, *args, **kwargs):
        nonlocal active, reuse_strategy, step_index, cached_reuse_delta
        reuse_strategy = getattr(self, _ACTION_HEAD_REUSE_STRATEGY_ATTR, None)
        active = True
        step_index = -1
        cached_reuse_delta = None
        try:
            if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            return original_get_action_with_features(*args, **kwargs)
        finally:
            active = False

    model.forward = types.MethodType(forward_with_compile_cache, model)
    action_head.get_action_with_features = types.MethodType(
        get_action_with_compile_cache,
        action_head,
    )
