from typing import Optional

from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import SinusoidalPositionalEmbedding, TimestepEmbedding, Timesteps
import torch
from torch import nn
import torch.nn.functional as F


EncoderKV = tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, bool]
EncoderKVCache = tuple[EncoderKV | None, ...]


def _project_encoder_kv(
    attention: Attention,
    encoder_hidden_states: torch.Tensor,
) -> EncoderKV:
    if attention.norm_cross:
        encoder_hidden_states = attention.norm_encoder_hidden_states(encoder_hidden_states)

    key = attention.to_k(encoder_hidden_states)
    value = attention.to_v(encoder_hidden_states)
    batch_size, sequence_length, inner_dim = key.shape
    head_dim = inner_dim // attention.heads
    key = key.view(batch_size, sequence_length, attention.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, sequence_length, attention.heads, head_dim).transpose(1, 2)
    if attention.norm_k is not None:
        key = attention.norm_k(key)
    return key, value


def _attention_with_projected_encoder_kv(
    attention: Attention,
    hidden_states: torch.Tensor,
    encoder_kv: EncoderKV,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Run diffusers SDPA while reusing already-projected encoder K/V."""
    residual = hidden_states
    if attention.spatial_norm is not None:
        hidden_states = attention.spatial_norm(hidden_states, None)

    input_ndim = hidden_states.ndim
    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
    else:
        batch_size = hidden_states.shape[0]

    key, value = encoder_kv
    sequence_length = key.shape[-2]
    if attention_mask is not None:
        attention_mask = attention.prepare_attention_mask(
            attention_mask,
            sequence_length,
            batch_size,
        )
        attention_mask = attention_mask.view(
            batch_size,
            attention.heads,
            -1,
            attention_mask.shape[-1],
        )

    if attention.group_norm is not None:
        hidden_states = attention.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = attention.to_q(hidden_states)
    head_dim = query.shape[-1] // attention.heads
    query = query.view(batch_size, -1, attention.heads, head_dim).transpose(1, 2)
    if attention.norm_q is not None:
        query = attention.norm_q(query)

    hidden_states = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
    )
    hidden_states = hidden_states.transpose(1, 2).reshape(
        batch_size,
        -1,
        attention.heads * head_dim,
    )
    hidden_states = hidden_states.to(query.dtype)
    hidden_states = attention.to_out[0](hidden_states)
    hidden_states = attention.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(
            batch_size,
            channel,
            height,
            width,
        )
    if attention.residual_connection:
        hidden_states = hidden_states + residual
    return hidden_states / attention.rescale_output_factor


class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim, compute_dtype=torch.float32):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps):
        dtype = next(self.parameters()).dtype
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)
        return timesteps_emb


class AdaLayerNorm(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()
        self.chunk_dim = chunk_dim
        output_dim = embedding_dim * 2
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(
        self,
        x: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.norm_elementwise_affine = norm_elementwise_affine
        self.positional_embeddings = positional_embeddings
        self.num_positional_embeddings = num_positional_embeddings
        self.norm_type = norm_type

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(
                dim, max_seq_length=num_positional_embeddings
            )
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        if final_dropout:
            self.final_dropout = nn.Dropout(dropout)
        else:
            self.final_dropout = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_kv: Optional[EncoderKV] = None,
        temb: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        # 0. Self-Attention
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        if encoder_kv is None:
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=(
                    encoder_attention_mask if encoder_hidden_states is not None else attention_mask
                ),
            )
        else:
            if encoder_hidden_states is None:
                raise ValueError("Projected encoder K/V requires cross-attention hidden states.")
            key, value, *metadata = encoder_kv
            compacted = bool(metadata[0]) if metadata else False
            attn_output = _attention_with_projected_encoder_kv(
                self.attn1,
                norm_hidden_states,
                (key, value),
                None if compacted else encoder_attention_mask,
            )
        if self.final_dropout:
            attn_output = self.final_dropout(attn_output)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states


class DiT(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        interleave_self_attention=False,
        cross_attention_dim: Optional[int] = None,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.gradient_checkpointing = False

        # Timestep encoder
        self.timestep_encoder = TimestepEncoder(
            embedding_dim=self.inner_dim, compute_dtype=self.compute_dtype
        )

        all_blocks = []
        for idx in range(self.config.num_layers):
            use_self_attn = idx % 2 == 1 and interleave_self_attention
            curr_cross_attention_dim = cross_attention_dim if not use_self_attn else None

            all_blocks += [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=self.config.max_num_positional_embeddings,
                    final_dropout=final_dropout,
                    cross_attention_dim=curr_cross_attention_dim,
                )
            ]
        self.transformer_blocks = nn.ModuleList(all_blocks)

        # Output blocks
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, self.output_dim)
        print(
            "Total number of DiT parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        encoder_hidden_states: torch.Tensor,  # Shape: (B, S, D)
        timestep: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False,
    ):
        # Encode timesteps
        temb = self.timestep_encoder(timestep)

        # Process through transformer blocks - single pass through the blocks
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        all_hidden_states = [hidden_states]

        # Process through transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1 and self.config.interleave_self_attention:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            all_hidden_states.append(hidden_states)

        # Output processing
        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states
        else:
            return self.proj_out_2(hidden_states)


class AlternateVLDiT(DiT):
    """
    Alternate Vision-Language DiT that separates image and non-image tokens
    during cross-attention processing.
    """

    def __init__(self, *args, attend_text_every_n_blocks: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.attend_text_every_n_blocks = attend_text_every_n_blocks
        self._compact_mask_index_cache = {}

    @staticmethod
    def _mask_cache_signature(mask: torch.Tensor) -> tuple:
        try:
            version = mask._version
        except RuntimeError:
            version = None
        return (
            id(mask),
            mask.data_ptr(),
            tuple(mask.shape),
            mask.device.type,
            mask.device.index,
            version,
        )

    def _compact_encoder_hidden_states(
        self,
        encoder_hidden_states: torch.Tensor,
        image_mask: Optional[torch.Tensor],
        backbone_attention_mask: Optional[torch.Tensor],
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Split batch-1 VLM memory into the tokens each cross block can attend."""
        if (
            self.training
            or encoder_hidden_states.shape[0] != 1
            or image_mask is None
            or backbone_attention_mask is None
        ):
            return None
        cache_key = (
            self._mask_cache_signature(image_mask),
            self._mask_cache_signature(backbone_attention_mask),
        )
        cached_indices = self._compact_mask_index_cache.get(cache_key)
        if cached_indices is None:
            image_attention_mask = image_mask & backbone_attention_mask
            non_image_attention_mask = (~image_mask) & backbone_attention_mask
            image_indices = torch.where(image_attention_mask[0])[0]
            non_image_indices = torch.where(non_image_attention_mask[0])[0]
            if len(self._compact_mask_index_cache) >= 8:
                self._compact_mask_index_cache.pop(next(iter(self._compact_mask_index_cache)))
            self._compact_mask_index_cache[cache_key] = (image_indices, non_image_indices)
        else:
            image_indices, non_image_indices = cached_indices
        if image_indices.numel() == 0 or non_image_indices.numel() == 0:
            return None
        return (
            encoder_hidden_states.index_select(1, image_indices).contiguous(),
            encoder_hidden_states.index_select(1, non_image_indices).contiguous(),
        )

    def build_encoder_kv_cache(
        self,
        encoder_hidden_states: torch.Tensor,
        image_mask: Optional[torch.Tensor] = None,
        backbone_attention_mask: Optional[torch.Tensor] = None,
    ) -> EncoderKVCache:
        """Project the fixed VLM representation once for every cross-attention block."""
        encoder_hidden_states = encoder_hidden_states.contiguous()
        compacted = self._compact_encoder_hidden_states(
            encoder_hidden_states,
            image_mask,
            backbone_attention_mask,
        )
        values = []
        for block_index, block in enumerate(self.transformer_blocks):
            if block_index % 2:
                values.append(None)
                continue
            selected_hidden_states = encoder_hidden_states
            if compacted is not None:
                image_hidden_states, non_image_hidden_states = compacted
                selected_hidden_states = (
                    non_image_hidden_states
                    if block_index % (2 * self.attend_text_every_n_blocks) == 0
                    else image_hidden_states
                )
            key, value = _project_encoder_kv(block.attn1, selected_hidden_states)
            values.append((key, value, True) if compacted is not None else (key, value))
        return tuple(values)

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        encoder_hidden_states: torch.Tensor,  # Shape: (B, S, D)
        timestep: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False,
        image_mask: Optional[torch.Tensor] = None,
        backbone_attention_mask: Optional[torch.Tensor] = None,
        encoder_kv_cache: Optional[EncoderKVCache] = None,
    ):
        assert image_mask is not None, "Image mask is required"

        # Encode timesteps
        temb = self.timestep_encoder(timestep)

        # Process through transformer blocks - single pass through the blocks
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        # Create attention masks for image and non-image tokens
        # image_mask shape: (B, S) where True indicates image tokens
        # For attention, we need to invert: False means "don't attend to this token"

        image_attention_mask = image_mask & backbone_attention_mask
        non_image_attention_mask = (~image_mask) & backbone_attention_mask
        compacted_encoder_hidden_states = self._compact_encoder_hidden_states(
            encoder_hidden_states,
            image_mask,
            backbone_attention_mask,
        )

        all_hidden_states = [hidden_states]
        assert self.config.interleave_self_attention, "Interleave self attention must be enabled"
        if encoder_kv_cache is not None and len(encoder_kv_cache) != len(self.transformer_blocks):
            raise ValueError(
                "Encoder K/V cache must contain one entry per DiT transformer block."
            )

        # Process through transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1:
                # Self-attention blocks
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                # Cross-attention blocks - alternate between non-image and image tokens
                if idx % (2 * self.attend_text_every_n_blocks) == 0:
                    # Attend to non-image tokens
                    curr_encoder_attention_mask = non_image_attention_mask
                    compact_encoder_hidden_states = (
                        None
                        if compacted_encoder_hidden_states is None
                        else compacted_encoder_hidden_states[1]
                    )
                else:
                    # Attend to image tokens
                    curr_encoder_attention_mask = image_attention_mask
                    compact_encoder_hidden_states = (
                        None
                        if compacted_encoder_hidden_states is None
                        else compacted_encoder_hidden_states[0]
                    )

                encoder_kv = None if encoder_kv_cache is None else encoder_kv_cache[idx]
                encoder_kv_is_compact = encoder_kv is not None and len(encoder_kv) > 2
                if compact_encoder_hidden_states is not None and (
                    encoder_kv is None or encoder_kv_is_compact
                ):
                    curr_encoder_hidden_states = compact_encoder_hidden_states
                    curr_encoder_attention_mask = None
                else:
                    curr_encoder_hidden_states = encoder_hidden_states

                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=curr_encoder_hidden_states,
                    encoder_attention_mask=curr_encoder_attention_mask,
                    encoder_kv=encoder_kv,
                    temb=temb,
                )
            all_hidden_states.append(hidden_states)

        # Output processing
        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states
        else:
            return self.proj_out_2(hidden_states)


class SelfAttentionTransformer(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        interleave_self_attention=False,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.gradient_checkpointing = False

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=self.config.max_num_positional_embeddings,
                    final_dropout=final_dropout,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        print(
            "Total number of SelfAttentionTransformer parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        return_all_hidden_states: bool = False,
    ):
        # Process through transformer blocks - single pass through the blocks
        hidden_states = hidden_states.contiguous()
        all_hidden_states = [hidden_states]

        # Process through transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            hidden_states = block(hidden_states)
            all_hidden_states.append(hidden_states)

        if return_all_hidden_states:
            return hidden_states, all_hidden_states
        else:
            return hidden_states
