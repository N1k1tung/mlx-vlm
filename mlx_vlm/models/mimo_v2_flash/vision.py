"""MiMo ViT with row/column sliding attention and a shared image patch merger."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..mlp import SwiGLUMLP
from ..qwen2_5_vl.vision import PatchEmbed, apply_rotary_pos_emb_vision
from .config import VisionConfig


class Attention(nn.Module):
    def __init__(self, config: VisionConfig, full_attention: bool):
        super().__init__()
        self.head_dim = config.qk_channels
        self.n_heads = config.num_heads
        self.n_kv_heads = config.num_key_value_heads
        self.window = None if full_attention else config.visual_token_window_size
        self.qkv = nn.Linear(
            config.hidden_size, (self.n_heads + 2 * self.n_kv_heads) * self.head_dim,
        )
        self.proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size)
        self.sinks = (
            mx.zeros((self.n_heads,)) if config.use_sink and not full_attention else None
        )

    def __call__(self, x, lengths, freqs):
        q_size, kv_size = self.n_heads * self.head_dim, self.n_kv_heads * self.head_dim
        q, k, v = mx.split(self.qkv(x), [q_size, q_size + kv_size], axis=-1)
        q = q.reshape(1, -1, self.n_heads, self.head_dim)
        k = k.reshape(1, -1, self.n_kv_heads, self.head_dim)
        q = apply_rotary_pos_emb_vision(q.astype(mx.float32), freqs).astype(x.dtype)
        k = apply_rotary_pos_emb_vision(k.astype(mx.float32), freqs).astype(x.dtype)
        q, k = q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3)
        v = v.reshape(1, -1, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        outputs, offset = [], 0
        for length in lengths:
            # Bound local attention masks to a query chunk plus its two halos.
            chunk = length if self.window is None else 128
            for start in range(0, length, chunk):
                end = min(start + chunk, length)
                left = 0 if self.window is None else max(0, start - self.window)
                right = length if self.window is None else min(length, end + self.window)
                mask = None
                if self.window is not None:
                    mask = mx.abs(mx.arange(start, end)[:, None] - mx.arange(left, right)) <= self.window
                out = mx.fast.scaled_dot_product_attention(
                    q[:, :, offset + start:offset + end],
                    k[:, :, offset + left:offset + right],
                    v[:, :, offset + left:offset + right],
                    scale=self.head_dim ** -0.5, mask=mask, sinks=self.sinks,
                )
                outputs.append(out.transpose(0, 2, 1, 3).reshape(end - start, -1))
            offset += length
        return self.proj(mx.concatenate(outputs))


class Block(nn.Module):
    def __init__(self, config, full_attention):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = Attention(config, full_attention)
        self.mlp = SwiGLUMLP(config.hidden_size, config.intermediate_size, bias=True)

    def __call__(self, x, lengths, freqs):
        x = x + self.attn(self.norm1(x), lengths, freqs)
        return x + self.mlp(self.norm2(x))


class PatchMerger(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size * config.spatial_merge_size ** 2
        self.ln_q = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = [
            nn.Linear(width, width, bias=False), nn.GELU(approx="precise"),
            nn.Linear(width, config.out_hidden_size, bias=False),
        ]
        self.width = width

    def __call__(self, x):
        x = self.ln_q(x).reshape(-1, self.width)
        for layer in self.mlp:
            x = layer(x)
        return x


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size, temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_chans, hidden_size=config.hidden_size,
        )
        self.blocks = [
            Block(config, i in config.fullatt_block_indexes) for i in range(config.depth)
        ]
        self.merger = PatchMerger(config)

    def __call__(self, pixel_values, grid_thw):
        config = self.config
        grid = np.asarray(grid_thw)
        merge = config.spatial_merge_size
        if (
            grid.ndim != 2 or grid.shape[1] != 3 or not len(grid)
            or not np.issubdtype(grid.dtype, np.integer) or np.any(grid <= 0)
            or np.any(grid[:, 0] != 1) or np.any(grid[:, 1:] % merge)
        ):
            raise ValueError("MiMo images require integer grids [1, H, W] divisible by the merge size")
        width = config.in_chans * config.temporal_patch_size * config.patch_size ** 2
        if pixel_values.shape != (int(np.prod(grid, axis=1).sum()), width):
            raise ValueError("MiMo image grids do not match the supplied patch dimensions")

        positions, columns, lengths, offset = [], [], [], 0
        for _, h, w in grid.tolist():
            rows, cols = np.indices((h, w))
            positions.append(np.stack([
                axis.reshape(h // merge, merge, w // merge, merge).transpose(0, 2, 1, 3).reshape(-1)
                for axis in (rows, cols)
            ], axis=-1))
            count = h * w // merge ** 2
            columns.append(np.arange(count).reshape(h // merge, w // merge).T.reshape(-1) + offset)
            offset += count
            lengths.append(h * w)
        column_index = mx.array(np.concatenate(columns))
        reverse_index = mx.argsort(column_index)
        inv_freq = 1.0 / (10000.0 ** (mx.arange(0, config.qk_channels // 2, 2) / (config.qk_channels // 2)))
        freqs = (mx.array(np.concatenate(positions))[..., None] * inv_freq).reshape(-1, config.qk_channels // 2)
        unit = merge ** 2
        column_freqs = freqs.reshape(-1, unit, freqs.shape[-1])[column_index].reshape(freqs.shape)
        x = self.patch_embed(pixel_values.astype(self.patch_embed.proj.weight.dtype))
        column_order = False
        for i, block in enumerate(self.blocks):
            use_columns = bool(config.vit_window_attn_types) and config.vit_window_attn_types[i] == 1
            if use_columns != column_order:
                index = column_index if use_columns else reverse_index
                x = x.reshape(-1, unit, x.shape[-1])[index].reshape(x.shape)
                column_order = use_columns
            x = block(x, lengths, column_freqs if use_columns else freqs)
        if column_order:
            x = x.reshape(-1, unit, x.shape[-1])[reverse_index].reshape(x.shape)
        return self.merger(x)
