from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from ....models.base import create_attention_mask
from ....models.cache import BatchRotatingKVCache, RotatingKVCache
from ....models.mimo_v2_flash.language import Attention
from ....models.mlp import SwiGLUMLP
from ..qwen3_5_mtp.qwen3_5_mtp import Qwen3_5MTPDraftModel
from .config import MimoV2MTPConfig


class MimoV2MTPDraftModel(Qwen3_5MTPDraftModel):
    """Standalone runtime for MiMo-V2's first native MTP predictor stage."""

    prefer_requested_block_size = False
    default_runtime_block_size = 2
    default_batched_sampling_block_size = 2

    def __init__(self, config: MimoV2MTPConfig):
        nn.Module.__init__(self)
        self.config = config
        text_config = config.text_config
        if text_config is None:
            raise ValueError("MimoV2MTPConfig.text_config must be set")

        self.args = text_config
        hidden_size = text_config.hidden_size
        self.enorm = nn.RMSNorm(hidden_size, eps=text_config.layernorm_epsilon)
        self.hnorm = nn.RMSNorm(hidden_size, eps=text_config.layernorm_epsilon)
        self.eh_proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.input_layernorm = nn.RMSNorm(
            hidden_size, eps=text_config.layernorm_epsilon
        )
        self.self_attn = Attention(text_config, is_sliding_window=True)
        self.pre_mlp_layernorm = nn.RMSNorm(
            hidden_size, eps=text_config.layernorm_epsilon
        )
        self.mlp = SwiGLUMLP(hidden_size, text_config.intermediate_size)
        self.final_layernorm = nn.RMSNorm(
            hidden_size, eps=text_config.layernorm_epsilon
        )

        self._input_embed = None
        self._input_embed_scale = 1.0
        self._lm_head_fn = None
        self._greedy_argmax_fn = None
        self._cache = []
        self._seed_token = None
        self._seed_hidden = None
        self._next_position = 0
        self._round_appended = 0
        self._kv_valid_len = 0
        self._position = 0
        self._draft_round = 0
        self.accept_lens = []
        self.draft_lens = []

    def validate_target_compatibility(self, target_model) -> None:
        target = getattr(target_model, "language_model", target_model)
        target_args = getattr(target, "args", None)
        model_type = getattr(target_args, "model_type", "")
        if model_type not in ("mimo_v2", "mimo_v2_flash"):
            raise ValueError(
                "MiMo-V2 MTP requires a MiMo-V2 target model, "
                f"got model_type={model_type!r}."
            )
        for field in (
            "hidden_size",
            "vocab_size",
            "swa_num_attention_heads",
            "swa_num_key_value_heads",
            "swa_head_dim",
            "swa_v_head_dim",
        ):
            if getattr(target_args, field, None) != getattr(self.args, field, None):
                raise ValueError(f"MiMo-V2 target and MTP {field} do not match.")

    def make_cache(self, left_padding=None) -> List[RotatingKVCache]:
        window = self.args.sliding_window_size
        if left_padding is not None:
            return [BatchRotatingKVCache(window, left_padding)]
        return [RotatingKVCache(max_size=window)]

    def _forward_hidden(
        self,
        token_embed: mx.array,
        hidden: mx.array,
        cache: Optional[List[RotatingKVCache]],
        position_ids: mx.array,
    ) -> mx.array:
        del position_ids
        h = self.eh_proj(
            mx.concatenate([self.enorm(token_embed), self.hnorm(hidden)], axis=-1)
        )
        layer_cache = cache[0] if cache else None
        mask = create_attention_mask(
            h,
            layer_cache,
            window_size=self.args.sliding_window_size,
            return_array=True,
        )
        h = h + self.self_attn(self.input_layernorm(h), mask, layer_cache)
        h = h + self.mlp(self.pre_mlp_layernorm(h))
        return self.final_layernorm(h)

    def draft_block(
        self,
        last_bonus,
        hidden: mx.array,
        cache,
        block_size: int,
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
    ) -> mx.array:
        # Only stage 0 has published runtime semantics; do not recursively use
        # it as the unreleased stage-1/stage-2 predictors.
        return super().draft_block(
            last_bonus,
            hidden,
            cache,
            min(block_size, 2),
            sampler,
            token_dtype,
            greedy,
        )
