from typing import Dict, Optional, Tuple

import mlx.core as mx

from ....fp8 import transform_fp8_weights
from ....models.mimo_v2_flash.checkpoint import split_mimo_qkv_weights
from ..mtp_split import MTPSplitter
from .mimo_v2_mtp import MimoV2MTPDraftModel

_MTP_PREFIX = "model.mtp.layers.0."


class MimoV2MTPSplitter(MTPSplitter):
    output_model_type = "mimo_v2_mtp"
    draft_model_cls = MimoV2MTPDraftModel
    require_text_config = False
    tie_word_embeddings_default = False
    block_size_extra = 1
    tokenizer_files = (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.jinja",
    )

    def read_text_config(self, source_config: dict) -> dict:
        text_config = dict(source_config.get("text_config") or source_config)
        text_config["num_nextn_predict_layers"] = 1
        text_config.pop("quantization", None)
        text_config.pop("quantization_config", None)
        return text_config

    def select_keys(self, key: str, text_config: dict) -> bool:
        del text_config
        return key.startswith(_MTP_PREFIX)

    def rename(
        self, tensors: Dict[str, mx.array], text_config: dict
    ) -> Dict[str, mx.array]:
        del text_config
        return {
            key[len(_MTP_PREFIX) :]: value
            for key, value in tensors.items()
            if key.startswith(_MTP_PREFIX)
        }

    def depth(self, text_config: dict) -> int:
        del text_config
        return 1

    def transform_source_quantization(
        self,
        tensors: Dict[str, mx.array],
        source_config: dict,
        target_quantization: Optional[dict],
    ) -> Tuple[Dict[str, mx.array], Optional[dict]]:
        tensors = dict(tensors)
        split_mimo_qkv_weights(
            tensors,
            source_config,
            _MTP_PREFIX + "self_attn.",
            sliding=True,
            target_quantization=target_quantization,
        )
        return transform_fp8_weights(
            tensors,
            source_config,
            target_quantization=target_quantization,
        )

    def quantization_from_source(self, tensors, source_config):
        if any(key.endswith(".scales") for key in tensors):
            return source_config.get("quantization")
        return None


def split_mimo_v2_mtp(source: str, output: str, **kwargs):
    return MimoV2MTPSplitter().split(source, output, **kwargs)
