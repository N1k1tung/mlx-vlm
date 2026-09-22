from dataclasses import dataclass, field
from typing import List, Optional

from ..base import BaseModelConfig


@dataclass
class VisionConfig(BaseModelConfig):
    depth: int = 0
    hidden_size: int = 1280
    intermediate_size: int = 4608
    out_hidden_size: int = 4096
    num_heads: int = 32
    num_key_value_heads: int = 8
    qk_channels: int = 64
    in_chans: int = 3
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    fullatt_block_indexes: List[int] = field(default_factory=list)
    vit_window_attn_types: List[int] = field(default_factory=list)
    visual_token_window_size: int = 64
    use_sink: bool = True
    rms_norm_eps: float = 1e-6


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str
    num_experts_per_tok: int
    hybrid_layer_pattern: List[int]
    moe_layer_freq: List[int]
    add_swa_attention_sink_bias: bool
    add_full_attention_sink_bias: bool
    sliding_window_size: int
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    n_shared_experts: Optional[int]
    n_routed_experts: Optional[int]
    routed_scaling_factor: Optional[float]
    topk_method: str
    scoring_func: str
    norm_topk_prob: bool
    n_group: int
    topk_group: int
    max_position_embeddings: int
    layernorm_epsilon: float
    rope_theta: float
    swa_rope_theta: float
    swa_num_attention_heads: int
    swa_num_key_value_heads: int
    head_dim: int
    v_head_dim: int
    swa_head_dim: int
    swa_v_head_dim: int
    partial_rotary_factor: float
    attention_value_scale: float = 1.0
    moe_router_dtype: Optional[str] = None
    vision_config: Optional[VisionConfig] = None
    image_token_id: int = 151655

    def __post_init__(self):
        if isinstance(self.vision_config, dict):
            self.vision_config = VisionConfig.from_dict(self.vision_config)
