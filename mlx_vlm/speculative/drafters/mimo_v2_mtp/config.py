import inspect
from dataclasses import dataclass
from typing import Optional

from ....models.base import BaseModelConfig
from ....models.mimo_v2_flash.config import ModelConfig as MimoV2Config


class TextConfig:
    @classmethod
    def from_dict(cls, params: dict):
        return MimoV2Config.from_dict(params)


@dataclass
class MimoV2MTPConfig(BaseModelConfig):
    model_type: str = "mimo_v2_mtp"
    text_config: Optional[TextConfig] = None
    block_size: int = 2
    runtime_block_size: int = 2
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)

    @classmethod
    def from_dict(cls, params: dict) -> "MimoV2MTPConfig":
        sig = inspect.signature(cls).parameters
        return cls(**{key: value for key, value in params.items() if key in sig})

    from_hf_dict = from_dict
