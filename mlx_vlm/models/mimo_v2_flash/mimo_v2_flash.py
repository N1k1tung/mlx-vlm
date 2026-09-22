from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures, LanguageModelOutput
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config)
        if config.vision_config is not None and config.vision_config.depth:
            self.visual = VisionModel(config.vision_config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ) -> InputEmbeddingsFeatures:
        if any(kwargs.get(key) is not None for key in (
            "audio_codes", "audio_embeds", "input_features", "pixel_values_videos", "video_pixel_values",
        )):
            raise ValueError("MiMo-V2.6 currently supports text and image inputs only")
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        features = kwargs.get("cached_image_features")
        image_mask = input_ids == self.config.image_token_id
        count = int(image_mask.sum().item())
        if pixel_values is not None and features is None:
            features = self.encode_images(pixel_values, **kwargs)
        if features is None:
            if count:
                raise ValueError("MiMo image placeholders require image features")
        else:
            if features.shape != (count, self.config.hidden_size) or not count:
                raise ValueError("MiMo image feature count does not match prompt placeholders")
            indices = mx.maximum(mx.cumsum(image_mask.reshape(-1)) - 1, 0)
            gathered = features[indices].reshape(inputs_embeds.shape).astype(inputs_embeds.dtype)
            inputs_embeds = mx.where(image_mask[..., None], gathered, inputs_embeds)
        return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

    def encode_images(self, pixel_values, **kwargs):
        if not hasattr(self, "visual"):
            raise ValueError("This MiMo checkpoint has no vision encoder")
        grid = kwargs.get("image_grid_thw")
        if grid is None:
            raise ValueError("MiMo images require image_grid_thw")
        return self.visual(pixel_values, grid)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array = None,
        mask: mx.array = None,
        cache=None,
        **kwargs,
    ) -> LanguageModelOutput:
        inputs_embeds = kwargs.pop("inputs_embeds", None)
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids, pixel_values, **kwargs).inputs_embeds
        return self.language_model(input_ids, cache=cache, inputs_embeds=inputs_embeds, mask=mask, **kwargs)

    def sanitize(self, weights):
        language, vision = {}, {}
        for key, value in weights.items():
            if key.startswith(("audio_encoder.", "speech_embeddings.", "model.mtp.")):
                continue
            if key.startswith("visual."):
                if key == "visual.patch_embed.proj.weight" and value.shape[1] == self.config.vision_config.in_chans:
                    value = value.transpose(0, 2, 3, 4, 1)
                vision[key] = value
            else:
                language[key.removeprefix("language_model.")] = value
        language = self.language_model.sanitize(language)
        return {**{f"language_model.{k}": v for k, v in language.items()}, **vision}

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()
