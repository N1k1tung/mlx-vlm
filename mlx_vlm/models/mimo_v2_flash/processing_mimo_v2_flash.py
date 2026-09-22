"""MiMo text/image processing using the existing Qwen visual patch format."""

import json

from transformers import AutoTokenizer

from ..base import install_auto_processor_patch, load_chat_template
from ..qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
from ..qwen3_vl.processing_qwen3_vl import Qwen3VLImageProcessor, _flatten_images


class MiMoProcessor(Qwen2_5_VLProcessor):
    def __call__(self, images=None, text=None, audio=None, videos=None, **kwargs):
        if audio is not None or videos is not None:
            raise ValueError("MiMo-V2.6 currently supports text and image inputs only")
        texts = [text] if isinstance(text, str) else list(text or [])
        images = _flatten_images(images) if images is not None else []
        count = sum(value.count(self.image_token) for value in texts)
        if count != len(images):
            raise ValueError(f"MiMo prompt contains {count} image markers for {len(images)} images")
        return super().__call__(images=images or None, text=texts, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from ...utils import get_model_path

        path = get_model_path(str(pretrained_model_name_or_path))
        config = json.loads((path / "config.json").read_text())
        vision = config.get("vision_config") or {}
        processor = config.get("processor_config") or {}
        kwargs.pop("use_fast", None)
        kwargs.setdefault("trust_remote_code", True)
        tokenizer = load_chat_template(AutoTokenizer.from_pretrained(path, **kwargs), path)
        return cls(
            tokenizer=tokenizer,
            image_processor=Qwen3VLImageProcessor(
                patch_size=vision.get("patch_size", 16),
                temporal_patch_size=vision.get("temporal_patch_size", 2),
                merge_size=vision.get("spatial_merge_size", 2),
                min_pixels=processor.get("image_min_pixels", 8192),
                max_pixels=processor.get("image_max_pixels", 8388608),
                # MiMo's production processor uses ImageNet normalization;
                # the generic Qwen preprocessor sidecar specifies CLIP instead.
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225],
            ),
            chat_template=tokenizer.chat_template,
        )


install_auto_processor_patch("mimo_v2", MiMoProcessor)
