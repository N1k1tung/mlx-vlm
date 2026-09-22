"""Small local checks: python -m mlx_vlm.models.mimo_v2_flash.check [--checkpoint PATH]."""

import argparse
import json
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from PIL import Image

from ...speculative.cache_state import commit_speculative_round
from ...speculative.common import verify_forward
from ...speculative.drafters.qwen3_dflash.config import DFlashConfig
from ...speculative.drafters.qwen3_dflash.dflash import DFlashDraftModel
from .checkpoint import transform_mimo_weights
from .config import ModelConfig, VisionConfig
from .mimo_v2_flash import Model
from .processing_mimo_v2_flash import MiMoProcessor
from .vision import VisionModel


def tiny_config():
    return ModelConfig(
        model_type="mimo_v2", num_experts_per_tok=2,
        hybrid_layer_pattern=[0, 1], moe_layer_freq=[0, 1],
        add_swa_attention_sink_bias=True, add_full_attention_sink_bias=False,
        sliding_window_size=4, vocab_size=128, hidden_size=64,
        intermediate_size=128, moe_intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=8, num_key_value_heads=4, n_shared_experts=None,
        n_routed_experts=4, routed_scaling_factor=None, topk_method="noaux_tc",
        scoring_func="sigmoid", norm_topk_prob=True, n_group=1, topk_group=1,
        max_position_embeddings=512, layernorm_epsilon=1e-6,
        rope_theta=10000, swa_rope_theta=10000, swa_num_attention_heads=8,
        swa_num_key_value_heads=4, head_dim=8, v_head_dim=8, swa_head_dim=8,
        swa_v_head_dim=8, partial_rotary_factor=0.5, attention_value_scale=0.707,
        image_token_id=110,
        vision_config=VisionConfig(
            depth=3, hidden_size=32, intermediate_size=64, out_hidden_size=64,
            num_heads=4, num_key_value_heads=2, qk_channels=8, patch_size=2,
            fullatt_block_indexes=[0], vit_window_attn_types=[-1, 0, 1],
            visual_token_window_size=4,
        ),
    )


def check_forward():
    mx.random.seed(4)
    model = Model(tiny_config())
    ids = mx.array([[1, *([110] * 12), 2, 3]])
    media = dict(pixel_values=mx.random.normal((48, 24)), image_grid_thw=mx.array([[1, 4, 6], [1, 6, 4]]))
    embeddings = model.get_input_embeddings(ids, **media).inputs_embeds
    reference = model(ids, **media).logits
    cache = model.make_cache()
    chunks = [model.language_model(ids[:, i:i+3], inputs_embeds=embeddings[:, i:i+3], cache=cache).logits for i in range(0, ids.shape[1], 3)]
    np.testing.assert_allclose(np.array(mx.concatenate(chunks, axis=1)), np.array(reference), atol=2e-4, rtol=2e-4)

    # Reject three tokens after the rotating window is already full.
    block = mx.array([[4, 5, 6, 7]])
    output, transaction = verify_forward(model.language_model, block, cache, capture_layer_ids=[0, 1])
    assert [h.shape for h in output.hidden_states] == [(1, 4, 64)] * 2
    commit_speculative_round(model.language_model, cache, transaction, 0, 4)
    actual = model.language_model(mx.array([[8]]), cache=cache).logits
    expected = model(mx.concatenate([ids, mx.array([[4, 8]])], axis=1), **media).logits[:, -1:]
    np.testing.assert_allclose(np.array(actual), np.array(expected), atol=2e-4, rtol=2e-4)
    try:
        model.get_input_embeddings(ids, pixel_values=media["pixel_values"][:24], image_grid_thw=media["image_grid_thw"][:1])
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched image features were accepted")

    config = DFlashConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=1,
                         num_attention_heads=8, num_key_value_heads=4, head_dim=8,
                         vocab_size=128, mask_token_id=111, target_layer_ids=[0, 1],
                         num_target_layers=2, partial_rotary_factor=0.5,
                         attention_value_scale=0.612, attention_sink_bias=True,
                         layer_types=["sliding_attention"], sliding_window=8, block_size=4)
    draft = DFlashDraftModel(config)
    draft.mask_embedding = mx.ones((64,))
    draft.bind(model)
    assert mx.array_equal(draft._embed_input_tokens(mx.array([[111]])), mx.ones((1, 1, 64))).item()
    logits = draft(mx.array([[2, 111, 111, 111]]), mx.concatenate(output.hidden_states, axis=-1), draft.make_cache())
    assert logits.shape == (1, 4, 128) and mx.all(mx.isfinite(logits)).item()
    print("Text/image prefill, cache rollback, and dFlash forward: OK")


def check_quantization():
    config = tiny_config().to_dict()
    config.update(head_dim=192, v_head_dim=128, swa_head_dim=192, swa_v_head_dim=128,
                  quantization_config={"quant_method": "fp8", "weight_block_size": [128, 128]})
    weights, expected = {}, {}
    for layer in range(2):
        prefix = f"model.layers.{layer}.self_attn."
        sizes = [384, 192, 128]
        rows = sum(sizes)
        scales = 2.0 ** (mx.arange(24).reshape(24, 1) % 5)
        weights[prefix + "qkv_proj.weight"] = mx.full((4 * rows, 64), 56, mx.uint8)  # E4M3 1.0
        weights[prefix + "qkv_proj.weight_scale_inv"] = scales
        shards = [np.repeat(np.array(scales[i*6:(i+1)*6]), 128, axis=0)[:rows].repeat(64, axis=1) for i in range(4)]
        for name, chunks in zip(("q_proj", "k_proj", "v_proj"), zip(*(np.split(s, np.cumsum(sizes)[:-1]) for s in shards))):
            expected[prefix + name] = np.concatenate(chunks)
    expert = "model.layers.1.mlp.experts.0.gate_proj"
    weights[expert + ".weight"] = mx.full((64, 32), 0x21, mx.uint8)
    weights[expert + ".weight_scale"] = mx.full((64, 2), 127, mx.uint8)
    normalized, quantization = transform_mimo_weights(weights, config)
    for prefix, reference in expected.items():
        decoded = mx.dequantize(normalized[prefix + ".weight"], normalized[prefix + ".scales"], group_size=32, bits=8, mode="mxfp8")
        np.testing.assert_array_equal(np.array(decoded.astype(mx.float32)), reference)
    decoded = mx.dequantize(normalized[expert + ".weight"], normalized[expert + ".scales"], group_size=32, bits=4, mode="mxfp4")
    np.testing.assert_array_equal(np.array(decoded.astype(mx.float32)), np.tile([0.5, 1.0], (64, 32)))
    assert quantization["language_model.model.layers.1.mlp.switch_mlp.gate_proj"]["mode"] == "mxfp4"
    print("TP-interleaved FP8 scales and packed MXFP4: OK")


def check_checkpoint(path):
    # Read headers only: never materialize the 173 GB language model.
    with (path / "model_pp0_ep0_shard0.safetensors").open("rb") as f:
        header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    config = ModelConfig.from_dict(json.loads((path / "config.json").read_text()))
    vision = VisionModel(config.vision_config)
    shapes = {"visual." + k: list(v.shape) for k, v in tree_flatten(vision.parameters())}
    actual = {k: v["shape"] for k, v in header.items() if k.startswith("visual.")}
    shape = actual["visual.patch_embed.proj.weight"]
    actual["visual.patch_embed.proj.weight"] = [shape[i] for i in (0, 2, 3, 4, 1)]
    assert shapes == actual, "Vision parameters differ from the released checkpoint"
    processor = MiMoProcessor.from_pretrained(path, trust_remote_code=True)
    from ...prompt_utils import apply_chat_template
    from ...utils import prepare_inputs
    prompt = apply_chat_template(processor, config, "Describe the image.", num_images=1)
    inputs = prepare_inputs(processor, prompts=prompt, images=[Image.new("RGB", (64, 64), "red")])
    count = int((inputs["input_ids"] == config.image_token_id).sum().item())
    assert count == inputs["pixel_values"].shape[0] // 4
    draft_config = DFlashConfig.from_dict(json.loads((path / "dflash/config.json").read_text()))
    draft = DFlashDraftModel(draft_config)
    draft.model_path = path / "dflash"
    draft._load_mask_embedding()
    assert draft.mask_embedding.shape == (config.hidden_size,)
    print("Released vision weight shapes, image preparation, and draft mask: OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    check_quantization()
    check_forward()
    if args.checkpoint:
        check_checkpoint(args.checkpoint)


if __name__ == "__main__":
    main()
