"""Normalize MiMo's TP-interleaved FP8 / packed MXFP4 release weights."""

import mlx.core as mx

from ...fp8 import (
    MLX_MXFP8_QUANTIZATION,
    _quantize_fp8_weight,
    transform_fp8_weights,
)


def split_mimo_qkv_weights(
    weights, config, prefix, *, sliding, target_quantization=None
):
    """Normalize one TP-interleaved MiMo QKV projection in place."""
    weight_key = prefix + "qkv_proj.weight"
    if weight_key not in weights:
        return

    weight = weights.pop(weight_key)
    scale_key = prefix + "qkv_proj.weight_scale_inv"
    scales = weights.pop(scale_key, None)
    tp = config["num_key_value_heads"]
    heads = config["swa_num_attention_heads" if sliding else "num_attention_heads"]
    kv_heads = config[
        "swa_num_key_value_heads" if sliding else "num_key_value_heads"
    ]
    head_dim = config["swa_head_dim" if sliding else "head_dim"]
    value_dim = config["swa_v_head_dim" if sliding else "v_head_dim"]
    if heads % tp or kv_heads % tp:
        raise ValueError("MiMo QKV heads must be divisible by the export TP size")

    sizes = (
        heads // tp * head_dim,
        kv_heads // tp * head_dim,
        kv_heads // tp * value_dim,
    )
    rows = sum(sizes)
    if weight.shape != (tp * rows, config["hidden_size"]):
        raise ValueError(f"Invalid TP-interleaved MiMo QKV weight shape at {prefix}")

    parts = {name: ([], [], []) for name in ("q_proj", "k_proj", "v_proj")}
    if scales is not None:
        scale_rows = (rows + 127) // 128
        expected = (tp * scale_rows, (config["hidden_size"] + 127) // 128)
        if scales.shape != expected:
            raise ValueError(f"Invalid TP-interleaved MiMo QKV scale shape at {prefix}")

    for rank in range(tp):
        shard = weight[rank * rows : (rank + 1) * rows]
        if scales is not None:
            shard_scale = scales[rank * scale_rows : (rank + 1) * scale_rows]
            quantized = _quantize_fp8_weight(
                shard, shard_scale, target_quantization
            )
            packed, native_scales = quantized[:2]
            native_biases = quantized[2] if len(quantized) == 3 else None
        else:
            packed, native_scales, native_biases = shard, None, None
        offset = 0
        for name, size in zip(parts, sizes):
            parts[name][0].append(packed[offset : offset + size])
            if native_scales is not None:
                parts[name][1].append(native_scales[offset : offset + size])
            if native_biases is not None:
                parts[name][2].append(native_biases[offset : offset + size])
            offset += size

    for name, (packed, native_scales, native_biases) in parts.items():
        weights[prefix + name + ".weight"] = mx.concatenate(packed)
        if native_scales:
            weights[prefix + name + ".scales"] = mx.concatenate(native_scales)
        if native_biases:
            weights[prefix + name + ".biases"] = mx.concatenate(native_biases)


def transform_mimo_weights(weights, config):
    # Audio and the older next-token heads are not part of text/image inference.
    weights = {
        k: v for k, v in weights.items()
        if not k.startswith(("model.mtp.", "audio_encoder.", "speech_embeddings."))
    }
    quantization = dict(MLX_MXFP8_QUANTIZATION)
    for layer, sliding in enumerate(config["hybrid_layer_pattern"]):
        prefix = f"model.layers.{layer}.self_attn."
        split_mimo_qkv_weights(weights, config, prefix, sliding=bool(sliding))

    for key in list(weights):
        if not key.endswith(".weight_scale") or ".mlp.experts." not in key:
            continue
        scale = weights.pop(key)
        weight_key = key.removesuffix("_scale")
        weight = weights[weight_key]
        if (
            weight.dtype != mx.uint8 or scale.dtype != mx.uint8
            or weight.ndim != 2 or scale.shape != (weight.shape[0], weight.shape[1] // 16)
            or weight.shape[1] % 16
        ):
            raise ValueError(f"Invalid MiMo MXFP4 expert layout: {weight_key}")
        weights[weight_key] = weight.view(mx.uint32)
        weights[key.removesuffix("weight_scale") + "scales"] = scale
        layer_prefix, expert_suffix = key.split(".experts.")
        projection = expert_suffix.split(".")[1]
        path = f"language_model.{layer_prefix}.switch_mlp.{projection}"
        quantization[path] = {"group_size": 32, "bits": 4, "mode": "mxfp4"}

    weights, _ = transform_fp8_weights(weights, config)
    return weights, quantization
