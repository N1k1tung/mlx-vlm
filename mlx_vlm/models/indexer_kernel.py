"""Fused scoring epilogue for DeepSeek-style sparse indexers.

An indexer scores *every* pool for *every* query before keeping the top few::

    score[b, l, p] = scale * Σ_h w[b, l, h] · relu(g[b, l, h, p])
    g[b, l, h, p]  = q[b, l, h, :] · k[b, p, :]

The GEMM that produces ``g`` is the cheap part: at 128k pools the
``[queries, heads, pools]`` FP32 tile is 2.1 GB per 512-query chunk and
``mx.matmul`` writes it in 2.7 ms on an M5 Ultra.  Doing ``relu``, the
head-weight product and the head sum as separate commands then streams that
same tile through memory three more times: 9.9 ms, ~two thirds of the whole
scoring step.  This kernel reads the tile once and emits only
``[queries, pools]``.
"""

from functools import lru_cache

import mlx.core as mx

_TG = 256

_INDEXER_EPILOGUE_SOURCE = r"""
    uint tid   = thread_position_in_threadgroup.x;
    uint bl    = threadgroup_position_in_grid.y;
    uint stile = threadgroup_position_in_grid.z;

    int Np = meta[0];
    int L  = meta[1];
    int b  = int(bl) / L;
    int l  = int(bl) - b * L;
    uint s = stile * TG + tid;

    threadgroup float wc[NHEADS];
    if (tid < (uint)NHEADS) {
        wc[tid] = weights[(b * L + l) * NHEADS + int(tid)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (s < (uint)Np) {
        // scores are [B, H, L, Np] by default, or [B, L, H, Np] when the
        // indexer keeps queries outside the head axis.
        long base = QUERY_MAJOR
            ? ((long(b) * L + l) * NHEADS) * Np + long(s)
            : ((long(b) * NHEADS) * L + l) * Np + long(s);
        long hstride = QUERY_MAJOR ? long(Np) : long(L) * Np;
        float acc = 0.0f;
        for (int h = 0; h < NHEADS; ++h) {
            acc += max(scores_in[base + long(h) * hstride], 0.0f) * wc[h];
        }
        scores[(b * L + l) * Np + int(s)] = acc * scale[0];
    }
"""


@lru_cache(maxsize=None)
def _indexer_epilogue_kernel(n_heads, query_major):
    return mx.fast.metal_kernel(
        name=f"indexer_epilogue_h{n_heads}{'_qm' if query_major else ''}",
        input_names=["scores_in", "weights", "scale", "meta"],
        output_names=["scores"],
        source=_INDEXER_EPILOGUE_SOURCE,
        ensure_row_contiguous=True,
    )


def indexer_head_reduce_available():
    """Whether :func:`indexer_head_reduce` can run (it has no CPU/grad path)."""
    return mx.default_device() == mx.gpu and mx.metal.is_available()


def indexer_head_reduce(scores, weights, scale, *, query_major=False):
    """``Σ_h weights[..., h] · relu(scores[..., h, :])`` in one pass.

    ``scores`` is the raw FP32 GEMM output, ``[B, H, L, pools]`` or
    ``[B, L, H, pools]`` with ``query_major=True``.  ``weights`` is
    ``[B, L, H]``; the result is ``[B, L, pools]``.
    """

    if scores.ndim != 4 or weights.ndim != 3:
        raise ValueError("expected [B, H, L, pools] scores and [B, L, H] weights")
    batch, axis_one, axis_two, pools = scores.shape
    # [B, H, L, pools] -> heads first;  [B, L, H, pools] -> queries first.
    queries = axis_one if query_major else axis_two
    heads = axis_two if query_major else axis_one
    if weights.shape != (batch, queries, heads):
        raise ValueError("weights must be [B, queries, heads]")
    if not indexer_head_reduce_available():
        raise RuntimeError("the indexer epilogue needs Metal")

    scores = mx.contiguous(scores)
    weights = mx.contiguous(weights.astype(mx.float32))
    kernel = _indexer_epilogue_kernel(heads, bool(query_major))
    (reduced,) = kernel(
        inputs=[
            scores,
            weights,
            mx.array([float(scale)], dtype=mx.float32),
            mx.array([pools, queries], dtype=mx.int32),
        ],
        template=[("NHEADS", heads), ("TG", _TG), ("QUERY_MAJOR", bool(query_major))],
        grid=(_TG, batch * queries, (pools + _TG - 1) // _TG),
        threadgroup=(_TG, 1, 1),
        output_shapes=[(batch, queries, pools)],
        output_dtypes=[mx.float32],
    )
    return reduced


def indexer_dense_scores(q, pooled, weights, scale):
    """Head-major convenience used by LongCat Flash Sparse."""

    return indexer_head_reduce(
        q.astype(mx.float32) @ pooled[:, None].swapaxes(-1, -2).astype(mx.float32),
        weights,
        scale,
    )


def indexer_dense_scores_available(dtype=None, n_heads=None, head_dim=None):
    return indexer_head_reduce_available()
