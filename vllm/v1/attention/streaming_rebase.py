# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged recent-window K rotation for streaming-KV position rebase.

When a SinkWindow request's M-RoPE base crosses ``--streaming-kv-rebase-at``,
the runner rotates the recent window's cached K in place by the RoPE angle of
−Δ (Δ = base − start_size), so the window's effective positions drop from
``[base, base+W)`` back to ``[start_size, start_size+W)``. Sinks are already
bounded and are never rotated; V carries no positional encoding and is never
touched (this module only ever receives K).

Uniform-Δ exactness (the proven prior art this module re-expresses): Qwen's
M-RoPE places the shared sequence base on all three T/H/W axes, and every
frequency pair uses the same ``inv_freq[j]`` regardless of which axis drives
it, so adding the SAME Δ to all three axes adds ``Δ·inv_freq[j]`` to every
pair — a single uniform-frequency rotation R(Δ) independent of the
(interleaved) section layout:

    R_uniform(−Δ) ∘ R_mrope(t, h, w)  ==  R_mrope(t−Δ, h−Δ, w−Δ)

Proven for genuine multimodal windows (t != h != w) at the fp32 floor by
``test_mrope_multimodal_delta.py::
test_structural_uniform_delta_equals_axis_shift_video_small_pos`` (commit
638d2092a, branch ``feat/streaming-kv-mrope-reindex-b``); ported to
``tests/v1/attention/test_streaming_rebase.py``.

Single-rotation invariant: config validation enforces
``rebase_at >= start_size + 2*recent_size``, so consecutive rebase events are
at least one full recent window apart and any cached K entry is rotated at
most once before eviction claims it — the fp32-staged, cache-dtype write-back
rounding never compounds.
"""

from typing import TYPE_CHECKING

import torch

from vllm.utils.math_utils import cdiv

if TYPE_CHECKING:
    from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding


def rebase_delta(base: int, start_size: int) -> int:
    """Rotation amount for a rebase event: Δ = base − start_size.

    ``base`` is the recent window's first effective position; subtracting Δ
    from every window position lands the front exactly on ``start_size`` (the
    first position after the pinned sinks).
    """
    assert base >= start_size, f"rebase with base={base} < start_size={start_size}"
    return base - start_size


def _apply_rotation(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, is_neox_style: bool
) -> torch.Tensor:
    """Rotate pairs of ``x`` [..., heads, rot_dim] by per-pair (cos, sin).

    Same pairing and formula as ``ApplyRotaryEmb.forward_static`` (the model's
    own application convention). ``cos``/``sin`` are [rot_dim/2] (uniform) or
    [N, rot_dim/2] (per token) and broadcast over the heads dim.
    """
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    return torch.stack((o1, o2), dim=-1).flatten(-2)


def _uniform_delta_cos_sin(
    rotary: "MRotaryEmbedding", delta: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 (cos, sin) of the uniform −Δ rotation at the module's actual
    frequencies.

    Uses ``_compute_inv_freq(rotary.base)`` rather than ``cos_sin_cache`` (the
    cache is model-dtype, too coarse for fp32 staging). Non-scaled rope only:
    ``MRotaryEmbedding._compute_inv_freq`` forwards its argument to YaRN as the
    SCALING FACTOR when ``scaling_factor`` is set, so a YaRN rotary would get
    silently wrong frequencies here — ``rotate_kv_pages`` rejects it up front.
    """
    inv_freq = rotary._compute_inv_freq(rotary.base).to(
        device=device, dtype=torch.float32
    )
    angles = -float(delta) * inv_freq
    return angles.cos(), angles.sin()


def rotate_kv_pages(
    key_cache: torch.Tensor,
    block_ids: list[int],
    token_start: int,
    token_end: int,
    delta: int,
    rotary: "MRotaryEmbedding",
    block_size: int,
) -> None:
    """In-place: rotate K for tokens [token_start, token_end) of the request's
    logical stream (mapped through ``block_ids`` pages) by RoPE angle of −delta
    on all M-RoPE axes (uniform Δ). fp32 staging, write back cache dtype.
    V untouched (this function only ever receives K).

    Args:
        key_cache: [num_blocks, block_size, num_kv_heads, head_size] K view of
            the paged cache (the FlashAttention packed layout's
            ``kv_cache.transpose(1, 2).split(head_size, dim=-1)[0]``); may be
            non-contiguous, written through in place.
        block_ids: Pages holding the logical stream: token ``t`` lives at
            ``key_cache[block_ids[t // block_size], t % block_size]``.
        token_start: First logical token to rotate; may be mid-block. NOTE for
            compacted rows (``compute_sinkwindow_rows``): sinks are packed as
            WHOLE blocks, so the recent tail starts at
            ``cdiv(start_size, block_size) * block_size`` — pass THAT, not
            ``start_size``; tokens in between are pinned near-prefix tokens.
        token_end: One past the last logical token to rotate (``capped[r]``
            for a compacted row); the last block may be partial.
        delta: Positions shift DOWN by this amount (``rebase_delta`` output).
        rotary: The model's rotary module — supplies inv_freq (theta),
            rotary_dim, head_size and neox/gpt-j pairing. Never hardcoded.
            Scaled-rope (YaRN) rotaries are rejected: see
            ``_uniform_delta_cos_sin``.
        block_size: Tokens per page; must match ``key_cache.shape[1]``.
    """
    assert key_cache.dtype in (torch.bfloat16, torch.float16, torch.float32), (
        f"streaming-kv rebase requires a bf16/fp16 KV cache (fp32 allowed for "
        f"tests), got {key_cache.dtype}; see --streaming-kv-rebase-at dtype "
        "requirements"
    )
    if delta == 0 or token_end <= token_start:
        return
    assert getattr(rotary, "scaling_factor", None) is None, (
        "streaming-kv rebase supports non-scaled rope only: MRotaryEmbedding."
        "_compute_inv_freq dispatches its argument as the YaRN scaling factor, "
        "so scaled-rope delta frequencies would be silently wrong"
    )
    assert key_cache.dim() == 4 and key_cache.shape[1] == block_size, (
        f"expected [num_blocks, {block_size}, kv_heads, head_size] K view, "
        f"got {tuple(key_cache.shape)}"
    )
    assert token_start >= 0 and token_end <= len(block_ids) * block_size
    assert key_cache.shape[-1] == rotary.head_size

    first_block = token_start // block_size
    last_block = cdiv(token_end, block_size)
    ids = torch.as_tensor(
        block_ids[first_block:last_block], dtype=torch.long, device=key_cache.device
    )
    lo = token_start - first_block * block_size
    hi = token_end - first_block * block_size

    # Gather the spanned pages -> view the valid token range -> rotate -> scatter.
    # The gather off a packed-layout K view is NOT contiguous, and flattening it
    # would silently copy (dropping the write-back); force a contiguous staging
    # buffer and use view(), which throws instead of copying.
    pages = key_cache[ids].contiguous()
    flat = pages.view(-1, *key_cache.shape[2:])
    rot_dim = rotary.rotary_dim
    seg = flat[lo:hi, :, :rot_dim].to(torch.float32)
    cos, sin = _uniform_delta_cos_sin(rotary, delta, key_cache.device)
    flat[lo:hi, :, :rot_dim] = _apply_rotation(seg, cos, sin, rotary.is_neox_style).to(
        key_cache.dtype
    )
    key_cache[ids] = pages


def _cos_sin_at(
    rotary: "MRotaryEmbedding", positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 per-token (cos, sin) at ``positions`` ([N] text or [3, N] M-RoPE),
    mirroring ``MRotaryEmbedding.forward_native``'s cache lookup and section
    handling."""
    cache = rotary.cos_sin_cache
    if cache.device != positions.device:
        cache = cache.to(positions.device)
    cos, sin = cache[positions].to(torch.float32).chunk(2, dim=-1)
    if positions.ndim == 2:
        from vllm.model_executor.layers.rotary_embedding.mrope import (
            apply_interleaved_rope,
        )

        assert rotary.mrope_section
        if rotary.mrope_interleaved:
            cos = apply_interleaved_rope(cos, rotary.mrope_section)
            sin = apply_interleaved_rope(sin, rotary.mrope_section)
        else:
            cos = torch.cat(
                [m[i] for i, m in enumerate(cos.split(rotary.mrope_section, dim=-1))],
                dim=-1,
            )
            sin = torch.cat(
                [m[i] for i, m in enumerate(sin.split(rotary.mrope_section, dim=-1))],
                dim=-1,
            )
    return cos, sin


def rotate_flat_reference(
    k: torch.Tensor,
    positions: torch.Tensor,
    delta: int,
    rotary: "MRotaryEmbedding",
) -> torch.Tensor:
    """Non-paged reference for tests: the definitional rebase
    ``R(positions − delta) · R(positions)⁻¹ · k`` via the rotary module's own
    per-position (per-axis) cos/sin — no uniform-Δ shortcut, so paged-vs-flat
    equality re-proves the structural identity against the real module.

    Args:
        k: [N, num_kv_heads, head_size] stored (already rotated) K.
        positions: [N] or [3, N] positions K was written at.
        delta: Positions shift down by this amount.
        rotary: The model's rotary module (``mscale == 1``; the two-step
            inverse/forward through the mscale-baked cache would otherwise
            rescale K).
    """
    assert getattr(rotary, "mscale", 1.0) == 1.0
    new_positions = positions - delta
    assert int(new_positions.min()) >= 0, "rebased positions must stay >= 0"
    rot_dim = rotary.rotary_dim
    cos_old, sin_old = _cos_sin_at(rotary, positions)
    cos_new, sin_new = _cos_sin_at(rotary, new_positions)
    seg = k[..., :rot_dim].to(torch.float32)
    seg = _apply_rotation(seg, cos_old, -sin_old, rotary.is_neox_style)
    seg = _apply_rotation(seg, cos_new, sin_new, rotary.is_neox_style)
    out = k.clone()
    out[..., :rot_dim] = seg.to(k.dtype)
    return out
