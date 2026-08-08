# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged recent-window K rotation for streaming-KV position rebase.

When a SinkWindow request's M-RoPE window crosses ``--streaming-kv-rebase-at``,
the runner rotates the recent window's cached K in place by the RoPE angle of
−Δ (Δ = the window's LOWEST effective position − the aligned sink boundary), so
the window's effective positions drop from ``[m, m+span]`` back to
``[start_size, start_size+span]``. Sinks are already bounded and are never
rotated; V carries no positional encoding and is never touched (this module
only ever receives K).

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

Single-rotation invariant, and where it is only near-exact. Config validation
enforces ``rebase_at >= start_size + 2*recent_size``, and the trigger fires on
``window_front + recent_size >= rebase_at``, so consecutive rebase events are
at least ``recent_size`` POSITIONS apart. For a stream whose positions advance
by at most one per token (text, and every measured Qwen3-Omni A/V workload —
positions ≈ 0.056 × tokens), that is at least ``recent_size`` TOKENS of front
advance, i.e. a full window turnover: any cached K entry is rotated at most
once before eviction claims it and the fp32-staged, cache-dtype write-back
rounding never compounds.

Δ is anchored to the window's minimum ``m`` rather than its front ``F`` (so
no alive token can be rebased onto a sink-block position), which costs
``F − m`` of the spacing budget: the strong form needs
``(F − m) <= W * (1 - r)`` with ``W = recent_size`` and ``r`` the mean
positions-per-token rate over the window. Beyond that, the two events'
windows can overlap by a few SEAM tokens, which are then rotated twice. That
is a cost, not a correctness break: uniform-Δ rotations compose exactly
(``R(−Δ₂)∘R(−Δ₁) == R(−(Δ₁+Δ₂))`` in fp32, pinned by the composition test),
so a seam token pays one extra cache-dtype write-back round-trip. The
shipping regime is covered with room to spare — at the minimum legal
``rebase_at`` and the measured ``r``, ``W*(1-r)`` exceeds the observed
``F − m`` by ~an order of magnitude.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv

if TYPE_CHECKING:
    from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
    from vllm.v1.kv_cache_interface import KVCacheSpec
    from vllm.v1.worker.gpu_input_batch import CachedRequestState

logger = init_logger(__name__)


def rebase_delta(base: int, start_size: int) -> int:
    """Rotation amount for a rebase event: Δ = base − start_size.

    ``base`` is the alive window's LOWEST effective position and
    ``start_size`` the boundary it should land on (callers pass the ALIGNED
    sink boundary): subtracting Δ from every window position puts the whole
    window at or above the first position after the pinned sinks. Anchoring on
    the minimum rather than the window front is what keeps multimodal windows
    (whose axes are spread within a token) off the sink positions; see the
    module docstring for the spacing it costs the single-rotation invariant.
    """
    assert base >= start_size, f"rebase with base={base} < start_size={start_size}"
    return base - start_size


def sinkwindow_row_geometry(
    total_kv: int, block_size: int, start_size: int, recent_size: int
) -> tuple[int, int, int, int]:
    """Compacted-row geometry of one SinkWindow request.

    Single source of truth for the ``[sink_blocks ++ recent_TAIL_blocks]``
    mapping: used by the FA metadata builder (``compute_sinkwindow_rows``) to
    rebuild block-table rows, and by the rebase event to address the same
    rows. Both must agree or the rotation lands on the wrong keys.

    Args:
        total_kv: Logical KV length of the request, in tokens.
        block_size: KV cache block size, in tokens.
        start_size: Number of pinned sink tokens at the prefix.
        recent_size: Length of the recent (tail) window, in tokens.

    Returns:
        ``(sink_blocks, tail_start, num_valid_blocks, capped)`` — the pinned
        prefix block count, the first recent-tail block, one past the last
        valid block, and the row's alive token count in compacted-row
        coordinates.
    """
    sink_blocks = cdiv(start_size, block_size)
    recent_blocks = recent_size // block_size
    assert recent_blocks > 0, (
        f"SinkWindow recent window ({recent_size} tokens) is smaller than the "
        f"KV cache block size ({block_size}); it would round down to zero tail "
        "blocks and decode would see only the sink prefix. Set "
        "--streaming-kv-recent-size to at least the block size."
    )
    num_valid_blocks = cdiv(total_kv, block_size)
    tail_start = max(sink_blocks, num_valid_blocks - recent_blocks)
    capped = sink_blocks * block_size + (total_kv - tail_start * block_size)
    return sink_blocks, tail_start, num_valid_blocks, capped


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


def _assert_positions_writable(positions: torch.Tensor) -> None:
    """Refuse to rebase a broadcast position tensor.

    Talker / code2wav stages synthesise positions as
    ``torch.arange(n).unsqueeze(0).expand(3, n)`` — a stride-0 view whose
    elements alias, so an in-place shift raises mid-way. Those requests are
    already excluded by the ``SinkWindowSpec`` gate; this fires BEFORE any K
    is rotated so a gating mistake can never leave K rotated with unshifted
    positions.
    """
    assert 0 not in positions.stride(), (
        "streaming-kv rebase got a stride-0 (expanded) M-RoPE position "
        f"tensor (strides={tuple(positions.stride())}); it cannot be shifted "
        "in place. Only SinkWindowSpec requests may be rebased — talker / "
        "code2wav stages share one broadcast position row."
    )


def apply_rebase_offset(req_state: "CachedRequestState") -> None:
    """Seat A: re-apply the request's persisted rebase offset.

    ``_init_mrope_positions`` re-derives the WHOLE absolute position tensor
    from scratch on every streaming-session chunk append, so a rebase applied
    once would be silently reverted by the next chunk. Call this after every
    recompute: effective positions are always ``raw - mrope_rebase_offset``.
    """
    offset = req_state.mrope_rebase_offset
    if not offset:
        return
    positions = req_state.mrope_positions
    assert positions is not None and req_state.mrope_position_delta is not None
    _assert_positions_writable(positions)
    positions -= offset
    req_state.mrope_position_delta -= offset


def clear_rebase_offset(req_state: "CachedRequestState") -> None:
    """Drop the rebase offset and un-shift the positions with it.

    The offset only ever describes cached K that was rotated in place, so it
    must not outlive that K. Preemption frees EVERY block of the request and
    resets ``num_computed_tokens`` to 0, and SinkWindow groups have prefix
    caching disabled, so the resumed request re-prefills from token 0 into
    fresh blocks: without this, the re-prefill would read the stored tensor
    from index 0, i.e. at ``raw - offset``, and those negative positions index
    ``cos_sin_cache`` from the END silently. Un-shifting (rather than
    asserting) keeps the session alive — the next threshold crossing simply
    rebases again.
    """
    offset = req_state.mrope_rebase_offset
    if not offset:
        return
    positions = req_state.mrope_positions
    assert positions is not None and req_state.mrope_position_delta is not None
    _assert_positions_writable(positions)
    positions += offset
    req_state.mrope_position_delta += offset
    req_state.mrope_rebase_offset = 0


def _effective_position(req_state: "CachedRequestState", token_idx: int) -> int:
    """Largest effective M-RoPE axis value of one token of the request."""
    positions = req_state.mrope_positions
    assert positions is not None
    if token_idx < positions.shape[1]:
        return int(positions[:, token_idx].max())
    # Past the prompt: decode positions are synthesised as delta + index
    # (MRotaryEmbedding.get_next_input_positions_tensor).
    assert req_state.mrope_position_delta is not None
    return req_state.mrope_position_delta + token_idx


def _alive_position_bounds(
    req_state: "CachedRequestState", start: int, end: int
) -> tuple[int, int]:
    """(min, max) effective M-RoPE position over logical tokens ``[start, end)``."""
    positions = req_state.mrope_positions
    assert positions is not None
    num_stored = positions.shape[1]
    bounds: list[int] = []
    stored_end = min(end, num_stored)
    if start < stored_end:
        window = positions[:, start:stored_end]
        bounds += [int(window.min()), int(window.max())]
    if end > num_stored:
        # Decode tail: delta + index, strictly increasing in the index.
        delta = req_state.mrope_position_delta
        assert delta is not None
        bounds += [delta + max(start, num_stored), delta + end - 1]
    assert bounds, f"empty alive range [{start}, {end})"
    return min(bounds), max(bounds)


def maybe_rebase_request(
    req_state: "CachedRequestState",
    *,
    key_caches: Sequence[torch.Tensor],
    block_ids: Sequence[int],
    kv_cache_spec: "KVCacheSpec",
    rebase_at: int,
    rotary: "MRotaryEmbedding",
) -> int:
    """Seat B: rebase one request's positions if its window crossed the wall.

    Rotates the alive recent window's cached K by ``-Δ`` on every layer,
    accumulates ``Δ`` into ``req_state.mrope_rebase_offset`` (Seat A re-applies
    it after each recompute) and shifts the live position state, so the
    session's effective positions restart just after the pinned sinks instead
    of growing past the rotary table's trained range.

    ``Δ`` is measured from the ALIGNED sink boundary
    (``cdiv(start_size, block_size) * block_size``) down to the alive window's
    MINIMUM effective position, so no alive token can be rebased onto a
    sink-block position and none can go negative.

    Args:
        req_state: The request's runner-side state; mutated in place.
        key_caches: K view of every layer in the SinkWindow group
            (``kv_cache.transpose(1, 2).split(head_size, dim=-1)[0]``). Layers
            share positions, so one Δ applies to all of them, but each layer
            owns its own K.
        block_ids: The request's LOGICAL block-table row for that group
            (evicted middle blocks still present as nulls).
        kv_cache_spec: The group's spec. Anything but ``SinkWindowSpec`` is a
            no-op — the gate is the spec, never ``uses_mrope``.
        rebase_at: ``--streaming-kv-rebase-at``.
        rotary: The model's rotary module (supplies theta / rotary_dim /
            pairing).

    Returns:
        The Δ applied, or 0 when nothing was rebased.
    """
    from vllm.v1.kv_cache_interface import SinkWindowSpec

    if not isinstance(kv_cache_spec, SinkWindowSpec):
        return 0
    if req_state.mrope_positions is None:
        return 0

    block_size = kv_cache_spec.block_size
    num_computed = req_state.num_computed_tokens
    if req_state.mrope_rebase_offset:
        # Backstop for every path that discards the rotated K without clearing
        # the offset (see clear_rebase_offset): this step reads the stored
        # tensor from `num_computed` onwards, and a stale offset shows up there
        # as a negative effective position, which cos_sin_cache would index
        # from the end SILENTLY.
        assert (
            _alive_position_bounds(req_state, num_computed, num_computed + 1)[0] >= 0
        ), (
            f"request {req_state.req_id} is about to read M-RoPE position "
            f"{_alive_position_bounds(req_state, num_computed, num_computed + 1)[0]} "
            f"< 0 at token {num_computed}: its rebase offset "
            f"({req_state.mrope_rebase_offset}) outlived the rotated KV it "
            "describes. Every path that drops a request's cached K must call "
            "clear_rebase_offset (preemption resume does)."
        )
    sink_blocks, tail_start, num_valid_blocks, capped = sinkwindow_row_geometry(
        num_computed, block_size, kv_cache_spec.start_size, kv_cache_spec.sliding_window
    )
    aligned_start = sink_blocks * block_size
    window_start = tail_start * block_size
    if capped <= aligned_start:
        # Nothing alive past the sinks yet.
        return 0

    # Cheap O(1) trigger probe: the window FRONT stands in for the whole
    # window, which assumes positions advance by at most one per token (see
    # the module docstring — true for text and for every measured Qwen3-Omni
    # A/V workload). A stream that advances positions FASTER than tokens
    # under-triggers here and trips the post-event assert below instead of
    # corrupting anything; that fail-loud posture is deliberate, the shipping
    # geometry is covered by the validation stage.
    recent_size = kv_cache_spec.sliding_window
    if _effective_position(req_state, window_start) + recent_size < rebase_at:
        return 0

    base, top = _alive_position_bounds(req_state, window_start, num_computed)
    delta = rebase_delta(base, aligned_start)
    if delta == 0:
        return 0
    positions = req_state.mrope_positions
    _assert_positions_writable(positions)

    compacted_ids = list(block_ids[:sink_blocks]) + list(
        block_ids[tail_start:num_valid_blocks]
    )
    for key_cache in key_caches:
        rotate_kv_pages(
            key_cache,
            compacted_ids,
            aligned_start,
            capped,
            delta,
            rotary,
            block_size,
        )

    req_state.mrope_rebase_offset += delta
    positions -= delta
    assert req_state.mrope_position_delta is not None
    req_state.mrope_position_delta -= delta
    assert top - delta < rebase_at, (
        f"rebase left request {req_state.req_id} at effective position "
        f"{top - delta} >= --streaming-kv-rebase-at {rebase_at}: the window "
        f"spans {top - base} positions over {num_computed - window_start} "
        "tokens, i.e. this stream advances MORE than one position per token, "
        "which the O(1) trigger probe (window front + recent_size) "
        "under-estimates. The bound is aligned_start + window span < "
        "rebase_at, so restore the headroom by RAISING "
        "--streaming-kv-rebase-at (more budget) or LOWERING "
        "--streaming-kv-recent-size (shorter window, hence smaller span; it "
        "also lowers the start + 2*recent validator floor)."
    )
    logger.info(
        "[streaming-kv] rebase req=%s delta=%d new_base=%d recent_tokens=%d",
        req_state.req_id,
        delta,
        aligned_start,
        capped - aligned_start,
    )
    return delta


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
