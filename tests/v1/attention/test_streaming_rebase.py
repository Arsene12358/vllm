# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the streaming-KV paged K rotation module (position rebase).

Test groups:
  (a) paged walk vs the non-paged flat reference on synthetic layouts
      (multiple shuffled pages, partial last block, token_start mid-block,
      gpt-j style), triangulated against the model's own ``forward_native``;
  (b) uniform-Δ multimodal exactness, ported from the prior-art suite
      ``tests/streaming_kv/test_mrope_multimodal_delta.py`` (206 lines, commit
      638d2092a on branch ``feat/streaming-kv-mrope-reindex-b``) — every
      numeric assertion and threshold kept, entry point swapped to
      ``rotate_kv_pages`` over a real packed paged cache;
  (c) rotate-then-rotate composition == single rotation of the summed Δ
      (unitarity, fp32);
  (d) sinks untouched — ``token_start`` honors S, ``rebase_delta`` arithmetic;
  (e) bf16 write-back single-rotation error bound (≤ 2x the measured bf16
      storage floor, the numerical content of the single-rotation invariant).

Prior-art sign convention: the old kernels rotated positions BY +δ (callers
passed δ = −50). ``rotate_kv_pages(delta)`` rotates by −Δ (positions shift
DOWN by Δ = base − start_size), so ported call sites use ``delta = −δ_old``;
targets (``pos + δ_old`` == ``pos − delta``) are unchanged.
"""

import contextlib

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
from vllm.v1.attention.streaming_rebase import (
    rebase_delta,
    rotate_flat_reference,
    rotate_kv_pages,
)

# Qwen3-Omni Thinker text rope config (as in the prior-art suite).
HEAD_DIM = 128
ROTARY_DIM = 128
MAX_POS = 65536
ROPE_THETA = 1_000_000.0
MROPE_SECTION = [24, 20, 20]
NUM_KV_HEADS = 2
BLOCK_SIZE = 16


@contextlib.contextmanager
def _config_ctx():
    # MRotaryEmbedding is a CustomOp; its __init__ reads the current vLLM
    # compilation config to pick a forward impl. Mirrors tests/conftest.py's
    # `default_vllm_config` fixture.
    with set_current_vllm_config(VllmConfig()):
        yield


def _make_rope(is_neox_style: bool = True) -> MRotaryEmbedding:
    return MRotaryEmbedding(
        head_size=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        max_position_embeddings=MAX_POS,
        base=ROPE_THETA,
        is_neox_style=is_neox_style,
        dtype=torch.float32,
        mrope_section=MROPE_SECTION,
        mrope_interleaved=True,
    )


def _video_positions(base0: int) -> torch.Tensor:
    """Genuine video positions: shared per-token base + intra-frame grid
    offsets, so t != h != w on every token. [3, N]."""
    base = base0 + torch.tensor([0, 3, 6, 50, 55, 60])
    t = base + torch.tensor([0, 0, 0, 1, 1, 1])  # temporal: per-frame
    h = base + torch.tensor([0, 1, 2, 0, 1, 2])  # height: grid row
    w = base + torch.tensor([0, 2, 4, 0, 2, 4])  # width: grid col
    pos = torch.stack([t, h, w]).to(torch.long)  # [3, N]
    # sanity: these are real multimodal positions, not degenerate text ones.
    assert not torch.equal(pos[0], pos[1]) and not torch.equal(pos[1], pos[2])
    return pos


def _video_positions_n(base0: int, n: int) -> torch.Tensor:
    """Like `_video_positions` but for arbitrary N (synthetic layouts)."""
    idx = torch.arange(n)
    base = base0 + idx * 3
    t = base + idx % 4
    h = base + idx % 5
    w = base + idx % 7
    pos = torch.stack([t, h, w]).to(torch.long)
    assert not torch.equal(pos[0], pos[1]) and not torch.equal(pos[1], pos[2])
    return pos


def _rotate_k(rope, pos, k_raw):
    _, k = rope.forward_native(pos, torch.randn_like(k_raw), k_raw.clone())
    return k.view(-1, NUM_KV_HEADS, HEAD_DIM)


# --------------------------------------------------------------------------
# Paged-cache harness: the exact v0.26.0 FlashAttention packed KV layout.
# --------------------------------------------------------------------------
def _make_packed_cache(num_blocks: int, dtype: torch.dtype):
    """Packed KV tensor (num_blocks, H, block_size, 2*D) and the K/V views
    the impl derives from it (flash_attn.py forward:
    ``kv_cache.transpose(1, 2).split(head_size, dim=-1)``)."""
    kv = torch.randn(
        num_blocks, NUM_KV_HEADS, BLOCK_SIZE, 2 * HEAD_DIM, dtype=torch.float32
    ).to(dtype)
    key_cache, value_cache = kv.transpose(1, 2).split(HEAD_DIM, dim=-1)
    return kv, key_cache, value_cache


def _scatter_tokens(key_cache, block_ids, token_start, k):
    for i in range(k.shape[0]):
        t = token_start + i
        key_cache[block_ids[t // BLOCK_SIZE], t % BLOCK_SIZE] = k[i].to(key_cache.dtype)


def _gather_tokens(key_cache, block_ids, token_start, token_end):
    rows = [
        key_cache[block_ids[t // BLOCK_SIZE], t % BLOCK_SIZE]
        for t in range(token_start, token_end)
    ]
    return torch.stack(rows)


def _paged_rotate(k_stored, delta, rope, dtype=torch.float32):
    """Round-trip `k_stored` [N, H, D] through a fresh packed paged cache and
    `rotate_kv_pages`; returns the rotated tokens as a flat [N, H, D]."""
    n = k_stored.shape[0]
    num_blocks = -(-n // BLOCK_SIZE) + 2
    _, key_cache, _ = _make_packed_cache(num_blocks, dtype)
    block_ids = list(range(num_blocks))
    _scatter_tokens(key_cache, block_ids, 0, k_stored)
    rotate_kv_pages(key_cache, block_ids, 0, n, delta, rope, BLOCK_SIZE)
    return _gather_tokens(key_cache, block_ids, 0, n)


# --------------------------------------------------------------------------
# (a) Paged walk vs flat reference on synthetic layouts.
# --------------------------------------------------------------------------
def test_paged_matches_reference_multi_page_partial_tail_video():
    """Shuffled pages + partial last block, real video positions. The paged
    fast path must match both the definitional flat reference and the model's
    own forward_native at the rebased positions (triangle closed)."""
    torch.manual_seed(10)
    with _config_ctx():
        rope = _make_rope()
        n, delta = 45, 37  # 3 blocks: full, full, partial (13 tokens)
        pos = _video_positions_n(200, n)
        k_raw = torch.randn(n, NUM_KV_HEADS * HEAD_DIM, dtype=torch.float32)
        k_stored = _rotate_k(rope, pos, k_raw)

        kv, key_cache, value_cache = _make_packed_cache(10, torch.float32)
        kv_before = kv.clone()
        block_ids = [5, 2, 7]  # shuffled, non-contiguous pages
        _scatter_tokens(key_cache, block_ids, 0, k_stored)
        kv_stored = kv.clone()

        rotate_kv_pages(key_cache, block_ids, 0, n, delta, rope, BLOCK_SIZE)

        got = _gather_tokens(key_cache, block_ids, 0, n)
        want = rotate_flat_reference(k_stored, pos, delta, rope)
        err_ref = (got - want).abs().max().item()
        want_native = _rotate_k(rope, pos - delta, k_raw)
        err_native = (got - want_native).abs().max().item()

    print(f"[paged] vs reference={err_ref:.3e} vs forward_native={err_native:.3e}")
    assert err_ref < 1e-4, f"paged != flat reference; max abs err={err_ref}"
    assert err_native < 1e-4, f"paged != forward_native target; err={err_native}"
    # V cache is never touched: the whole V half is bitwise intact.
    assert torch.equal(value_cache, kv_before.transpose(1, 2)[..., HEAD_DIM:])
    # Pages not in block_ids are bitwise intact.
    untouched = [b for b in range(10) if b not in block_ids]
    assert torch.equal(kv[untouched], kv_stored[untouched])
    # Unwritten tail slots of the partial last block are bitwise intact.
    assert torch.equal(
        key_cache[block_ids[-1], n % BLOCK_SIZE :],
        kv_stored.transpose(1, 2)[block_ids[-1], n % BLOCK_SIZE :, :, :HEAD_DIM],
    )


def test_paged_token_start_mid_block():
    """token_start lands mid-block: everything below it is bitwise intact,
    everything in [token_start, token_end) matches the flat reference."""
    torch.manual_seed(11)
    with _config_ctx():
        rope = _make_rope()
        n, start, delta = 45, 21, 12
        pos = _video_positions_n(150, n)
        k_stored = _rotate_k(
            rope, pos, torch.randn(n, NUM_KV_HEADS * HEAD_DIM, dtype=torch.float32)
        )
        _, key_cache, _ = _make_packed_cache(6, torch.float32)
        block_ids = [4, 0, 3]
        _scatter_tokens(key_cache, block_ids, 0, k_stored)
        before = _gather_tokens(key_cache, block_ids, 0, n)

        rotate_kv_pages(key_cache, block_ids, start, n, delta, rope, BLOCK_SIZE)

        got = _gather_tokens(key_cache, block_ids, 0, n)
        want = rotate_flat_reference(k_stored[start:], pos[:, start:], delta, rope)
    assert torch.equal(got[:start], before[:start])
    err = (got[start:] - want).abs().max().item()
    assert err < 1e-4, f"mid-block rotated range != reference; err={err}"


def test_paged_matches_reference_gptj_style_text():
    """Non-neox (gpt-j interleaved-pair) style is honored via the rotary
    module's own is_neox_style, text (1-D) positions."""
    torch.manual_seed(12)
    with _config_ctx():
        rope = MRotaryEmbedding(
            head_size=HEAD_DIM,
            rotary_dim=ROTARY_DIM,
            max_position_embeddings=MAX_POS,
            base=ROPE_THETA,
            is_neox_style=False,
            dtype=torch.float32,
        )
        n, delta = 20, 9
        pos = torch.arange(100, 100 + n, dtype=torch.long)
        k_raw = torch.randn(n, NUM_KV_HEADS * HEAD_DIM, dtype=torch.float32)
        k_stored = _rotate_k(rope, pos, k_raw)

        got = _paged_rotate(k_stored, delta, rope)
        want = rotate_flat_reference(k_stored, pos, delta, rope)
        err_ref = (got - want).abs().max().item()
        err_native = (got - _rotate_k(rope, pos - delta, k_raw)).abs().max().item()
    assert err_ref < 1e-4, f"gptj paged != reference; err={err_ref}"
    assert err_native < 1e-4, f"gptj paged != forward_native; err={err_native}"


# --------------------------------------------------------------------------
# (b) Ported exactness suite (every assertion/threshold kept; entry point
# swapped to rotate_kv_pages through the packed paged cache).
# --------------------------------------------------------------------------
def test_structural_uniform_delta_equals_axis_shift_video_small_pos():
    """rotate(R_mrope(t,h,w)·k, Δ)  ==  R_mrope(t−Δ,h−Δ,w−Δ)·k  for t!=h!=w.
    Small positions => fp32 cos/sin exact => isolates the structural identity."""
    torch.manual_seed(0)
    with _config_ctx():
        rope = _make_rope()
        pos = _video_positions(base0=300)
        n = pos.shape[1]
        delta = 50  # prior art: δ_old = −50 with the shift-by-+δ kernel
        k_raw = torch.randn(n, NUM_KV_HEADS * HEAD_DIM, dtype=torch.float32)

        k_stored = _rotate_k(rope, pos, k_raw)
        k_rerot = _paged_rotate(k_stored, delta, rope)
        k_target = _rotate_k(rope, pos - delta, k_raw)

    err = (k_rerot - k_target).abs().max().item()
    print(f"[structural] video t!=h!=w, small pos: max abs err = {err:.3e}")
    assert err < 1e-4, f"uniform Δ != per-axis shift for video; max abs err={err}"


def test_structural_relative_preserved_text_query_video_key_small_pos():
    """A text query attending to a recent *video* key gets the SAME score
    after the rebase rotation as the unclamped baseline. Small positions."""
    torch.manual_seed(1)
    with _config_ctx():
        rope = _make_rope()
        pos_k = _video_positions(base0=300)
        n = pos_k.shape[1]
        q_true, safe_bound = 400, 350
        delta = q_true - safe_bound  # 50 (prior art: δ_old = −50)

        k_raw = torch.randn(n, NUM_KV_HEADS * HEAD_DIM, dtype=torch.float32)
        q_raw = torch.randn(1, NUM_KV_HEADS * HEAD_DIM, dtype=torch.float32)

        def q_at(p):
            row = torch.tensor([p])
            return _rotate_k(rope, torch.stack([row, row, row]), q_raw)

        k_base = _rotate_k(rope, pos_k, k_raw)
        score_base = torch.einsum("qhd,khd->hqk", q_at(q_true), k_base)
        k_bnd = _paged_rotate(k_base, delta, rope)
        score_bnd = torch.einsum("qhd,khd->hqk", q_at(safe_bound), k_bnd)

    err = (score_base - score_bnd).abs().max().item()
    print(f"[structural] relative score (text q / video k): max abs err = {err:.3e}")
    assert err < 1e-3, f"rebased relative != baseline; max abs err={err}"


def test_production_scale_rebased_not_worse_than_baseline():
    """Ground truth = same relatives computed at small positions (exact fp32).
    Compare baseline (true large pos) and rebased (clamp + re-rotate) against
    it. Assert the rebased deviation is no worse than the baseline's."""
    torch.manual_seed(2)
    with _config_ctx():
        rope = _make_rope()
        q_true, safe_bound = 64000, 63488
        delta = q_true - safe_bound  # 512 (prior art: δ_old = −512)
        base0 = 60000
        n = 6
        k_raw = torch.randn(n, NUM_KV_HEADS * HEAD_DIM, dtype=torch.float32)
        q_raw = torch.randn(1, NUM_KV_HEADS * HEAD_DIM, dtype=torch.float32)

        def q_at(p):
            row = torch.tensor([p])
            return _rotate_k(rope, torch.stack([row, row, row]), q_raw)

        # Ground truth: shift everything down by a constant c (relatives are
        # invariant), into the precise small-number regime.
        c = base0 - 200
        pos_small = _video_positions(base0=base0 - c)  # keys near pos ~200
        k_small = _rotate_k(rope, pos_small, k_raw)
        score_truth = torch.einsum("qhd,khd->hqk", q_at(q_true - c), k_small)

        # Baseline at true large positions (fp32 imprecise).
        pos_large = _video_positions(base0=base0)
        k_large = _rotate_k(rope, pos_large, k_raw)
        score_base = torch.einsum("qhd,khd->hqk", q_at(q_true), k_large)

        # Rebased: query clamped at safe_bound, keys re-rotated by −Δ.
        k_bnd = _paged_rotate(k_large, delta, rope)
        score_bnd = torch.einsum("qhd,khd->hqk", q_at(safe_bound), k_bnd)

    err_base = (score_base - score_truth).abs().max().item()
    err_bnd = (score_bnd - score_truth).abs().max().item()
    print(
        f"[prod-scale] fp32 err vs truth: baseline={err_base:.3e} "
        f"rebased={err_bnd:.3e}  (both are shared large-angle fp32 error)"
    )
    # Rebased must not be materially worse than baseline's own fp32 error.
    assert err_bnd < 3 * err_base + 1e-3, (
        f"re-rotation adds error beyond baseline fp32 floor: "
        f"baseline={err_base}, rebased={err_bnd}"
    )


def test_negative_control_nonzero_delta_changes_video_k():
    """Guard: a non-zero Δ must actually change K (catch a no-op kernel)."""
    torch.manual_seed(3)
    with _config_ctx():
        rope = _make_rope()
        pos = _video_positions(base0=300)
        n = pos.shape[1]
        k_raw = torch.randn(n, NUM_KV_HEADS * HEAD_DIM, dtype=torch.float32)
        k_stored = _rotate_k(rope, pos, k_raw)
        k_rerot = _paged_rotate(k_stored, 50, rope)
    assert (k_rerot - k_stored).abs().max().item() > 1e-2


# --------------------------------------------------------------------------
# (c) Composition / unitarity.
# --------------------------------------------------------------------------
def test_composition_two_rotations_equal_summed_delta_fp32():
    """rotate(Δ1) ∘ rotate(Δ2) == rotate(Δ1+Δ2); rotations preserve norms."""
    torch.manual_seed(20)
    with _config_ctx():
        rope = _make_rope()
        n = 45
        pos = _video_positions_n(400, n)
        k_stored = _rotate_k(
            rope, pos, torch.randn(n, NUM_KV_HEADS * HEAD_DIM, dtype=torch.float32)
        )
        once = _paged_rotate(_paged_rotate(k_stored, 30, rope), 70, rope)
        summed = _paged_rotate(k_stored, 100, rope)
    # Guard against a vacuous pass: a no-op kernel satisfies once == summed.
    assert (summed - k_stored).abs().max().item() > 1e-2
    err = (once - summed).abs().max().item()
    assert err < 1e-4, f"composition != summed Δ; err={err}"
    norm_drift = (once.norm(dim=-1) - k_stored.norm(dim=-1)).abs().max().item()
    assert norm_drift < 1e-4, f"rotation not unitary; norm drift={norm_drift}"


# --------------------------------------------------------------------------
# (d) Sinks untouched + rebase_delta arithmetic.
# --------------------------------------------------------------------------
def test_rebase_delta_arithmetic():
    """Δ = base − start_size; the rebased window front lands exactly at S."""
    assert rebase_delta(200, 24) == 176
    s, w = 128, 2048
    base = 49152 - w  # window front when B+W hits rebase_at=49152
    delta = rebase_delta(base, s)
    assert base - delta == s  # post-rebase window front == S


def test_sinks_untouched_token_start_honors_sink_block_boundary():
    """Realistic compacted row (block-aligned S, the validated production
    shape): compute_sinkwindow_rows packs sinks as whole blocks and the recent
    tail starts at slot ``sink_blocks * block_size``. [0, S) bitwise intact,
    [S, end) rotated to the reference; unwritten tail slots intact."""
    torch.manual_seed(21)
    with _config_ctx():
        rope = _make_rope()
        s, n_recent = 32, 36  # S = 2 whole sink blocks; 68 tokens, partial tail
        n = s + n_recent
        base = 200
        delta = rebase_delta(base, s)  # 168
        pos_sink = torch.arange(s, dtype=torch.long)
        pos_recent = base + torch.arange(n_recent, dtype=torch.long)
        pos = torch.cat([pos_sink, pos_recent])
        k_raw = torch.randn(n, NUM_KV_HEADS * HEAD_DIM, dtype=torch.float32)
        k_stored = _rotate_k(rope, pos, k_raw)

        _, key_cache, _ = _make_packed_cache(8, torch.float32)
        block_ids = [1, 4, 2, 5, 7]  # compacted row: sink blocks ++ tail blocks
        _scatter_tokens(key_cache, block_ids, 0, k_stored)
        before = _gather_tokens(key_cache, block_ids, 0, n)

        rotate_kv_pages(key_cache, block_ids, s, n, delta, rope, BLOCK_SIZE)

        got = _gather_tokens(key_cache, block_ids, 0, n)
        want = rotate_flat_reference(k_stored[s:], pos_recent, delta, rope)
    assert torch.equal(got[:s], before[:s]), "sink tokens were rotated"
    err = (got[s:] - want).abs().max().item()
    assert err < 1e-4, f"recent window != reference; err={err}"
    # The rebased window's effective positions start exactly at S.
    assert int((pos_recent - delta).min()) == s


def test_non_aligned_start_size_slack_tokens_protected():
    """Non-block-aligned start_size: sinks are packed as WHOLE blocks, so the
    compacted-row rotation boundary is cdiv(start_size, bs) * bs — the slack
    tokens in [start_size, boundary) are pinned near-prefix tokens and must
    survive the rebase bitwise (rotating them by −Δ would corrupt them)."""
    torch.manual_seed(23)
    with _config_ctx():
        rope = _make_rope()
        start_size, n_recent = 24, 28
        boundary = -(-start_size // BLOCK_SIZE) * BLOCK_SIZE  # cdiv * bs = 32
        n = boundary + n_recent  # 60 tokens over 4 blocks (partial tail)
        base = 200
        delta = rebase_delta(base, start_size)  # 176
        # Pinned prefix occupies its whole blocks: positions 0..boundary-1
        # (sinks [0, start_size) AND slack [start_size, boundary)).
        pos_prefix = torch.arange(boundary, dtype=torch.long)
        pos_recent = base + torch.arange(n_recent, dtype=torch.long)
        pos = torch.cat([pos_prefix, pos_recent])
        k_raw = torch.randn(n, NUM_KV_HEADS * HEAD_DIM, dtype=torch.float32)
        k_stored = _rotate_k(rope, pos, k_raw)

        _, key_cache, _ = _make_packed_cache(6, torch.float32)
        block_ids = [3, 0, 5, 2]
        _scatter_tokens(key_cache, block_ids, 0, k_stored)
        before = _gather_tokens(key_cache, block_ids, 0, n)

        rotate_kv_pages(key_cache, block_ids, boundary, n, delta, rope, BLOCK_SIZE)

        got = _gather_tokens(key_cache, block_ids, 0, n)
        want = rotate_flat_reference(k_stored[boundary:], pos_recent, delta, rope)
    assert torch.equal(got[:start_size], before[:start_size]), "sinks rotated"
    assert torch.equal(got[start_size:boundary], before[start_size:boundary]), (
        "slack tokens in [start_size, sink-block boundary) were rotated"
    )
    err = (got[boundary:] - want).abs().max().item()
    assert err < 1e-4, f"recent window != reference; err={err}"


def test_zero_delta_is_a_bitwise_noop():
    """Δ == 0 must not touch the cache at all (early-out, no fp32 round trip)."""
    torch.manual_seed(22)
    with _config_ctx():
        rope = _make_rope()
        n = 20
        k_stored = _rotate_k(
            rope,
            _video_positions_n(100, n),
            torch.randn(n, NUM_KV_HEADS * HEAD_DIM, dtype=torch.float32),
        )
        kv, key_cache, _ = _make_packed_cache(4, torch.bfloat16)
        block_ids = [0, 1]
        _scatter_tokens(key_cache, block_ids, 0, k_stored)
        snap = kv.clone()
        rotate_kv_pages(key_cache, block_ids, 0, n, 0, rope, BLOCK_SIZE)
    assert torch.equal(kv, snap)


def test_yarn_scaled_rotary_rejected():
    """``MRotaryEmbedding._compute_inv_freq`` forwards its argument to YaRN as
    the SCALING FACTOR (not the base), so a scaled-rope rotary would silently
    get wrong delta frequencies. The module must fail loudly, before any cache
    mutation, instead of corrupting K."""
    torch.manual_seed(24)
    with _config_ctx():
        rope = MRotaryEmbedding(
            head_size=HEAD_DIM,
            rotary_dim=ROTARY_DIM,
            max_position_embeddings=1024,
            base=ROPE_THETA,
            is_neox_style=True,
            dtype=torch.float32,
            mrope_section=MROPE_SECTION,
            mrope_interleaved=True,
            scaling_factor=4.0,
        )
        kv, key_cache, _ = _make_packed_cache(4, torch.bfloat16)
        snap = kv.clone()
        with pytest.raises(AssertionError, match="non-scaled rope"):
            rotate_kv_pages(key_cache, [0, 1], 0, 20, 50, rope, BLOCK_SIZE)
    assert torch.equal(kv, snap)  # rejected before any cache mutation


def test_fp8_kv_cache_rejected():
    """Belt-and-suspenders for the config-time dtype gate: compressed-tensors
    checkpoints flip cache_dtype to fp8 at model load (Attention.__init__),
    AFTER every config validator has run, so the rotation itself must refuse
    a quantized cache instead of silently writing back ``seg.to(fp8)``."""
    torch.manual_seed(24)
    with _config_ctx():
        rope = _make_rope()
        kv, key_cache, _ = _make_packed_cache(4, torch.float8_e4m3fn)
        snap = kv.clone()
        with pytest.raises(AssertionError, match="bf16/fp16 KV cache"):
            rotate_kv_pages(key_cache, [0, 1], 0, 20, 50, rope, BLOCK_SIZE)
    # Rejected before any cache mutation (uint8 view: fp8 has no CPU eq).
    assert torch.equal(kv.view(torch.uint8), snap.view(torch.uint8))


# --------------------------------------------------------------------------
# (e) bf16 write-back single-rotation error bound.
# --------------------------------------------------------------------------
def test_bf16_writeback_single_rotation_error_bound():
    """One fp32-staged rotation on a bf16 cache stays within the bf16 storage
    floor (measured in-test, the way the prior-art suite measured its fp32
    floor). This is the numerical content of the single-rotation invariant:
    each K entry is rotated at most once before eviction, so the write-back
    rounding never compounds.

    Two bounds, because unitarity constrains L2, not max: the rotation
    preserves the L2 norm of the pre-existing input-rounding error exactly and
    the write-back adds one fresh rounding of the same scale, so the RMS error
    is genuinely <= 2x the RMS storage floor (measured ~1.33x, stable across
    seeds 0-11). The max-metric worst case for round -> unitary-rotate ->
    round is (1 + sqrt(2)) * floor ~= 2.41x when the errors align, plus
    max-statistics scatter (measured up to 2.65x across seeds 0-11) — bounded
    at 3x as a coarse guard."""
    torch.manual_seed(4)
    with _config_ctx():
        rope = _make_rope()
        pos = _video_positions_n(300, 45)
        n = pos.shape[1]
        delta = 50
        k_raw = torch.randn(n, NUM_KV_HEADS * HEAD_DIM, dtype=torch.float32)
        k_stored = _rotate_k(rope, pos, k_raw)  # fp32-exact stored K
        k_target = _rotate_k(rope, pos - delta, k_raw)  # fp32-exact target

        got = _paged_rotate(k_stored, delta, rope, dtype=torch.bfloat16)

        # The unavoidable bf16 storage floor the cache already carries.
        floor_vec = k_stored.to(torch.bfloat16).float() - k_stored
        err_vec = got.float() - k_target
        floor_max = floor_vec.abs().max().item()
        err_max = err_vec.abs().max().item()
        floor_rms = floor_vec.pow(2).mean().sqrt().item()
        err_rms = err_vec.pow(2).mean().sqrt().item()
    print(
        f"[bf16] single-rotation rms={err_rms:.3e} (floor {floor_rms:.3e}, "
        f"ratio {err_rms / floor_rms:.2f}) max={err_max:.3e} "
        f"(floor {floor_max:.3e}, ratio {err_max / floor_max:.2f})"
    )
    assert err_rms <= 2 * floor_rms, (
        f"bf16 single-rotation RMS error {err_rms} exceeds 2x the RMS "
        f"storage floor {floor_rms}"
    )
    assert err_max <= 3 * floor_max, (
        f"bf16 single-rotation max error {err_max} exceeds 3x the max "
        f"storage floor {floor_max}"
    )
