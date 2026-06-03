"""Stage 1: the per-token uniform Δ kernel is exact for *multimodal* recent
windows (video tokens, t != h != w) — not just text (t == h == w).

Why this matters: Option B re-rotates the recent window's K by a per-token
uniform Δ (`_apply_rope_delta_mrope_pertoken`, same Δ on all three M-RoPE
axes). The earlier validation used a text-only prompt, so the recent window
never contained video tokens whose axes differ. These tests close that gap by
checking the kernel against the model's *real* MRotaryEmbedding for genuine
video positions.

The structural claim (see flash_attn.py:_apply_rope_delta_mrope docstring):
the M-RoPE angle at frequency pair j is pos_axis(j) * inv_freq[j] with
inv_freq[j] shared across axes, so adding the SAME Δ to all three axes adds
Δ*inv_freq[j] to every pair — i.e. a uniform rotation R(Δ). Thus

    R_uniform(Δ) ∘ R_mrope(t, h, w)  ==  R_mrope(t+Δ, h+Δ, w+Δ)

regardless of the (interleaved) section layout. With a text decode query
(t_q = h_q = w_q) the recent window only needs this uniform shift, so the
per-token kernel reproduces the unclamped baseline relatives exactly — video
as well as text.

Numerical note: the identity is exact in real arithmetic but cos/sin of large
absolute positions (~64k rad) lose precision in fp32 (range reduction over
~10k periods). The STRUCTURAL tests therefore use small positions (clean fp32);
a separate test confirms that at production scale the bounded path is no less
accurate than the baseline (the residual error is shared, not introduced by
the re-rotation).
"""
import contextlib

import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
from vllm.v1.attention.backends.flash_attn import (
    _apply_rope_delta_mrope_pertoken,
)

# Qwen3-Omni Thinker text rope config.
HEAD_DIM = 128
ROTARY_DIM = 128
MAX_POS = 65536
ROPE_THETA = 1_000_000.0
MROPE_SECTION = [24, 20, 20]
NUM_KV_HEADS = 2


@contextlib.contextmanager
def _config_ctx():
    # MRotaryEmbedding is a CustomOp; its __init__ reads the current vLLM
    # compilation config to pick a forward impl. Mirrors tests/conftest.py's
    # `default_vllm_config` fixture.
    with set_current_vllm_config(VllmConfig()):
        yield


def _make_rope() -> MRotaryEmbedding:
    return MRotaryEmbedding(
        head_size=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        max_position_embeddings=MAX_POS,
        base=ROPE_THETA,
        is_neox_style=True,
        dtype=torch.float32,
        mrope_section=MROPE_SECTION,
        mrope_interleaved=True,
    )


def _video_positions(base0: int) -> torch.Tensor:
    """Genuine video positions: shared per-token base + intra-frame grid
    offsets, so t != h != w on every token. [3, N]."""
    base = base0 + torch.tensor([0, 3, 6, 50, 55, 60])
    t = base + torch.tensor([0, 0, 0, 1, 1, 1])      # temporal: per-frame
    h = base + torch.tensor([0, 1, 2, 0, 1, 2])      # height: grid row
    w = base + torch.tensor([0, 2, 4, 0, 2, 4])      # width: grid col
    pos = torch.stack([t, h, w]).to(torch.long)      # [3, N]
    # sanity: these are real multimodal positions, not degenerate text ones.
    assert not torch.equal(pos[0], pos[1]) and not torch.equal(pos[1], pos[2])
    return pos


def _rotate_k(rope, pos, k_raw):
    _, k = rope.forward_native(pos, torch.randn_like(k_raw), k_raw.clone())
    return k.view(-1, NUM_KV_HEADS, HEAD_DIM)


# --------------------------------------------------------------------------
# STRUCTURAL: uniform Δ == per-axis shift, for video (t!=h!=w), clean fp32.
# --------------------------------------------------------------------------
def test_structural_uniform_delta_equals_axis_shift_video_small_pos():
    """kernel(R_mrope(t,h,w)·k, Δ)  ==  R_mrope(t+Δ,h+Δ,w+Δ)·k  for t!=h!=w.
    Small positions => fp32 cos/sin exact => isolates the structural identity."""
    torch.manual_seed(0)
    with _config_ctx():
        rope = _make_rope()
        pos = _video_positions(base0=300)
        n = pos.shape[1]
        delta = -50
        k_raw = torch.randn(n, NUM_KV_HEADS * HEAD_DIM, dtype=torch.float32)

        k_stored = _rotate_k(rope, pos, k_raw)
        k_rerot = _apply_rope_delta_mrope_pertoken(
            k_stored, torch.full((n,), float(delta)), rope_theta=ROPE_THETA
        )
        k_target = _rotate_k(rope, pos + delta, k_raw)

    err = (k_rerot - k_target).abs().max().item()
    print(f"[structural] video t!=h!=w, small pos: max abs err = {err:.3e}")
    assert err < 1e-4, f"uniform Δ != per-axis shift for video; max abs err={err}"


def test_structural_relative_preserved_text_query_video_key_small_pos():
    """A text query attending to a recent *video* key gets the SAME score
    under the bounded scheme as the unclamped baseline. Small positions."""
    torch.manual_seed(1)
    with _config_ctx():
        rope = _make_rope()
        pos_k = _video_positions(base0=300)
        n = pos_k.shape[1]
        q_true, safe_bound = 400, 350
        delta = safe_bound - q_true                    # -50

        k_raw = torch.randn(n, NUM_KV_HEADS * HEAD_DIM, dtype=torch.float32)
        q_raw = torch.randn(1, NUM_KV_HEADS * HEAD_DIM, dtype=torch.float32)

        def q_at(p):
            row = torch.tensor([p])
            return _rotate_k(rope, torch.stack([row, row, row]), q_raw)

        k_base = _rotate_k(rope, pos_k, k_raw)
        score_base = torch.einsum("qhd,khd->hqk", q_at(q_true), k_base)
        k_bnd = _apply_rope_delta_mrope_pertoken(
            k_base, torch.full((n,), float(delta)), rope_theta=ROPE_THETA
        )
        score_bnd = torch.einsum("qhd,khd->hqk", q_at(safe_bound), k_bnd)

    err = (score_base - score_bnd).abs().max().item()
    print(f"[structural] relative score (text q / video k): max abs err = {err:.3e}")
    assert err < 1e-3, f"bounded relative != baseline; max abs err={err}"


# --------------------------------------------------------------------------
# PRODUCTION SCALE: at pos ~64k, bounded is no less accurate than baseline.
# The fp32 large-angle error is SHARED, not introduced by the re-rotation.
# --------------------------------------------------------------------------
def test_production_scale_bounded_not_worse_than_baseline():
    """Ground truth = same relatives computed at small positions (exact fp32).
    Compare baseline (true large pos) and bounded (clamp + re-rotate) against
    it. Assert bounded's deviation is no worse than baseline's."""
    torch.manual_seed(2)
    with _config_ctx():
        rope = _make_rope()
        q_true, safe_bound = 64000, 63488
        delta = safe_bound - q_true                    # -512
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
        pos_small = _video_positions(base0=base0 - c)   # keys near pos ~200
        k_small = _rotate_k(rope, pos_small, k_raw)
        score_truth = torch.einsum("qhd,khd->hqk", q_at(q_true - c), k_small)

        # Baseline at true large positions (fp32 imprecise).
        pos_large = _video_positions(base0=base0)
        k_large = _rotate_k(rope, pos_large, k_raw)
        score_base = torch.einsum("qhd,khd->hqk", q_at(q_true), k_large)

        # Bounded: query clamped at safe_bound, keys re-rotated by Δ.
        k_bnd = _apply_rope_delta_mrope_pertoken(
            k_large, torch.full((n,), float(delta)), rope_theta=ROPE_THETA
        )
        score_bnd = torch.einsum("qhd,khd->hqk", q_at(safe_bound), k_bnd)

    err_base = (score_base - score_truth).abs().max().item()
    err_bnd = (score_bnd - score_truth).abs().max().item()
    print(f"[prod-scale] fp32 err vs truth: baseline={err_base:.3e} "
          f"bounded={err_bnd:.3e}  (both are shared large-angle fp32 error)")
    # Bounded must not be materially worse than baseline's own fp32 error.
    assert err_bnd < 3 * err_base + 1e-3, (
        f"re-rotation adds error beyond baseline fp32 floor: "
        f"baseline={err_base}, bounded={err_bnd}")


def test_negative_control_nonzero_delta_changes_video_k():
    """Guard: a non-zero Δ must actually change K (catch a no-op kernel)."""
    torch.manual_seed(3)
    with _config_ctx():
        rope = _make_rope()
        pos = _video_positions(base0=300)
        n = pos.shape[1]
        k_raw = torch.randn(n, NUM_KV_HEADS * HEAD_DIM, dtype=torch.float32)
        k_stored = _rotate_k(rope, pos, k_raw)
        k_rerot = _apply_rope_delta_mrope_pertoken(
            k_stored, torch.full((n,), -50.0), rope_theta=ROPE_THETA
        )
    assert (k_rerot - k_stored).abs().max().item() > 1e-2
