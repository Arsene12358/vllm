"""Stage 1: per-token M-RoPE Δ kernel sanity.

With a uniform Δ, the per-token kernel must equal option (a)'s all-axis scalar
kernel (same rotation applied to every token / every axis).
"""
import torch

from vllm.v1.attention.backends.flash_attn import (
    _apply_rope_delta_mrope,
    _apply_rope_delta_mrope_pertoken,
)


def test_pertoken_matches_scalar_when_uniform():
    torch.manual_seed(0)
    k = torch.randn(5, 2, 128)
    scalar = _apply_rope_delta_mrope(
        k, delta_t=-7, mrope_section=[24, 20, 20], delta_h=-7, delta_w=-7)
    vec = _apply_rope_delta_mrope_pertoken(k, torch.full((5,), -7.0))
    assert torch.allclose(scalar, vec, atol=1e-4), (scalar - vec).abs().max()


def test_pertoken_varies_per_token():
    # distinct Δ per token => distinct rotations (token 0 with Δ=0 is identity)
    torch.manual_seed(1)
    k = torch.randn(3, 1, 128)
    out = _apply_rope_delta_mrope_pertoken(k, torch.tensor([0.0, -5.0, -50.0]))
    assert torch.allclose(out[0], k[0], atol=1e-5)        # Δ=0 -> unchanged
    assert not torch.allclose(out[1], k[1], atol=1e-3)    # Δ=-5 -> changed
    assert not torch.allclose(out[1], out[2], atol=1e-3)  # different Δ -> different
