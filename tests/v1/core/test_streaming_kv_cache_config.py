# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the streaming-kv config flags and SinkWindowSpec emission.

U7 — CacheConfig.streaming_kv_{start_size,recent_size} validator:
     - both unset      -> ok (default; streaming KV disabled)
     - both set, > 0   -> ok
     - one set, other not -> ValueError
     - either <= 0     -> ValueError
U8 — CLI: --streaming-kv-start-size / --streaming-kv-recent-size round-trip
     through EngineArgs.add_cli_args / from_cli_args.
U9 — Attention.get_kv_cache_spec emits SinkWindowSpec for decoder layers when
     the flags are set (taking precedence over the sliding-window branch),
     keeps the default spec when unset, and rejects MLA loudly (both the
     Attention assert and the MLAAttention layer class).
U10 — unify_hybrid_kv_cache_specs (disable-hybrid-kv-cache-manager path) never
      down-converts SinkWindowSpec: uniform sink-window specs pass through
      untouched, mixed specs raise a clear error instead of silently dropping
      the eviction feature.
"""

import argparse
from types import SimpleNamespace

import pytest
import torch

from vllm.config.cache import CacheConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.v1.attention.backend import AttentionType
from vllm.v1.core.kv_cache_utils import unify_hybrid_kv_cache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    SinkWindowSpec,
    get_kv_quant_mode,
)

pytestmark = pytest.mark.cpu_test


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
class _FakeAttentionLayer:
    """Minimal stand-in for Attention with the attributes get_kv_cache_spec
    reads. Building a real Attention requires a full ModelConfig + backend."""

    attn_type = AttentionType.DECODER
    kv_cache_dtype = "auto"
    kv_cache_torch_dtype = torch.bfloat16
    layer_name = "model.layers.0.self_attn.attn"
    num_kv_heads = 4
    head_size = 128
    head_size_v = 128
    sliding_window = None


def _make_vllm_config(start=None, recent=None, use_mla=False, block_size=16):
    cache_config = CacheConfig(
        block_size=block_size,
        streaming_kv_start_size=start,
        streaming_kv_recent_size=recent,
    )
    return SimpleNamespace(
        cache_config=cache_config,
        model_config=SimpleNamespace(use_mla=use_mla),
    )


def _make_sink_spec(start_size=64, sliding_window=512, num_kv_heads=4):
    return SinkWindowSpec(
        block_size=16,
        num_kv_heads=num_kv_heads,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=sliding_window,
        start_size=start_size,
    )


# ----------------------------------------------------------------------------
# U7 — CacheConfig validator
# ----------------------------------------------------------------------------
def test_streaming_kv_disabled_by_default():
    c = CacheConfig()
    assert c.streaming_kv_start_size is None
    assert c.streaming_kv_recent_size is None


def test_streaming_kv_both_set_ok():
    c = CacheConfig(
        streaming_kv_start_size=64,
        streaming_kv_recent_size=8192,
    )
    assert c.streaming_kv_start_size == 64
    assert c.streaming_kv_recent_size == 8192


@pytest.mark.parametrize(
    "kwargs",
    [
        {"streaming_kv_start_size": 64},
        {"streaming_kv_recent_size": 8192},
    ],
)
def test_streaming_kv_partial_rejected(kwargs):
    with pytest.raises(ValueError, match="must be set together"):
        CacheConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"streaming_kv_start_size": 0, "streaming_kv_recent_size": 8192},
        {"streaming_kv_start_size": -1, "streaming_kv_recent_size": 8192},
        {"streaming_kv_start_size": 64, "streaming_kv_recent_size": 0},
        {"streaming_kv_start_size": 64, "streaming_kv_recent_size": -100},
    ],
)
def test_streaming_kv_nonpositive_rejected(kwargs):
    with pytest.raises(ValueError, match=r"must be > 0"):
        CacheConfig(**kwargs)


# ----------------------------------------------------------------------------
# U8 — CLI plumbing
# ----------------------------------------------------------------------------
def test_streaming_kv_cli_args_roundtrip():
    parser = argparse.ArgumentParser()
    EngineArgs.add_cli_args(parser)
    args = parser.parse_args(
        [
            "--streaming-kv-start-size",
            "128",
            "--streaming-kv-recent-size",
            "1024",
        ]
    )
    engine_args = EngineArgs.from_cli_args(args)
    assert engine_args.streaming_kv_start_size == 128
    assert engine_args.streaming_kv_recent_size == 1024


def test_streaming_kv_cli_defaults_to_none():
    parser = argparse.ArgumentParser()
    EngineArgs.add_cli_args(parser)
    engine_args = EngineArgs.from_cli_args(parser.parse_args([]))
    assert engine_args.streaming_kv_start_size is None
    assert engine_args.streaming_kv_recent_size is None


# ----------------------------------------------------------------------------
# U9 — SinkWindowSpec emission in Attention.get_kv_cache_spec
# ----------------------------------------------------------------------------
def test_get_kv_cache_spec_emits_sink_window_spec():
    cfg = _make_vllm_config(start=128, recent=1024)
    spec = Attention.get_kv_cache_spec(_FakeAttentionLayer(), cfg)
    assert isinstance(spec, SinkWindowSpec)
    assert spec.block_size == 16
    assert spec.num_kv_heads == 4
    assert spec.head_size == 128
    assert spec.dtype == torch.bfloat16
    assert spec.kv_quant_mode == get_kv_quant_mode("auto")
    assert spec.sliding_window == 1024
    assert spec.start_size == 128


def test_get_kv_cache_spec_streaming_overrides_sliding_window():
    """The streaming-kv branch must run before the per-layer sliding-window
    branch so every decoder layer emits SinkWindowSpec uniformly."""
    layer = _FakeAttentionLayer()
    layer.sliding_window = 4096
    cfg = _make_vllm_config(start=64, recent=512)
    spec = Attention.get_kv_cache_spec(layer, cfg)
    assert isinstance(spec, SinkWindowSpec)
    assert spec.sliding_window == 512
    assert spec.start_size == 64


def test_get_kv_cache_spec_default_path_unchanged_when_unset():
    cfg = _make_vllm_config()
    spec = Attention.get_kv_cache_spec(_FakeAttentionLayer(), cfg)
    assert isinstance(spec, FullAttentionSpec)
    assert not isinstance(spec, SinkWindowSpec)


def test_streaming_kv_mla_rejected():
    cfg = _make_vllm_config(start=128, recent=1024, use_mla=True)
    with pytest.raises(AssertionError, match="MLA is not supported for sink"):
        Attention.get_kv_cache_spec(_FakeAttentionLayer(), cfg)


def test_streaming_kv_mla_attention_layer_rejected():
    """MLA models route through MLAAttention.get_kv_cache_spec, which must
    fail loudly instead of silently ignoring the streaming-kv flags."""
    cfg = _make_vllm_config(start=128, recent=1024, use_mla=True)
    with pytest.raises(AssertionError, match="MLA is not supported for sink"):
        MLAAttention.get_kv_cache_spec(SimpleNamespace(), cfg)


# ----------------------------------------------------------------------------
# U10 — disable-hybrid down-conversion must never touch SinkWindowSpec
# ----------------------------------------------------------------------------
def test_unify_hybrid_specs_keeps_uniform_sink_window():
    specs = {f"model.layers.{i}.self_attn.attn": _make_sink_spec() for i in range(4)}
    unify_hybrid_kv_cache_specs(specs)
    assert all(isinstance(spec, SinkWindowSpec) for spec in specs.values())


def test_unify_hybrid_specs_rejects_mixed_sink_window():
    specs = {
        "model.layers.0.self_attn.attn": _make_sink_spec(),
        "model.layers.1.self_attn.attn": FullAttentionSpec(
            block_size=16,
            num_kv_heads=4,
            head_size=128,
            dtype=torch.bfloat16,
        ),
    }
    with pytest.raises(ValueError, match="must not be down-converted"):
        unify_hybrid_kv_cache_specs(specs)
