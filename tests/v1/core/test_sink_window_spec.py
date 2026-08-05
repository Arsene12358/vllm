# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SinkWindowSpec.

Covers:
  U1 - instantiation with valid args populates fields, frozen dataclass
  U2 - rejects invalid args (start_size <= 0, sliding_window <= 0)
  U3 - equality + hashability (so the spec works as a dict / set key)
  U4 - UniformTypeKVCacheSpecs.is_uniform_type dispatches on SinkWindowSpec
  U5 - max_memory_usage_bytes formula matches the layout
       (sinks + recent window + new tokens, capped at max_model_len)
"""

from dataclasses import FrozenInstanceError

import pytest
import torch

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    SinkWindowSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def make_spec(
    block_size=16,
    num_kv_heads=4,
    head_size=128,
    dtype=torch.bfloat16,
    sliding_window=512,
    start_size=64,
) -> SinkWindowSpec:
    return SinkWindowSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        sliding_window=sliding_window,
        start_size=start_size,
    )


# ----------------------------------------------------------------------------
# U1 — instantiation
# ----------------------------------------------------------------------------
def test_sinkwindow_spec_basic_construction():
    spec = make_spec()
    assert spec.block_size == 16
    assert spec.num_kv_heads == 4
    assert spec.head_size == 128
    assert spec.dtype == torch.bfloat16
    assert spec.sliding_window == 512
    assert spec.start_size == 64


def test_sinkwindow_spec_is_frozen():
    spec = make_spec()
    with pytest.raises(FrozenInstanceError):
        spec.start_size = 99  # type: ignore[misc]


# ----------------------------------------------------------------------------
# U2 — validation
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("bad_start", [0, -1, -100])
def test_sinkwindow_spec_rejects_nonpositive_start_size(bad_start):
    with pytest.raises(ValueError, match="start_size must be > 0"):
        make_spec(start_size=bad_start)


@pytest.mark.parametrize("bad_window", [0, -1, -100])
def test_sinkwindow_spec_rejects_nonpositive_sliding_window(bad_window):
    with pytest.raises(ValueError, match="sliding_window must be > 0"):
        make_spec(sliding_window=bad_window)


# ----------------------------------------------------------------------------
# U3 — equality + hashability
# ----------------------------------------------------------------------------
def test_sinkwindow_spec_equality():
    a = make_spec(start_size=64, sliding_window=512)
    b = make_spec(start_size=64, sliding_window=512)
    assert a == b
    assert hash(a) == hash(b)


def test_sinkwindow_spec_inequality_on_any_field():
    base = make_spec(start_size=64, sliding_window=512)
    assert base != make_spec(start_size=65, sliding_window=512)
    assert base != make_spec(start_size=64, sliding_window=513)
    assert base != make_spec(start_size=64, sliding_window=512, head_size=64)


def test_sinkwindow_spec_set_membership():
    """Specs must be hashable so they can index grouping dicts
    (e.g. `same_type_layers` in kv_cache_utils.py)."""
    a = make_spec(start_size=64, sliding_window=512)
    b = make_spec(start_size=128, sliding_window=512)
    s = {a, b, make_spec(start_size=64, sliding_window=512)}  # dup of a
    assert len(s) == 2


def test_sinkwindow_spec_not_equal_to_sliding_window_spec():
    """Type discrimination: SinkWindowSpec is its own type."""
    sink = make_spec(start_size=64, sliding_window=512)
    sliding = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=512,
    )
    assert sink != sliding


# ----------------------------------------------------------------------------
# U4 — UniformTypeKVCacheSpecs handling
# ----------------------------------------------------------------------------
def test_uniform_type_all_same_sink_window():
    specs = {
        f"layer.{i}": make_spec(start_size=64, sliding_window=512) for i in range(4)
    }
    assert UniformTypeKVCacheSpecs.is_uniform_type(specs)


def test_uniform_type_mixed_start_size_rejected():
    specs = {
        "layer.0": make_spec(start_size=64, sliding_window=512),
        "layer.1": make_spec(start_size=128, sliding_window=512),
    }
    assert not UniformTypeKVCacheSpecs.is_uniform_type(specs)


def test_uniform_type_mixed_window_rejected():
    specs = {
        "layer.0": make_spec(start_size=64, sliding_window=512),
        "layer.1": make_spec(start_size=64, sliding_window=1024),
    }
    assert not UniformTypeKVCacheSpecs.is_uniform_type(specs)


def test_uniform_type_sink_window_mixed_with_full_attention_rejected():
    specs = {
        "layer.0": make_spec(start_size=64, sliding_window=512),
        "layer.1": FullAttentionSpec(
            block_size=16,
            num_kv_heads=4,
            head_size=128,
            dtype=torch.bfloat16,
            sliding_window=None,
        ),
    }
    assert not UniformTypeKVCacheSpecs.is_uniform_type(specs)


# ----------------------------------------------------------------------------
# U5 — max_memory_usage_bytes formula
# ----------------------------------------------------------------------------
class _MockVllmConfig:
    """Minimal stub that satisfies max_memory_usage_bytes' attribute reads."""

    class _Parallel:
        decode_context_parallel_size = 1
        prefill_context_parallel_size = 1

    class _Model:
        def __init__(self, max_model_len: int) -> None:
            self.max_model_len = max_model_len

    def __init__(self, max_model_len: int, max_in_flight_tokens: int) -> None:
        self.parallel_config = self._Parallel()
        self.model_config = self._Model(max_model_len)
        self.max_in_flight_tokens = max_in_flight_tokens


def test_max_memory_usage_bytes_typical():
    """Sink+window typical: 64 + 512 + 2048 in-flight = 2624 tokens budget."""
    spec = make_spec(
        block_size=16,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.bfloat16,
        start_size=64,
        sliding_window=512,
    )
    cfg = _MockVllmConfig(max_model_len=65536, max_in_flight_tokens=2048)
    # page_size_bytes = 2 * 16 * 4 * 128 * 2 (bf16) = 32768 bytes/block
    # num_tokens = min(64 + 512 - 1 + 2048, 65536) = 2623
    # blocks = cdiv(2623, 16) + 1 = 165
    # total = 165 * 32768 = 5_406_720 bytes
    assert spec.page_size_bytes == 32768
    assert spec.max_memory_usage_bytes(cfg) == 165 * 32768


def test_max_memory_usage_bytes_capped_at_max_model_len():
    """When start+window+in-flight exceeds max_model_len, we cap."""
    spec = make_spec(start_size=64, sliding_window=8192)
    cfg = _MockVllmConfig(max_model_len=4096, max_in_flight_tokens=2048)
    expected_tokens = 4096  # cap
    expected_blocks = (expected_tokens + spec.block_size - 1) // spec.block_size + 1
    assert spec.max_memory_usage_bytes(cfg) == expected_blocks * spec.page_size_bytes


def test_max_memory_usage_bytes_dcp_unsupported():
    spec = make_spec()
    cfg = _MockVllmConfig(max_model_len=65536, max_in_flight_tokens=2048)
    cfg.parallel_config.decode_context_parallel_size = 2
    with pytest.raises(AssertionError, match="DCP not supported"):
        spec.max_memory_usage_bytes(cfg)


def test_max_memory_usage_bytes_pcp_unsupported():
    spec = make_spec()
    cfg = _MockVllmConfig(max_model_len=65536, max_in_flight_tokens=2048)
    cfg.parallel_config.prefill_context_parallel_size = 2
    with pytest.raises(AssertionError, match="PCP not supported"):
        spec.max_memory_usage_bytes(cfg)
