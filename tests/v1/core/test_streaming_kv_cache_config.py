# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the streaming-kv fields on CacheConfig (Phase 1, Change C).

U7 — CacheConfig.streaming_kv_{start_size,recent_size} validator:
     - both unset      -> ok (default; streaming KV disabled)
     - both set, > 0   -> ok
     - one set, other not -> ValueError
     - either <= 0     -> ValueError
"""
import pytest

from vllm.config.cache import CacheConfig

pytestmark = pytest.mark.cpu_test


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
