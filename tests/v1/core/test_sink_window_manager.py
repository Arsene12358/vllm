# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SinkWindowManager (Phase 1, Change B).

  U6 - spec_manager_map[SinkWindowSpec] resolves to SinkWindowManager
       (so KVCacheManager finds the right policy for our new spec).
  U5 - remove_skipped_blocks is a no-op while the eviction frontier is
       still inside or at the pinned sink prefix.
  U4 - remove_skipped_blocks pins the first start_block_count blocks
       even when num_computed_tokens grows past start+sliding_window.
"""
import pytest
import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import (
    SinkWindowManager,
    SlidingWindowManager,
    spec_manager_map,
)
from vllm.v1.kv_cache_interface import SinkWindowSpec

pytestmark = pytest.mark.cpu_test


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def make_spec(block_size=2, sliding_window=4, start_size=4) -> SinkWindowSpec:
    return SinkWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=sliding_window,
        start_size=start_size,
    )


def make_manager(spec: SinkWindowSpec, block_pool: BlockPool) -> SinkWindowManager:
    # v0.20.0 manager construction: keyword args, no positional block_pool.
    return SinkWindowManager(
        spec,
        block_pool=block_pool,
        enable_caching=True,
        kv_cache_group_id=0,
    )


# ----------------------------------------------------------------------------
# U6 — registry
# ----------------------------------------------------------------------------
def test_spec_manager_map_resolves_sink_window():
    assert spec_manager_map[SinkWindowSpec] is SinkWindowManager
    # And SinkWindowManager is its own class, not SlidingWindowManager:
    assert SinkWindowManager is not SlidingWindowManager
    # But it is a SlidingWindowManager subclass (we reuse the eviction skeleton).
    assert issubclass(SinkWindowManager, SlidingWindowManager)


def test_sink_window_manager_records_start_block_count():
    """cdiv: a partial last sink block still counts as pinned."""
    spec = make_spec(block_size=4, sliding_window=8, start_size=4)
    bp = BlockPool(num_gpu_blocks=128, enable_caching=True)
    m = make_manager(spec, bp)
    assert m.start_size == 4
    assert m.start_block_count == 1

    spec2 = make_spec(block_size=4, sliding_window=8, start_size=5)
    m2 = make_manager(spec2, bp)
    # cdiv(5, 4) = 2 — partial 5th token forces a second pinned block
    assert m2.start_block_count == 2


# ----------------------------------------------------------------------------
# U4 + U5 — eviction semantics
# ----------------------------------------------------------------------------
def test_sink_window_remove_skipped_blocks():
    """block_size=2, sliding_window=4, start_size=4 → start_block_count=2.

    Blocks 0 and 1 are pinned (sink); eviction targets only [2, last_useful_block).
    """
    spec = make_spec(block_size=2, sliding_window=4, start_size=4)
    bp = BlockPool(num_gpu_blocks=2000, enable_caching=True)
    manager = make_manager(spec, bp)
    null_id = bp.null_block.block_id

    def id_to_blocks(ids):
        return [
            KVCacheBlock(i) if i != null_id else bp.null_block for i in ids
        ]

    def assert_ids(block_table, expected_ids):
        for block, exp in zip(block_table, expected_ids):
            if exp == null_id:
                assert block == bp.null_block
            else:
                assert block.block_id == exp

    original = [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]
    block_table = id_to_blocks(original)
    manager.req_to_blocks["req"] = block_table

    # U5 — num_computed=0 → no eviction
    manager.remove_skipped_blocks("req", 0)
    assert_ids(block_table, original)

    # U5 — last_useful_token=4-4+1=1, last_useful_block=0, ≤ start_block_count=2.
    manager.remove_skipped_blocks("req", 4)
    assert_ids(block_table, original)

    # U5 — num_computed=7 → last_useful_token=4, last_useful_block=2;
    # 2 ≤ start_block_count, still no eviction.
    manager.remove_skipped_blocks("req", 7)
    assert_ids(block_table, original)

    # U4 — num_computed=9 → last_useful_token=6, last_useful_block=3.
    # 3 > 2, so block index 2 (id=1002) is evicted.
    # Sink blocks 0, 1 (ids 1000, 1001) MUST stay.
    manager.remove_skipped_blocks("req", 9)
    assert_ids(
        block_table,
        [1000, 1001, null_id, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010],
    )

    # U4 — num_computed=13 → last_useful_token=10, last_useful_block=5.
    # Loop walks i=4, 3, 2; index 2 is already null → break.
    # Net: indices 3 and 4 evicted, sinks still intact.
    manager.remove_skipped_blocks("req", 13)
    assert_ids(
        block_table,
        [1000, 1001, null_id, null_id, null_id, 1005, 1006, 1007, 1008, 1009, 1010],
    )

    # U4 — much later: num_computed=21 → last_useful_token=18, last_useful_block=9.
    # Indices [8, 7, 6, 5] all evictable; index 4 already null → break.
    manager.remove_skipped_blocks("req", 21)
    assert_ids(
        block_table,
        [1000, 1001] + [null_id] * 7 + [1009, 1010],
    )


def test_sink_window_remove_skipped_blocks_zero_window_offset():
    """Edge case: start_size exactly equals block_size — start_block_count=1.

    Verifies the boundary condition where only the first single block is pinned.
    """
    spec = make_spec(block_size=4, sliding_window=8, start_size=4)
    bp = BlockPool(num_gpu_blocks=2000, enable_caching=True)
    manager = make_manager(spec, bp)
    null_id = bp.null_block.block_id

    original = [2000, 2001, 2002, 2003, 2004, 2005]
    block_table = [
        KVCacheBlock(i) if i != null_id else bp.null_block for i in original
    ]
    manager.req_to_blocks["req"] = block_table

    # num_computed=20 → last_useful_token=13, last_useful_block=3.
    # 3 > start_block_count=1 → evict indices [2, 1] (id 2002, 2001).
    # Sink block 0 (id 2000) MUST stay.
    manager.remove_skipped_blocks("req", 20)
    expected = [2000, null_id, null_id, 2003, 2004, 2005]
    for block, exp in zip(block_table, expected):
        if exp == null_id:
            assert block == bp.null_block
        else:
            assert block.block_id == exp


def test_sink_window_find_longest_cache_hit_disabled():
    """Phase 1 disables prefix caching for SinkWindow groups.

    The discontiguous post-eviction layout breaks the inherited matcher;
    returning empty forces re-prefill, which is correct (if slower).
    """
    spec = make_spec()
    bp = BlockPool(num_gpu_blocks=128, enable_caching=True)
    out = SinkWindowManager.find_longest_cache_hit(
        block_hashes=[],
        max_length=0,
        kv_cache_group_ids=[0, 1, 2],
        block_pool=bp,
        kv_cache_spec=spec,
        use_eagle=False,
    )
    assert out == ([], [], [])
