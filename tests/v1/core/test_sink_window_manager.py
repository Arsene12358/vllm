# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SinkWindowManager (Phase 1, Change B).

U6 - KVCacheSpecRegistry resolves SinkWindowSpec to SinkWindowManager
     (so KVCacheManager finds the right policy for our new spec).
U5 - remove_skipped_blocks is a no-op while the eviction frontier is
     still inside or at the pinned sink prefix.
U4 - remove_skipped_blocks pins the first start_block_count blocks
     even when num_computed_tokens grows past start+sliding_window.
U11 - cross-module property: the manager's eviction frontier never passes the
     first block the compacted row (`sinkwindow_row_geometry`, shared by the FA
     metadata builder and the rebase event) still addresses.
"""

import logging

import pytest
import torch

from vllm.utils.math_utils import cdiv
from vllm.v1.attention.streaming_rebase import sinkwindow_row_geometry
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import (
    SinkWindowManager,
    SlidingWindowManager,
    get_manager_for_kv_cache_spec,
)
from vllm.v1.kv_cache_interface import SinkWindowSpec
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

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


def make_block_pool(num_gpu_blocks: int, block_size: int) -> BlockPool:
    return BlockPool(
        num_gpu_blocks=num_gpu_blocks,
        enable_caching=True,
        hash_block_size=block_size,
    )


def make_manager(spec: SinkWindowSpec, block_pool: BlockPool) -> SinkWindowManager:
    # v0.26.0 manager construction: keyword args + scheduler_block_size.
    return SinkWindowManager(
        spec,
        block_pool=block_pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=spec.block_size,
    )


# ----------------------------------------------------------------------------
# U6 — registry
# ----------------------------------------------------------------------------
def test_registry_resolves_sink_window():
    spec = make_spec()
    assert KVCacheSpecRegistry.get_manager_class(spec) is SinkWindowManager
    # SinkWindowSpec is its own uniform-type base: never grouped with
    # SlidingWindowSpec layers.
    assert KVCacheSpecRegistry.get_uniform_type_base_spec(spec) is SinkWindowSpec
    # And SinkWindowManager is its own class, not SlidingWindowManager:
    assert SinkWindowManager is not SlidingWindowManager
    # But it is a SlidingWindowManager subclass (we reuse the eviction skeleton).
    assert issubclass(SinkWindowManager, SlidingWindowManager)


def test_get_manager_for_kv_cache_spec_wires_admission_cap():
    """SinkWindow recycles blocks like SWA, so the runtime admission cap must
    be wired from the spec method (the single source of truth shared with
    startup pool sizing)."""
    spec = make_spec(block_size=4, sliding_window=8, start_size=4)
    bp = make_block_pool(128, spec.block_size)
    manager = get_manager_for_kv_cache_spec(
        spec,
        max_in_flight_tokens=2048,
        max_model_len=65536,
        block_pool=bp,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=spec.block_size,
    )
    assert type(manager) is SinkWindowManager
    assert manager._max_admission_blocks_per_request == (
        spec.max_admission_blocks_per_request(
            max_in_flight_tokens=2048, max_model_len=65536
        )
    )


def test_sink_window_manager_records_start_block_count():
    """cdiv: a partial last sink block still counts as pinned."""
    spec = make_spec(block_size=4, sliding_window=8, start_size=4)
    bp = make_block_pool(128, spec.block_size)
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
    bp = make_block_pool(2000, spec.block_size)
    manager = make_manager(spec, bp)
    null_id = bp.null_block.block_id

    def id_to_blocks(ids):
        return [KVCacheBlock(i) if i != null_id else bp.null_block for i in ids]

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
    bp = make_block_pool(2000, spec.block_size)
    manager = make_manager(spec, bp)
    null_id = bp.null_block.block_id

    original = [2000, 2001, 2002, 2003, 2004, 2005]
    block_table = [KVCacheBlock(i) if i != null_id else bp.null_block for i in original]
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


def test_sink_window_eviction_log_line():
    """The eviction log line is a frozen demo contract — assert it verbatim."""
    spec = make_spec(block_size=2, sliding_window=4, start_size=4)
    bp = make_block_pool(2000, spec.block_size)
    manager = make_manager(spec, bp)
    manager.req_to_blocks["req"] = [KVCacheBlock(i) for i in range(1000, 1011)]

    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    log = logging.getLogger("vllm.v1.core.single_type_kv_cache_manager")
    handler = _Capture(level=logging.INFO)
    old_level = log.level
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        # Evicts exactly block index 2 → total=11, alive=10, pinned=2, freed=1.
        manager.remove_skipped_blocks("req", 9)
    finally:
        log.removeHandler(handler)
        log.setLevel(old_level)

    messages = [r.getMessage() for r in records if "[streaming-kv]" in r.getMessage()]
    assert messages == [
        "[streaming-kv] eviction req=req computed=9 total_blocks=11 "
        "alive=10 (start_pinned=2) freed=1"
    ]


# ----------------------------------------------------------------------------
# U11 — eviction frontier vs the compacted row's tail
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("block_size", "start_size", "recent_size", "n_max"),
    [
        (4, 8, 16, 120),  # start_size block-aligned
        (4, 6, 16, 120),  # start_size mid-block: cdiv pins a 2nd sink block
        (8, 16, 32, 200),  # bigger blocks, longer sweep
        (16, 128, 1024, 3000),  # production shape (block_size 16)
    ],
)
def test_eviction_frontier_never_reaches_the_compacted_row_tail(
    block_size, start_size, recent_size, n_max
):
    """Cross-commit contract, untested until now: the manager decides what to
    free, `sinkwindow_row_geometry` decides what the FA metadata builder and
    the rebase event read — two modules, no shared code, and a freed block that
    the compacted row still addresses is silent attention over reused KV.

    The relation that keeps them consistent is `last_useful_block <=
    tail_start` at every `num_computed`: the manager's frontier
    (`(N - recent_size + 1) // block_size`) never passes the geometry's first
    alive tail block. Swept step by step over a growing row, with the real
    manager doing the freeing.
    """
    spec = make_spec(
        block_size=block_size, sliding_window=recent_size, start_size=start_size
    )
    bp = make_block_pool(2 * cdiv(n_max, block_size) + 16, block_size)
    manager = make_manager(spec, bp)
    blocks: list[KVCacheBlock] = []
    manager.req_to_blocks["req"] = blocks

    for num_computed in range(1, n_max + 1):
        while len(blocks) < cdiv(num_computed, block_size):
            blocks.extend(bp.get_new_blocks(1))

        sink_blocks, tail_start, num_valid_blocks, _ = sinkwindow_row_geometry(
            num_computed, block_size, start_size, recent_size
        )
        # The manager's own frontier, capped the way it caps it.
        last_useful_block = min(
            (num_computed - recent_size + 1) // block_size, len(blocks)
        )
        assert last_useful_block <= tail_start, (
            f"eviction frontier {last_useful_block} passed the compacted row's "
            f"first alive tail block {tail_start} at num_computed={num_computed}"
        )

        manager.remove_skipped_blocks("req", num_computed)

        # ...and the observable consequence: every block the compacted row
        # addresses (pinned sinks ++ alive tail) is still real.
        addressed = list(range(min(sink_blocks, num_valid_blocks))) + list(
            range(tail_start, num_valid_blocks)
        )
        for i in addressed:
            assert blocks[i] != bp.null_block, (
                f"block {i} is addressed by the compacted row but was freed at "
                f"num_computed={num_computed} (sink_blocks={sink_blocks}, "
                f"tail_start={tail_start})"
            )

    # Not vacuous: the sweep did reach the eviction regime.
    assert any(b == bp.null_block for b in blocks)


# ----------------------------------------------------------------------------
# Prefix caching is disabled for SinkWindow groups
# ----------------------------------------------------------------------------
def test_sink_window_find_longest_cache_hit_disabled():
    """Phase 1 disables prefix caching for SinkWindow groups.

    The discontiguous post-eviction layout breaks the inherited matcher;
    returning empty forces re-prefill, which is correct (if slower).
    """
    spec = make_spec()
    bp = make_block_pool(128, spec.block_size)
    hit_blocks, hit_length = SinkWindowManager.find_longest_cache_hit(
        block_hashes=[],
        max_length=0,
        kv_cache_group_ids=[0, 1, 2],
        block_pool=bp,
        kv_cache_spec=spec,
        drop_eagle_block=False,
        alignment_tokens=spec.block_size,
    )
    assert hit_blocks == ([], [], [])
    assert hit_length == 0


def test_sink_window_reachable_block_mask_disables_caching():
    """No block can ever serve a prefix-cache hit (see above), so
    `cache_blocks` must keep every block out of the prefix-cache hash map.
    The inherited SWA mask would also assert on a non-SlidingWindowSpec."""
    spec = make_spec()
    mask = SinkWindowManager.reachable_block_mask(
        start_block=0,
        end_block=8,
        alignment_tokens=spec.block_size,
        kv_cache_spec=spec,
        use_eagle=False,
    )
    assert mask == [False] * 8
