# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the SinkWindow single-pass block-table compaction.

The KV manager evicts middle blocks but leaves the logical block table
length intact (nulls in the hole). The FA metadata builder rebuilds each
row as ``[sink_blocks ++ recent_TAIL_blocks]`` and caps ``seq_lens`` to the
alive token count. Tail anchoring is the load-bearing property: reading the
recent window from the front (right after the sinks) makes decode miss its
own newest tokens.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionMetadataBuilder,
    compute_sinkwindow_rows,
    sinkwindow_max_seq_len,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, SinkWindowSpec
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def test_compaction_tail_anchored():
    bs, start, recent = 16, 32, 64  # 2 sink blocks, 4 recent blocks
    raw = torch.arange(1, 33, dtype=torch.int32).reshape(1, 32)  # blocks 1..32
    dst = torch.zeros_like(raw)
    capped = compute_sinkwindow_rows(
        raw, torch.tensor([300]), 1, bs, start, recent, dst
    )
    # 300 tokens -> 19 valid blocks; tail_start = max(2, 19-4) = 15
    assert dst[0, :2].tolist() == [1, 2]  # sinks kept
    assert dst[0, 2:6].tolist() == [16, 17, 18, 19]  # tail blocks 15..18 (0-idx)
    assert capped == [2 * bs + (300 - 15 * bs)]  # 32 + 60 = 92


def test_no_eviction_yet_degenerates_to_full():
    bs, start, recent = 16, 32, 64
    raw = torch.arange(1, 5, dtype=torch.int32).reshape(1, 4)  # 4 blocks, 50 toks
    dst = torch.zeros_like(raw)
    capped = compute_sinkwindow_rows(raw, torch.tensor([50]), 1, bs, start, recent, dst)
    assert dst[0, :4].tolist() == [1, 2, 3, 4]
    assert capped == [50]


def test_not_front_anchored_after_sinks():
    """Guard against the regression that collapses decode to EOS: the window
    must come from the END of the row, never from the blocks right after the
    sinks."""
    bs, start, recent = 16, 32, 64
    raw = torch.arange(1, 33, dtype=torch.int32).reshape(1, 32)
    dst = torch.zeros_like(raw)
    compute_sinkwindow_rows(raw, torch.tensor([300]), 1, bs, start, recent, dst)
    front_anchored = [3, 4, 5, 6]  # raw[0, 2:6] — what a front read would give
    assert dst[0, 2:6].tolist() != front_anchored
    # The newest block (index nvb-1 = 18, value 19) must be present.
    assert 19 in dst[0].tolist()


def test_capped_len_matches_written_blocks():
    """capped[r] must be coverable by the sink+tail blocks actually written,
    otherwise the kernel reads past the compacted row into stale entries."""
    bs, start, recent = 16, 32, 64
    sink_blocks, recent_blocks = 2, 4
    raw = torch.arange(1, 65, dtype=torch.int32).reshape(1, 64)
    for total_kv in (1, 16, 17, 32, 48, 49, 96, 97, 300, 512):
        dst = torch.zeros_like(raw)
        capped = compute_sinkwindow_rows(
            raw, torch.tensor([total_kv]), 1, bs, start, recent, dst
        )
        nvb = -(-total_kv // bs)
        tail_start = max(sink_blocks, nvb - recent_blocks)
        n_written = sink_blocks + max(0, nvb - tail_start)
        assert 0 < capped[0] <= n_written * bs, (total_kv, capped, n_written)
        assert capped[0] <= total_kv
        # ...and never above the batch-independent bound used for max_seq_len.
        assert capped[0] <= sinkwindow_max_seq_len(bs, start, recent)


def test_recent_window_below_block_size_is_rejected():
    """recent_size // block_size floors to 0, which would delete the whole
    recent tail and leave decode attending to a truncated prefix."""
    bs, start, recent = 16, 32, 8
    raw = torch.arange(1, 33, dtype=torch.int32).reshape(1, 32)
    dst = torch.zeros_like(raw)
    with pytest.raises(AssertionError, match="block size"):
        compute_sinkwindow_rows(raw, torch.tensor([300]), 1, bs, start, recent, dst)


def test_multi_request_rows_are_independent():
    bs, start, recent = 16, 32, 64
    raw = torch.stack(
        [
            torch.arange(1, 33, dtype=torch.int32),  # req 0: blocks 1..32
            torch.arange(101, 133, dtype=torch.int32),  # req 1: blocks 101..132
            torch.arange(201, 233, dtype=torch.int32),  # req 2 (padding row)
        ]
    )
    dst = torch.zeros_like(raw)
    # num_reqs=2, seq_lens padded to 3 rows.
    capped = compute_sinkwindow_rows(
        raw, torch.tensor([300, 50, 999]), 2, bs, start, recent, dst
    )
    assert len(capped) == 2
    assert dst[0, :6].tolist() == [1, 2, 16, 17, 18, 19]
    assert dst[1, :4].tolist() == [101, 102, 103, 104]  # no eviction yet
    assert capped == [92, 50]
    # Rows beyond num_reqs are left untouched.
    assert dst[2].tolist() == [0] * 32


BLOCK_SIZE = 16
START_SIZE = 32
RECENT_SIZE = 64
MAX_MODEL_LEN = 512
MAX_NUM_SEQS = 8


def _make_vllm_config():
    """Minimal mock VllmConfig with only the fields the FA builder touches,
    avoiding any model download / HF config inspection."""
    return SimpleNamespace(
        model_config=SimpleNamespace(
            get_num_attention_heads=lambda _pc: 4,
            get_num_kv_heads=lambda _pc: 2,
            get_head_size=lambda: 64,
            max_model_len=MAX_MODEL_LEN,
            rswa_window=None,
        ),
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: False),
            max_cudagraph_capture_size=None,
        ),
        attention_config=SimpleNamespace(flash_attn_max_num_splits_for_cuda_graph=1),
        scheduler_config=SimpleNamespace(max_num_seqs=MAX_NUM_SEQS),
    )


def _make_sinkwindow_builder():
    spec = SinkWindowSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.bfloat16,
        sliding_window=RECENT_SIZE,
        start_size=START_SIZE,
    )
    return FlashAttentionMetadataBuilder(
        spec, ["layer.0"], _make_vllm_config(), torch.device("cpu")
    )


def _make_common_metadata(
    seq_lens: list[int], block_table: torch.Tensor, max_seq_len: int | None = None
):
    num_reqs = len(seq_lens)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=torch.arange(num_reqs + 1, dtype=torch.int32),
        query_start_loc_cpu=torch.arange(num_reqs + 1, dtype=torch.int32),
        seq_lens=seq_lens_t,
        _seq_lens_cpu=seq_lens_t,
        num_reqs=num_reqs,
        num_actual_tokens=num_reqs,
        max_query_len=1,
        max_seq_len=max_seq_len if max_seq_len is not None else max(seq_lens),
        block_table_tensor=block_table,
        slot_mapping=torch.zeros(num_reqs, dtype=torch.int64),
        causal=True,
    )


def test_builder_disables_cached_metadata_fast_path():
    """The runner's cached-metadata fast path swaps in the RAW block table
    without re-running build(); for SinkWindow that would read evicted
    blocks, so full builds must be forced."""
    assert _make_sinkwindow_builder().supports_update_block_table is False
    # ...and only for SinkWindow: other specs keep the optimization.
    full_spec = FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=2, head_size=64, dtype=torch.bfloat16
    )
    full_builder = FlashAttentionMetadataBuilder(
        full_spec, ["layer.0"], _make_vllm_config(), torch.device("cpu")
    )
    assert full_builder.supports_update_block_table is True


def test_build_swaps_in_compacted_row_and_capped_lens():
    builder = _make_sinkwindow_builder()
    raw = torch.arange(1, 33, dtype=torch.int32).reshape(1, 32)
    md = builder.build(
        common_prefix_len=0,
        common_attn_metadata=_make_common_metadata([300], raw),
    )
    assert md.block_table[0, :6].tolist() == [1, 2, 16, 17, 18, 19]
    assert md.seq_lens.tolist() == [92]
    # max_seq_len is the batch-independent bound, not max(capped) — see
    # test_cudagraph_capture_max_seq_len_is_not_below_replay.
    assert md.max_seq_len == sinkwindow_max_seq_len(BLOCK_SIZE, START_SIZE, RECENT_SIZE)
    assert md.max_seq_len >= 92
    # Ordinary causal varlen path, never cascade.
    assert md.use_cascade is False
    assert md.common_prefix_len == 0


def test_build_does_not_apply_a_kernel_sliding_window():
    """The spec's sliding_window is the recent-window length used to build the
    compacted row, not a kernel mask: an FA window would mask out the sinks,
    which sit at the front of the compacted row."""
    builder = _make_sinkwindow_builder()
    raw = torch.arange(1, 33, dtype=torch.int32).reshape(1, 32)
    md = builder.build(
        common_prefix_len=0,
        common_attn_metadata=_make_common_metadata([300], raw),
    )
    assert md.sliding_window == (-1, -1)


def test_build_reuses_persistent_buffers():
    """CUDA-graph safety: tensor identities must stay stable across builds
    while the contents change."""
    builder = _make_sinkwindow_builder()
    raw = torch.arange(1, 33, dtype=torch.int32).reshape(1, 32)
    first = builder.build(
        common_prefix_len=0,
        common_attn_metadata=_make_common_metadata([300], raw),
    )
    first_bt_ptr = first.block_table.data_ptr()
    first_sl_ptr = first.seq_lens.data_ptr()
    second = builder.build(
        common_prefix_len=0,
        common_attn_metadata=_make_common_metadata([301], raw),
    )
    assert second.block_table.data_ptr() == first_bt_ptr
    assert second.seq_lens.data_ptr() == first_sl_ptr
    assert second.seq_lens.tolist() == [93]
    assert builder._sinkwin_block_table.shape == (MAX_NUM_SEQS, MAX_MODEL_LEN // 16)


def test_cudagraph_capture_max_seq_len_is_not_below_replay():
    """max_seq_len is baked into a captured full CUDA graph. The runner
    inflates it to max_model_len for capture (gpu_model_runner: for_cudagraph
    _capture -> max_seq_len = self.max_model_len) while filling the dummy
    per-request lengths with max_query_len (1 for uniform decode). Deriving
    max_seq_len from that batch would capture ~1 and truncate every replay."""
    builder = _make_sinkwindow_builder()
    raw = torch.arange(1, 65, dtype=torch.int32).reshape(2, 32)
    capture = builder.build_for_cudagraph_capture(
        _make_common_metadata([1, 1], raw, max_seq_len=MAX_MODEL_LEN)
    )
    replay = builder.build(
        common_prefix_len=0,
        common_attn_metadata=_make_common_metadata([300, 300], raw),
    )
    assert replay.seq_lens.tolist() == [92, 92]  # actual per-request lengths
    assert capture.max_seq_len >= replay.max_seq_len
    assert capture.max_seq_len >= max(replay.seq_lens.tolist())
    # Batch-independent: identical scalar at capture and replay.
    assert capture.max_seq_len == replay.max_seq_len
    assert capture.max_seq_len == sinkwindow_max_seq_len(
        BLOCK_SIZE, START_SIZE, RECENT_SIZE
    )


def test_build_rejects_mm_prefix_ranges():
    """PrefixLM mm ranges are absolute token positions; the compacted row
    re-indexes the KV, so the mask would land on the wrong keys."""
    builder = _make_sinkwindow_builder()
    raw = torch.arange(1, 33, dtype=torch.int32).reshape(1, 32)
    cm = _make_common_metadata([300], raw)
    cm.mm_req_doc_ranges = {0: [(10, 40)]}
    with pytest.raises(AssertionError, match="streaming-kv"):
        builder.build(common_prefix_len=0, common_attn_metadata=cm)


def test_build_rejects_cascade_prefix():
    builder = _make_sinkwindow_builder()
    raw = torch.arange(1, 33, dtype=torch.int32).reshape(1, 32)
    with pytest.raises(AssertionError, match="cascade"):
        builder.build(
            common_prefix_len=32,
            common_attn_metadata=_make_common_metadata([300], raw),
        )


def test_runner_returns_zero_common_prefix_for_sinkwindow():
    """The cascade decision must short-circuit for SinkWindowSpec: cascade's
    common-prefix concept is batch-global, the compaction is per-request."""
    spec = SinkWindowSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.bfloat16,
        sliding_window=RECENT_SIZE,
        start_size=START_SIZE,
    )
    # `self` and the builder are never touched on this path.
    prefix_len = GPUModelRunner._compute_cascade_attn_prefix_len(
        object(),
        num_scheduled_tokens=np.array([1]),
        num_computed_tokens=np.array([300]),
        num_common_prefix_blocks=4,
        kv_cache_spec=spec,
        attn_metadata_builder=None,
    )
    assert prefix_len == 0
