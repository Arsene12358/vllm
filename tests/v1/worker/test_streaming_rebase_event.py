# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the streaming-KV position rebase EVENT (runner seats).

The rotation kernel itself is proven in
``tests/v1/attention/test_streaming_rebase.py``; this suite pins the event
that drives it from the runner:

  * Seat B (``maybe_rebase_request``) — threshold, compacted-row handoff
    (``token_start`` = the ALIGNED sink boundary, ``token_end`` = the capped
    alive count), rotation of EVERY layer's K cache, offset bump, position
    shift, the FROZEN log line;
  * Seat A (``apply_rebase_offset``) — re-applying the persisted offset after
    ``_init_mrope_positions`` recomputes positions from scratch (without it
    the next streaming chunk silently reverts the rebase);
  * the fences — non-``SinkWindowSpec`` groups are excluded, and a stride-0
    (``expand``ed) position tensor raises BEFORE anything is mutated.

Everything runs against a real ``MRotaryEmbedding``, a real
``CachedRequestState``, a real ``SinkWindowSpec`` and a synthetic packed
paged K cache in the v0.26.0 FlashAttention layout.
"""

import contextlib
import logging
from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
from vllm.v1.attention.streaming_rebase import (
    apply_rebase_offset,
    maybe_rebase_request,
    rotate_kv_pages,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, SinkWindowSpec
from vllm.v1.worker import gpu_model_runner
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.utils import AttentionGroup

# Qwen3-Omni Thinker text rope config, as in the rotation-module suite.
HEAD_DIM = 128
ROTARY_DIM = 128
MAX_POS = 65536
ROPE_THETA = 1_000_000.0
MROPE_SECTION = [24, 20, 20]
NUM_KV_HEADS = 2

# Geometry: 2 sink blocks, 4 recent blocks, minimum legal rebase threshold.
BLOCK_SIZE = 16
START_SIZE = 32
RECENT_SIZE = 64
REBASE_AT = START_SIZE + 2 * RECENT_SIZE  # 160
NUM_COMPUTED = 300  # 19 valid blocks; tail_start = 15
NUM_PROMPT = 320
ALIGNED_START = 32  # cdiv(START_SIZE, BLOCK_SIZE) * BLOCK_SIZE
WINDOW_START = 240  # tail_start * BLOCK_SIZE
CAPPED = 92  # ALIGNED_START + (NUM_COMPUTED - WINDOW_START)
NUM_CACHE_BLOCKS = 25

# Logical block table: [sinks ++ evicted nulls ++ recent tail].
SINK_IDS = [11, 12]
TAIL_IDS = [21, 22, 23, 24]
BLOCK_IDS = SINK_IDS + [0] * 13 + TAIL_IDS
COMPACTED_IDS = SINK_IDS + TAIL_IDS


@contextlib.contextmanager
def _config_ctx():
    # MRotaryEmbedding is a CustomOp; its __init__ reads the current vLLM
    # compilation config to pick a forward impl.
    with set_current_vllm_config(VllmConfig()):
        yield


def _make_rope(cls: type[MRotaryEmbedding] = MRotaryEmbedding) -> MRotaryEmbedding:
    return cls(
        head_size=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        max_position_embeddings=MAX_POS,
        base=ROPE_THETA,
        is_neox_style=True,
        dtype=torch.float32,
        mrope_section=MROPE_SECTION,
        mrope_interleaved=True,
    )


def _make_spec(**kwargs) -> SinkWindowSpec:
    return SinkWindowSpec(
        block_size=kwargs.pop("block_size", BLOCK_SIZE),
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_DIM,
        dtype=torch.float32,
        sliding_window=kwargs.pop("sliding_window", RECENT_SIZE),
        start_size=kwargs.pop("start_size", START_SIZE),
        **kwargs,
    )


def _make_key_caches(num_layers: int, seed: int = 0):
    """`num_layers` packed KV tensors + the K views the FA impl derives."""
    torch.manual_seed(seed)
    packed = [
        torch.randn(
            NUM_CACHE_BLOCKS,
            NUM_KV_HEADS,
            BLOCK_SIZE,
            2 * HEAD_DIM,
            dtype=torch.float32,
        )
        for _ in range(num_layers)
    ]
    key_caches = [kv.transpose(1, 2).split(HEAD_DIM, dim=-1)[0] for kv in packed]
    return packed, key_caches


def _text_positions(n: int = NUM_PROMPT) -> torch.Tensor:
    """Text-like positions: axis i of token t is t + i (min axis == t)."""
    idx = torch.arange(n, dtype=torch.long)
    return torch.stack([idx, idx + 1, idx + 2])


def _video_positions(n: int = NUM_PROMPT) -> torch.Tensor:
    """Genuine multimodal positions: a shared per-token base plus intra-frame
    grid offsets, so t != h != w and a token's axes are spread over more than
    one position."""
    idx = torch.arange(n, dtype=torch.long)
    pos = torch.stack([idx, idx + idx % 5, idx + (idx % 7) * 2])
    assert not torch.equal(pos[0], pos[1]) and not torch.equal(pos[1], pos[2])
    return pos


def _make_req_state(positions: torch.Tensor | None = None) -> CachedRequestState:
    if positions is None:
        positions = _text_positions()
    return CachedRequestState(
        req_id="req0",
        prompt_token_ids=[0] * NUM_PROMPT,
        mm_features=[],
        sampling_params=None,
        generator=None,
        block_ids=(list(BLOCK_IDS),),
        num_computed_tokens=NUM_COMPUTED,
        output_token_ids=[],
        mrope_positions=positions,
        mrope_position_delta=-42,
    )


def _rebase(req_state, key_caches, *, spec=None, rebase_at=REBASE_AT, rotary=None):
    return maybe_rebase_request(
        req_state,
        key_caches=key_caches,
        block_ids=req_state.block_ids[0],
        kv_cache_spec=spec if spec is not None else _make_spec(),
        rebase_at=rebase_at,
        rotary=rotary,
    )


def _capture_streaming_kv_logs():
    """Pin the FROZEN log line the same way the eviction-line test does."""
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    log = logging.getLogger("vllm.v1.attention.streaming_rebase")
    handler = _Capture(level=logging.INFO)
    old_level = log.level

    @contextlib.contextmanager
    def _ctx():
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        try:
            yield records
        finally:
            log.removeHandler(handler)
            log.setLevel(old_level)

    return _ctx()


# ----------------------------------------------------------------------------
# Seat B — the rebase event
# ----------------------------------------------------------------------------
def test_rebase_rotates_the_aligned_recent_range_on_every_layer():
    """The handoff is pinned exactly: compacted row, token_start == the
    ALIGNED sink boundary (never raw start_size), token_end == capped."""
    packed, key_caches = _make_key_caches(3, seed=1)
    before = [kv.clone() for kv in packed]
    req_state = _make_req_state()

    with _config_ctx():
        rope = _make_rope()
        delta = _rebase(req_state, key_caches, rotary=rope)

        assert delta == WINDOW_START - ALIGNED_START  # 208

        # Independent replay of the expected call, on untouched copies.
        want_packed = [kv.clone() for kv in before]
        for kv in want_packed:
            want_k = kv.transpose(1, 2).split(HEAD_DIM, dim=-1)[0]
            rotate_kv_pages(
                want_k, COMPACTED_IDS, ALIGNED_START, CAPPED, delta, rope, BLOCK_SIZE
            )

    for got, want in zip(packed, want_packed):
        assert torch.equal(got, want)

    # Sinks, evicted pages and pages outside the row are bitwise intact...
    untouched = [b for b in range(NUM_CACHE_BLOCKS) if b not in TAIL_IDS]
    for got, orig in zip(packed, before):
        assert torch.equal(got[untouched], orig[untouched])
        # ...and so are the alive window's unwritten tail slots.
        tail_slots = CAPPED - ALIGNED_START - 3 * BLOCK_SIZE  # 12 of 16
        got_k = got.transpose(1, 2).split(HEAD_DIM, dim=-1)[0]
        orig_k = orig.transpose(1, 2).split(HEAD_DIM, dim=-1)[0]
        assert torch.equal(
            got_k[TAIL_IDS[-1], tail_slots:], orig_k[TAIL_IDS[-1], tail_slots:]
        )
        # The rotated range really changed.
        assert not torch.equal(got_k[TAIL_IDS[0]], orig_k[TAIL_IDS[0]])


def test_rebase_bumps_offset_and_shifts_both_position_fields():
    _, key_caches = _make_key_caches(1, seed=2)
    req_state = _make_req_state()
    positions_before = req_state.mrope_positions.clone()

    with _config_ctx():
        delta = _rebase(req_state, key_caches, rotary=_make_rope())

    assert delta == 208
    assert req_state.mrope_rebase_offset == 208
    assert torch.equal(req_state.mrope_positions, positions_before - 208)
    assert req_state.mrope_position_delta == -42 - 208


def test_rebase_emits_the_frozen_log_line():
    _, key_caches = _make_key_caches(1, seed=3)
    req_state = _make_req_state()

    with _config_ctx(), _capture_streaming_kv_logs() as records:
        _rebase(req_state, key_caches, rotary=_make_rope())

    messages = [r.getMessage() for r in records if "[streaming-kv]" in r.getMessage()]
    assert messages == [
        "[streaming-kv] rebase req=req0 delta=208 new_base=32 recent_tokens=60"
    ]


def test_no_rebase_below_threshold():
    packed, key_caches = _make_key_caches(2, seed=4)
    before = [kv.clone() for kv in packed]
    req_state = _make_req_state()
    positions_before = req_state.mrope_positions.clone()

    with _config_ctx(), _capture_streaming_kv_logs() as records:
        delta = _rebase(req_state, key_caches, rebase_at=100_000, rotary=_make_rope())

    assert delta == 0
    assert req_state.mrope_rebase_offset == 0
    assert req_state.mrope_position_delta == -42
    assert torch.equal(req_state.mrope_positions, positions_before)
    for got, orig in zip(packed, before):
        assert torch.equal(got, orig)
    assert not [r for r in records if "rebase" in r.getMessage()]


def test_second_call_at_the_same_base_is_a_no_op():
    """Idempotence: once the window sits at the sink boundary the threshold no
    longer trips, so no session can be double-rotated at one base."""
    packed, key_caches = _make_key_caches(2, seed=5)
    req_state = _make_req_state()

    with _config_ctx():
        rope = _make_rope()
        first = _rebase(req_state, key_caches, rotary=rope)
        after_first = [kv.clone() for kv in packed]
        second = _rebase(req_state, key_caches, rotary=rope)

    assert first == 208
    assert second == 0
    assert req_state.mrope_rebase_offset == 208
    for got, once in zip(packed, after_first):
        assert torch.equal(got, once)


def test_rebased_window_never_lands_below_the_sink_boundary():
    """Aligned-Δ policy: Δ comes from the alive window's MINIMUM effective
    position, so no alive token can be rebased onto sink-block positions —
    and the whole window ends up under the threshold."""
    _, key_caches = _make_key_caches(1, seed=6)
    req_state = _make_req_state(_video_positions())
    front_column = req_state.mrope_positions[:, WINDOW_START]
    assert int(front_column.max()) > int(front_column.min())  # genuinely mm

    with _config_ctx():
        delta = _rebase(req_state, key_caches, rotary=_make_rope())

    assert delta > 0
    alive = req_state.mrope_positions[:, WINDOW_START:NUM_COMPUTED]
    assert int(alive.min()) == ALIGNED_START
    assert int(alive.max()) < REBASE_AT


def test_delta_is_measured_from_the_aligned_boundary_not_raw_start_size():
    """`start_size` unaligned to the block size (rejected in rebase mode by
    config validation, but neither the Δ nor the rotation range may depend on
    that): the compacted row packs sinks as WHOLE blocks, so rotating from raw
    `start_size` would rotate pinned sink tokens."""
    packed, key_caches = _make_key_caches(1, seed=7)
    before = packed[0].clone()
    req_state = _make_req_state()

    with _config_ctx():
        delta = _rebase(
            req_state,
            key_caches,
            spec=_make_spec(start_size=30),
            rotary=_make_rope(),
        )

    # cdiv(30, 16) * 16 == 32, not 30.
    assert delta == WINDOW_START - 32
    # ...and the sink pages (which hold compacted tokens 0..31) are intact.
    assert torch.equal(packed[0][SINK_IDS], before[SINK_IDS])


# ----------------------------------------------------------------------------
# Seat B — runner wiring (unbound-call idiom: a real GPUModelRunner needs a GPU)
# ----------------------------------------------------------------------------
def test_seat_b_drives_every_batched_request_with_its_own_row():
    packed, key_caches = _make_key_caches(1, seed=10)
    req_state = _make_req_state()
    with _config_ctx():
        rope = _make_rope()
        runner = SimpleNamespace(
            cache_config=SimpleNamespace(streaming_kv_rebase_at=REBASE_AT),
            _streaming_rebase_context=(0, _make_spec(), key_caches, rope),
            input_batch=SimpleNamespace(req_ids=["req0"]),
            requests={"req0": req_state},
        )
        GPUModelRunner._maybe_rebase_streaming_positions(runner)

    assert req_state.mrope_rebase_offset == 208
    # The rotation went through the request's own compacted row.
    assert not torch.equal(
        packed[0].transpose(1, 2).split(HEAD_DIM, dim=-1)[0][TAIL_IDS[0]],
        _make_key_caches(1, seed=10)[1][0][TAIL_IDS[0]],
    )


def test_rebase_context_resolves_k_views_and_one_shared_rotary():
    """The K view handed to the rotation must alias the layer's KV cache (the
    rotation writes through it) and every layer of the group must be there."""
    packed, _ = _make_key_caches(2, seed=11)
    layer_names = ["l0", "l1"]
    spec = _make_spec()
    with _config_ctx():
        # vllm-omni's patch.py rebinds `MRotaryEmbedding` process-wide to its
        # subclass in every vllm module, so build the rope from whatever class
        # the runner module resolved — that is the one `get_rope` would have
        # instantiated in the same process.
        rope = _make_rope(cls=gpu_model_runner.MRotaryEmbedding)
        model = torch.nn.Module()
        model.add_module("rope", rope)
        runner = SimpleNamespace(
            attn_groups=[
                [
                    AttentionGroup(
                        backend=None,
                        layer_names=layer_names,
                        kv_cache_spec=spec,
                        kv_cache_group_id=0,
                    )
                ]
            ],
            compilation_config=SimpleNamespace(
                static_forward_context={
                    name: SimpleNamespace(kv_cache=kv)
                    for name, kv in zip(layer_names, packed)
                }
            ),
            runner_only_attn_layers=set(),
            get_model=lambda: model,
        )
        group_id, got_spec, key_caches, rotary = (
            GPUModelRunner._build_streaming_rebase_context(runner)
        )

    assert (group_id, got_spec, rotary) == (0, spec, rope)
    assert len(key_caches) == 2
    for view, kv in zip(key_caches, packed):
        assert view.shape == (NUM_CACHE_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
        assert view.data_ptr() == kv.data_ptr()  # aliases K, not a copy


# ----------------------------------------------------------------------------
# Seat A — the persisted offset survives a from-scratch recompute
# ----------------------------------------------------------------------------
def test_seat_a_reapplies_the_offset_after_a_recompute():
    req_state = _make_req_state()
    req_state.mrope_rebase_offset = 208
    # `_init_mrope_positions` rebuilds both fields from scratch (raw values).
    req_state.mrope_positions = _text_positions()
    req_state.mrope_position_delta = -42
    raw = req_state.mrope_positions.clone()

    apply_rebase_offset(req_state)

    assert torch.equal(req_state.mrope_positions, raw - 208)
    assert req_state.mrope_position_delta == -42 - 208
    assert req_state.mrope_rebase_offset == 208  # not consumed


def test_seat_a_is_a_no_op_without_an_offset():
    req_state = _make_req_state()
    raw = req_state.mrope_positions.clone()

    apply_rebase_offset(req_state)

    assert torch.equal(req_state.mrope_positions, raw)
    assert req_state.mrope_position_delta == -42


# ----------------------------------------------------------------------------
# Fences
# ----------------------------------------------------------------------------
def test_non_sinkwindow_group_is_excluded():
    """Fence 1: the gate is the KV spec, never `uses_mrope`."""
    packed, key_caches = _make_key_caches(1, seed=8)
    before = packed[0].clone()
    req_state = _make_req_state()
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_DIM,
        dtype=torch.float32,
    )

    with _config_ctx():
        delta = _rebase(req_state, key_caches, spec=spec, rotary=_make_rope())

    assert delta == 0
    assert req_state.mrope_rebase_offset == 0
    assert torch.equal(packed[0], before)


def test_stride0_positions_raise_before_any_mutation():
    """Fence 1 backstop: talker/code2wav positions are stride-0 `expand()`
    views. In-place ops on them raise — the event must refuse up front, not
    after rotating K."""
    packed, key_caches = _make_key_caches(1, seed=9)
    before = packed[0].clone()
    expanded = torch.arange(NUM_PROMPT, dtype=torch.long).unsqueeze(0).expand(3, -1)
    assert 0 in expanded.stride()
    req_state = _make_req_state(expanded)

    with _config_ctx(), pytest.raises((AssertionError, RuntimeError), match="stride"):
        _rebase(req_state, key_caches, rotary=_make_rope())

    assert torch.equal(packed[0], before)
    assert req_state.mrope_rebase_offset == 0
