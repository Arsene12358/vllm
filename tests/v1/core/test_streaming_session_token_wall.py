# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A streaming session that outgrows max_model_len must fail alone.

A session's prompt grows on every append (_update_request_as_session folds the
prior chunk's computed output back into the prompt and then extends it), and
nothing downstream re-validates that length: InputProcessor only ever sees the
small chunk. The model runner's token_ids_cpu row is max_model_len wide, so an
overflowing append used to take the whole EngineCore down with it. It must be a
per-request error instead: the append is rejected, the session finishes with
FinishReason.ERROR, and the engine keeps serving.
"""

import logging

import pytest

from tests.v1.core.utils import create_scheduler
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import FinishReason
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus, StreamingUpdate


def _session_chunk(prompt_token_ids: list[int], max_tokens: int = 1) -> Request:
    return Request(
        request_id="session",
        prompt_token_ids=prompt_token_ids,
        sampling_params=SamplingParams(max_tokens=max_tokens),
        pooling_params=None,
        resumable=True,
    )


def _step(scheduler) -> dict:
    """Run one schedule/execute/update cycle, sampling one token per request.

    Mirrors EngineCore.step, INCLUDING its `has_requests()` early return: a
    rejection that leaves the session excluded from the unfinished count would
    stop the engine stepping and never reach update_from_output, so the gate has
    to be part of every test's step or the hang is invisible.
    """
    assert scheduler.has_requests(), (
        "engine would stop stepping here: has_requests() is False while a "
        "request is still owned by the scheduler"
    )
    scheduler_output = scheduler.schedule()
    req_ids = list(scheduler_output.num_scheduled_tokens)
    model_output = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        sampled_token_ids=[[42] for _ in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    _LAST_SCHEDULED.clear()
    _LAST_SCHEDULED.extend(req_ids)
    return scheduler.update_from_output(scheduler_output, model_output)


_LAST_SCHEDULED: list[str] = []


def _idle_session(scheduler) -> Request:
    """Seed a session and drive it to the parked WAITING_FOR_STREAMING_REQ state."""
    session = _session_chunk(list(range(8)))
    scheduler.add_request(session)
    for _ in range(4):
        _step(scheduler)
        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            break
    assert session.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    return session


def test_session_append_past_max_model_len_fails_only_the_request():
    scheduler = create_scheduler()
    session = _idle_session(scheduler)
    limit = scheduler.max_model_len
    prompt_before = list(session.prompt_token_ids)
    computed_before = session.num_computed_tokens

    # The append is rejected and the session's token state is left untouched --
    # nothing oversized ever reaches the model runner's max_model_len-wide row.
    update = StreamingUpdate.from_request(_session_chunk(list(range(limit))))
    assert update is not None
    scheduler._update_request_as_session(session, update)
    assert list(session.prompt_token_ids) == prompt_before
    assert session.num_computed_tokens == computed_before
    assert scheduler.streaming_overflow_error_reqs == {"session"}

    # The session stays parked (a blocked status, so nothing schedules it), but
    # the engine must still consider itself to have work, or step() returns
    # before update_from_output can finish the request and the session hangs.
    assert session.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert scheduler.num_waiting_for_streaming_input == 1
    assert scheduler.get_num_unfinished_requests() == 1
    assert scheduler.has_requests()

    # The next engine step finishes just that request, with an error naming the
    # limit -- no exception escapes to kill the EngineCore.
    engine_core_outputs = _step(scheduler)
    outputs = [o for eco in engine_core_outputs.values() for o in eco.outputs]
    errored = [o for o in outputs if o.request_id == "session"]
    assert len(errored) == 1
    assert errored[0].finish_reason == FinishReason.ERROR
    assert "max_model_len" in errored[0].stop_reason
    assert str(limit) in errored[0].stop_reason

    assert "session" not in scheduler.requests
    assert not scheduler.streaming_overflow_error_reqs
    assert scheduler.num_waiting_for_streaming_input == 0


def test_queued_overflow_append_parks_then_errors_the_session():
    """The other call path: the append arrives while the session is generating,
    so _handle_stopped_request pops it. It must park the session (not schedule
    it) and let the next step finish it with an error."""
    scheduler = create_scheduler()
    session = _idle_session(scheduler)
    limit = scheduler.max_model_len

    # Resume the session normally, then queue an overflowing chunk behind it.
    resume = StreamingUpdate.from_request(_session_chunk([7, 8, 9]))
    assert resume is not None
    scheduler._update_request_as_session(session, resume)
    assert session.streaming_queue is not None
    session.streaming_queue.append(
        StreamingUpdate.from_request(_session_chunk(list(range(limit))))
    )

    # One step: the session generates its token, stops, pops the queued chunk,
    # is parked by the rejection, and is finished with an error by the same
    # update_from_output -- so the wall costs exactly one engine step.
    engine_core_outputs = _step(scheduler)
    outputs = [o for eco in engine_core_outputs.values() for o in eco.outputs]
    errored = [o for o in outputs if o.finish_reason == FinishReason.ERROR]
    assert len(errored) == 1
    assert "max_model_len" in errored[0].stop_reason
    assert session.status == RequestStatus.FINISHED_ERROR
    assert not session.streaming_queue  # queued chunks dropped
    assert not scheduler.streaming_overflow_error_reqs
    assert "session" not in scheduler.requests
    assert scheduler.num_waiting_for_streaming_input == 0

    # The engine keeps serving: a fresh request still schedules normally.
    scheduler.add_request(_session_chunk([1, 2, 3]))
    assert _step(scheduler) is not None


def test_rejected_idle_session_keeps_the_engine_stepping_to_its_error():
    """The dominant path at a paced frame rate: chunks arrive while the session
    is parked, so add_request rejects. EngineCore.step() early-returns unless
    has_requests() stays True, so drive the whole loop the way step() does and
    assert the session is never scheduled, never hangs, and ends in an error."""
    scheduler = create_scheduler()
    session = _idle_session(scheduler)
    limit = scheduler.max_model_len

    update = StreamingUpdate.from_request(_session_chunk(list(range(limit))))
    assert update is not None
    scheduler._update_request_as_session(session, update)

    errored = []
    for _ in range(4):  # each iteration asserts has_requests() inside _step
        engine_core_outputs = _step(scheduler)
        assert "session" not in _LAST_SCHEDULED  # parked, never scheduled
        errored += [
            o
            for eco in engine_core_outputs.values()
            for o in eco.outputs
            if o.request_id == "session"
        ]
        assert scheduler.num_waiting_for_streaming_input >= 0  # no double decrement
        if "session" not in scheduler.requests:
            break

    assert len(errored) == 1
    assert errored[0].finish_reason == FinishReason.ERROR
    assert "max_model_len" in errored[0].stop_reason
    assert scheduler.num_waiting_for_streaming_input == 0
    # One more step flushes the finished id, then the engine quiesces cleanly
    # instead of spinning on a phantom request.
    _step(scheduler)
    assert not scheduler.has_requests()


def test_a_later_fitting_chunk_cannot_revive_a_rejected_session():
    """Rejection must be sticky. Chunk sizes vary -- a query chunk is far
    smaller than a frame chunk -- so a later append can still fit under the
    limit. Applying it would un-park the session, let it be scheduled, and
    finish it as LENGTH_CAPPED before the error drain, clearing the set with no
    error output and leaving the next chunk to open a brand-new empty-prompt
    session under the same id: a silent context reset."""
    scheduler = create_scheduler()
    session = _idle_session(scheduler)
    limit = scheduler.max_model_len

    scheduler._update_request_as_session(
        session, StreamingUpdate.from_request(_session_chunk(list(range(limit))))
    )
    assert scheduler.streaming_overflow_error_reqs == {"session"}
    prompt_after_rejection = list(session.prompt_token_ids)

    # A small chunk that comfortably fits must be ignored, not applied.
    scheduler._update_request_as_session(
        session, StreamingUpdate.from_request(_session_chunk([1, 2, 3]))
    )
    assert list(session.prompt_token_ids) == prompt_after_rejection
    assert session.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert scheduler.num_waiting_for_streaming_input == 1  # still parked, once
    assert scheduler.streaming_overflow_error_reqs == {"session"}

    engine_core_outputs = _step(scheduler)
    assert "session" not in _LAST_SCHEDULED  # never revived into scheduling
    outputs = [o for eco in engine_core_outputs.values() for o in eco.outputs]
    errored = [o for o in outputs if o.request_id == "session"]
    assert len(errored) == 1
    assert errored[0].finish_reason == FinishReason.ERROR
    assert "max_model_len" in errored[0].stop_reason
    assert scheduler.num_waiting_for_streaming_input == 0


def test_rejection_is_logged_once_per_session(caplog):
    """A paced source keeps sending chunks; the reason must not be re-logged."""
    scheduler = create_scheduler()
    session = _idle_session(scheduler)
    limit = scheduler.max_model_len

    with caplog.at_level(logging.ERROR):
        for _ in range(3):
            update = StreamingUpdate.from_request(_session_chunk(list(range(limit))))
            assert update is not None
            scheduler._update_request_as_session(session, update)

    assert sum("max_model_len" in r.getMessage() for r in caplog.records) == 1


def test_session_append_within_max_model_len_still_appends():
    """The guard must not fire on an append that still fits."""
    scheduler = create_scheduler()
    session = _idle_session(scheduler)
    computed_before = session.num_computed_tokens

    update = StreamingUpdate.from_request(_session_chunk([1, 2, 3]))
    assert update is not None
    scheduler._update_request_as_session(session, update)
    assert session.num_prompt_tokens == computed_before + 3
    assert session.status == RequestStatus.WAITING
    assert not scheduler.streaming_overflow_error_reqs


@pytest.mark.parametrize("overflow_by", [0, 1])
def test_session_append_rejected_at_the_boundary(overflow_by: int):
    """Reaching max_model_len exactly is already fatal: no room left to sample."""
    scheduler = create_scheduler()
    session = _idle_session(scheduler)
    room = scheduler.max_model_len - session.num_computed_tokens

    update = StreamingUpdate.from_request(
        _session_chunk(list(range(room + overflow_by)))
    )
    assert update is not None
    scheduler._update_request_as_session(session, update)
    assert scheduler.streaming_overflow_error_reqs == {"session"}
    assert session.resumable is False


def test_session_append_just_under_max_model_len_is_allowed():
    """One token of headroom is enough: the session must not be rejected."""
    scheduler = create_scheduler()
    session = _idle_session(scheduler)
    room = scheduler.max_model_len - session.num_computed_tokens

    update = StreamingUpdate.from_request(_session_chunk(list(range(room - 1))))
    assert update is not None
    scheduler._update_request_as_session(session, update)
    assert session.num_prompt_tokens == scheduler.max_model_len - 1
    assert not scheduler.streaming_overflow_error_reqs
