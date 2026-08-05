# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming sessions must honor each chunk's max_tokens.

check_stop (vllm/v1/core/sched/utils.py) reads the cached
Request.max_tokens, so _update_request_as_session has to refresh it from
every StreamingUpdate. Otherwise a session keeps the FIRST chunk's budget
forever and cannot interleave input-only chunks (max_tokens=1) with
chunks that generate a full answer.
"""

from tests.v1.core.utils import create_scheduler
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request, StreamingUpdate


def _session_chunk(max_tokens: int, prompt_token_ids: list[int]) -> Request:
    return Request(
        request_id="session",
        prompt_token_ids=prompt_token_ids,
        sampling_params=SamplingParams(max_tokens=max_tokens),
        pooling_params=None,
        resumable=True,
    )


def test_streaming_session_applies_per_chunk_max_tokens():
    scheduler = create_scheduler()

    # First chunk: an input-only append with a generation budget of 1.
    session = _session_chunk(max_tokens=1, prompt_token_ids=[1, 2, 3])
    session.num_computed_tokens = session.num_prompt_tokens
    assert session.max_tokens == 1

    # Second chunk: a query that should generate a full answer.
    update = StreamingUpdate.from_request(
        _session_chunk(max_tokens=300, prompt_token_ids=[4, 5, 6])
    )
    assert update is not None and update.max_tokens == 300
    scheduler._update_request_as_session(session, update)
    assert session.max_tokens == 300

    # Third chunk: back to an input-only append.
    session.num_computed_tokens = session.num_prompt_tokens
    update = StreamingUpdate.from_request(
        _session_chunk(max_tokens=1, prompt_token_ids=[7])
    )
    assert update is not None
    scheduler._update_request_as_session(session, update)
    assert session.max_tokens == 1
