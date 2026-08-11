# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What each penalty is scored over for a streaming-session request.

A streaming session folds every prior chunk's generated tokens into the
request's *prompt* and clears its output ids, so for a long-lived session:

  * ``repetition_penalty`` is scored over prompt UNION output, i.e. over the
    whole session history including every earlier answer, and it never resets;
  * ``frequency_penalty`` / ``presence_penalty`` are scored over the output
    only, which resets on every append, i.e. over the current chunk's answer.

Drivers that keep ONE engine request alive across many queries depend on that
asymmetry when they choose per-chunk penalties, so pin both halves of it.
"""

import torch

from tests.v1.core.utils import create_scheduler
from vllm.model_executor.layers.utils import apply_penalties
from vllm.sampling_params import RepetitionDetectionParams, SamplingParams
from vllm.v1.engine import FinishReason
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus, StreamingUpdate

VOCAB_SIZE = 8
PROMPT_ONLY_TOKEN = 3
OUTPUT_ONLY_TOKEN = 5
UNSEEN_TOKEN = 7


def _penalized(
    *,
    presence: float = 0.0,
    frequency: float = 0.0,
    repetition: float = 1.0,
    output_tokens: list[int] | None = None,
) -> torch.Tensor:
    logits = torch.ones((1, VOCAB_SIZE), dtype=torch.float32)
    return apply_penalties(
        logits,
        torch.tensor([[PROMPT_ONLY_TOKEN]], dtype=torch.int64),
        torch.tensor([output_tokens or [OUTPUT_ONLY_TOKEN]], dtype=torch.int64),
        torch.tensor([presence], dtype=torch.float32),
        torch.tensor([frequency], dtype=torch.float32),
        torch.tensor([repetition], dtype=torch.float32),
    )[0]


def test_repetition_penalty_is_scored_over_prompt_and_output():
    out = _penalized(repetition=2.0)
    assert out[PROMPT_ONLY_TOKEN].item() == 0.5
    assert out[OUTPUT_ONLY_TOKEN].item() == 0.5
    assert out[UNSEEN_TOKEN].item() == 1.0


def test_frequency_and_presence_penalties_are_scored_over_output_only():
    freq = _penalized(frequency=1.0)
    assert freq[PROMPT_ONLY_TOKEN].item() == 1.0  # prompt is not scored
    assert freq[OUTPUT_ONLY_TOKEN].item() == 0.0

    pres = _penalized(presence=1.0)
    assert pres[PROMPT_ONLY_TOKEN].item() == 1.0
    assert pres[OUTPUT_ONLY_TOKEN].item() == 0.0


def test_frequency_penalty_scales_with_repeats_but_presence_does_not():
    """The graded/flat split is why frequency is the usable anti-loop knob."""
    loop = [OUTPUT_ONLY_TOKEN] * 3
    freq = _penalized(frequency=1.0, output_tokens=loop)
    pres = _penalized(presence=1.0, output_tokens=loop)
    assert freq[OUTPUT_ONLY_TOKEN].item() == -2.0
    assert pres[OUTPUT_ONLY_TOKEN].item() == 0.0


def test_repetition_penalty_saturates_once_the_prompt_covers_the_vocabulary():
    """A long session's prompt masks nearly everything, so at temperature 0 the
    penalty stops reordering the plausible tokens and only promotes tokens the
    session has never emitted."""
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.5]], dtype=torch.float32)
    saturated_prompt = torch.tensor([[0, 1, 2]], dtype=torch.int64)  # token 3 unseen
    out = apply_penalties(
        logits.clone(),
        saturated_prompt,
        torch.tensor([[0]], dtype=torch.int64),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([1.3]),
    )[0]
    # Ranking among the masked tokens is untouched (all divided by 1.3)...
    assert out[0] > out[1] > out[2]
    # ...while the never-emitted token gains ground on all of them.
    assert out[3] / logits[0][3] > out[2] / logits[0][2]


def test_session_append_folds_output_into_prompt_and_clears_output_ids():
    """The scheduler-side half: this is what makes the two scopes diverge."""
    scheduler = create_scheduler()
    session = Request(
        request_id="session",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        resumable=True,
    )
    session.append_output_token_ids([41, 42])
    session.num_computed_tokens = session.num_prompt_tokens + 2
    assert list(session.output_token_ids) == [41, 42]

    update = StreamingUpdate.from_request(
        Request(
            request_id="session",
            prompt_token_ids=[9],
            sampling_params=SamplingParams(max_tokens=8),
            pooling_params=None,
            resumable=True,
        )
    )
    assert update is not None
    scheduler._update_request_as_session(session, update)

    # The answer moved into the prompt (repetition_penalty keeps seeing it)...
    assert list(session.prompt_token_ids) == [1, 2, 3, 41, 42, 9]
    # ...and the output ids reset, so frequency/presence start from scratch.
    assert list(session.output_token_ids) == []


# ---------------------------------------------------------------------------
# repetition_detection: the output-scoped anti-loop guard a forever-request
# uses INSTEAD of repetition_penalty. Pin that it survives a session append,
# is scored over the current chunk's output only, and leaves the session alive.
# ---------------------------------------------------------------------------

LOOP_TOKEN = 42
_DETECTION = RepetitionDetectionParams(
    min_pattern_size=1, max_pattern_size=4, min_count=4
)


def _detecting_chunk(prompt_token_ids: list[int], max_tokens: int) -> Request:
    return Request(
        request_id="session",
        prompt_token_ids=prompt_token_ids,
        sampling_params=SamplingParams(
            max_tokens=max_tokens, repetition_detection=_DETECTION
        ),
        pooling_params=None,
        resumable=True,
    )


def _step_sampling(scheduler, token_id: int) -> list:
    """One engine step in which every scheduled request samples `token_id`."""
    scheduler_output = scheduler.schedule()
    req_ids = list(scheduler_output.num_scheduled_tokens)
    model_output = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        sampled_token_ids=[[token_id] for _ in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    engine_core_outputs = scheduler.update_from_output(scheduler_output, model_output)
    return [o for eco in engine_core_outputs.values() for o in eco.outputs]


def test_repetition_detection_is_scored_per_chunk_and_keeps_the_session_alive():
    scheduler = create_scheduler()
    session = _detecting_chunk([1, 2, 3], max_tokens=32)
    scheduler.add_request(session)

    # Two identical tokens in the first chunk: short of min_count=4.
    for _ in range(2):
        _step_sampling(scheduler, LOOP_TOKEN)
    assert list(session.output_token_ids) == [LOOP_TOKEN] * 2
    assert not session.is_finished()

    # Append a query chunk that itself carries loop tokens. The computed loop
    # token folds into the prompt and the output ids reset, so the request's
    # token state now ENDS in four consecutive loop tokens while its output is
    # empty: a detector scored over the whole token state would trip on the very
    # next sampled token, an output-scoped one needs four more.
    update = StreamingUpdate.from_request(
        _detecting_chunk([LOOP_TOKEN] * 3, max_tokens=32)
    )
    assert update is not None
    scheduler._update_request_as_session(session, update)
    assert list(session.prompt_token_ids)[-4:] == [LOOP_TOKEN] * 4
    assert list(session.output_token_ids) == []
    assert session.sampling_params.repetition_detection == _DETECTION

    # If the folded history counted, this would trip on the very first token.
    outputs = []
    for _ in range(3):
        outputs = _step_sampling(scheduler, LOOP_TOKEN)
        assert not any(o.finish_reason is not None for o in outputs)
    assert list(session.output_token_ids) == [LOOP_TOKEN] * 3

    # The fourth repeat inside THIS chunk trips it.
    outputs = _step_sampling(scheduler, LOOP_TOKEN)
    finished = [o for o in outputs if o.finish_reason is not None]
    assert len(finished) == 1
    assert finished[0].finish_reason == FinishReason.REPETITION
    assert finished[0].stop_reason == "repetition_detected"

    # The degenerate answer ended, but the SESSION survives: still owned by the
    # scheduler and parked waiting for the next input chunk.
    assert "session" in scheduler.requests
    assert session.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert not session.is_finished()
