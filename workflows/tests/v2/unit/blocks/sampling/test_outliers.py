"""``roboflow_core/identify_outliers@v1``: vMF window rank per source, bounded and checked.

``V1_*`` were recorded by running V1 ``identify_outliers/v1_tensor.py`` in
float32 on the camera sequences of ``test_changes.camera`` (threshold 0.2,
warmup 3, window 5); the recorder lives in the harness task
``ecosystem-e4-embedding-sampling/v1_trace``. They hold V1's left ranks. V2
ranks ties at their middle, so on these untied traces each V1 rank ``r / n``
becomes ``(2r + 1) / (2n)`` and the flags follow; ``mid_cdf`` derives that.
"""

import math
from typing import Any, Dict, List, Optional

import pytest
import torch
from pydantic import ValidationError
from roboflow_workflows.v2.blocks.sampling import (
    IdentifyOutliersBlock,
    SamplingCapacityError,
)
from roboflow_workflows.v2.blocks.sampling.identify_outliers.v1 import _mean_direction, _percentile
from roboflow_workflows.v2.data.batch import SampleContext
from roboflow_workflows.v2.engine.context import (
    ExecutionContext,
    use_execution_context,
)

PARAMS = {"threshold_percentile": 0.2, "warmup": 3, "window_size": 5}


def camera(phase: float, shift_at: int, steps: int) -> List[List[float]]:
    """4-dim embeddings wobbling around one direction, then another from shift_at."""
    rows = []
    for i in range(steps):
        wobble = (
            round(0.08 * math.sin(1.7 * i + phase), 3),
            round(0.06 * math.cos(1.1 * i + phase), 3),
        )
        if i < shift_at:
            row = [1.0, 0.2 + wobble[0], 0.1 + wobble[1], 0.05]
        else:
            row = [0.1 + wobble[0], 1.0, 0.3 + wobble[1], -0.2]
        rows.append(row)

    return rows


A = camera(0.0, 9, 14)
B = camera(2.0, 6, 10)
INTERLEAVED = [row for pair in zip(A, B) for row in pair] + A[len(B) :]
SOURCES = ["cam-a", "cam-b"] * len(B) + ["cam-a"] * (len(A) - len(B))

# (is_outlier, percentile) per call; the first two calls warm up.
V1_A = [
    (False, 0.5), (False, 0.5), (True, 0.0), (True, 0.0), (False, 0.8),
    (False, 0.4), (False, 0.4), (False, 0.8), (False, 0.2), (True, 0.0),
    (True, 0.0), (False, 0.6), (False, 0.8), (False, 0.8),
]  # fmt: skip
V1_B = [
    (False, 0.5), (False, 0.5), (False, 1 / 3), (True, 0.0), (False, 0.4),
    (False, 0.6), (True, 0.0), (False, 0.2), (False, 0.6), (False, 0.2),
]  # fmt: skip
V1_INTERLEAVED = [
    (False, 0.5), (False, 0.5), (False, 2 / 3), (True, 0.0), (False, 0.8),
    (False, 0.2), (False, 0.2), (True, 0.0), (False, 0.4), (False, 0.2),
    (False, 0.4), (True, 0.0), (False, 0.2), (True, 0.0), (False, 0.4),
    (False, 0.2), (False, 0.8), (False, 0.6), (False, 0.6), (False, 0.2),
    (False, 0.4), (False, 0.6), (True, 0.0), (False, 0.6),
]  # fmt: skip


def mid_cdf(v1_trace, *, warmup=3, window=5, threshold=0.2):
    """V2's expected (is_outlier, percentile) from a V1 left-rank trace."""
    expected = []
    for call, (is_outlier, percentile) in enumerate(v1_trace):
        if call + 1 < warmup:
            expected.append((is_outlier, percentile))
            continue

        n = min(call + 1, window)
        twice_rank = 2 * round(percentile * n) + 1
        mid = twice_rank / (2 * n)
        expected.append((mid < threshold or mid > 1 - threshold, mid))

    return expected


class _Scope:
    """Engine side of a call whose ``embedding`` binding has context ``sample``."""

    def __init__(self, sample: Optional[SampleContext]) -> None:
        self._sample = sample

    def sample_at(self, index, *, field=None):
        # The state key must come from the embedding, not the whole call.
        assert field == "embedding", field
        return self._sample

    def temporal_at(self, index, *, field=None):
        return None


def _call(
    block: IdentifyOutliersBlock,
    embedding: Any,
    *,
    source: Optional[str] = "cam-a",
    index=(0,),
    **overrides: Any,
) -> Dict[str, Any]:
    sample = SampleContext(source_id=source) if source is not None else None
    context = ExecutionContext(
        step_path=("outliers",),
        block_type=IdentifyOutliersBlock.type,
        session_id="session",
        run_id="run",
        indices=(index,),
        call_scope=_Scope(sample),
    )
    if not isinstance(embedding, torch.Tensor):
        embedding = torch.tensor(embedding)
    params = {**PARAMS, "scope": "source", "max_sources": 256, **overrides}
    with use_execution_context(context):
        result = IdentifyOutliersBlock.run(block, embedding=embedding, **params)

    return result


def _pair(result: Dict[str, Any]) -> tuple:
    pair = (result["is_outlier"], result["percentile"])

    return pair


# ------------------------------------------------------------------ V1 trace
def test_matches_the_recorded_v1_trace_ranked_at_the_middle():
    block = IdentifyOutliersBlock()

    results = [_call(block, row) for row in A]

    assert [_pair(r) for r in results] == mid_cdf(V1_A)
    assert [r["warming_up"] for r in results] == [True, True] + [False] * 12
    assert all(type(r["percentile"]) is float for r in results)


def test_mid_cdf_changes_these_v1_flags_on_purpose():
    # Lowest of 5: V1 0.0 -> 0.1, still flagged. Highest of 5: V1 0.8 was not
    # above 1 - 0.2, its middle rank 0.9 is. Second lowest: 0.2 -> 0.3.
    changed = [(call, v1, v2) for call, (v1, v2) in enumerate(zip(V1_A, mid_cdf(V1_A)))]

    assert [(c, v1, v2) for c, v1, v2 in changed if v1[0] != v2[0]] == [
        (4, (False, 0.8), (True, 0.9)),
        (7, (False, 0.8), (True, 0.9)),
        (12, (False, 0.8), (True, 0.9)),
        (13, (False, 0.8), (True, 0.9)),
    ]


@pytest.mark.parametrize("threshold, flagged", [(0.1, False), (0.11, True)])
def test_the_threshold_is_strict_unlike_identify_changes(threshold, flagged):
    block = IdentifyOutliersBlock()
    rows = [[1.0, 0.0]] * 4 + [[0.0, 1.0]]

    results = [_call(block, row, threshold_percentile=threshold) for row in rows]

    # The new direction is the unique lowest of 5: percentile 1 / 10.
    assert _pair(results[-1]) == (flagged, 0.1)


def test_two_cameras_keep_separate_windows():
    block = IdentifyOutliersBlock()

    got = [_pair(_call(block, row, source=s)) for row, s in zip(INTERLEAVED, SOURCES)]

    assert [p for p, s in zip(got, SOURCES) if s == "cam-a"] == mid_cdf(V1_A)
    assert [p for p, s in zip(got, SOURCES) if s == "cam-b"] == mid_cdf(V1_B)


def test_global_scope_shares_one_window_like_v1():
    block = IdentifyOutliersBlock()

    got = [
        _pair(_call(block, row, source=s, scope="global"))
        for row, s in zip(INTERLEAVED, SOURCES)
    ]

    assert got == mid_cdf(V1_INTERLEAVED)


def test_without_a_source_each_top_level_position_is_a_source():
    block = IdentifyOutliersBlock()

    first = [_call(block, row, source=None, index=(0, 2)) for row in A[:3]]
    other = _call(block, B[0], source=None, index=(1,))

    assert [r["warming_up"] for r in first] == [True, True, False]
    assert other["warming_up"] is True


def test_a_source_too_many_fails_and_keeps_the_others():
    block = IdentifyOutliersBlock()
    _call(block, A[0], source="cam-a", max_sources=1)

    with pytest.raises(SamplingCapacityError, match="max_sources=1"):
        _call(block, A[0], source="cam-b", max_sources=1)

    assert _call(block, A[1], source="cam-a", max_sources=1)["warming_up"] is True
    assert _call(block, A[2], source="cam-a", max_sources=1)["warming_up"] is False


def test_reset_state_forgets_every_window():
    block = IdentifyOutliersBlock()
    for row in A[:4]:
        _call(block, row)

    block.reset_state()

    assert _call(block, A[0])["warming_up"] is True


# ----------------------------------------------------- warmup and windows
def test_warmup_counts_the_current_call():
    block = IdentifyOutliersBlock()

    first = _call(block, [1.0, 0.0], warmup=1)

    # One embedding is its own mean direction and ties with itself: the middle.
    assert first == {"is_outlier": False, "percentile": 0.5, "warming_up": False}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("window_size", [1, 32])
def test_a_constant_stream_is_central_not_an_outlier(dtype, window_size):
    # V1 ranked ties from the left: percentile 0 and an alarm on every call.
    block = IdentifyOutliersBlock()
    embedding = torch.tensor([0.0, 3.0, 4.0], dtype=dtype)

    results = [
        _call(block, embedding, threshold_percentile=0.05, window_size=window_size)
        for _ in range(6)
    ]

    assert [_pair(r) for r in results] == [(False, 0.5)] * 6


def test_tied_scores_share_the_middle_of_their_ranks():
    block = IdentifyOutliersBlock()
    # Three equal high scores, then two equal low ones.
    rows = [[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 2

    results = [_call(block, row, warmup=1) for row in rows]

    # Highs tie at ranks 0..n-1; the first low is unique, the second ties
    # with it at ranks 0 and 1: (0 + 2) / (2 * 5).
    assert [r["percentile"] for r in results] == [0.5, 0.5, 0.5, 1 / 8, 0.2]


def test_cancelling_directions_fall_back_to_the_current_embedding():
    block = IdentifyOutliersBlock()

    results = [_call(block, row, warmup=2) for row in ([1.0, 0.0], [-1.0, 0.0])]

    # mu = e: e scores 1 and ranks highest of 2, the middle of rank 1 of 2.
    assert _pair(results[1]) == (False, 0.75)


def test_the_rank_uses_the_newest_embeddings_own_score():
    # QA input: seeded unit rows whose newest row has the lowest score. V1
    # ranked a separate dot(mu, e) that rounds above that row here (on this
    # BLAS), so it ranked e second instead of first.
    kept = torch.randn((8, 16), generator=torch.Generator().manual_seed(23))
    kept = kept / kept.norm(dim=1, keepdim=True)
    scores = kept @ _mean_direction(kept)
    assert int(scores.argmin()) == 7

    assert _percentile(kept) == 1 / 16


def test_eight_embeddings_cannot_reach_the_default_tail():
    # Small-sample limit: the lowest of n untied ranks at 1 / (2n); at n = 8
    # that is 0.0625, not below the default 0.05.
    block = IdentifyOutliersBlock()
    rows = [[1.0, 0.01 * n] for n in range(7)] + [[0.0, 1.0]]

    results = [
        _call(block, row, threshold_percentile=0.05, window_size=8) for row in rows
    ]

    assert _pair(results[-1]) == (False, 1 / 16)


def test_a_real_change_fires_the_default_tail_in_a_default_window():
    block = IdentifyOutliersBlock()
    generator = torch.Generator().manual_seed(7)
    scene = torch.randn(16, generator=generator)
    rows = [scene + 0.05 * torch.randn(16, generator=generator) for _ in range(20)]
    other = torch.randn(16, generator=generator)

    steady = [
        _call(block, row, threshold_percentile=0.05, window_size=32) for row in rows
    ]
    changed = _call(block, other, threshold_percentile=0.05, window_size=32)

    # Unique lowest of 21: 1 / 42 < 0.05.
    assert _pair(changed) == (True, 1 / 42)
    assert not steady[-1]["warming_up"]


def test_a_smaller_selected_window_trims_at_once():
    block = IdentifyOutliersBlock()
    for row in A[:5]:
        _call(block, row, window_size=5)
    kept = block._states.get(("source", "cam-a")).embeddings

    sizes = []
    for row, size in zip(A[5:9], [2, 2, 1, 3]):
        result = _call(block, row, window_size=size)
        sizes.append(len(kept))

    assert sizes == [2, 2, 1, 2]
    newest_two = IdentifyOutliersBlock()
    for row in A[7:9]:
        expected = _call(newest_two, row, warmup=1, window_size=3)
    assert result == expected
    assert block._states.get(("source", "cam-a")).samples == 9


def test_a_zero_vector_is_kept_as_a_new_zero_tensor():
    block = IdentifyOutliersBlock()
    zero = torch.zeros(3)

    _call(block, zero)

    kept = block._states.get(("source", "cam-a")).embeddings[0]
    assert kept is not zero
    torch.testing.assert_close(kept, zero)


def test_zero_and_finite_vectors_mix_without_nan():
    block = IdentifyOutliersBlock()
    rows = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]

    results = [_call(block, row, warmup=1) for row in rows]

    percentiles = [r["percentile"] for r in results]
    assert all(0 <= p <= 1 for p in percentiles)
    # Every call alone and all-zero windows tie: the middle.
    assert percentiles[0] == 0.5
    assert _call(IdentifyOutliersBlock(), [0.0, 0.0], warmup=1)["percentile"] == 0.5


# ----------------------------------------------------------- invalid inputs
@pytest.mark.parametrize(
    "changed",
    [torch.tensor(A[1][:3]), torch.tensor(A[1], dtype=torch.float64)],
    ids=["length", "dtype"],
)
def test_an_embedding_unlike_the_window_fails_even_during_warmup(changed):
    # V1 accepted it during warmup and then failed on every later call.
    block = IdentifyOutliersBlock()
    _call(block, A[0])

    with pytest.raises(ValueError, match="does not match this source"):
        _call(block, changed)

    assert [_pair(_call(block, row)) for row in A[1:4]] == mid_cdf(V1_A)[1:4]


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("scope", ["source", "global"])
def test_a_nonfinite_embedding_fails_and_changes_nothing(bad, scope):
    block = IdentifyOutliersBlock()

    # A rejected first call of cam-b takes no max_sources slot.
    with pytest.raises(ValueError, match="must be finite"):
        _call(block, [bad, 1.0, 0.0, 0.0], source="cam-b", scope=scope, max_sources=1)
    results = [_call(block, A[0], scope=scope, max_sources=1)]
    with pytest.raises(ValueError, match="must be finite"):
        _call(block, [1.0, bad, 0.0, 0.0], scope=scope, max_sources=1)
    results += [_call(block, row, scope=scope, max_sources=1) for row in A[1:6]]

    assert [_pair(r) for r in results] == mid_cdf(V1_A)[:6]
    assert block._states.get(("source", "cam-b")) is None


def test_a_finite_embedding_whose_norm_overflows_fails():
    with pytest.raises(ValueError, match="got norm inf"):
        _call(IdentifyOutliersBlock(), [1e30, 1e30])


@pytest.mark.parametrize(
    "embedding",
    [torch.zeros(0), torch.zeros(2, 3), torch.tensor([1, 2, 3])],
    ids=["empty", "2-D", "integer"],
)
def test_an_embedding_must_be_a_non_empty_1d_float_tensor(embedding):
    with pytest.raises(ValueError, match="non-empty 1-D floating tensor"):
        _call(IdentifyOutliersBlock(), embedding)


@pytest.mark.parametrize(
    "overrides",
    [
        {"threshold_percentile": -0.1},
        {"window_size": None},
        {"window_size": 0},
    ],
)
def test_selected_parameters_are_checked_before_any_update(overrides):
    block = IdentifyOutliersBlock()

    with pytest.raises(ValueError):
        _call(block, A[0], **overrides)

    assert block._states.get(("source", "cam-a")) is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"threshold_percentile": 1.5},
        {"window_size": None},
        {"window_size": 0},
        {"scope": "camera"},
        {"max_sources": 0},
    ],
)
def test_literal_parameters_are_checked_when_the_workflow_compiles(overrides):
    with pytest.raises(ValidationError):
        IdentifyOutliersBlock.Params(embedding="$inputs.embedding", **overrides)


def test_defaults_match_v1():
    params = IdentifyOutliersBlock.Params(embedding="$inputs.embedding")

    assert (params.threshold_percentile, params.warmup, params.window_size) == (
        0.05,
        3,
        32,
    )
    assert (params.scope, params.max_sources) == ("source", 256)
