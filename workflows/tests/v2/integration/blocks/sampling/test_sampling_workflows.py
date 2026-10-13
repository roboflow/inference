"""``v2/identify_changes@v1`` and ``v2/identify_outliers@v1`` in compiled workflows.

Two synthetic cameras send 4-dim embeddings; each changes scene once
(camera A at call 9, camera B at call 6)::

    pulse     0  1  2  ...  9  10  ...  13
    cam A     a0 a1 a2      a9 a10      a13
    cam B     b0 b1 b2      b9  -        -

``V1_*`` are the outputs of V1's tensor blocks, recorded in float32 by the
harness task ``ecosystem-e4-embedding-sampling/v1_trace``: camera A alone,
camera B alone, and both cameras through V1's one shared state in pulse
order (a0 b0 a1 b1 ...). The same embeddings run passively (with and
without sample contexts) and actively, serially and pipelined, in both
block execution modes. ``scope="source"`` must give V1_A and V1_B;
``scope="global"`` must give V1_INTERLEAVED_*.

identify_changes must match V1 exactly. identify_outliers ranks ties at
their middle where V1 ranked from the left, so its expectations are derived
from the V1 traces by ``mid_cdf``: V1 rank ``r / n`` becomes ``(2r + 1) / (2n)``
on these untied windows.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
from roboflow_workflows.v2.blocks.sampling import (
    IdentifyChangesBlock,
    IdentifyOutliersBlock,
    SamplingCapacityError,
)
from roboflow_workflows.v2.data.batch import (
    Axis,
    Batch,
    EntryLayout,
    EntryMetadata,
    InputValue,
    SampleContext,
)
from roboflow_workflows.v2.data.kinds import BOOLEAN_KIND, FLOAT_KIND, INTEGER_KIND
from roboflow_workflows.v2.data.predictions import EMBEDDING_KIND
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from roboflow_workflows.v2.engine.errors import StepExecutionError
from roboflow_workflows.v2.engine.pipelining.options import PipelineOptions
from roboflow_workflows.v2.engine.plan import CompileOptions
from roboflow_workflows.v2.engine.sources import Emission, Source, SourceOutput

WAIT = 20.0
MODES = ["run", "phases"]


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
PULSES = [[a, b] for a, b in zip(A, B)] + [[a] for a in A[len(B) :]]

# (is_outlier, percentile) of identify_changes, EMA, threshold 0.2, warmup 3.
V1_CHANGES_A = [
    (False, 0.5), (False, 0.5), (False, 0.999217298871),
    (True, 0.999057853856), (False, 0.149371460106), (False, 0.11207926459),
    (False, 0.194180085309), (False, 0.113777804982), (True, 0.917148729834),
    (True, 0.999217053267), (True, 0.985127958211), (True, 0.909153523913),
    (False, 0.753986812983), (False, 0.729011583959),
]  # fmt: skip
V1_CHANGES_B = [
    (False, 0.5), (False, 0.5), (False, 0.999217298871),
    (True, 0.00081357538409), (True, 0.042421596866), (False, 0.796071649883),
    (True, 0.999217074563), (True, 0.960532730706), (True, 0.917166965079),
    (False, 0.878789752042),
]  # fmt: skip
V1_CHANGES_INTERLEAVED = [
    (False, 0.5), (False, 0.5), (False, 0.000782701129001),
    (True, 0.916950981087), (True, 0.0195596092189), (False, 0.122557884851),
    (False, 0.661774702272), (True, 0.0240424852372), (True, 0.0311205849383),
    (True, 0.0813192526269), (False, 0.102269888058), (False, 0.323813461184),
    (False, 0.198828879802), (True, 0.999217023912), (False, 0.382983486559),
    (True, 0.970108236457), (False, 0.335200717396), (True, 0.942915267898),
    (False, 0.848500672437), (False, 0.827478626722), (False, 0.714420171281),
    (False, 0.536457422494), (False, 0.37266431741), (False, 0.394854605691),
]  # fmt: skip
# (is_outlier, percentile) of identify_outliers, threshold 0.2, warmup 3, window 5.
V1_OUTLIERS_A = [
    (False, 0.5), (False, 0.5), (True, 0.0), (True, 0.0), (False, 0.8),
    (False, 0.4), (False, 0.4), (False, 0.8), (False, 0.2), (True, 0.0),
    (True, 0.0), (False, 0.6), (False, 0.8), (False, 0.8),
]  # fmt: skip
V1_OUTLIERS_B = [
    (False, 0.5), (False, 0.5), (False, 1 / 3), (True, 0.0), (False, 0.4),
    (False, 0.6), (True, 0.0), (False, 0.2), (False, 0.6), (False, 0.2),
]  # fmt: skip
V1_OUTLIERS_INTERLEAVED = [
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
        mid = (2 * round(percentile * n) + 1) / (2 * n)
        expected.append((mid < threshold or mid > 1 - threshold, mid))

    return expected


OUTLIERS_A = mid_cdf(V1_OUTLIERS_A)
OUTLIERS_B = mid_cdf(V1_OUTLIERS_B)
OUTLIERS_INTERLEAVED = mid_cdf(V1_OUTLIERS_INTERLEAVED)
# Per pulse member: ((changes outlier, percentile), (outliers outlier, percentile))
Rows = Dict[Tuple[int, int], Tuple[Tuple[bool, float], Tuple[bool, float]]]


def _per_source() -> Rows:
    rows = {}
    for pulse, members in enumerate(PULSES):
        rows[(pulse, 0)] = (V1_CHANGES_A[pulse], OUTLIERS_A[pulse])
        if len(members) == 2:
            rows[(pulse, 1)] = (V1_CHANGES_B[pulse], OUTLIERS_B[pulse])

    return rows


def _shared() -> Rows:
    rows, n = {}, 0
    for pulse, members in enumerate(PULSES):
        for cam in range(len(members)):
            rows[(pulse, cam)] = (V1_CHANGES_INTERLEAVED[n], OUTLIERS_INTERLEAVED[n])
            n += 1

    return rows


class TwoCameras(Source):
    """Stationary ``camera`` axis; camera B stops after its last embedding."""

    type = "test/two_embedding_cameras@v1"
    outputs = {
        "embedding": SourceOutput(
            EMBEDDING_KIND,
            layout=EntryLayout((Axis(id="camera", kind="sample", stationary=True),)),
        )
    }

    def open(self) -> None:
        self._pulse = 0

    def read(self) -> Optional[Emission]:
        if self._pulse == len(PULSES):
            return None

        pulse, self._pulse = self._pulse, self._pulse + 1
        members = PULSES[pulse]
        indices = [(cam,) for cam in range(len(members))]
        value = Batch.of([torch.tensor(row) for row in members], indices=indices)
        sample = {(cam,): SampleContext(source_id=f"cam-{cam}") for cam, in indices}
        emission = Emission(
            {"embedding": InputValue(value, EntryMetadata(sample=sample))}
        )

        return emission


class PerEmbeddingConfig(Block):
    """Emits ``value`` once per embedding, without the embedding's context.

    ``source="value"`` takes the outputs' context from the literal ``value``,
    which has none, so a call selecting them has no common sample context.
    """

    type = "test/per_embedding_config@v1"
    outputs = {
        "threshold": Output(FLOAT_KIND, source="value"),
        "window": Output(INTEGER_KIND, source="value"),
    }

    class Params(BlockParams):
        anchor: Ref(EMBEDDING_KIND)
        value: Any | Ref()

    def run(self, *, anchor: Any, value: Any) -> Dict[str, Any]:
        return value


CATALOGUE = Catalogue(
    [IdentifyChangesBlock, IdentifyOutliersBlock, PerEmbeddingConfig],
    sources=[TwoCameras],
    kinds=[EMBEDDING_KIND, BOOLEAN_KIND, FLOAT_KIND, INTEGER_KIND],
)
EMBEDDING_INPUT = {
    "type": "WorkflowBatchInput",
    "name": "embedding",
    "kind": [EMBEDDING_KIND.name],
}
FIELDS = {
    "changed": "$steps.changes.is_outlier",
    "change_percentile": "$steps.changes.percentile",
    "average": "$steps.changes.average",
    "std": "$steps.changes.std",
    "outlier": "$steps.outliers.is_outlier",
    "outlier_percentile": "$steps.outliers.percentile",
}


def _steps(embedding: str, *, scope: str = "source", **changes: Any) -> List[dict]:
    steps = [
        {
            "type": IdentifyChangesBlock.type,
            "name": "changes",
            "embedding": embedding,
            "window_size": 4,
            "scope": scope,
            **changes,
        },
        {
            "type": IdentifyOutliersBlock.type,
            "name": "outliers",
            "embedding": embedding,
            "threshold_percentile": 0.2,
            "window_size": 5,
            "scope": scope,
        },
    ]

    return steps


def _outputs() -> List[dict]:
    outputs = [
        {"type": "JsonField", "name": name, "selector": selector}
        for name, selector in FIELDS.items()
    ]

    return outputs


def _row(values: Dict[str, Any]) -> Tuple[Tuple[bool, float], Tuple[bool, float]]:
    assert values["average"].shape == values["std"].shape == (4,)
    row = (
        (values["changed"], values["change_percentile"]),
        (values["outlier"], values["outlier_percentile"]),
    )

    return row


def _assert_rows(got: Rows, expected: Rows) -> None:
    assert got.keys() == expected.keys()
    for key, ((changed, p_change), outlier) in expected.items():
        (got_changed, got_p_change), got_outlier = got[key]

        assert (got_changed, got_outlier) == (changed, outlier), key
        assert got_p_change == pytest.approx(p_change, abs=1e-9), key


# ------------------------------------------------------------------ passive
def _passive_plan(mode: str, *, inputs=(), **step_options: Any):
    definition = {
        "version": "2.0",
        "inputs": [EMBEDDING_INPUT, *inputs],
        "steps": _steps("$inputs.embedding", **step_options),
        "outputs": _outputs(),
    }
    plan = compile_workflow(
        definition, catalogue=CATALOGUE, options=CompileOptions(block_execution=mode)
    )

    return plan


def _passive_rows(session, *, with_sources: bool, extra=None) -> Rows:
    rows = {}
    for pulse, members in enumerate(PULSES):
        tensors = [torch.tensor(row) for row in members]
        embedding: Any = tensors
        if with_sources:
            samples = {
                (cam,): SampleContext(source_id=f"cam-{cam}")
                for cam in range(len(members))
            }
            embedding = InputValue(tensors, EntryMetadata(sample=samples))
        result = session.run({"embedding": embedding, **(extra or {})})
        for cam, values in enumerate(result.rows()):
            rows[(pulse, cam)] = _row(values)

    return rows


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("with_sources", [True, False], ids=["sources", "positions"])
def test_passive_runs_keep_one_state_per_camera(mode, with_sources):
    session = _passive_plan(mode).create_session()

    rows = _passive_rows(session, with_sources=with_sources)
    session.close()

    _assert_rows(rows, _per_source())


@pytest.mark.parametrize("mode", MODES)
def test_passive_global_scope_reproduces_v1_shared_state(mode):
    session = _passive_plan(mode, scope="global").create_session()

    rows = _passive_rows(session, with_sources=True)
    session.close()

    _assert_rows(rows, _shared())


def test_selected_threshold_and_warmup_reach_the_block():
    inputs = [
        {"type": "WorkflowParameter", "name": "threshold"},
        {"type": "WorkflowParameter", "name": "warmup"},
    ]
    plan = _passive_plan(
        "run",
        inputs=inputs,
        threshold_percentile="$inputs.threshold",
        warmup="$inputs.warmup",
    )
    session = plan.create_session()

    rows = _passive_rows(
        session, with_sources=True, extra={"threshold": 0.2, "warmup": 100}
    )
    session.close()

    assert not any(changes[0] for changes, _ in rows.values())
    expected = _per_source()
    assert [rows[key][0][1] for key in expected] == pytest.approx(
        [row[0][1] for row in expected.values()], abs=1e-9
    )


def _window_session(scope: str):
    definition = {
        "version": "2.0",
        "inputs": [EMBEDDING_INPUT, {"type": "WorkflowParameter", "name": "window"}],
        "steps": [
            {
                "type": IdentifyOutliersBlock.type,
                "name": "outliers",
                "embedding": "$inputs.embedding",
                "threshold_percentile": 0.2,
                "warmup": 1,
                "window_size": "$inputs.window",
                "scope": scope,
            }
        ],
        "outputs": [
            {"type": "JsonField", "name": "p", "selector": "$steps.outliers.percentile"}
        ],
    }
    session = compile_workflow(definition, catalogue=CATALOGUE).create_session()

    return session


def _window_run(session, pulses: List[int], *, window: int) -> List[float]:
    for pulse in pulses:
        tensors = [torch.tensor(row) for row in PULSES[pulse]]
        samples = {(cam,): SampleContext(source_id=f"cam-{cam}") for cam in range(2)}
        embedding = InputValue(tensors, EntryMetadata(sample=samples))
        result = session.run({"embedding": embedding, "window": window})
    percentiles = [values["p"] for values in result.rows()]

    return percentiles


@pytest.mark.parametrize(
    "scope, kept_pulses", [("source", [7, 8, 9]), ("global", [8, 9])]
)
def test_a_selected_smaller_window_trims_in_the_same_call(scope, kept_pulses):
    session = _window_session(scope)
    _window_run(session, list(range(9)), window=5)

    shrunk = _window_run(session, [9], window=3)
    fresh = _window_session(scope)
    expected = _window_run(fresh, kept_pulses, window=3)
    session.close()
    fresh.close()

    # Pulse 9 ranks against only the newest 3 embeddings of its window: the
    # ones a fresh session holds after those pulses (global: both cameras).
    assert shrunk == expected


def test_one_source_too_many_fails_the_run():
    session = _passive_plan("run", max_sources=1).create_session()
    two = [torch.tensor(row) for row in PULSES[0]]

    with pytest.raises(StepExecutionError, match="max_sources=1") as raised:
        session.run({"embedding": two})
    session.close()

    assert isinstance(raised.value.__cause__, SamplingCapacityError)


def _config_session(mode: str, *, scope: str):
    sampling = [
        {
            "type": block.type,
            "name": name,
            "embedding": "$inputs.embedding",
            "threshold_percentile": "$steps.config.threshold",
            "window_size": "$steps.config.window",
            "warmup": 3,
            "scope": scope,
        }
        for name, block in [
            ("changes", IdentifyChangesBlock),
            ("outliers", IdentifyOutliersBlock),
        ]
    ]
    config = {
        "type": PerEmbeddingConfig.type,
        "name": "config",
        "anchor": "$inputs.embedding",
        "value": {"threshold": 0.2, "window": 5},
    }
    definition = {
        "version": "2.0",
        "inputs": [EMBEDDING_INPUT],
        "steps": [config, *sampling],
        "outputs": [
            {"type": "JsonField", "name": name, "selector": f"$steps.{name}.warming_up"}
            for name in ("changes", "outliers")
        ],
    }
    plan = compile_workflow(
        definition, catalogue=CATALOGUE, options=CompileOptions(block_execution=mode)
    )
    session = plan.create_session()

    return session


# warming_up per call of (changes, outliers), warmup 3: changes counts the
# embeddings before the call, outliers includes the call's own embedding.
OWN_STATES = ([True] * 5, [True, True, True, True, False])
ONE_STATE = ([True, True, True, False, False], [True, True, False, False, False])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "scope, sources, warming_up",
    [
        ("source", ["a", "b", "a", "b", "a"], OWN_STATES),
        ("global", ["a", "b", "a", "b", "a"], ONE_STATE),
        ("source", [None] * 5, ONE_STATE),
    ],
    ids=["sources", "global", "no-context"],
)
def test_context_free_selected_parameters_keep_the_embedding_source(
    mode, scope, sources, warming_up
):
    # One embedding per run, always at batch position 0. Cameras a and b keep
    # their own states; global scope, or position 0 without contexts, keeps one.
    session = _config_session(mode, scope=scope)
    got = []
    for source in sources:
        embedding: Any = [torch.tensor([1.0, 0.0] if source == "a" else [0.0, 1.0])]
        if source is not None:
            sample = {(0,): SampleContext(source_id=source)}
            embedding = InputValue(embedding, EntryMetadata(sample=sample))
        result = session.run({"embedding": embedding})
        got.append(result.rows()[0])
    session.close()

    changes, outliers = warming_up
    assert got == [{"changes": c, "outliers": o} for c, o in zip(changes, outliers)]


# ------------------------------------------------------------------- active
PIPELINES = [
    pytest.param(None, id="serial"),
    pytest.param(
        PipelineOptions(max_in_flight=3, max_workflow_parallelism=2), id="P3-W2"
    ),
]


def _run_active(mode: str, pipeline: Optional[PipelineOptions], *, scope: str):
    definition = {
        "version": "2.0",
        "inputs": [],
        "sources": [{"type": TwoCameras.type, "name": "cams"}],
        "steps": _steps("$sources.cams.embedding", scope=scope),
        "outputs": [
            {
                "type": "OutputGroup",
                "name": "sampled",
                "anchor": "$sources.cams.embedding",
                "outputs": _outputs(),
            }
        ],
    }
    plan = compile_workflow(
        definition, catalogue=CATALOGUE, options=CompileOptions(block_execution=mode)
    )
    session = plan.create_session()
    results = []
    run = session.start(handlers={"sampled": results.append}, pipeline=pipeline)
    try:
        assert run.wait(WAIT), "active run did not finish"
    finally:
        run.stop()
        session.close()

    rows = {}
    for pulse, result in enumerate(results):
        data = result.outputs.data
        for index, _ in data["changed"].iter_with_indices():
            values = {
                name: dict(data[name].iter_with_indices())[index] for name in data
            }
            rows[(pulse, index[0])] = _row(values)

    return rows


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("pipeline", PIPELINES)
def test_active_cameras_keep_their_own_state(mode, pipeline):
    rows = _run_active(mode, pipeline, scope="source")

    _assert_rows(rows, _per_source())


@pytest.mark.parametrize("pipeline", PIPELINES)
def test_active_global_scope_reproduces_v1_shared_state(pipeline):
    rows = _run_active("run", pipeline, scope="global")

    _assert_rows(rows, _shared())
