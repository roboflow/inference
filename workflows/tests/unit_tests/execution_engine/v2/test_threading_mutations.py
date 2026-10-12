"""Output groups are readers in mutation analysis (threading decision 001).

A group is delivered once its own dependencies finished; nothing else in the
pulse waits for the delivery. A mutator of the same payload outside those
dependencies may therefore change it while the group's handler reads it::

    $sources.cam.meta ──▶ group "frames" (waits for nothing)
                     └──▶ bump (mutates meta in place)       unordered: reported

    $sources.cam.meta ──▶ bump ──▶ group "frames" reads $steps.bump.value
                                                             ordered: valid
"""

import threading
from typing import Any, Dict, List

import pytest
from roboflow_workflows.execution_engine.v2 import (
    Catalogue,
    Source,
    SourceOutput,
    compile_workflow,
)
from roboflow_workflows.execution_engine.v2.compilation import CompileOptions
from roboflow_workflows.execution_engine.v2.errors import MutationConflictError
from roboflow_workflows.execution_engine.v2.kinds import DICTIONARY_KIND, FLOAT_KIND
from roboflow_workflows.execution_engine.v2.sources import Emission

from tests.unit_tests.execution_engine.v2.compilation.fixture_blocks import (
    ALL_BLOCKS,
    gate,
    nested,
    parameter,
    step,
    workflow,
)
from tests.unit_tests.execution_engine.v2.test_operator_compilation import (
    compile_definition,
    group,
    window,
)

STRICT = CompileOptions(mutation_conflicts="error")
GROUP_CONFLICT = "output group 'frames' field"


def _group_warnings(plan) -> List[str]:
    found = [warning for warning in plan.warnings if "output group" in warning]

    return found


def test_a_group_reading_a_payload_an_unordered_step_mutates_is_reported() -> None:
    steps = [step("increment", "bump", value="$sources.cam.meta")]
    outputs = [group("frames", "$sources.cam.image", raw="$sources.cam.meta")]

    plan = compile_definition(steps, outputs=outputs)

    [warning] = plan.warnings
    assert warning.startswith("$steps.bump value ($sources.cam.meta) is mutated")
    assert f"{GROUP_CONFLICT} 'raw' ($sources.cam.meta)" in warning
    assert "without waiting for $steps.bump" in warning
    with pytest.raises(MutationConflictError, match=GROUP_CONFLICT) as rejected:
        compile_definition(steps, outputs=outputs, options=STRICT)
    assert rejected.value.step_path == ("bump",)
    assert rejected.value.field_path == ("value",)


@pytest.mark.parametrize(
    "fields",
    [
        # The group waits for the mutator: it reads its output.
        {"bumped": "$steps.bump.value", "raw": "$sources.cam.meta"},
        # The group waits for a descendant of the mutator.
        {"echoed": "$steps.echo.value"},
        # The group never reads the mutated payload.
        {"image": "$sources.cam.image", "seen": "$steps.read.seen"},
    ],
    ids=["reads-mutator", "reads-descendant", "disjoint"],
)
def test_ordered_or_disjoint_groups_compile_strictly(fields: Dict[str, str]) -> None:
    steps = [
        step("increment", "bump", value="$sources.cam.meta"),
        step("echo", "echo", value="$steps.bump.value"),
        step("reader", "read", value="$steps.bump.value"),
    ]
    outputs = [group("frames", "$sources.cam.image", **fields)]

    plan = compile_definition(steps, outputs=outputs, options=STRICT)

    assert plan.warnings == ()


def test_a_wildcard_field_reads_every_output_of_its_step() -> None:
    plan = compile_definition(
        [
            step("echo", "echo", value="$sources.cam.meta"),
            step("increment", "bump", value="$sources.cam.meta"),
        ],
        outputs=[group("frames", "$sources.cam.image", all="$steps.echo.*")],
    )

    [warning] = _group_warnings(plan)
    assert f"{GROUP_CONFLICT} 'all' ($steps.echo.*)" in warning


def test_nested_mutators_and_forwarded_child_outputs_keep_their_provenance() -> None:
    child = workflow(
        [step("increment", "bump", value="$inputs.x")],
        {"bumped": "$steps.bump.value", "forwarded": "$inputs.x"},
        inputs=[parameter("x", kind=["dictionary"])],
    )
    steps = [
        nested(
            "child",
            workflow_definition=child,
            parameter_bindings={"x": "$sources.cam.meta"},
        )
    ]

    unordered = compile_definition(
        steps,
        outputs=[group("frames", "$sources.cam.image", raw="$steps.child.forwarded")],
    )
    ordered = compile_definition(
        steps,
        outputs=[group("frames", "$sources.cam.image", out="$steps.child.bumped")],
        options=STRICT,
    )

    [warning] = _group_warnings(unordered)
    assert warning.startswith("$steps.child/bump value ($inputs.x) is mutated")
    assert "($steps.child.forwarded)" in warning
    assert ordered.warnings == ()


def test_a_mutator_of_another_pulse_domain_is_not_a_group_reader_conflict() -> None:
    # ``make`` is static: every pulse of every domain gets its own payload, so
    # a mutation in a cam pulse never touches what a probe pulse delivers.
    steps = [
        step("echo", "make", value="$inputs.payload"),
        gate("on_cam", "$sources.cam.image", ["bump"]),
        step("increment", "bump", value="$steps.make.value"),
    ]
    inputs = [parameter("payload", kind=["dictionary"])]

    other_domain = compile_definition(
        steps,
        outputs=[group("probes", "$sources.probe.value", made="$steps.make.value")],
        inputs=inputs,
    )
    same_domain = compile_definition(
        steps,
        outputs=[group("frames", "$sources.cam.image", made="$steps.make.value")],
        inputs=inputs,
    )

    assert other_domain.step(("bump",)).domain == "cam"
    # The shared workflow input is reported on its own in both plans.
    assert any("shares its workflow inputs" in item for item in other_domain.warnings)
    assert _group_warnings(other_domain) == []
    [warning] = _group_warnings(same_domain)
    assert f"{GROUP_CONFLICT} 'made' ($steps.make.value)" in warning


def test_an_operator_domain_group_races_its_mutator_and_upstream_groups_do_not() -> (
    None
):
    plan = compile_definition(
        [step("increment", "bump", value="$operators.clip.meta")],
        [window("clip", {"meta": "$sources.cam.meta"})],
        [
            group("frames", "$sources.cam.image", raw="$sources.cam.meta"),
            group("clips", "$operators.clip.meta", meta="$operators.clip.meta"),
        ],
    )

    retained, raced = plan.warnings
    assert "may retain or emit it again" in retained
    # Within one clip pulse the delivery and bump share the collected
    # payloads; the cam group's pulse ended before the clip pulse began.
    assert "output group 'clips' field 'meta'" in raced
    assert "'frames'" not in " ".join(plan.warnings)


# --- the overlap the diagnostic describes -------------------------------------


class MetaFeed(Source):
    """Emits the dictionaries of the ``payloads`` resource, one per pulse."""

    type = "test/meta_feed@v1"
    outputs = {
        "image": SourceOutput(FLOAT_KIND),
        "meta": SourceOutput(DICTIONARY_KIND),
    }

    def __init__(self, *, payloads: List[dict]) -> None:
        self.payloads = payloads
        self.items = iter(())

    def open(self) -> None:
        self.items = iter(self.payloads)

    def read(self):
        payload = next(self.items, None)
        if payload is None:
            return None

        emission = Emission({"image": 1.0, "meta": payload})

        return emission

    def close(self) -> None:
        pass


def test_a_delivered_payload_is_changed_by_a_mutator_the_group_does_not_await() -> None:
    catalogue = Catalogue(ALL_BLOCKS, sources=[MetaFeed], namespace="test")
    definition = {
        "version": "2.0",
        "inputs": [],
        "sources": [{"type": MetaFeed.type, "name": "cam"}],
        "steps": [step("increment", "bump", value="$sources.cam.meta")],
        "outputs": [group("frames", "$sources.cam.image", raw="$sources.cam.meta")],
    }
    plan = compile_workflow(definition, catalogue=catalogue)
    session = plan.create_session(resources={"payloads": [{"frame": 0}]})
    seen: List[Any] = []
    delivered = threading.Event()

    def handle(result) -> None:
        payload = result.rows()[0]["raw"]
        seen.append((payload, dict(payload)))
        delivered.set()

    run = session.start(handlers={"frames": handle})

    assert run.wait(10.0)
    assert delivered.is_set()
    [(payload, at_delivery)] = seen
    # The handler held the payload before bump ran; bump then changed it in
    # place. With W > 1 that change can happen while the handler reads it.
    assert at_delivery == {"frame": 0}
    assert payload == {"frame": 0, "count": 1}
    assert plan.warnings and GROUP_CONFLICT in plan.warnings[0]
