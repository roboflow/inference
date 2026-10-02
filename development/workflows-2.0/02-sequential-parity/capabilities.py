"""V2 capability examples beyond the 45 parity cases.

Each example compiles a definition from ``workflows/``, creates a session and
runs it, then checks the real rows, call shapes and errors against expected
values written down in advance. Where a V1 measurement exists (generated roots,
vectorized calls), the expected values are that measurement; where V2 decides
differently, the check is labelled.

Change a definition in ``workflows/`` or an input below and rerun
``run_demo.py --scenario capabilities`` to observe the effect.
"""

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import boundary_examples
import dynamic_examples
import gate_examples
import numpy as np
import plugin_examples
from blocks import create_fixture_catalogue
from boundary_blocks import create_boundary_catalogue
from capability_blocks import create_capability_catalogue
from roboflow_workflows.execution_engine.v2.blocks import create_catalogue
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import (
    WorkflowReference,
    compile_workflow,
)
from roboflow_workflows.execution_engine.v2.errors import (
    LineageError,
    MutationConflictError,
    NestedWorkflowError,
)
from roboflow_workflows.execution_engine.v2.introspection import (
    describe_catalogue,
    describe_workflow,
    discover_connections,
    discover_workload,
)
from roboflow_workflows.execution_engine.v2.plan import CompileOptions
from v2_runner import InvocationRecorder, plain

WORKFLOWS_DIR = Path(__file__).resolve().parent / "workflows"


@dataclass
class ExampleReport:
    """Checks of one example and the definitions it used."""

    name: str
    checks: List[Dict[str, Any]] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def check(self, description: str, actual: Any, expected: Any) -> None:
        """Record whether ``actual`` equals ``expected``.

        Args:
            description: What is checked.
            actual: Value observed from the engine.
            expected: Value written down in advance.
        """
        self.checks.append(
            {
                "check": description,
                "passed": actual == expected,
                "actual": plain(actual),
                "expected": plain(expected),
            }
        )

    @property
    def ok(self) -> bool:
        """Whether every check passed."""
        return all(check["passed"] for check in self.checks)


def load(name: str) -> Dict[str, Any]:
    """Read a definition from ``workflows/``.

    Args:
        name: Path relative to ``workflows/`` without ``.json``.

    Returns:
        The parsed definition.
    """
    definition = json.loads((WORKFLOWS_DIR / f"{name}.json").read_text())

    return definition


def catalogue() -> Catalogue:
    """Blocks available to the examples: fixture, capability, boundary and image blocks.

    Returns:
        The merged catalogue.
    """
    merged = Catalogue.merge(
        create_fixture_catalogue(),
        create_capability_catalogue(),
        create_boundary_catalogue(),
        create_catalogue(),
    )

    return merged


def _calls(recorder: InvocationRecorder, step: str) -> List[Dict[str, Any]]:
    return [entry for entry in recorder.invocations if entry["step"] == step]


def _run(
    name: str,
    inputs: Dict[str, Any],
    *,
    options: CompileOptions = CompileOptions(),
) -> tuple:
    recorder = InvocationRecorder()
    plan = compile_workflow(load(name), catalogue=catalogue(), options=options)
    result = plan.create_session(observer=recorder).run(inputs)

    return result, recorder


def selector_crop_mosaic(report: ExampleReport) -> None:
    """Crop rectangles and mosaic tile size come from inputs, not from JSON."""
    images = [
        np.full((120, 200, 3), 50, dtype=np.uint8),
        np.full((110, 110, 3), 60, dtype=np.uint8),
    ]
    one, two = [[0, 0, 10, 10]], [[0, 0, 10, 10], [20, 20, 40, 40]]
    plan = compile_workflow(load("selector_crop_mosaic"), catalogue=catalogue())
    session = plan.create_session()

    first = session.run({"images": images, "regions": [one, two]}).rows()
    report.check("default tile 24: crop counts", [r["count"] for r in first], [1, 2])
    report.check(
        "default tile 24: canvas shapes",
        [list(r["mosaic"].size_hw) for r in first],
        [[24, 24], [24, 48]],
    )
    second = session.run(
        {"images": images, "regions": [two, one], "tile_size": 10}
    ).rows()
    report.check("swapped regions: crop counts", [r["count"] for r in second], [2, 1])
    report.check(
        "tile_size input 10: canvas shapes",
        [list(r["mosaic"].size_hw) for r in second],
        [[10, 20], [10, 10]],
    )


def vote_and_csv(report: ExampleReport) -> None:
    """A list of batch selectors gives one call with list[Batch]; CSV leaves mix."""
    result, recorder = _run(
        "vote_and_csv",
        {"first": ["cat", "dog", "cat"], "second": ["cat", "cat", "cat"]},
    )
    (vote,) = _calls(recorder, "vote")
    report.check(
        "vote: one call receiving two Batches with full indices",
        vote["arguments"]["predictions"],
        [
            {"batch": ["cat", "dog", "cat"], "indices": [[0], [1], [2]]},
            {"batch": ["cat", "cat", "cat"], "indices": [[0], [1], [2]]},
        ],
    )
    rows = result.rows()
    report.check("vote rows", [r["unanimous"] for r in rows], [True, False, True])
    (literal,) = _calls(recorder, "csv_literal")
    report.check(
        "csv with only literal columns: one plain call",
        (literal["index"], literal["arguments"]["columns"]),
        ([], {"camera": "north", "enabled": True}),
    )
    (varying,) = _calls(recorder, "csv_varying")
    report.check(
        "csv with a varying column: one vectorized call, literal leaf stays plain",
        (varying["index"], varying["arguments"]["columns"]),
        (
            None,
            {
                "label": {"batch": ["cat", "dog", "cat"], "indices": [[0], [1], [2]]},
                "camera": "north",
            },
        ),
    )
    report.check(
        "csv rows",
        [(r["csv_literal"], r["csv_varying"]) for r in rows],
        [
            ("north,True", "cat,north"),
            ("north,True", "dog,north"),
            ("north,True", "cat,north"),
        ],
    )


def compound_group_cast(report: ExampleReport) -> None:
    """A scalar beside child groups becomes a one-element group under each parent."""
    result, recorder = _run("compound_group_cast", {"parents": [1, 2]})
    calls = _calls(recorder, "named")
    report.check("one call per parent", [call["index"] for call in calls], [[0], [1]])
    report.check(
        "label cast to a singleton group at (parent, 0); children keep indices",
        [call["result"]["summary"]["groups"] for call in calls],
        [
            {
                "children": {"values": [10], "indices": [[0, 0]]},
                "label": {"values": ["tag"], "indices": [[0, 0]]},
            },
            {
                "children": {"values": [20, 21], "indices": [[1, 0], [1, 1]]},
                "label": {"values": ["tag"], "indices": [[1, 0]]},
            },
        ],
    )


def per_output_layouts(report: ExampleReport) -> None:
    """One block returns a parent-level total and a child-level output."""
    result, recorder = _run("per_output_layouts", {"parents": [1, 2]})
    report.check(
        "rows",
        result.rows(),
        [{"total": 11, "shifted": [11]}, {"total": 43, "shifted": [22, 23]}],
    )
    layouts = {
        name: len(result.outputs.layout[entry].axes)
        for name, ports in result.selections.items()
        for entry in ports.values()
    }
    report.check("layout depth per output", layouts, {"total": 1, "shifted": 2})
    report.check("one call per parent", len(_calls(recorder, "combine")), 2)


def configured_output_names(report: ExampleReport) -> None:
    """Output names come from a literal parameter and may be numeric or hyphenated."""
    result, _ = _run("configured_output_names", {"text": "2026,dog"})
    report.check(
        "rows select $steps.split.2026, $steps.split.class-name and $steps.split.*",
        result.rows(),
        [
            {
                "year": "2026",
                "label": "dog",
                "all": {"2026": "2026", "class-name": "dog"},
            }
        ],
    )


def generated_roots(report: ExampleReport) -> None:
    """Input-free sources create their own groups; rows follow V1's measurements."""
    expected_rows = {
        "source_only": [{"generated": [10, 20]}],
        "source_plus_input_outputs": [
            {"generated": [10, 20, 30], "item": "x"},
            {"generated": [10, 20, 30], "item": "y"},
        ],
        "two_independent_roots": [{"left": [10, 20], "right": [100, 200, 300]}],
        "source_through_reducer": [{"generated": [10, 20, 30], "total": 60}],
        "source_only_output_with_unused_batch_input": [{"generated": [10, 20, 30]}],
    }
    for case, rows in expected_rows.items():
        inputs = {"items": ["x", "y"]} if "input" in case else {}
        result, recorder = _run(f"generated_roots/{case}", inputs)
        report.check(f"{case}: rows equal V1", result.rows(), rows)
    _, recorder = _run("generated_roots/source_through_reducer", {})
    (total,) = _calls(recorder, "total")
    report.check(
        "reducer: one call with the generated Batch",
        total["arguments"]["values"],
        {"batch": [10, 20, 30], "indices": [[0], [1], [2]]},
    )
    result, recorder = _run("generated_roots/empty_source_through_reducer", {})
    report.check(
        "[V1-QUIRK-EMPTY-EXPANSION] empty source: V2 reduces the genuine empty group"
        " (V1: no call, total null)",
        (result.rows(), len(_calls(recorder, "total"))),
        ([{"generated": [], "total": 0}], 1),
    )
    try:
        compile_workflow(
            load("generated_roots/independent_roots_combined"), catalogue=catalogue()
        )
        error = None
    except LineageError as raised:
        error = type(raised).__name__
    report.check(
        "two independent roots into one item consumer: compile error (V1 too)",
        error,
        "LineageError",
    )


def vectorized_calls(report: ExampleReport) -> None:
    """Vectorized blocks get one call over ragged children, as measured in V1."""
    _, recorder = _run("vectorized/ragged_vectorized", {"roots": [2, 1]})
    calls = _calls(recorder, "consumer")
    report.check(
        "batch consumer: one call over all children of both roots",
        [call["arguments"]["item"] for call in calls],
        [{"batch": [20, 21, 10], "indices": [[0, 0], [0, 1], [1, 0]]}],
    )
    result, recorder = _run("vectorized/ragged_groups_vectorized", {"roots": [2, 1]})
    report.check(
        "group consumer: one call, rows carry call number 1",
        (len(_calls(recorder, "consumer")), result.rows()),
        (
            1,
            [
                {"result": {"parent": 2, "children": [20, 21], "call": 1}},
                {"result": {"parent": 1, "children": [10], "call": 1}},
            ],
        ),
    )
    # Root 0 expands to a genuinely empty group. V1 treats it as a missing parent:
    # its default consumer skips parent 0 (row null) and its empty-accepting
    # consumer receives None. V2 keeps the empty group (decision 008).
    for definition in ("ragged_groups_vectorized", "ragged_groups_accept_empty"):
        result, recorder = _run(f"vectorized/{definition}", {"roots": [0, 2]})
        report.check(
            f"[V1-QUIRK-EMPTY-EXPANSION] {definition} with roots [0, 2]: one call,"
            " parent 0 gets an empty group (V1: skipped, or None if accepting empty)",
            (len(_calls(recorder, "consumer")), result.rows()),
            (
                1,
                [
                    {"result": {"parent": 0, "children": [], "call": 1}},
                    {"result": {"parent": 2, "children": [20, 21], "call": 1}},
                ],
            ),
        )
    result, recorder = _run("vectorized/ragged_vectorized_control", {"roots": [2, 1]})
    report.check(
        "vectorized gate: one gate call, consumer gets survivors [20, 10] in one call",
        (
            len(_calls(recorder, "gate")),
            [call["arguments"]["item"] for call in _calls(recorder, "consumer")],
        ),
        (1, [{"batch": [20, 10], "indices": [[0, 0], [1, 0]]}]),
    )
    report.check(
        "vectorized gate rows",
        result.rows(),
        [
            {"result": [{"item": 20, "call": 1}, None]},
            {"result": [{"item": 10, "call": 1}]},
        ],
    )


def kind_codecs(report: ExampleReport) -> None:
    """Kinds convert caller inputs and serialize outputs at the boundary."""
    plan = compile_workflow(load("kind_codecs"), catalogue=catalogue())
    session = plan.create_session()
    result = session.run({"temperature": "21.5C"})
    report.check(
        "'21.5C' is deserialized to 21.5", result.rows(), [{"fahrenheit": 70.7}]
    )
    report.check(
        "rows(serialize=True) uses the fahrenheit serializer",
        result.rows(serialize=True),
        [{"fahrenheit": "70.7F"}],
    )
    report.check(
        "a plain number needs no deserializer",
        session.run({"temperature": 30}).rows(serialize=True),
        [{"fahrenheit": "86.0F"}],
    )


def mutation_policy(report: ExampleReport) -> None:
    """Declared in-place mutation: unordered readers warn, strict mode rejects."""
    unordered = load("mutation_unordered")
    plan = compile_workflow(unordered, catalogue=catalogue())
    report.check(
        "unordered writer and reader: compiles with one warning",
        len(plan.warnings),
        1,
    )
    report.details["warnings"] = list(plan.warnings)
    try:
        compile_workflow(
            unordered,
            catalogue=catalogue(),
            options=CompileOptions(mutation_conflicts="error"),
        )
        error = None
    except MutationConflictError as raised:
        error = type(raised).__name__
    report.check(
        "strict mode rejects the same definition", error, "MutationConflictError"
    )

    ordered = compile_workflow(load("mutation_ordered"), catalogue=catalogue())
    report.check("reader after writer: no warning", list(ordered.warnings), [])
    tags = ["a"]
    rows = ordered.create_session().run({"tags": tags}).rows()
    report.check("reader sees the appended tag", rows, [{"count": 2}])
    report.check("the caller's list was changed in place", tags, ["a", "seen"])


def caller_inputs(report: ExampleReport) -> None:
    """[D012-INPUT-PREPARATION] Input preparation leaves the caller's mapping alone."""
    inputs = {"a": [1, 2], "b": 3}
    snapshot = copy.deepcopy(inputs)
    result, _ = _run("caller_inputs", inputs)
    report.check(
        "broadcast b and default f are used",
        result.rows(),
        [{"scaled": 3, "offset": 10}, {"scaled": 6, "offset": 20}],
    )
    report.check(
        "the caller's mapping is unchanged (V1 writes b=[3.0, 3.0], f=10, a=[1.0, 2.0])",
        inputs,
        snapshot,
    )


def resource_precedence(report: ExampleReport) -> None:
    """Resources: scoped caller value, then unscoped, then catalogue, then default."""
    plan = compile_workflow(load("resource_precedence"), catalogue=catalogue())

    def text(resources: Optional[Dict[str, Any]]) -> str:
        return plan.create_session(resources).run({}).rows()[0]["text"]

    report.check("catalogue provider only", text(None), "hello from the catalogue!")
    report.check(
        "[V1 difference] unscoped caller value beats the catalogue provider",
        text({"greeting": "hi"}),
        "hi!",
    )
    report.check(
        "namespaced caller value beats the unscoped one",
        text({"greeting": "hi", "capability.greeting": "scoped"}),
        "scoped!",
    )
    report.check(
        "explicit None replaces a constructor default",
        text({"greeting": "hi", "suffix": None}),
        "hi",
    )


def _saved_resolver(calls: List[str]) -> Callable[[WorkflowReference], dict]:
    def resolve(reference: WorkflowReference) -> dict:
        calls.append(reference.workflow_id)

        return load(f"nested/saved/{reference.workflow_id}")

    return resolve


def nested_composition(report: ExampleReport) -> None:
    """Saved references, limits and duplicate dynamic definitions."""
    calls: List[str] = []
    plan = compile_workflow(
        load("nested/diamond"),
        catalogue=catalogue(),
        reference_resolver=_saved_resolver(calls),
    )
    report.check(
        "diamond: each saved workflow fetched once", calls, ["left", "leaf", "right"]
    )
    report.check(
        "diamond: both branches run the shared leaf",
        (
            ["__".join(step.path) for step in plan.steps],
            plan.create_session().run({"x": 3}).rows(),
        ),
        (["left__leaf__scale", "right__leaf__scale"], [{"left": 30, "right": 30}]),
    )

    for label, options, fetched in (
        ("depth limit 1", CompileOptions(max_nested_depth=1), ["left"]),
        (
            "count limit 3",
            CompileOptions(max_nested_count=3),
            ["left", "leaf", "right"],
        ),
    ):
        calls.clear()
        try:
            compile_workflow(
                load("nested/diamond"),
                catalogue=catalogue(),
                options=options,
                reference_resolver=_saved_resolver(calls),
            )
            error = None
        except NestedWorkflowError as raised:
            error = type(raised).__name__
        report.check(
            f"{label}: NestedWorkflowError; nothing fetched beyond the limit",
            (error, calls),
            ("NestedWorkflowError", fetched),
        )

    duplicate = compile_workflow(
        load("nested/duplicate_dynamic"),
        catalogue=catalogue(),
        options=CompileOptions(allow_local_code=True),
    )
    report.check(
        "duplicate dynamic block type: the root definition wins, with a warning",
        (duplicate.create_session().run({"x": 5}).rows(), len(duplicate.warnings)),
        ([{"parent": 6, "child": 6}], 1),
    )
    report.details["duplicate warnings"] = list(duplicate.warnings)


def inspection_without_execution(report: ExampleReport, *, output_dir: Path) -> None:
    """Inspecting a plan never runs submitted code; a session does, when allowed."""
    sentinel = output_dir / "dynamic_code_ran.txt"
    sentinel.unlink(missing_ok=True)
    code = (
        "import pathlib\n"
        f"pathlib.Path({str(sentinel)!r}).write_text('module loaded')\n\n"
        "def run(self, value) -> BlockResult:\n"
        "    return {'doubled': value * 2}\n"
    )
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "x"}],
        "dynamic_blocks_definitions": [
            {
                "type": "DynamicBlockDefinition",
                "manifest": {
                    "type": "ManifestDescription",
                    "block_type": "Doubler",
                    "inputs": {
                        "value": {
                            "type": "DynamicInputDefinition",
                            "selector_types": ["input_parameter"],
                        }
                    },
                    "outputs": {
                        "doubled": {"type": "DynamicOutputDefinition", "kind": []}
                    },
                },
                "code": {"type": "PythonCode", "run_function_code": code},
            }
        ],
        "steps": [{"type": "Doubler", "name": "double", "value": "$inputs.x"}],
        "outputs": [
            {
                "type": "JsonField",
                "name": "doubled",
                "selector": "$steps.double.doubled",
            }
        ],
    }
    report.details["definition"] = definition

    plan = compile_workflow(definition, catalogue=catalogue())
    description = describe_workflow(plan)
    connections = discover_connections(plan)
    workload = discover_workload(plan)
    report.details["describe_workflow"] = plain(description)
    report.details["connections"] = [plain(repr(item)) for item in connections]
    report.details["workload"] = plain(repr(workload))
    report.check(
        "compile + describe_workflow + discover_connections + discover_workload: code not run",
        sentinel.exists(),
        False,
    )
    report.check(
        "catalogue description lists the fixture blocks without constructing them",
        "fixture/echo@v1" in json.dumps(plain(describe_catalogue(catalogue()))),
        True,
    )
    try:
        plan.create_session()
        refused = None
    except Exception as raised:
        refused = [type(raised).__name__, type(raised.__cause__).__name__]
    report.check(
        "session without allow_local_code is refused (constructor error caused by"
        " LocalCodeNotAllowedError); code still not run",
        (refused, sentinel.exists()),
        (["ResourceError", "LocalCodeNotAllowedError"], False),
    )
    allowed = compile_workflow(
        definition, catalogue=catalogue(), options=CompileOptions(allow_local_code=True)
    )
    rows = allowed.create_session().run({"x": 4}).rows()
    report.check(
        "opted-in session loads the code and runs it",
        (sentinel.exists(), rows),
        (True, [{"doubled": 8}]),
    )


EXAMPLES: Dict[str, Callable[..., None]] = {
    "selector_crop_mosaic": selector_crop_mosaic,
    "vote_and_csv": vote_and_csv,
    "compound_group_cast": compound_group_cast,
    "per_output_layouts": per_output_layouts,
    "configured_output_names": configured_output_names,
    "generated_roots": generated_roots,
    "vectorized_calls": vectorized_calls,
    "kind_codecs": kind_codecs,
    "mutation_policy": mutation_policy,
    "caller_inputs": caller_inputs,
    "resource_precedence": resource_precedence,
    "nested_composition": nested_composition,
    "inspection_without_execution": inspection_without_execution,
}


def _register(function: Callable[..., None]) -> None:
    # Examples in the themed modules receive the catalogue and loader explicitly.
    def example(report: ExampleReport) -> None:
        def recording_load(name: str) -> Dict[str, Any]:
            report.details.setdefault("definitions", []).append(
                f"workflows/{name}.json"
            )
            return load(name)

        function(report, catalogue=catalogue(), load=recording_load)

    EXAMPLES[function.__name__] = example


for _function in (
    boundary_examples.validated_parameters,
    boundary_examples.nested_inputs,
    boundary_examples.execution_context,
    dynamic_examples.dynamic_shared_state,
    dynamic_examples.dynamic_representation,
    plugin_examples.plugin_catalogue,
    gate_examples.nested_forwarding_gates,
):
    _register(_function)


def run_examples(names: List[str], *, output_dir: Path) -> bool:
    """Run the named examples, print one line per check and write reports.

    Args:
        names: Example names from ``EXAMPLES``.
        output_dir: Root directory; reports go to ``capabilities/<name>.json``.

    Returns:
        Whether every check of every example passed.
    """
    import click

    directory = output_dir / "capabilities"
    directory.mkdir(parents=True, exist_ok=True)
    all_ok = True
    for name in names:
        report = ExampleReport(name=name)
        example = EXAMPLES[name]
        try:
            if name == "inspection_without_execution":
                example(report, output_dir=directory)
            else:
                example(report)
        except Exception as error:
            report.checks.append(
                {
                    "check": "example ran without an unexpected exception",
                    "passed": False,
                    "actual": f"{type(error).__name__}: {error}",
                    "expected": None,
                }
            )
        click.echo(f"  {name}")
        for check in report.checks:
            marker = "PASS" if check["passed"] else "FAIL"
            click.echo(f"    [{marker}] {check['check']}")
            if not check["passed"]:
                click.echo(f"           actual:   {json.dumps(check['actual'])[:300]}")
                click.echo(
                    f"           expected: {json.dumps(check['expected'])[:300]}"
                )
        (directory / f"{name}.json").write_text(
            json.dumps(report.__dict__, indent=2, default=repr) + "\n"
        )
        all_ok = all_ok and report.ok

    return all_ok
