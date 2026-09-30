"""Named demo scenarios for the passive V2 foundation.

Every scenario loads an ordinary JSON definition from ``workflows/``, compiles
it with ``compile_workflow``, creates an execution session and runs it. The
scenario code only prepares inputs, checks expectations against the real
result objects and writes artifacts. It never groups, gates or executes steps
itself.

Each scenario returns a :class:`ScenarioReport`; ``ok`` is ``False`` when an
expectation was not met, and the CLI turns that into a non-zero exit code.
"""

import json
import statistics
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import torch
from author_blocks import (
    CallCounter,
    ExpandBlock,
    ScaleBlock,
    SumWithParentBlock,
    create_author_catalogue,
)
from fixtures import (
    BRIGHTNESS_MINIMUM,
    CROP_REGIONS,
    EXPECTED_CROP_COUNTS,
    Fixture,
    make_fixtures,
    make_input_metadata,
)
from inspection import (
    batch_local_indices,
    batch_to_nested,
    describe_result,
    image_describer,
    plain_describer,
    save_png,
    write_json,
)
from roboflow_workflows.execution_engine.v2.blocks import (
    CropBlock,
    InvertBlock,
    create_catalogue,
)
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.data import (
    Batch,
    EntryMetadata,
    InputValue,
    SampleContext,
)
from roboflow_workflows.execution_engine.v2.plan import RunResult

WORKFLOWS_DIR = Path(__file__).resolve().parent / "workflows"

SCENARIO_NAMES = (
    "nested",
    "filtered",
    "invalid-bindings",
    "author-block",
    "metadata-cost",
)


@dataclass
class ScenarioReport:
    """Outcome of one scenario run."""

    name: str
    ok: bool = True
    checks: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def check(self, description: str, passed: bool, detail: Any = None) -> None:
        """Record one expectation.

        Args:
            description: What was expected.
            passed: Whether the real result met it.
            detail: Optional evidence.
        """
        self.checks.append(
            {"check": description, "passed": bool(passed), "detail": detail}
        )
        if not passed:
            self.ok = False


def load_definition(name: str) -> Dict[str, Any]:
    """Read a JSON workflow definition from the ``workflows`` directory.

    Args:
        name: Path relative to ``workflows/`` without the ``.json`` suffix.

    Returns:
        Parsed definition.
    """
    definition = json.loads((WORKFLOWS_DIR / f"{name}.json").read_text())

    return definition


def run_scenario(name: str, *, output_dir: Path) -> ScenarioReport:
    """Run one named scenario and write its artifacts.

    Args:
        name: One of ``SCENARIO_NAMES``.
        output_dir: Root directory; the scenario writes into ``<root>/<name>``.

    Returns:
        The scenario report; it is also written to ``<root>/<name>/report.json``.

    Raises:
        ValueError: On an unknown scenario name.
    """
    runners: Dict[str, Callable[[Path], ScenarioReport]] = {
        "nested": _run_nested,
        "filtered": _run_filtered,
        "invalid-bindings": _run_invalid_bindings,
        "author-block": _run_author_block,
        "metadata-cost": _run_metadata_cost,
    }
    if name not in runners:
        raise ValueError(f"Unknown scenario {name!r}; choose from {list(runners)}")

    scenario_dir = output_dir / name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    report = runners[name](scenario_dir)
    write_json(report.__dict__, scenario_dir / "report.json")

    return report


def run_workflow(
    name: str, *, catalogue: Catalogue, inputs: Mapping[str, Any]
) -> RunResult:
    """Compile a definition from ``workflows/``, create a session and run it once.

    Args:
        name: Definition name relative to ``workflows/``.
        catalogue: Blocks the definition may use.
        inputs: Workflow inputs.

    Returns:
        The engine's run result.
    """
    plan = compile_workflow(load_definition(name), catalogue=catalogue)
    result = plan.create_session().run(inputs)

    return result


def output_entry(result: RunResult, name: str) -> Optional[str]:
    """Return the buffer entry key selected by workflow output ``name``.

    Args:
        result: Run result.
        name: Workflow output name.

    Returns:
        The entry key, or ``None`` when the whole output was filtered.
    """
    (entry,) = result.selections[name].values()
    if entry not in result.outputs.data:
        return None

    return entry


def output_data(result: RunResult, name: str) -> Any:
    """Return the data of workflow output ``name``; ``None`` if fully filtered.

    Args:
        result: Run result.
        name: Workflow output name.

    Returns:
        Payload or nested ``Batch``.
    """
    entry = output_entry(result, name)
    data = None if entry is None else result.outputs.data[entry]

    return data


def _image_inputs(fixtures: List[Fixture]) -> Dict[str, InputValue]:
    images = Batch.of([fixture.image for fixture in fixtures])
    inputs = {"images": InputValue(images, metadata=make_input_metadata(fixtures))}

    return inputs


def _save_fixture_pngs(
    fixtures: List[Fixture], directory: Path, report: ScenarioReport
) -> None:
    for position, fixture in enumerate(fixtures):
        path = directory / f"input_{position}_{fixture.name}.png"
        save_png(fixture.image, path)
        report.artifacts.append(str(path))


def _group_sizes(batch: Batch) -> List[int]:
    sizes = [len(child) if isinstance(child, Batch) else -1 for child in batch]

    return sizes


def _run_nested(scenario_dir: Path) -> ScenarioReport:
    report = ScenarioReport(name="nested")
    fixtures = make_fixtures()
    _save_fixture_pngs(fixtures, scenario_dir / "inputs", report)

    result = run_workflow(
        "nested", catalogue=create_catalogue(), inputs=_image_inputs(fixtures)
    )

    artifact_dir = scenario_dir / "outputs"
    document = describe_result(result, describe_payload=image_describer(artifact_dir))
    write_json(document, scenario_dir / "result.json")
    report.artifacts.append(str(scenario_dir / "result.json"))
    report.artifacts.extend(str(path) for path in sorted(artifact_dir.glob("*.png")))

    outputs = result.outputs
    crops = output_data(result, "crops")
    report.check(
        "crop child counts are [2, 0, 1]",
        _group_sizes(crops) == list(EXPECTED_CROP_COUNTS),
        _group_sizes(crops),
    )
    crops_axes = [
        axis.kind for axis in outputs.layout[output_entry(result, "crops")].axes
    ]
    summary_axes = [
        axis.kind for axis in outputs.layout[output_entry(result, "summary")].axes
    ]
    report.check(
        "crops layout is [sample, dynamic_nesting] while summary layout is [sample]",
        crops_axes == ["sample", "dynamic_nesting"] and summary_axes == ["sample"],
        {"crops": crops_axes, "summary": summary_axes},
    )
    mosaic_axes = [
        axis.kind for axis in outputs.layout[output_entry(result, "mosaic")].axes
    ]
    report.check(
        "mosaic layout collapsed back to [sample]",
        mosaic_axes == ["sample"],
        mosaic_axes,
    )
    report.check(
        "mosaic_all layout is ungrouped ()",
        outputs.layout[output_entry(result, "mosaic_all")].axes == (),
        [axis.id for axis in outputs.layout[output_entry(result, "mosaic_all")].axes],
    )

    direct_crop = CropBlock()
    direct_invert = InvertBlock()
    regions = [tuple(region) for region in CROP_REGIONS]
    pixel_equal = True
    for parent_position, fixture in enumerate(fixtures):
        expected = direct_crop.run(image=fixture.image, regions=regions)
        engine_group = crops[parent_position]
        engine_inverted = output_data(result, "inverted")[parent_position]
        if list(engine_group.indices) != [
            (parent_position,) + index for index in expected["crops"].indices
        ]:
            pixel_equal = False
        for engine_crop, expected_crop in zip(engine_group, expected["crops"]):
            if not torch.equal(engine_crop.tensor_image, expected_crop.tensor_image):
                pixel_equal = False
        for engine_inv, expected_crop in zip(engine_inverted, expected["crops"]):
            if not torch.equal(
                engine_inv.tensor_image,
                direct_invert.run(image=expected_crop)["image"].tensor_image,
            ):
                pixel_equal = False
    report.check(
        "engine crops/inverted crops equal direct block calls pixel for pixel",
        pixel_equal,
    )

    mosaic = output_data(result, "mosaic")
    counts = list(output_data(result, "mosaic_count"))
    report.check("mosaic counts per parent are [2, 0, 1]", counts == [2, 0, 1], counts)
    blank = mosaic[1]
    report.check(
        "empty group (beta) produced the documented blank 48x48 canvas with background 128",
        blank.size_hw == (48, 48)
        and int(blank.tensor_image.min()) == 128
        and int(blank.tensor_image.max()) == 128,
        list(blank.tensor_image.shape),
    )
    report.check(
        "mosaic_all over the sample axis counted all 3 inputs",
        output_data(result, "mosaic_all_count") == 3,
        output_data(result, "mosaic_all_count"),
    )
    sample_ids = [
        outputs.metadata[output_entry(result, "crops")].sample_at(index).source_id
        for index, _ in _leaf_indices(crops)
    ]
    report.check(
        "every crop inherits its parent's source_id",
        sample_ids == ["fixture:alpha", "fixture:alpha", "fixture:gamma"],
        sample_ids,
    )
    mosaic_metadata = outputs.metadata[output_entry(result, "mosaic")]
    beta_temporal = mosaic_metadata.temporal_at((1,))
    alpha_temporal = mosaic_metadata.temporal_at((0,))
    report.check(
        "collapsed mosaic keeps alpha's temporal context and beta stays without one",
        alpha_temporal is not None and beta_temporal is None,
        {"alpha": alpha_temporal is not None, "beta": beta_temporal is None},
    )
    report.check(
        "all outputs complete",
        all(status == "complete" for status in result.statuses.values()),
        dict(result.statuses),
    )
    single = run_workflow(
        "single_image_mosaic",
        catalogue=create_catalogue(),
        inputs={"image": fixtures[2].image},
    )
    single_canvas = output_data(single, "mosaic")
    report.check(
        "a single ungrouped image bound to the mosaic group is cast into a one-image group",
        output_data(single, "count") == 1 and single_canvas.size_hw == (32, 32),
        {
            "count": output_data(single, "count"),
            "shape": list(single_canvas.tensor_image.shape),
        },
    )
    report.notes.append(
        "The [2,0,1] grouping, per-output layouts and blank canvas come from the compiled graph;"
        " the demo only compares them with direct block calls."
    )

    return report


def _leaf_indices(
    value: Any, prefix: Tuple[int, ...] = ()
) -> List[Tuple[Tuple[int, ...], Any]]:
    if not isinstance(value, Batch):
        return [(prefix, value)]

    leaves: List[Tuple[Tuple[int, ...], Any]] = []
    for index, child in value.iter_with_indices():
        leaves.extend(_leaf_indices(child, index))

    return leaves


def _run_filtered(scenario_dir: Path) -> ScenarioReport:
    report = ScenarioReport(name="filtered")
    fixtures = make_fixtures()
    _save_fixture_pngs(fixtures, scenario_dir / "inputs", report)

    result = run_workflow(
        "filtered", catalogue=create_catalogue(), inputs=_image_inputs(fixtures)
    )

    artifact_dir = scenario_dir / "outputs"
    document = describe_result(result, describe_payload=image_describer(artifact_dir))
    write_json(document, scenario_dir / "result.json")
    report.artifacts.append(str(scenario_dir / "result.json"))
    report.artifacts.extend(str(path) for path in sorted(artifact_dir.glob("*.png")))

    outputs = result.outputs
    keep_values = [
        (index, value) for index, value in _leaf_indices(output_data(result, "keep"))
    ]
    report.check(
        f"has_brightness(minimum={BRIGHTNESS_MINIMUM}) keeps only alpha's first crop",
        keep_values == [((0, 0), True), ((0, 1), False), ((2, 0), False)],
        [[list(index), value] for index, value in keep_values],
    )

    inverted = output_data(result, "inverted")
    inverted_leaves = (
        [index for index, _ in _leaf_indices(inverted)]
        if inverted is not None
        else None
    )
    report.check(
        "the v2/continue_if gate let invert run only for the surviving crop, index (0, 0)",
        inverted_leaves == [(0, 0)],
        (
            [list(index) for index in inverted_leaves]
            if inverted_leaves is not None
            else None
        ),
    )

    mosaic = output_data(result, "mosaic")
    mosaic_indices = list(mosaic.indices) if isinstance(mosaic, Batch) else None
    mosaic_count = output_data(result, "mosaic_count")
    mosaic_counts = (
        {index: value for index, value in _leaf_indices(mosaic_count)}
        if mosaic_count is not None
        else {}
    )
    report.check(
        "downstream mosaic: alpha partial (count 1), beta genuinely empty (count 0), gamma absent because all its candidates were filtered",
        mosaic_indices == [(0,), (1,)] and mosaic_counts == {(0,): 1, (1,): 0},
        {
            "indices": (
                [list(i) for i in mosaic_indices] if mosaic_indices else mosaic_indices
            ),
            "counts": {str(k): v for k, v in mosaic_counts.items()},
        },
    )
    unfiltered_counts = list(output_data(result, "mosaic_unfiltered_count"))
    report.check(
        "ungated sibling mosaic still sees all groups: counts [2, 0, 1]",
        unfiltered_counts == [2, 0, 1],
        unfiltered_counts,
    )
    report.check(
        "a status is reported for every declared output",
        set(result.selections)
        == {o["name"] for o in load_definition("filtered")["outputs"]}
        and set(result.statuses)
        == {key for ports in result.selections.values() for key in ports.values()},
        dict(result.statuses),
    )
    report.notes.append(
        "Filtered versus empty: beta has a mosaic (blank canvas, count 0) because its crop group is"
        " genuinely empty; gamma has no mosaic entry because its only crop was gated out."
    )
    report.notes.append(
        "The gate is an ordinary v2/continue_if step with next_steps; it replaces the earlier"
        " provisional 'when' field. Per-output statuses come from result.statuses."
    )

    return report


# (definition name, what is wrong, error class the compiler must raise)
INVALID_CASES: Tuple[Tuple[str, str, str], ...] = (
    (
        "unknown_selector",
        "step parameter references a step that does not exist",
        "SelectorError",
    ),
    ("unknown_block_type", "step type not in the catalogue", "UnknownBlockError"),
    ("kind_mismatch", "crop_summary bound to an image parameter", "KindMismatchError"),
    (
        "unrelated_lineages_equal_shape",
        "parent and children from unrelated sample axes",
        "LineageError",
    ),
    (
        "sibling_expansions_as_parent_child",
        "parent at [N,K_a] beside children of [N,K_b]",
        "LineageError",
    ),
    ("cycle", "two steps feeding each other", "CycleError"),
    (
        "bad_block_config",
        "crop region literal with three numbers",
        "ParamsValidationError",
    ),
    (
        "unknown_config_key",
        "crop step with unsupported parameter 'padding'",
        "ParamsValidationError",
    ),
    (
        "gate_not_boolean",
        "continue_if condition bound to a crop_summary output",
        "KindMismatchError",
    ),
    (
        "missing_required_input",
        "crop without its image parameter",
        "ParamsValidationError",
    ),
    (
        "unknown_section",
        "definition with an unsupported 'sources' section",
        "WorkflowCompileError",
    ),
    (
        "unknown_output_selector",
        "workflow output selecting a non-existent block output",
        "SelectorError",
    ),
    (
        "temporal_axis_input",
        "input declaring a time axis; the sequential engine rejects it",
        "WorkflowCompileError",
    ),
    (
        "wrong_version",
        "definition version 1.0 handed to the V2 compiler",
        "WorkflowCompileError",
    ),
)

RUNTIME_CASES: Tuple[Tuple[str, str], ...] = (
    ("broken_empty_mapping", "block returns {} (missing result, not a filter signal)"),
    ("broken_missing_output", "block omits its declared output name"),
    ("broken_raise", "block raises; original exception must be the cause"),
    ("broken_wrong_kind", "block returns a string where a number is declared"),
)


def _catalogue_with_author_blocks() -> Tuple[Catalogue, CallCounter]:
    counter = CallCounter()
    catalogue = Catalogue.merge(create_catalogue(), create_author_catalogue(counter))

    return catalogue, counter


def _exception_record(error: BaseException) -> Dict[str, Any]:
    record = {
        "type": type(error).__name__,
        "message": str(error),
        "cause": (
            None
            if error.__cause__ is None
            else {
                "type": type(error.__cause__).__name__,
                "message": str(error.__cause__),
            }
        ),
    }

    return record


def _run_invalid_bindings(scenario_dir: Path) -> ScenarioReport:
    report = ScenarioReport(name="invalid-bindings")
    cases: List[Dict[str, Any]] = []

    for case_name, description, expected_error in INVALID_CASES:
        catalogue, counter = _catalogue_with_author_blocks()
        definition = load_definition(f"invalid/{case_name}")
        record: Dict[str, Any] = {
            "case": case_name,
            "description": description,
            "stage": "compile",
        }
        try:
            compile_workflow(definition, catalogue=catalogue)
            record["error"] = None
        except Exception as error:
            record["error"] = _exception_record(error)
        record["block_runs_before_error"] = counter.total
        cases.append(record)
        error_type = record["error"]["type"] if record["error"] else None
        report.check(
            f"compile rejects '{case_name}' with {expected_error} before any block run",
            error_type == expected_error and counter.total == 0,
            error_type or "compiled without error",
        )

    for case_name, description in RUNTIME_CASES:
        catalogue, counter = _catalogue_with_author_blocks()
        definition = load_definition(f"runtime/{case_name}")
        record = {"case": case_name, "description": description, "stage": "run"}
        try:
            plan = compile_workflow(definition, catalogue=catalogue)
            record["compiled"] = True
        except Exception as error:
            record["compiled"] = False
            record["error"] = _exception_record(error)
            cases.append(record)
            report.check(
                f"'{case_name}' compiles (failure is a runtime contract violation)",
                False,
                record["error"],
            )
            continue

        try:
            plan.create_session().run({"values": Batch.of([1, 2])})
            record["error"] = None
        except Exception as error:
            record["error"] = _exception_record(error)
        record["block_runs"] = dict(counter.runs)
        cases.append(record)
        report.check(
            f"run raises for '{case_name}' with step context in the message",
            record["error"] is not None and "broken" in record["error"]["message"],
            record["error"],
        )

    write_json(cases, scenario_dir / "cases.json")
    report.artifacts.append(str(scenario_dir / "cases.json"))
    report.notes.append(
        "Every message above is the real exception text produced by the V2 compiler/executor."
    )

    return report


def _author_oracle(values: List[float]) -> Dict[str, Any]:
    """Direct calls of the author blocks, mirroring workflows/author_block.json.

    This is the comparison oracle only. It calls the block objects the way a
    unit test would; it does not stand in for the engine.
    """
    counter = CallCounter()
    scale = ScaleBlock(counter=counter)
    expand = ExpandBlock(counter=counter)
    reduce = SumWithParentBlock(counter=counter)

    def scale_two(value):
        return scale.run(value=value, factor=2)["scaled"]

    def scale_three(value):
        return scale.run(value=value, factor=3)["scaled"]

    def expand_shallow(value):
        return expand.run(value=value, offsets=[1, 2], limit=31)["children"]

    def expand_deep(value):
        return expand.run(value=value, offsets=[1], limit=60)["children"]

    def reduce_plain(parent, children):
        return reduce.run(parent=parent, children=children, parent_weight=0)

    def reduce_parent(parent, children):
        return reduce.run(parent=parent, children=children, parent_weight=1)

    oracle: Dict[str, Any] = {
        key: []
        for key in (
            "scaled_root",
            "children",
            "children_local",
            "scaled",
            "sums",
            "counts",
            "with_parent",
            "deep_children",
            "deep_scaled",
            "deep_totals",
            "deep_counts",
            "twice_reduced",
        )
    }
    for value in values:
        oracle["scaled_root"].append(scale_two(value))
        children = expand_shallow(value)
        oracle["children"].append(list(children))
        oracle["children_local"].append([index[0] for index in children.indices])
        scaled = [scale_two(child) for child in children]
        oracle["scaled"].append(scaled)
        plain = reduce_plain(value, children)
        oracle["sums"].append(plain["total"])
        oracle["counts"].append(plain["count"])
        oracle["with_parent"].append(reduce_parent(value, scaled)["total"])
        deep_children = [list(expand_deep(s)) for s in scaled]
        oracle["deep_children"].append(deep_children)
        deep_scaled = [[scale_three(d) for d in group] for group in deep_children]
        oracle["deep_scaled"].append(deep_scaled)
        deep_totals = [
            reduce_parent(s, group)["total"] for s, group in zip(scaled, deep_scaled)
        ]
        oracle["deep_totals"].append(deep_totals)
        oracle["deep_counts"].append([len(group) for group in deep_scaled])
        oracle["twice_reduced"].append(reduce_plain(value, deep_totals)["total"])

    return oracle


def _run_author_block(scenario_dir: Path) -> ScenarioReport:
    report = ScenarioReport(name="author-block")
    catalogue, counter = _catalogue_with_author_blocks()
    plan = compile_workflow(load_definition("author_block"), catalogue=catalogue)
    session = plan.create_session()

    runs = {
        "ragged": [10, 200, 30],
        "singleton": [30],
        "empty": [],
    }
    comparison: Dict[str, Any] = {}
    for run_name, values in runs.items():
        metadata = EntryMetadata(
            sample={
                (n,): SampleContext(source_id=f"value:{n}") for n in range(len(values))
            }
        )
        runs_before = counter.total
        result = session.run(
            {"values": InputValue(Batch.of(values), metadata=metadata)}
        )
        document = describe_result(result, describe_payload=plain_describer)
        write_json(document, scenario_dir / f"result_{run_name}.json")
        report.artifacts.append(str(scenario_dir / f"result_{run_name}.json"))

        oracle = _author_oracle(values)
        engine = {
            name: batch_to_nested(output_data(result, name))
            for name in oracle
            if name != "children_local"
        }
        engine["children_local"] = batch_local_indices(output_data(result, "children"))
        agreement = {name: engine[name] == oracle[name] for name in oracle}
        comparison[run_name] = {
            "values": values,
            "engine": engine,
            "direct": oracle,
            "agreement": agreement,
            "engine_block_runs_this_invocation": counter.total - runs_before,
            "run_id": result.run_id,
            "session_id": result.session_id,
            "layouts": {
                name: [axis.id for axis in result.outputs.layout[entry].axes]
                for name in result.selections
                if (entry := output_entry(result, name)) is not None
            },
        }
        report.check(
            f"[{run_name}] engine outputs equal direct block calls for all {len(oracle)} outputs",
            all(agreement.values()),
            {name: ok for name, ok in agreement.items() if not ok} or "all agree",
        )

    ragged = comparison["ragged"]["engine"]
    report.check(
        "ragged children are [[11, 12], [], [31]]",
        ragged["children"] == [[11, 12], [], [31]],
        ragged["children"],
    )
    report.check(
        "counts are [2, 0, 1] and sums are [23, 0, 31]",
        ragged["counts"] == [2, 0, 1] and ragged["sums"] == [23, 0, 31],
        {"counts": ragged["counts"], "sums": ragged["sums"]},
    )
    report.check(
        "parent+child reducer: [10+2*23, 200+0, 30+2*31] == [56, 200, 92]",
        ragged["with_parent"] == [56, 200, 92],
        ragged["with_parent"],
    )
    layouts = comparison["ragged"]["layouts"]
    report.check(
        "deep_scaled has three axes, deep_totals two, twice_reduced one",
        len(layouts["deep_scaled"]) == 3
        and len(layouts["deep_totals"]) == 2
        and len(layouts["twice_reduced"]) == 1,
        layouts,
    )
    run_ids = {run["run_id"] for run in comparison.values()}
    session_ids = {run["session_id"] for run in comparison.values()}
    report.check(
        "three runs of one session have distinct run ids and one session id",
        len(run_ids) == 3 and len(session_ids) == 1,
        {"run_ids": sorted(run_ids), "session_ids": sorted(session_ids)},
    )
    write_json(comparison, scenario_dir / "comparison.json")
    report.artifacts.append(str(scenario_dir / "comparison.json"))
    report.notes.append(
        "Author blocks are single classes with nested Params; the shared call counter is a"
        " constructor resource supplied by the catalogue."
    )

    return report


def _run_metadata_cost(scenario_dir: Path) -> ScenarioReport:
    report = ScenarioReport(name="metadata-cost")
    repeats = 20
    shapes = {
        "[N]": (8,),
        "[N,C]": (8, 16),
        "[N,C,D]": (8, 16, 4),
        "[N,C] ragged [2,0,1]": None,
    }
    measurements: List[Dict[str, Any]] = []
    for label, shape in shapes.items():
        leaves = _leaf_paths(shape)
        parents = sorted({path[:1] for path in leaves})
        inherited_map = {
            parent: SampleContext(source_id=f"source:{parent[0]}") for parent in parents
        }
        materialized_map = dict(inherited_map)
        for path in leaves:
            materialized_map[path] = SampleContext(source_id=f"source:{path[0]}")
        for variant, context_map in (
            ("inherited", inherited_map),
            ("materialized", materialized_map),
        ):
            construction = []
            lookup = []
            for _ in range(repeats):
                start = time.perf_counter()
                metadata = EntryMetadata(sample=context_map)
                construction.append(time.perf_counter() - start)
                start = time.perf_counter()
                for path in leaves:
                    metadata.sample_at(path)
                lookup.append(time.perf_counter() - start)
            tracemalloc.start()
            before, _ = tracemalloc.get_traced_memory()
            metadata = EntryMetadata(sample=context_map)
            after, _ = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            measurements.append(
                {
                    "layout": label,
                    "variant": variant,
                    "leaf_count": len(leaves),
                    "map_entries": len(context_map),
                    "construction_median_us": round(
                        statistics.median(construction) * 1e6, 1
                    ),
                    "lookup_all_leaves_median_us": round(
                        statistics.median(lookup) * 1e6, 1
                    ),
                    "lookup_per_leaf_median_ns": round(
                        statistics.median(lookup) / max(len(leaves), 1) * 1e9
                    ),
                    "metadata_allocation_bytes": after - before,
                }
            )

    fixtures = make_fixtures()
    plan = compile_workflow(load_definition("nested"), catalogue=create_catalogue())
    session = plan.create_session()
    workflow_timings: Dict[str, Any] = {}
    for variant in ("without_metadata", "with_metadata"):
        if variant == "with_metadata":
            inputs = _image_inputs(fixtures)
        else:
            inputs = {"images": Batch.of([fixture.image for fixture in fixtures])}
        session.run(inputs)  # warm-up, not timed
        durations = []
        for _ in range(5):
            start = time.perf_counter()
            session.run(inputs)
            durations.append(time.perf_counter() - start)
        tracemalloc.start()
        session.run(inputs)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        workflow_timings[variant] = {
            "run_median_ms": round(statistics.median(durations) * 1e3, 2),
            "run_min_ms": round(min(durations) * 1e3, 2),
            "tracemalloc_peak_bytes": peak,
        }

    document = {
        "statement": (
            "Measurements only. These numbers describe metadata construction, lookup and allocation"
            " on this machine. They are not a benchmark of temporal execution and make no speedup claim."
        ),
        "repeats": repeats,
        "synthetic_metadata": measurements,
        "nested_image_workflow": workflow_timings,
    }
    write_json(document, scenario_dir / "metadata_cost.json")
    report.artifacts.append(str(scenario_dir / "metadata_cost.json"))
    report.check(
        "all synthetic layouts measured",
        len(measurements) == 2 * len(shapes),
        len(measurements),
    )
    report.check(
        "nested workflow measured with and without metadata", len(workflow_timings) == 2
    )
    report.notes.append(document["statement"])

    return report


def _leaf_paths(shape: Optional[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
    if shape is None:
        ragged = [2, 0, 1] * 3
        paths = [
            (parent, child)
            for parent, count in enumerate(ragged)
            for child in range(count)
        ]
        return paths

    paths: List[Tuple[int, ...]] = [()]
    for size in shape:
        paths = [path + (position,) for path in paths for position in range(size)]

    return paths
