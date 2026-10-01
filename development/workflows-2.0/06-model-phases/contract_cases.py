"""Contract examples: selection, inspection, private phases, mutation, errors."""

import copy
from typing import Any, Callable, Dict

from assets import load_image
from host import DemoContext, compile_for, load_definition, output_value, run_passive
from native_comparison import compare_predictions
from observations import expect
from pydantic import Field
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
)
from roboflow_workflows.execution_engine.v2.errors import (
    DeclarationError,
    MutationConflictError,
    StepExecutionError,
    WorkflowCompileError,
)
from roboflow_workflows.execution_engine.v2.implementations import Implementation
from roboflow_workflows.execution_engine.v2.introspection import (
    describe_workflow,
    discover_workload,
)
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND
from roboflow_workflows.execution_engine.v2.phases import phase
from roboflow_workflows.execution_engine.v2.targets import (
    Target,
    UnsupportedTargetError,
)

BATCHED_TOLERANCE = 1e-5


def selection(context: DemoContext) -> Dict[str, Any]:
    """Compile-time choice per target, unphased fallback, unsupported target."""
    definition = load_definition("classify.json")
    targets = {
        "cpu": Target.cpu(),
        "cpu+batched_views": Target(frozenset({"cpu", "batched_views"})),
        "mps": Target(frozenset({"mps"})),
    }
    chosen = {}
    for label, target in targets.items():
        plan = compile_for(definition, target=target, execution="phases")
        description = describe_workflow(plan)
        chosen[label] = {
            step["node_id"]: {
                "implementation": step["implementation"]["name"],
                "execution": step["execution"],
                "considered": step["implementation"]["considered"],
            }
            for step in description["steps"]
            if step["type"] == "model_demo/flip_averaged_classifier@v1"
        }
    expect(
        "choice and execution of the classify step per target",
        {
            label: steps["$steps.classify"]["implementation"]
            + "/"
            + steps["$steps.classify"]["execution"]
            for label, steps in chosen.items()
        },
        {
            "cpu": "cpu/phases",
            "cpu+batched_views": "cpu-batched-views/run",
            "mps": "mps/phases",
        },
    )

    try:
        compile_for(definition, target=Target(frozenset({"cuda"})))
        unsupported = None
    except UnsupportedTargetError as error:
        unsupported = {
            "error": str(error),
            "considered": [list(item) for item in error.considered],
        }
    expect("a cuda-only target is rejected", unsupported is not None, True)

    image = load_image("beagle")
    batched, separate = (
        output_value(
            run_passive(
                compile_for(load_definition("classify_one.json"), target=target),
                context.state_dict,
                {"image": image},
            ),
            "predictions",
        )
        for target in (targets["cpu+batched_views"], targets["cpu"])
    )
    batched_vs_separate = compare_predictions(
        separate, batched, tolerance=BATCHED_TOLERANCE
    )
    expect("batched views agree within tolerance", batched_vs_separate["equal"], True)

    workload = discover_workload(compile_for(definition))
    weights = workload.step("$steps.classify").describe()

    evidence = {
        "chosen": chosen,
        "unsupported_target": unsupported,
        "batched_vs_separate_views": batched_vs_separate,
        "workload_without_loading_weights": weights,
    }

    return evidence


def private_phase(context: DemoContext) -> Dict[str, Any]:
    """Phase results are private: selecting one as a workflow output is rejected."""
    try:
        compile_for(load_definition("invalid/private_phase.json"))
        rejected = None
    except WorkflowCompileError as error:
        rejected = f"{type(error).__name__}: {error}"
    expect("selecting $steps.classify.probabilities fails", rejected is not None, True)

    evidence = {"rejected": rejected}

    return evidence


def mutation(context: DemoContext) -> Dict[str, Any]:
    """In-place redaction inside a phase: warn, strict rejection, ordered effect."""
    unordered = load_definition("mutation_unordered.json")
    warnings = list(compile_for(unordered).warnings)
    expect("unordered reader of a mutated image: one warning", len(warnings), 1)
    try:
        compile_for(unordered, mutation_conflicts="error")
        strict = None
    except MutationConflictError as error:
        strict = str(error)
    expect("strict mode rejects it", strict is not None, True)

    ordered = compile_for(load_definition("mutation_ordered.json"), execution="phases")
    expect("reader after the writer: no warning", list(ordered.warnings), [])
    untouched = output_value(
        run_passive(
            compile_for(load_definition("classify_one.json")),
            context.state_dict,
            {"image": load_image("beagle")},
        ),
        "top_class",
    )
    image = load_image("beagle")
    tensor = image.tensor_image
    result = run_passive(ordered, context.state_dict, {"image": image})
    redacted = output_value(result, "redacted")
    after = output_value(result, "top_class")
    expect(
        "the caller's tensor itself was changed",
        bool((tensor[:, :640, :] == 128).all()),
        True,
    )
    expect(
        "no copy: the output shares the input storage",
        redacted.tensor_image.data_ptr(),
        tensor.data_ptr(),
    )
    expect("hiding the head changes the class", after != untouched, True)

    context.gallery.section("mutation: redact in place, then classify")
    context.gallery.add(
        redacted,
        name="mutation-redacted",
        lines=[f"before redaction: {untouched}", f"after redaction: {after}"],
    )

    evidence = {
        "warning": warnings[0],
        "strict_error": strict,
        "top_class_before": untouched,
        "top_class_after": after,
    }

    return evidence


def phase_error(context: DemoContext) -> Dict[str, Any]:
    """A failing phase is named in both modes, with the step and index."""
    definition = copy.deepcopy(load_definition("mutation_ordered.json"))
    definition["steps"][0]["region"] = [2000, 2000, 2100, 2100]
    errors = {}
    for mode in ("run", "phases"):
        try:
            run_passive(
                compile_for(definition, execution=mode),
                context.state_dict,
                {"image": load_image("beagle")},
            )
            errors[mode] = None
        except StepExecutionError as error:
            errors[mode] = {
                "phase": error.phase,
                "step": list(error.step_path),
                "message": str(error),
            }
    expect(
        "phase and step in both modes",
        {mode: (error or {}).get("phase") for mode, error in errors.items()},
        {"run": "bounds", "phases": "bounds"},
    )

    return errors


def _declare(build: Callable[[], type]) -> str:
    try:
        build()
    except DeclarationError as error:
        return str(error)

    raise AssertionError(f"{build.__name__} was accepted")


def _cycle() -> type:
    class Cyclic(Block):
        type = "authoring_demo/cyclic@v1"
        outputs = {"value": Output(FLOAT_KIND)}

        class Params(BlockParams):
            value: float = Field(description="Input.")

        @phase
        def scaled(self, value: float, shifted: float) -> float:
            return value * shifted

        @phase
        def shifted(self, scaled: float) -> float:
            return scaled + 1

        @phase
        def result(self, shifted: float) -> dict:
            return {"value": shifted}

        def run(self, *, value: float) -> dict:
            raise NotImplementedError

    return Cyclic


def _unknown_name() -> type:
    class UnknownName(Block):
        type = "authoring_demo/unknown_name@v1"
        outputs = {"value": Output(FLOAT_KIND)}

        class Params(BlockParams):
            value: float = Field(description="Input.")

        @phase
        def result(self, value: float, scale: float) -> dict:
            return {"value": value * scale}

        def run(self, *, value: float) -> dict:
            raise NotImplementedError

    return UnknownName


def _restated_contract() -> type:
    class Restating(Implementation):
        name = "restating"
        outputs = {"value": Output(FLOAT_KIND)}

        def run(self, *, value: float) -> dict:
            return {"value": value}

    class Contract(Block):
        type = "authoring_demo/restated@v1"
        implementations = (Restating,)
        outputs = {"value": Output(FLOAT_KIND)}

        class Params(BlockParams):
            value: float = Field(description="Input.")

    return Contract


def authoring_errors(context: DemoContext) -> Dict[str, Any]:
    """Declaration mistakes fail when the class is defined, with a clear reason."""
    evidence = {
        "phase cycle": _declare(_cycle),
        "unknown phase parameter": _declare(_unknown_name),
        "implementation restates the contract": _declare(_restated_contract),
    }

    return evidence
