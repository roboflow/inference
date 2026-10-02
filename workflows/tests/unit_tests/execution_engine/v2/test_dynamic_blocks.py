"""Dynamic blocks: V1-shaped custom Python definitions compiled and run by V2.

Workflows here go through the real compiler (``compile_workflow``) and the
real execution session. Submitted code records its own calls, so the tests
check invocation counts and state, not only returned values.
"""

import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from roboflow_workflows.execution_engine.v2 import (
    Batch,
    Catalogue,
    CompileOptions,
    compile_workflow,
    spec_of,
)
from roboflow_workflows.execution_engine.v2.blocks.control import ContinueIfBlock
from roboflow_workflows.execution_engine.v2.context import (
    ExecutionContext,
    NoExecutionContextError,
    use_execution_context,
)
from roboflow_workflows.execution_engine.v2.dynamic_blocks import (
    BLOCK_SOURCE,
    LEGACY_REPRESENTATION,
    DynamicBlockError,
    DynamicCodeError,
    LocalCodeNotAllowedError,
    RepresentationError,
    RepresentationPolicy,
    build_dynamic_catalogue,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ResourceError,
    StepExecutionError,
)
from roboflow_workflows.execution_engine.v2.kinds import BUILTIN_KINDS, FLOAT_KIND, Kind

CATALOGUE = Catalogue([ContinueIfBlock], kinds=BUILTIN_KINDS)
LOCAL = CompileOptions(allow_local_code=True)


def dynamic_input(
    *selector_types: str, kinds: Optional[List[str]] = None, **fields: Any
) -> Dict[str, Any]:
    definition = {
        "type": "DynamicInputDefinition",
        "selector_types": list(selector_types),
        **fields,
    }
    if kinds is not None:
        definition["selector_data_kind"] = {name: kinds for name in selector_types}

    return definition


def dynamic_block(
    block_type: str,
    *,
    inputs: Dict[str, Any],
    outputs: Dict[str, List[str]],
    run: str,
    init: Optional[str] = None,
    imports: Optional[List[str]] = None,
    **manifest: Any,
) -> Dict[str, Any]:
    code = {"type": "PythonCode", "run_function_code": run, "imports": imports or []}
    if init is not None:
        code["init_function_code"] = init

    definition = {
        "type": "DynamicBlockDefinition",
        "manifest": {
            "type": "ManifestDescription",
            "block_type": block_type,
            "inputs": inputs,
            "outputs": {
                name: {"type": "DynamicOutputDefinition", "kind": kinds}
                for name, kinds in outputs.items()
            },
            **manifest,
        },
        "code": code,
    }

    return definition


def workflow(
    *,
    dynamic: List[Dict[str, Any]],
    steps: List[Dict[str, Any]],
    outputs: Dict[str, str],
    inputs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    definition = {
        "version": "2.0",
        "inputs": (
            inputs
            if inputs is not None
            else [{"type": "WorkflowBatchInput", "name": "values", "kind": ["integer"]}]
        ),
        "dynamic_blocks_definitions": dynamic,
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs.items()
        ],
    }

    return definition


def block_class(definition: Dict[str, Any], *, allow_local_code: bool = True) -> type:
    catalogue = build_dynamic_catalogue(
        [definition], catalogue=CATALOGUE, allow_local_code=allow_local_code
    )
    found = catalogue.entry(definition["manifest"]["block_type"]).spec.block_class

    return found


COUNTER_INIT = "def init():\n    return {'calls': 0}\n"

COUNTING = dynamic_block(
    "Counting",
    inputs={"value": dynamic_input("step_output", "input_parameter")},
    outputs={"seen": []},
    init=COUNTER_INIT,
    run=(
        "def run(self, value):\n"
        "    self._init_results['calls'] += 1\n"
        "    return {'seen': (value, self._init_results['calls'])}\n"
    ),
)

EXPAND = dynamic_block(
    "Expand",
    inputs={"value": dynamic_input("step_output", "input_parameter")},
    outputs={"child": ["integer"]},
    output_dimensionality_offset=1,
    run="def run(self, value):\n    return [{'child': value * 10 + i} for i in range(value)]\n",
)


def recording_block(
    block_type: str, *, sentinel: Path, imports: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Echo block whose module top and init append to ``sentinel`` when executed."""
    record = (
        "def record(text):\n"
        f"    with open({str(sentinel)!r}, 'a') as log:\n"
        "        log.write(text + '\\n')\n"
    )
    definition = dynamic_block(
        block_type,
        inputs={"value": dynamic_input("input_parameter")},
        outputs={"value": []},
        imports=imports,
        run=record
        + "record('module')\ndef run(self, value):\n    return {'value': value}\n",
        init="def init():\n    record('init')\n    return {}\n",
    )

    return definition


class TestSubmittedCodeRunsOnlyInAnOptedInSession:
    def test_building_compiling_and_inspecting_never_execute_code(
        self, tmp_path: Path
    ) -> None:
        # given
        sentinel = tmp_path / "executed.txt"
        definition = recording_block(
            "Sentinel",
            sentinel=sentinel,
            imports=["import module_that_does_not_exist_anywhere"],
        )
        document = workflow(
            dynamic=[definition],
            inputs=[{"type": "WorkflowParameter", "name": "x"}],
            steps=[{"type": "Sentinel", "name": "s", "value": "$inputs.x"}],
            outputs={"value": "$steps.s.value"},
        )

        # when
        plan = compile_workflow(document, catalogue=CATALOGUE, options=LOCAL)
        plan.describe()
        spec_of(plan.step(("s",)).spec.block_class).describe()

        # then
        assert not sentinel.exists()
        with pytest.raises(ResourceError) as raised:
            plan.create_session()
        assert isinstance(raised.value.__cause__, DynamicCodeError)
        assert isinstance(raised.value.__cause__.__cause__, ModuleNotFoundError)

    def test_code_runs_once_per_step_when_a_session_is_created(
        self, tmp_path: Path
    ) -> None:
        # given
        sentinel = tmp_path / "executed.txt"
        definition = recording_block("Sentinel", sentinel=sentinel)
        document = workflow(
            dynamic=[definition],
            inputs=[{"type": "WorkflowParameter", "name": "x"}],
            steps=[
                {"type": "Sentinel", "name": "first", "value": "$inputs.x"},
                {"type": "Sentinel", "name": "second", "value": "$inputs.x"},
            ],
            outputs={"value": "$steps.second.value"},
        )
        plan = compile_workflow(document, catalogue=CATALOGUE, options=LOCAL)
        assert not sentinel.exists()

        # when
        session = plan.create_session()
        written_at_creation = sentinel.read_text().splitlines()
        rows = [session.run({"x": value}).rows() for value in (1, 2)]

        # then
        assert written_at_creation == ["module", "init", "module", "init"]
        assert sentinel.read_text().splitlines() == written_at_creation
        assert rows == [[{"value": 1}], [{"value": 2}]]

    def test_disallowed_local_code_compiles_but_refuses_session_creation(
        self, tmp_path: Path
    ) -> None:
        # given
        sentinel = tmp_path / "executed.txt"
        definition = recording_block("Refused", sentinel=sentinel)
        document = workflow(
            dynamic=[definition],
            inputs=[{"type": "WorkflowParameter", "name": "x"}],
            steps=[{"type": "Refused", "name": "s", "value": "$inputs.x"}],
            outputs={"value": "$steps.s.value"},
        )

        # when
        plan = compile_workflow(document, catalogue=CATALOGUE)

        # then
        assert plan.step(("s",)).namespace == BLOCK_SOURCE
        with pytest.raises(ResourceError) as raised:
            plan.create_session()
        assert isinstance(raised.value.__cause__, LocalCodeNotAllowedError)
        assert "allow_local_code=True" in str(raised.value.__cause__)
        assert "not sandboxed" in str(raised.value.__cause__)
        assert not sentinel.exists()

    def test_code_raising_at_module_top_fails_only_at_session_creation(self) -> None:
        # given
        definition = dynamic_block(
            "Broken",
            inputs={"value": dynamic_input("input_parameter")},
            outputs={"value": []},
            run="raise RuntimeError('top-level boom')\ndef run(self, value):\n    return {}\n",
        )
        document = workflow(
            dynamic=[definition],
            inputs=[{"type": "WorkflowParameter", "name": "x"}],
            steps=[{"type": "Broken", "name": "s", "value": "$inputs.x"}],
            outputs={"value": "$steps.s.value"},
        )
        plan = compile_workflow(document, catalogue=CATALOGUE, options=LOCAL)

        # when
        with pytest.raises(ResourceError) as raised:
            plan.create_session()

        # then
        error = raised.value.__cause__
        assert isinstance(error, DynamicCodeError)
        assert "'Broken'" in str(error) and "code.run_function_code" in str(error)
        assert str(error.__cause__) == "top-level boom"


class TestStateOfInitAndRun:
    def test_init_state_persists_per_step_and_session_and_is_never_shared(
        self,
    ) -> None:
        # given
        child = {
            "version": "2.0",
            "inputs": [{"type": "WorkflowParameter", "name": "x"}],
            "steps": [{"type": "Counting", "name": "count", "value": "$inputs.x"}],
            "outputs": [
                {"type": "JsonField", "name": "seen", "selector": "$steps.count.seen"}
            ],
        }
        document = workflow(
            dynamic=[COUNTING],
            inputs=[{"type": "WorkflowParameter", "name": "x"}],
            steps=[
                {"type": "Counting", "name": "a", "value": "$inputs.x"},
                {"type": "Counting", "name": "b", "value": "$inputs.x"},
                {
                    "type": "roboflow_core/inner_workflow@v1",
                    "name": "child_one",
                    "workflow_definition": child,
                    "parameter_bindings": {"x": "$inputs.x"},
                },
                {
                    "type": "roboflow_core/inner_workflow@v1",
                    "name": "child_two",
                    "workflow_definition": child,
                    "parameter_bindings": {"x": "$inputs.x"},
                },
            ],
            outputs={
                "a": "$steps.a.seen",
                "b": "$steps.b.seen",
                "one": "$steps.child_one.seen",
                "two": "$steps.child_two.seen",
            },
        )
        plan = compile_workflow(document, catalogue=CATALOGUE, options=LOCAL)

        # when
        session = plan.create_session()
        session.run({"x": "first"})
        session.run({"x": "second"})
        a_third = session.run({"x": "third"}).rows()
        fresh = plan.create_session().run({"x": "fresh"}).rows()

        # then
        assert a_third == [
            {
                "a": ("third", 3),
                "b": ("third", 3),
                "one": ("third", 3),
                "two": ("third", 3),
            }
        ]
        assert fresh == [
            {
                "a": ("fresh", 1),
                "b": ("fresh", 1),
                "one": ("fresh", 1),
                "two": ("fresh", 1),
            }
        ]
        instances = session.instances
        assert len({id(instance._init_results) for instance in instances.values()}) == 4

    def test_submitted_code_can_keep_state_on_self_and_import_explicitly(self) -> None:
        # given
        definition = dynamic_block(
            "Accumulate",
            inputs={"value": dynamic_input("input_parameter")},
            outputs={"history": [], "mean": ["float"]},
            imports=["import statistics"],
            run=(
                "def run(self, value):\n"
                "    self.history = getattr(self, 'history', []) + [value]\n"
                "    return {'history': list(self.history), 'mean': statistics.mean(self.history)}\n"
            ),
        )
        block = block_class(definition)()

        # when
        results = [block.run(value=value) for value in (2, 4)]

        # then
        assert results == [{"history": [2], "mean": 2}, {"history": [2, 4], "mean": 3}]

    def test_v1_implicit_imports_are_not_provided_and_errors_say_what_to_add(
        self,
    ) -> None:
        # given
        definition = dynamic_block(
            "UsesNumpy",
            inputs={"value": dynamic_input("input_parameter")},
            outputs={"value": []},
            run="def run(self, value):\n    return {'value': np.array(value)}\n",
        )
        block = block_class(definition)()

        # when
        with pytest.raises(NameError) as raised:
            block.run(value=[1])

        # then
        assert "'import numpy as np'" in str(raised.value)
        assert "code.imports" in str(raised.value)

    def test_submitted_code_sees_the_v2_batch_type(self) -> None:
        # given
        definition = dynamic_block(
            "WhichBatch",
            inputs={},
            outputs={"same": []},
            imports=[
                "from roboflow_workflows.execution_engine.v2.data import Batch as V2"
            ],
            run="def run(self):\n    return {'same': Batch is V2}\n",
        )

        # when
        result = block_class(definition)().run()

        # then
        assert result == {"same": True}

    def test_run_failure_names_the_nested_step_index_and_submitted_line(
        self,
    ) -> None:
        # given
        failing = dynamic_block(
            "Fails",
            inputs={"value": dynamic_input("step_output", "input_parameter")},
            outputs={"out": []},
            run=(
                "def run(self, value):\n"
                "    if value == 2:\n"
                "        raise ValueError(f'cannot handle {value}')\n"
                "    return {'out': value}\n"
            ),
        )
        child = {
            "version": "2.0",
            "inputs": [
                {"type": "WorkflowBatchInput", "name": "x", "kind": ["integer"]}
            ],
            "steps": [{"type": "Fails", "name": "fails", "value": "$inputs.x"}],
            "outputs": [
                {"type": "JsonField", "name": "out", "selector": "$steps.fails.out"}
            ],
        }
        document = workflow(
            dynamic=[failing],
            steps=[
                {
                    "type": "roboflow_core/inner_workflow@v1",
                    "name": "child",
                    "workflow_definition": child,
                    "parameter_bindings": {"x": "$inputs.values"},
                }
            ],
            outputs={"out": "$steps.child.out"},
        )
        session = compile_workflow(
            document, catalogue=CATALOGUE, options=LOCAL
        ).create_session()

        # when
        with pytest.raises(StepExecutionError) as raised:
            session.run({"values": [1, 2]})

        # then
        error = raised.value
        assert error.step_path == ("child", "fails") and error.index == (1,)
        assert isinstance(error.__cause__, ValueError)
        frame = traceback.extract_tb(error.__cause__.__traceback__)[-1]
        assert frame.filename == "<dynamic block Fails: run_function_code>"
        assert frame.lineno == 3
        assert frame.line == "raise ValueError(f'cannot handle {value}')"


class TestParametersFromTheManifest:
    def test_literals_selectors_defaults_and_optional_values(self) -> None:
        # given
        definition = dynamic_block(
            "Scale",
            inputs={
                "value": dynamic_input(
                    "step_output",
                    "input_parameter",
                    kinds=["float"],
                    value_types=["float"],
                ),
                "factor": dynamic_input(
                    "input_parameter",
                    value_types=["integer", "float"],
                    has_default_value=True,
                    default_value=2,
                ),
                "label": dynamic_input(
                    value_types=["string"], is_optional=True, has_default_value=True
                ),
                "tags": dynamic_input(
                    value_types=["list"], has_default_value=True, default_value=["base"]
                ),
            },
            outputs={"scaled": ["float"], "label": [], "tags": []},
            run=(
                "def run(self, value, factor, label, tags):\n"
                "    tags.append('seen')\n"
                "    return {'scaled': value * factor, 'label': label, 'tags': tags}\n"
            ),
        )
        document = workflow(
            dynamic=[definition],
            inputs=[
                {"type": "WorkflowParameter", "name": "value", "kind": ["float"]},
                {"type": "WorkflowParameter", "name": "factor", "default_value": 10},
            ],
            steps=[
                {"type": "Scale", "name": "defaults", "value": 1.5},
                {
                    "type": "Scale",
                    "name": "selected",
                    "value": "$inputs.value",
                    "factor": "$inputs.factor",
                    "label": "$inputs.not-a-selector-in-a-string-field",
                },
            ],
            outputs={"defaults": "$steps.defaults.*", "selected": "$steps.selected.*"},
        )
        plan = compile_workflow(document, catalogue=CATALOGUE, options=LOCAL)

        # when
        rows = plan.create_session().run({"value": 0.5}).rows()

        # then
        assert rows[0]["defaults"] == {
            "scaled": 3.0,
            "label": None,
            "tags": ["base", "seen"],
        }
        assert rows[0]["selected"] == {
            "scaled": 5.0,
            "label": "$inputs.not-a-selector-in-a-string-field",
            "tags": ["base", "seen"],
        }

    def test_generated_params_keep_defaults_distinct_from_explicit_values(self) -> None:
        # given
        spec = spec_of(
            block_class(
                dynamic_block(
                    "Optional",
                    inputs={
                        "limit": dynamic_input(
                            "input_parameter",
                            value_types=["float"],
                            is_optional=True,
                            has_default_value=True,
                            default_value=0.5,
                        ),
                        "required": dynamic_input("input_parameter", is_optional=True),
                    },
                    outputs={},
                    run="def run(self, limit, required):\n    return {}\n",
                )
            )
        )

        # when
        defaulted = spec.validate_params({"required": None})
        explicit = spec.validate_params({"required": None, "limit": None})

        # then
        assert spec.fields["required"].required is True
        assert defaulted.limit == 0.5 and "limit" not in defaulted.model_fields_set
        assert explicit.limit is None and "limit" in explicit.model_fields_set

    def test_kinds_are_the_catalogue_objects(self) -> None:
        # given
        detection = Kind(name="object_detection_prediction")
        catalogue = Catalogue(kinds=[detection, FLOAT_KIND])
        definition = dynamic_block(
            "Detections",
            inputs={
                "predictions": dynamic_input("step_output", kinds=[detection.name]),
                "image": dynamic_input("input_image"),
            },
            outputs={"count": ["float"]},
            run="def run(self, predictions, image):\n    return {'count': 0}\n",
        )
        image = Kind(name="image")

        # when
        dynamic = build_dynamic_catalogue(
            [definition], catalogue=Catalogue.merge(catalogue, Catalogue(kinds=[image]))
        )

        # then
        spec = dynamic.entry("Detections").spec
        assert spec.fields["predictions"].whole.kinds == (detection,)
        assert spec.fields["image"].whole.kinds == (image,)
        assert spec.outputs["count"].kinds == (FLOAT_KIND,)
        assert Catalogue.merge(catalogue, dynamic).kind(detection.name) is detection


class TestInvalidDefinitionsAreReportedWithContext:
    @pytest.mark.parametrize(
        ("manifest_changes", "code_changes", "expected"),
        [
            (
                {
                    "inputs": {
                        "value": dynamic_input("step_output", kinds=["no_such_kind"])
                    }
                },
                {},
                ["'Faulty'", "inputs.value", "unknown kind 'no_such_kind'", "'float'"],
            ),
            (
                {"inputs": {"image": dynamic_input("input_image")}},
                {"run_function_code": "def run(self, image):\n    return {}\n"},
                ["inputs.image", "unknown kind 'image'"],
            ),
            (
                {
                    "outputs": {
                        "out": {"type": "DynamicOutputDefinition", "kind": ["nope"]}
                    }
                },
                {},
                ["outputs.out", "unknown kind 'nope'"],
            ),
            (
                {"inputs": {"value": {"type": "DynamicInputDefinition"}}},
                {},
                ["inputs.value", "neither selector_types nor value_types"],
            ),
            (
                {"inputs": {"name": dynamic_input("input_parameter")}},
                {"run_function_code": "def run(self, name):\n    return {}\n"},
                ["inputs.name", "reserved for the step"],
            ),
            (
                {},
                {
                    "run_function_code": "def run(self, value):\n    return {'value': value\n"
                },
                ["code.run_function_code line 2", "syntax error"],
            ),
            (
                {},
                {"run_function_code": "def run(self):\n    return {}\n"},
                ["line 1", "does not accept input(s) ['value']"],
            ),
            (
                {},
                {"run_function_code": "def run(self, value, extra):\n    return {}\n"},
                ["requires ['extra']", "not manifest inputs"],
            ),
            (
                {},
                {"run_function_code": "def execute(self, value):\n    return {}\n"},
                ["defines no top-level function 'run'"],
            ),
            (
                {},
                {"run_function_code": "async def run(self, value):\n    return {}\n"},
                ["must be a plain function"],
            ),
            (
                {},
                {"init_function_code": "def init(size):\n    return {}\n"},
                ["init() is called once without arguments"],
            ),
            (
                {},
                {"init_function_code": "def setup():\n    return {}\n"},
                ["code.init_function_code", "no top-level function 'init'"],
            ),
            (
                {
                    "inputs": {
                        "a": dynamic_input(
                            "step_output", is_dimensionality_reference=True
                        ),
                        "b": dynamic_input(
                            "step_output", is_dimensionality_reference=True
                        ),
                    }
                },
                {"run_function_code": "def run(self, a, b):\n    return {}\n"},
                ["several dimensionality references ['a', 'b']"],
            ),
            (
                {
                    "inputs": {
                        "a": dynamic_input("step_output"),
                        "b": dynamic_input("step_output", dimensionality_offset=1),
                    }
                },
                {"run_function_code": "def run(self, a, b):\n    return {}\n"},
                ["offsets {'b': 1}", "is_dimensionality_reference"],
            ),
            (
                {
                    "inputs": {"a": dynamic_input("input_parameter")},
                    "output_dimensionality_offset": -1,
                },
                {"run_function_code": "def run(self, a):\n    return {}\n"},
                ["output_dimensionality_offset -1", "no input can receive batch data"],
            ),
            (
                {"batch_oriented_parameters": ["values"]},
                {},
                ["manifest.batch_oriented_parameters", "'values'"],
            ),
        ],
    )
    def test_definition_errors_name_the_block_and_the_faulty_part(
        self,
        manifest_changes: Dict[str, Any],
        code_changes: Dict[str, Any],
        expected: List[str],
    ) -> None:
        # given
        definition = dynamic_block(
            "Faulty",
            inputs={"value": dynamic_input("step_output")},
            outputs={"out": []},
            run="def run(self, value):\n    return {'out': value}\n",
        )
        definition["manifest"].update(manifest_changes)
        definition["code"].update(code_changes)

        # when
        with pytest.raises(DynamicBlockError) as raised:
            build_dynamic_catalogue([definition], catalogue=CATALOGUE)

        # then
        for fragment in expected:
            assert fragment in str(raised.value)

    def test_malformed_definition_reports_its_position_and_schema_problem(self) -> None:
        # when
        with pytest.raises(DynamicBlockError) as raised:
            build_dynamic_catalogue(
                [
                    COUNTING,
                    {"type": "DynamicBlockDefinition", "manifest": {"block_type": "X"}},
                ],
                catalogue=CATALOGUE,
            )

        # then
        assert "'X'" in str(raised.value)
        assert "definition 1" in str(raised.value)
        assert "code: Field required" in str(raised.value)

    def test_one_definition_per_block_type_and_no_catalogue_collision(self) -> None:
        # when
        with pytest.raises(DynamicBlockError) as twice:
            build_dynamic_catalogue([COUNTING, COUNTING], catalogue=CATALOGUE)
        with pytest.raises(DynamicBlockError) as collision:
            build_dynamic_catalogue(
                [
                    dynamic_block(
                        "v2/continue_if",
                        inputs={},
                        outputs={},
                        run="def run(self):\n    return {}\n",
                    )
                ],
                catalogue=CATALOGUE,
            )

        # then
        assert "entries 0 and 1" in str(twice.value)
        assert "ContinueIfBlock" in str(collision.value)

    def test_run_accepting_kwargs_receives_every_input(self) -> None:
        # given
        definition = dynamic_block(
            "Kwargs",
            inputs={
                "a": dynamic_input("input_parameter"),
                "b": dynamic_input(
                    value_types=["integer"], has_default_value=True, default_value=3
                ),
            },
            outputs={"received": []},
            run="def run(self, **inputs):\n    return {'received': sorted(inputs.items())}\n",
        )

        # when
        result = block_class(definition)().run(a=1, b=3)

        # then
        assert result == {"received": [("a", 1), ("b", 3)]}


class TestBatchDeclarations:
    @pytest.mark.parametrize(
        ("selector_types", "flags", "expected"),
        [
            (["step_output"], {}, "never"),
            (["generic"], {"batch_oriented_parameters": ["value"]}, "always"),
            (["step_output"], {"accepts_batch_input": True}, "always"),
            (["input_parameter"], {"accepts_batch_input": True}, "never"),
            (["generic"], {"accepts_batch_input": True}, "never"),
            (
                ["generic"],
                {"parameters_with_scalars_and_batches": ["value"]},
                "if_varying",
            ),
            (
                ["generic"],
                {
                    "parameters_with_scalars_and_batches": ["value"],
                    "get_parameters_enforcing_auto_batch_casting": ["value"],
                },
                "always",
            ),
            (
                ["step_output", "input_parameter"],
                {"accepts_batch_input": True},
                "if_varying",
            ),
        ],
    )
    def test_v1_batch_flags_become_leaf_batch_modes(
        self, selector_types: List[str], flags: Dict[str, Any], expected: str
    ) -> None:
        # given
        definition = dynamic_block(
            "Batched",
            inputs={"value": dynamic_input(*selector_types)},
            outputs={},
            run="def run(self, value):\n    return {}\n",
            **flags,
        )

        # when
        spec = spec_of(block_class(definition))

        # then
        assert spec.fields["value"].whole.batch == expected

    def test_batch_oriented_block_is_called_once_with_a_v2_batch(self) -> None:
        # given
        definition = dynamic_block(
            "BatchDouble",
            inputs={"values": dynamic_input("generic")},
            outputs={"doubled": []},
            batch_oriented_parameters=["values"],
            init=COUNTER_INIT,
            run=(
                "def run(self, values):\n"
                "    self._init_results['calls'] += 1\n"
                "    assert isinstance(values, Batch)\n"
                "    return [{'doubled': (v * 2, self._init_results['calls'])} for v in values]\n"
            ),
        )
        document = workflow(
            dynamic=[definition],
            steps=[
                {"type": "BatchDouble", "name": "double", "values": "$inputs.values"}
            ],
            outputs={"doubled": "$steps.double.doubled"},
        )
        plan = compile_workflow(document, catalogue=CATALOGUE, options=LOCAL)

        # when
        rows = plan.create_session().run({"values": [1, 2, 3]}).rows()

        # then
        assert plan.step(("double",)).delivers_batches is True
        assert [row["doubled"] for row in rows] == [(2, 1), (4, 1), (6, 1)]

    def test_scalar_or_batch_parameter_receives_a_plain_scalar_when_constant(
        self,
    ) -> None:
        # given: V1 case scalar_input_batch_capable (tasks/m1r-batch-call-check)
        definition = dynamic_block(
            "Either",
            inputs={"value": dynamic_input("generic")},
            outputs={"item": []},
            parameters_with_scalars_and_batches=["value"],
            run=(
                "def run(self, value):\n"
                "    if isinstance(value, Batch):\n"
                "        return [{'item': ('batch', v)} for v in value]\n"
                "    return {'item': ('scalar', value)}\n"
            ),
        )
        document = workflow(
            dynamic=[definition],
            inputs=[
                {"type": "WorkflowBatchInput", "name": "values", "kind": ["integer"]},
                {"type": "WorkflowParameter", "name": "constant"},
            ],
            steps=[
                {"type": "Either", "name": "scalar", "value": "$inputs.constant"},
                {"type": "Either", "name": "batch", "value": "$inputs.values"},
            ],
            outputs={"scalar": "$steps.scalar.item", "batch": "$steps.batch.item"},
        )
        plan = compile_workflow(document, catalogue=CATALOGUE, options=LOCAL)

        # when
        rows = plan.create_session().run({"values": [1, 2], "constant": "c"}).rows()

        # then
        assert rows == [
            {"scalar": ("scalar", "c"), "batch": ("batch", 1)},
            {"scalar": ("scalar", "c"), "batch": ("batch", 2)},
        ]


class TestDimensionality:
    def test_output_offset_one_expands_v1_child_lists_into_a_new_axis(self) -> None:
        # given
        document = workflow(
            dynamic=[EXPAND],
            steps=[{"type": "Expand", "name": "expand", "value": "$inputs.values"}],
            outputs={"children": "$steps.expand.child"},
        )
        plan = compile_workflow(document, catalogue=CATALOGUE, options=LOCAL)

        # when
        rows = plan.create_session().run({"values": [2, 0, 1]}).rows()

        # then
        output = plan.step(("expand",)).outputs["child"]
        assert output.transform == "expand"
        assert output.layout.axis_ids == ("inputs", "expand:children")
        assert rows == [{"children": [20, 21]}, {"children": []}, {"children": [10]}]

    def test_batch_block_expands_one_v1_child_list_per_invocation(self) -> None:
        # given
        definition = dynamic_block(
            "BatchExpand",
            inputs={"values": dynamic_input("step_output", "input_parameter")},
            outputs={"child": []},
            output_dimensionality_offset=1,
            batch_oriented_parameters=["values"],
            init=COUNTER_INIT,
            run=(
                "def run(self, values):\n"
                "    self._init_results['calls'] += 1\n"
                "    return [\n"
                "        [{'child': (v, i, self._init_results['calls'])} for i in range(v)]\n"
                "        for v in values\n"
                "    ]\n"
            ),
        )
        document = workflow(
            dynamic=[definition],
            steps=[
                {"type": "BatchExpand", "name": "expand", "values": "$inputs.values"}
            ],
            outputs={"children": "$steps.expand.child"},
        )
        plan = compile_workflow(document, catalogue=CATALOGUE, options=LOCAL)

        # when
        rows = plan.create_session().run({"values": [2, 1]}).rows()

        # then
        assert rows == [
            {"children": [(2, 0, 1), (2, 1, 1)]},
            {"children": [(1, 0, 1)]},
        ]

    def test_expanded_children_must_carry_exactly_the_declared_outputs(self) -> None:
        # given
        definition = dynamic_block(
            "BadExpand",
            inputs={},
            outputs={"a": [], "b": []},
            output_dimensionality_offset=1,
            run="def run(self):\n    return [{'a': 1, 'b': 2}, {'a': 3}]\n",
        )

        # when
        with pytest.raises(DynamicCodeError) as raised:
            block_class(definition)().run()

        # then
        assert "child 1" in str(raised.value) and "['a', 'b']" in str(raised.value)

    def test_v2_native_expanded_batches_pass_through(self) -> None:
        # given
        definition = dynamic_block(
            "NativeExpand",
            inputs={},
            outputs={"a": []},
            output_dimensionality_offset=1,
            run="def run(self):\n    return {'a': Batch.of([1, 2])}\n",
        )

        # when
        result = block_class(definition)().run()

        # then
        assert result == {"a": Batch.of([1, 2])}

    def test_output_offset_minus_one_collapses_each_parent_group(self) -> None:
        # given
        total = dynamic_block(
            "Total",
            inputs={"value": dynamic_input("step_output")},
            outputs={"total": []},
            output_dimensionality_offset=-1,
            init=COUNTER_INIT,
            run=(
                "def run(self, value):\n"
                "    self._init_results['calls'] += 1\n"
                "    return {'total': (sum(value), list(value.indices))}\n"
            ),
        )
        document = workflow(
            dynamic=[EXPAND, total],
            steps=[
                {"type": "Expand", "name": "expand", "value": "$inputs.values"},
                {"type": "Total", "name": "total", "value": "$steps.expand.child"},
            ],
            outputs={"total": "$steps.total.total"},
        )
        plan = compile_workflow(document, catalogue=CATALOGUE, options=LOCAL)

        # when
        session = plan.create_session()
        rows = session.run({"values": [2, 1]}).rows()

        # then
        step = plan.step(("total",))
        assert step.bindings[0].mode == "group"
        assert step.invocation_layout.axis_ids == ("inputs",)
        assert rows == [
            {"total": (41, [(0, 0), (0, 1)])},
            {"total": (10, [(1, 0)])},
        ]
        assert session.instances[("total",)]._init_results["calls"] == 2

    def test_scalar_bound_to_a_reducer_input_arrives_as_a_one_element_group(
        self,
    ) -> None:
        # given: V2 difference DYN-D5; V1 passes this generic input as a scalar
        reducer = dynamic_block(
            "Reduce",
            inputs={
                "data": dynamic_input("step_output"),
                "extra": dynamic_input("generic"),
            },
            outputs={"out": []},
            output_dimensionality_offset=-1,
            run="def run(self, data, extra):\n    return {'out': (list(data), extra)}\n",
        )
        document = workflow(
            dynamic=[EXPAND, reducer],
            inputs=[
                {"type": "WorkflowBatchInput", "name": "values", "kind": ["integer"]},
                {"type": "WorkflowParameter", "name": "extra"},
            ],
            steps=[
                {"type": "Expand", "name": "expand", "value": "$inputs.values"},
                {
                    "type": "Reduce",
                    "name": "reduce",
                    "data": "$steps.expand.child",
                    "extra": "$inputs.extra",
                },
            ],
            outputs={"out": "$steps.reduce.out"},
        )
        plan = compile_workflow(document, catalogue=CATALOGUE, options=LOCAL)

        # when
        rows = plan.create_session().run({"values": [1], "extra": "x"}).rows()

        # then
        assert plan.step(("reduce",)).binding_for("extra").mode == "constant_group"
        [(data, extra)] = [row["out"] for row in rows]
        assert data == [10] and isinstance(extra, Batch) and list(extra) == ["x"]

    def test_reference_input_and_offset_input_match_the_measured_v1_call(self) -> None:
        # given: V1 case ragged_groups_vectorized (tasks/m1r-batch-call-check):
        # one call, parent Batch([2, 1]) and value Batch([Batch([20, 21]), Batch([10])])
        stitch = dynamic_block(
            "Stitch",
            inputs={
                "parent": dynamic_input(
                    "step_output", "input_parameter", is_dimensionality_reference=True
                ),
                "value": dynamic_input("step_output", dimensionality_offset=1),
            },
            outputs={"result": []},
            batch_oriented_parameters=["parent", "value"],
            init=COUNTER_INIT,
            run=(
                "def run(self, parent, value):\n"
                "    self._init_results['calls'] += 1\n"
                "    calls = self._init_results['calls']\n"
                "    return [\n"
                "        {'result': {'parent': p, 'children': list(v), 'call': calls}}\n"
                "        for p, v in zip(parent, value)\n"
                "    ]\n"
            ),
        )
        document = workflow(
            dynamic=[EXPAND, stitch],
            steps=[
                {"type": "Expand", "name": "expand", "value": "$inputs.values"},
                {
                    "type": "Stitch",
                    "name": "stitch",
                    "parent": "$inputs.values",
                    "value": "$steps.expand.child",
                },
            ],
            outputs={"result": "$steps.stitch.result"},
        )
        plan = compile_workflow(document, catalogue=CATALOGUE, options=LOCAL)

        # when
        rows = plan.create_session().run({"values": [2, 1]}).rows()

        # then
        assert [row["result"] for row in rows] == [
            {"parent": 2, "children": [20, 21], "call": 1},
            {"parent": 1, "children": [10], "call": 1},
        ]

    def test_empty_accepting_reducer_also_runs_for_fully_filtered_groups(self) -> None:
        # given
        above = dynamic_block(
            "Above",
            inputs={"value": dynamic_input("step_output")},
            outputs={"above": ["boolean"]},
            run="def run(self, value):\n    return {'above': value > 25}\n",
        )
        keep = dynamic_block(
            "Keep",
            inputs={"value": dynamic_input("step_output")},
            outputs={"kept": []},
            run="def run(self, value):\n    return {'kept': value}\n",
        )

        def reducer(block_type: str, accepts_empty: bool) -> Dict[str, Any]:
            return dynamic_block(
                block_type,
                inputs={"value": dynamic_input("step_output")},
                outputs={"kept": []},
                output_dimensionality_offset=-1,
                accepts_empty_values=accepts_empty,
                run="def run(self, value):\n    return {'kept': list(value)}\n",
            )

        document = workflow(
            dynamic=[
                EXPAND,
                above,
                keep,
                reducer("Strict", False),
                reducer("Lenient", True),
            ],
            steps=[
                {"type": "Expand", "name": "expand", "value": "$inputs.values"},
                {"type": "Above", "name": "above", "value": "$steps.expand.child"},
                {
                    "type": "v2/continue_if",
                    "name": "gate",
                    "condition": "$steps.above.above",
                    "next_steps": ["$steps.keep"],
                },
                {"type": "Keep", "name": "keep", "value": "$steps.expand.child"},
                {"type": "Strict", "name": "strict", "value": "$steps.keep.kept"},
                {"type": "Lenient", "name": "lenient", "value": "$steps.keep.kept"},
            ],
            outputs={"strict": "$steps.strict.kept", "lenient": "$steps.lenient.kept"},
        )
        plan = compile_workflow(document, catalogue=CATALOGUE, options=LOCAL)

        # when
        rows = plan.create_session().run({"values": [2, 3]}).rows()

        # then: children [[20, 21], [30, 31, 32]]; the first group is fully filtered
        assert rows == [
            {"strict": None, "lenient": []},
            {"strict": [30, 31, 32], "lenient": [30, 31, 32]},
        ]

    def test_output_free_side_effect_block_runs_per_element(
        self, tmp_path: Path
    ) -> None:
        # given
        log = tmp_path / "log.txt"
        definition = dynamic_block(
            "Log",
            inputs={
                "value": dynamic_input("step_output", "input_parameter"),
                "path": dynamic_input(value_types=["string"]),
            },
            outputs={},
            run=(
                "def run(self, value, path):\n"
                "    with open(path, 'a') as log:\n"
                "        log.write(f'{value}\\n')\n"
                "    return {}\n"
            ),
        )
        document = workflow(
            dynamic=[definition],
            steps=[
                {
                    "type": "Log",
                    "name": "log",
                    "value": "$inputs.values",
                    "path": str(log),
                }
            ],
            outputs={},
        )
        plan = compile_workflow(document, catalogue=CATALOGUE, options=LOCAL)

        # when
        plan.create_session().run({"values": [5, 6, 7]})

        # then
        assert log.read_text().splitlines() == ["5", "6", "7"]


class TestNestedDefinitionsThroughTheCompiler:
    def test_nested_definitions_compile_and_conflicting_duplicates_keep_the_root(
        self,
    ) -> None:
        # given
        def scale(block_type: str, factor: int) -> Dict[str, Any]:
            return dynamic_block(
                block_type,
                inputs={"value": dynamic_input("step_output", "input_parameter")},
                outputs={"out": []},
                run=f"def run(self, value):\n    return {{'out': value * {factor}}}\n",
            )

        child = {
            "version": "2.0",
            "inputs": [{"type": "WorkflowParameter", "name": "x"}],
            "dynamic_blocks_definitions": [scale("Double", 100), scale("Triple", 3)],
            "steps": [
                {"type": "Double", "name": "double", "value": "$inputs.x"},
                {"type": "Triple", "name": "triple", "value": "$steps.double.out"},
            ],
            "outputs": [
                {"type": "JsonField", "name": "y", "selector": "$steps.triple.out"}
            ],
        }
        document = workflow(
            dynamic=[scale("Double", 2)],
            steps=[
                {"type": "Double", "name": "double", "value": "$inputs.values"},
                {
                    "type": "roboflow_core/inner_workflow@v1",
                    "name": "child",
                    "workflow_definition": child,
                    "parameter_bindings": {"x": "$steps.double.out"},
                },
            ],
            outputs={"y": "$steps.child.y"},
        )

        # when
        plan = compile_workflow(document, catalogue=CATALOGUE, options=LOCAL)
        rows = plan.create_session().run({"values": [1, 2]}).rows()

        # then: root Double (x2) is used inside the child too: 1*2*2*3, 2*2*2*3
        assert rows == [{"y": 12}, {"y": 24}]
        assert [step.path for step in plan.steps] == [
            ("double",),
            ("child", "double"),
            ("child", "triple"),
        ]
        assert len(plan.warnings) == 1
        assert "redefines the block type" in plan.warnings[0]


SHARED_STATE_KEY = f"{BLOCK_SOURCE}.shared_state"
POLICY_KEY = f"{BLOCK_SOURCE}.representation_policy"

TALLY = dynamic_block(
    "Tally",
    inputs={"value": dynamic_input("step_output", "input_parameter")},
    outputs={"seen": []},
    run=(
        "def run(self, value):\n"
        "    globals['count'] = globals.get('count', 0) + 1\n"
        "    return {'seen': (value, globals['count'], self.shared_state is globals)}\n"
    ),
)

READ_TALLY = dynamic_block(
    "ReadTally",
    inputs={"value": dynamic_input("step_output", "input_parameter")},
    outputs={"count": []},
    run="def run(self, value):\n    return {'count': globals.get('count')}\n",
)


def shared_state_workflow() -> Dict[str, Any]:
    child = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "x"}],
        "dynamic_blocks_definitions": [READ_TALLY],
        "steps": [
            {"type": "Tally", "name": "tally", "value": "$inputs.x"},
            {"type": "ReadTally", "name": "read", "value": "$steps.tally.seen"},
        ],
        "outputs": [
            {"type": "JsonField", "name": "count", "selector": "$steps.read.count"}
        ],
    }
    document = workflow(
        dynamic=[TALLY],
        inputs=[{"type": "WorkflowParameter", "name": "x"}],
        steps=[
            {"type": "Tally", "name": "tally", "value": "$inputs.x"},
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": "child",
                "workflow_definition": child,
                "parameter_bindings": {"x": "$inputs.x"},
            },
        ],
        outputs={"root": "$steps.tally.seen", "child": "$steps.child.count"},
    )

    return document


class TestSharedState:
    def test_direct_and_nested_dynamic_steps_share_one_mapping_per_session(
        self,
    ) -> None:
        # given
        plan = compile_workflow(
            shared_state_workflow(), catalogue=CATALOGUE, options=LOCAL
        )

        # when
        session = plan.create_session()
        first = session.run({"x": "a"}).rows()
        second = session.run({"x": "b"}).rows()
        fresh = plan.create_session().run({"x": "c"}).rows()

        # then: both Tally steps and the nested reader see one counter
        assert first == [{"root": ("a", 1, True), "child": 2}]
        assert second == [{"root": ("b", 3, True), "child": 4}]
        assert fresh == [{"root": ("c", 1, True), "child": 2}], "sessions are isolated"
        states = {id(instance.shared_state) for instance in session.instances.values()}
        assert len(states) == 1
        chosen = session.resources[("child", "read")]["shared_state"]
        assert chosen.source == f"catalogue:{SHARED_STATE_KEY}"

    def test_caller_mapping_is_shared_across_sessions_by_identity(self) -> None:
        # given
        plan = compile_workflow(
            shared_state_workflow(), catalogue=CATALOGUE, options=LOCAL
        )
        mapping: Dict[str, Any] = {"count": 10}

        # when
        rows = [
            plan.create_session(resources={SHARED_STATE_KEY: mapping})
            .run({"x": name})
            .rows()
            for name in ("a", "b")
        ]

        # then
        assert rows == [
            [{"root": ("a", 11, True), "child": 12}],
            [{"root": ("b", 13, True), "child": 14}],
        ]
        assert mapping == {"count": 14}

    def test_per_step_self_and_init_state_stay_separate_from_shared_state(
        self,
    ) -> None:
        # given
        definition = dynamic_block(
            "Both",
            inputs={"value": dynamic_input("input_parameter")},
            outputs={"mine": [], "shared": []},
            init=COUNTER_INIT,
            run=(
                "def run(self, value):\n"
                "    self._init_results['calls'] += 1\n"
                "    globals['calls'] = globals.get('calls', 0) + 1\n"
                "    return {'mine': self._init_results['calls'], 'shared': globals['calls']}\n"
            ),
        )
        document = workflow(
            dynamic=[definition],
            inputs=[{"type": "WorkflowParameter", "name": "x"}],
            steps=[
                {"type": "Both", "name": "first", "value": "$inputs.x"},
                {"type": "Both", "name": "second", "value": "$inputs.x"},
            ],
            outputs={"first": "$steps.first.*", "second": "$steps.second.*"},
        )
        session = compile_workflow(
            document, catalogue=CATALOGUE, options=LOCAL
        ).create_session()

        # when
        rows = session.run({"x": 1}).rows()

        # then
        assert rows == [
            {"first": {"mine": 1, "shared": 1}, "second": {"mine": 1, "shared": 2}}
        ]

    def test_shared_state_resource_must_be_a_mutable_mapping(self) -> None:
        # given
        plan = compile_workflow(
            shared_state_workflow(), catalogue=CATALOGUE, options=LOCAL
        )

        # when
        with pytest.raises(ResourceError) as raised:
            plan.create_session(resources={SHARED_STATE_KEY: ("not", "a", "mapping")})

        # then
        assert isinstance(raised.value.__cause__, TypeError)
        assert "mutable mapping" in str(raised.value.__cause__)


CONTEXT_PROBE = dynamic_block(
    "ContextProbe",
    inputs={"value": dynamic_input("step_output", "input_parameter")},
    outputs={"context": []},
    init="def init():\n    return {}\n",
    run=(
        "def run(self, value):\n"
        "    return {'context': self.get_workflow_context()}\n"
    ),
)


class TestWorkflowContext:
    def test_submitted_code_reads_the_context_of_its_own_call(self) -> None:
        # given
        child = {
            "version": "2.0",
            "inputs": [
                {"type": "WorkflowBatchInput", "name": "x", "kind": ["integer"]}
            ],
            "steps": [{"type": "ContextProbe", "name": "probe", "value": "$inputs.x"}],
            "outputs": [
                {
                    "type": "JsonField",
                    "name": "context",
                    "selector": "$steps.probe.context",
                }
            ],
        }
        document = workflow(
            dynamic=[CONTEXT_PROBE],
            steps=[
                {
                    "type": "roboflow_core/inner_workflow@v1",
                    "name": "child",
                    "workflow_definition": child,
                    "parameter_bindings": {"x": "$inputs.values"},
                }
            ],
            outputs={"context": "$steps.child.context"},
        )
        session = compile_workflow(
            document, catalogue=CATALOGUE, options=LOCAL
        ).create_session()

        # when
        result = session.run({"values": [5, 6]})

        # then
        contexts = [row["context"] for row in result.rows()]
        assert contexts[1] == {
            "step_name": "probe",
            "step_selector": "$steps.child/probe",
            "block_type": "ContextProbe",
            "workflow_execution_id": result.run_id,
            "session_id": session.session_id,
            "step_path": ["child", "probe"],
            "indices": [[1]],
        }

    def test_construction_has_a_context_without_a_run_and_nothing_leaks(
        self,
    ) -> None:
        # given: module-top code runs while the session constructs the step
        definition = dynamic_block(
            "WhenBuilt",
            inputs={"value": dynamic_input("input_parameter")},
            outputs={"built": [], "now": []},
            imports=[
                "from roboflow_workflows.execution_engine.v2.context import "
                "get_execution_context"
            ],
            run=(
                "BUILT = get_execution_context()\n"
                "def run(self, value):\n"
                "    if value == 'fail':\n"
                "        raise RuntimeError('boom')\n"
                "    now = self.get_workflow_context()\n"
                "    return {\n"
                "        'built': (BUILT.run_id, BUILT.session_id, BUILT.indices),\n"
                "        'now': (now['workflow_execution_id'], now['session_id']),\n"
                "    }\n"
            ),
        )
        document = workflow(
            dynamic=[definition],
            inputs=[{"type": "WorkflowParameter", "name": "x"}],
            steps=[{"type": "WhenBuilt", "name": "step", "value": "$inputs.x"}],
            outputs={"built": "$steps.step.built", "now": "$steps.step.now"},
        )
        session = compile_workflow(
            document, catalogue=CATALOGUE, options=LOCAL
        ).create_session()

        # when
        result = session.run({"x": "ok"})
        with pytest.raises(StepExecutionError):
            session.run({"x": "fail"})

        # then
        [row] = result.rows()
        assert row["built"] == (None, session.session_id, ())
        assert row["now"] == (result.run_id, session.session_id)
        with pytest.raises(NoExecutionContextError):
            session.instances[("step",)].get_workflow_context()

    def test_workflow_context_reads_an_explicitly_entered_context(self) -> None:
        # given
        block = block_class(CONTEXT_PROBE)()
        context = ExecutionContext(("a", "b"), "ContextProbe", "s-1", "r-1", [[2, 0]])

        # when
        with use_execution_context(context):
            seen = block.get_workflow_context()

        # then
        assert seen["step_selector"] == "$steps.a/b" and seen["indices"] == [[2, 0]]


class RecordingPolicy(RepresentationPolicy):
    """Lightweight carrier bridge: engine ``Carrier`` <-> legacy ``list``."""

    def __init__(self, events: List[Any], *, native: bool = False):
        self.events = events
        self.native = native

    def check_compatibility(self, manifest) -> None:
        self.events.append(("check", manifest.tensor_compatibility.value))
        if not self.native:
            super().check_compatibility(manifest)

    def prepare_inputs(self, inputs, *, manifest):
        self.events.append(("inputs", sorted(inputs)))
        return {name: list(value.items) for name, value in inputs.items()}

    def prepare_outputs(self, result, *, manifest):
        self.events.append(("outputs", sorted(result)))
        return {name: Carrier(value) for name, value in result.items()}


class Carrier:
    """Engine-side payload whose legacy representation is a list."""

    def __init__(self, items):
        self.items = tuple(items)

    def __eq__(self, other) -> bool:
        return isinstance(other, Carrier) and other.items == self.items

    def __repr__(self) -> str:
        return f"Carrier{self.items}"


def representation_block(
    block_type: str, *, tensor_compatibility: str = "legacy_compatibility"
) -> Dict[str, Any]:
    definition = dynamic_block(
        block_type,
        inputs={"values": dynamic_input("input_parameter")},
        outputs={"values": []},
        tensor_compatibility=tensor_compatibility,
        run=(
            "globals['events'].append(('module', None))\n"
            "def run(self, values):\n"
            "    globals['events'].append(('run', type(values).__name__))\n"
            "    return {'values': values + [len(values)]}\n"
        ),
        init="def init():\n    globals['events'].append(('init', None))\n    return {}\n",
    )

    return definition


def representation_plan(definition: Dict[str, Any]):
    document = workflow(
        dynamic=[definition],
        inputs=[{"type": "WorkflowParameter", "name": "values"}],
        steps=[
            {
                "type": definition["manifest"]["block_type"],
                "name": "adapt",
                "values": "$inputs.values",
            }
        ],
        outputs={"values": "$steps.adapt.values"},
    )
    plan = compile_workflow(document, catalogue=CATALOGUE, options=LOCAL)

    return plan


class TestRepresentationPolicy:
    def test_policy_hooks_surround_imports_init_and_each_call_in_order(self) -> None:
        # given
        events: List[Any] = []
        plan = representation_plan(representation_block("Adapt"))

        # when
        session = plan.create_session(
            resources={
                SHARED_STATE_KEY: {"events": events},
                POLICY_KEY: RecordingPolicy(events),
            }
        )
        rows = session.run({"values": Carrier([1, 2])}).rows()

        # then
        assert events == [
            ("check", "legacy_compatibility"),
            ("module", None),
            ("init", None),
            ("inputs", ["values"]),
            ("run", "list"),
            ("outputs", ["values"]),
        ]
        assert rows == [{"values": Carrier([1, 2, 2])}]

    def test_default_legacy_policy_keeps_payload_identity(self) -> None:
        # given
        definition = dynamic_block(
            "Identity",
            inputs={"payload": dynamic_input("input_parameter")},
            outputs={"payload": [], "same": []},
            run=(
                "def run(self, payload):\n"
                "    payload.append('touched')\n"
                "    return {'payload': payload, 'same': True}\n"
            ),
        )
        document = workflow(
            dynamic=[definition],
            inputs=[{"type": "WorkflowParameter", "name": "payload"}],
            steps=[{"type": "Identity", "name": "same", "payload": "$inputs.payload"}],
            outputs={"payload": "$steps.same.payload"},
        )
        session = compile_workflow(
            document, catalogue=CATALOGUE, options=LOCAL
        ).create_session()
        payload: List[str] = []

        # when
        rows = session.run({"payload": payload}).rows()

        # then
        assert rows[0]["payload"] is payload and payload == ["touched"]
        assert session.resources[("same",)]["representation_policy"].source == "default"
        assert (
            session.instances[("same",)].representation_policy is LEGACY_REPRESENTATION
        )

    def test_tensor_native_is_rejected_before_imports_or_init_by_default(self) -> None:
        # given
        events: List[Any] = []
        plan = representation_plan(
            representation_block("Native", tensor_compatibility="tensor_native")
        )

        # when
        with pytest.raises(ResourceError) as raised:
            plan.create_session(resources={SHARED_STATE_KEY: {"events": events}})

        # then
        assert isinstance(raised.value.__cause__, RepresentationError)
        assert "tensor_native" in str(raised.value.__cause__)
        assert POLICY_KEY in str(raised.value.__cause__)
        assert events == [], "no import, module code or init ran"

    def test_capable_policy_runs_a_tensor_native_block(self) -> None:
        # given
        events: List[Any] = []
        plan = representation_plan(
            representation_block("Native", tensor_compatibility="tensor_native")
        )

        # when
        rows = (
            plan.create_session(
                resources={
                    SHARED_STATE_KEY: {"events": events},
                    POLICY_KEY: RecordingPolicy(events, native=True),
                }
            )
            .run({"values": Carrier([7])})
            .rows()
        )

        # then
        assert events[0] == ("check", "tensor_native")
        assert rows == [{"values": Carrier([7, 1])}]

    def test_scoped_policy_resource_beats_unscoped_one(self) -> None:
        # given
        scoped_events: List[Any] = []
        unscoped_events: List[Any] = []
        plan = representation_plan(representation_block("Adapt"))

        # when
        session = plan.create_session(
            resources={
                SHARED_STATE_KEY: {"events": []},
                POLICY_KEY: RecordingPolicy(scoped_events),
                "representation_policy": RecordingPolicy(unscoped_events),
            }
        )
        unscoped_only = plan.create_session(
            resources={
                SHARED_STATE_KEY: {"events": []},
                "representation_policy": RecordingPolicy(unscoped_events),
            }
        )

        # then
        assert session.resources[("adapt",)]["representation_policy"].source == (
            f"provided:{POLICY_KEY}"
        )
        assert scoped_events and unscoped_events == [("check", "legacy_compatibility")]
        assert unscoped_only.resources[("adapt",)]["representation_policy"].source == (
            "provided:representation_policy"
        )

    @pytest.mark.parametrize("hook", ["prepare_inputs", "prepare_outputs"])
    def test_policy_failures_name_the_policy_hook_and_keep_the_cause(
        self, hook: str
    ) -> None:
        # given
        class Failing(RecordingPolicy):
            def prepare_inputs(self, inputs, *, manifest):
                if hook == "prepare_inputs":
                    raise ValueError("cannot convert input")
                return super().prepare_inputs(inputs, manifest=manifest)

            def prepare_outputs(self, result, *, manifest):
                raise ValueError("cannot convert output")

        events: List[Any] = []
        session = representation_plan(representation_block("Adapt")).create_session(
            resources={
                SHARED_STATE_KEY: {"events": events},
                POLICY_KEY: Failing(events),
            }
        )

        # when
        with pytest.raises(StepExecutionError) as raised:
            session.run({"values": Carrier([1])})

        # then
        error = raised.value.__cause__
        assert isinstance(error, RepresentationError)
        assert f"Failing.{hook} failed: ValueError" in str(error)
        assert isinstance(error.__cause__, ValueError)
        ran = [name for name, _ in events if name == "run"]
        assert ran == ([] if hook == "prepare_inputs" else ["run"])

    def test_objects_without_the_policy_methods_are_rejected(self) -> None:
        # given
        plan = representation_plan(representation_block("Adapt"))

        # when
        with pytest.raises(ResourceError) as raised:
            plan.create_session(
                resources={SHARED_STATE_KEY: {"events": []}, POLICY_KEY: object()}
            )

        # then
        assert "lacks ['check_compatibility', 'prepare_inputs', 'prepare_outputs']" in (
            str(raised.value.__cause__)
        )

    def test_small_cpu_tensor_policy_bridges_lists_and_arrays(self) -> None:
        # given: a host policy for tensor-native code; NumPy lives in the test only
        np = pytest.importorskip("numpy")

        class ArrayPolicy(RepresentationPolicy):
            def check_compatibility(self, manifest) -> None:
                return None

            def prepare_inputs(self, inputs, *, manifest):
                return {
                    name: np.asarray(value, dtype=np.float32)
                    for name, value in inputs.items()
                }

            def prepare_outputs(self, result, *, manifest):
                return {name: value.tolist() for name, value in result.items()}

        definition = dynamic_block(
            "Normalize",
            inputs={"values": dynamic_input("step_output", "input_parameter")},
            outputs={"values": []},
            tensor_compatibility="tensor_native",
            run=(
                "def run(self, values):\n"
                "    assert type(values).__name__ == 'ndarray'\n"
                "    return {'values': values / values.sum()}\n"
            ),
        )
        document = workflow(
            dynamic=[definition],
            inputs=[{"type": "WorkflowBatchInput", "name": "values"}],
            steps=[{"type": "Normalize", "name": "norm", "values": "$inputs.values"}],
            outputs={"values": "$steps.norm.values"},
        )
        plan = compile_workflow(document, catalogue=CATALOGUE, options=LOCAL)

        # when
        rows = (
            plan.create_session(resources={POLICY_KEY: ArrayPolicy()})
            .run({"values": [[1.0, 3.0], [2.0, 2.0]]})
            .rows()
        )

        # then
        assert rows == [{"values": [0.25, 0.75]}, {"values": [0.5, 0.5]}]

    def test_inspection_calls_no_policy_and_creates_no_shared_state(self) -> None:
        # given
        calls: List[Any] = []

        class Watching(RecordingPolicy):
            def check_compatibility(self, manifest) -> None:
                calls.append("check")

        policy = Watching(calls)
        definition = representation_block(
            "Native", tensor_compatibility="tensor_native"
        )

        # when
        plan = representation_plan(definition)
        plan.describe()
        spec = plan.step(("adapt",)).spec

        # then
        assert calls == [] and policy.events == []
        assert [resource.name for resource in spec.resources] == [
            "shared_state",
            "representation_policy",
        ]
        restriction_codes = {
            item.code for item in spec.block_class.discover_restrictions(None).items
        }
        assert "custom_python_tensor_native_requires_policy" in restriction_codes
        assert spec.metadata["dynamic"]["tensor_compatibility"] == "tensor_native"
