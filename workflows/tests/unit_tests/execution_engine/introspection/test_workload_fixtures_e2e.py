"""End-to-end fixtures for workload introspection, against the REAL registry.

Four things are checked here that no single-owner suite can check:

1. the branched reference example (image -> detection -> crop -> detection on
   crops -> custom Python -> classification) that the published JSON artifacts
   and the HTTP E2E test both use, so the documented counts have exactly one
   source of truth in the package;
2. the declaration census across the whole registry in FRESH subprocesses, for
   tensor off/on and for core / core+enterprise, because the Workflows
   configuration is process-level and frozen at the first import - an
   in-process test can only ever see one of those registries;
3. numpy/tensor parity of the declared VALUES (not only of the hook source
   text), for every block type in that census;
4. that introspecting a definition and compiling the same definition for
   execution do not contaminate each other, in both orders.

Nothing here loads a model, initialises a block or runs a workflow.
"""

import json
import os
import subprocess
import sys
from typing import Dict, List, Tuple
from unittest.mock import MagicMock

import pytest
from roboflow_workflows.execution_engine.introspection.workload import (
    describe_workflow_workload,
)
from roboflow_workflows.execution_engine.introspection.workload_entities import (
    WorkflowIntrospection,
)
from roboflow_workflows.execution_engine.v1.compiler.core import (
    COMPILATION_CACHE,
    compile_workflow,
)
from roboflow_workflows.prototypes.block import WorkflowBlockManifest

from tests.unit_tests.execution_engine.dynamic_blocs._workspace_resolver_stub import (
    StubResolver,
)

OBJECT_DETECTION_MODEL = "roboflow_core/roboflow_object_detection_model@v3"
CLASSIFICATION_MODEL = "roboflow_core/roboflow_classification_model@v2"
DYNAMIC_CROP = "roboflow_core/dynamic_crop@v1"

DECLARATION_HOOKS = (
    "discover_work_operations",
    "discover_portable_restrictions",
    "discover_dependent_resources",
)
# The single registered manifest allowed to keep the base `None` for dependent
# resources: its dispatched child may pull any model or project, so `[]` would
# be a lie (see the coverage test in tests/unit_tests/core_steps and the
# builder, which forces remote-dispatch children to incomplete anyway).
RESOURCES_INTENTIONALLY_UNKNOWN = {"roboflow_core/inner_workflow@v1"}
ENTERPRISE_LOADER = "roboflow_workflows.enterprise_blocks.loader"

CENSUS_SENTINEL = "<<<CENSUS>>>"
# Enumerates the loaded registry in a child process. The standalone package
# ignores ENABLE_TENSOR_DATA_REPRESENTATION (`default_configuration()` never
# reads the environment), so the configuration is installed explicitly before
# the first registry import - exactly what workflows/tests/conftest.py does.
CENSUS_CHILD = r"""
import dataclasses, json, sys

from roboflow_workflows import configuration as workflows_configuration

TENSOR_MODE = sys.argv[1] == "on"
_BASE = workflows_configuration.default_configuration()
workflows_configuration.reset_configuration()
workflows_configuration.configure_process(
    dataclasses.replace(
        _BASE,
        tensor=dataclasses.replace(
            _BASE.tensor,
            representation_enabled=TENSOR_MODE,
            image_tensor_device=workflows_configuration.resolve_image_tensor_device(
                TENSOR_MODE
            ),
        ),
        # the flag-gated 3D block is part of the registry we must audit
        models=dataclasses.replace(_BASE.models, sam3_3d_objects_enabled=True),
    )
)

from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    RestrictionMetadata,
    WorkOperation,
    normalize_declaration,
)
from roboflow_workflows.execution_engine.introspection.blocks_loader import (
    get_manifest_type_identifiers,
    load_workflow_blocks,
)


def declared_value(instance, hook, wrapper):
    # The VALUE a manifest declares, normalised exactly as the builder
    # normalises it. A hook that raises (or reads a field this placeholder
    # instance does not carry) records its error instead - that, too, must
    # match between the numpy and the tensor sibling.
    try:
        raw = getattr(instance, hook)()
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}
    if hook == "discover_dependent_resources":
        if raw is None:
            return {"value": None}
        if isinstance(raw, Discovery):
            return {
                "value": {
                    "complete": raw.complete,
                    "unknown_reasons": list(raw.unknown_reasons),
                    "items": [item.to_dict() for item in raw.items],
                }
            }
        return {"value": [item.to_dict() for item in raw]}
    normalised = normalize_declaration(raw, "unknown:step")
    rewrapped = wrapper(
        items=list(normalised.items),
        complete=normalised.complete,
        unknown_reasons=list(normalised.unknown_reasons),
    )
    return {"value": rewrapped.model_dump(mode="json")}


HOOKS = json.loads(sys.argv[2])
WRAPPERS = {
    "discover_work_operations": Discovery[WorkOperation],
    "discover_portable_restrictions": Discovery[RestrictionMetadata],
    "discover_dependent_resources": None,
}
rows = []
for block in load_workflow_blocks():
    manifest = block.manifest_class
    identifiers = get_manifest_type_identifiers(
        block_schema=manifest.model_json_schema(),
        block_source=block.block_source,
        block_identifier=block.identifier,
    )
    placeholder = manifest.model_construct(name="step", type=identifiers[0])
    rows.append(
        {
            "block_type": identifiers[0],
            "block_source": block.block_source,
            "manifest_module": manifest.__module__,
            "declared": {hook: hook in vars(manifest) for hook in HOOKS},
            "values": {
                hook: declared_value(placeholder, hook, WRAPPERS[hook])
                for hook in HOOKS
            },
        }
    )
print("<<<CENSUS>>>" + json.dumps(rows))
"""


def _image_input(name: str = "image") -> dict:
    return {"type": "WorkflowImage", "name": name}


def branched_definition() -> dict:
    """The reference example: two model branches sharing one model id, a crop
    that raises dimensionality and a custom Python block on the crops."""
    return {
        "version": "1.0",
        "inputs": [_image_input()],
        "steps": [
            {
                "type": OBJECT_DETECTION_MODEL,
                "name": "detection",
                "images": "$inputs.image",
                "model_id": "my-project/3",
            },
            {
                "type": DYNAMIC_CROP,
                "name": "crop",
                "images": "$inputs.image",
                "predictions": "$steps.detection.predictions",
            },
            {
                "type": OBJECT_DETECTION_MODEL,
                "name": "crop_detection",
                "images": "$steps.crop.crops",
                "model_id": "my-project/3",
            },
            {
                "type": "CountDetections",
                "name": "counter",
                "predictions": "$steps.crop_detection.predictions",
            },
            {
                "type": CLASSIFICATION_MODEL,
                "name": "classification",
                "images": "$steps.crop.crops",
                "model_id": "my-other-project/1",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "crop_detections",
                "selector": "$steps.crop_detection.predictions",
            },
            {
                "type": "JsonField",
                "name": "counts",
                "selector": "$steps.counter.count",
            },
            {
                "type": "JsonField",
                "name": "classes",
                "selector": "$steps.classification.predictions",
            },
        ],
        "dynamic_blocks_definitions": [
            {
                "type": "DynamicBlockDefinition",
                "manifest": {
                    "type": "ManifestDescription",
                    "block_type": "CountDetections",
                    "inputs": {
                        "predictions": {
                            "type": "DynamicInputDefinition",
                            "selector_types": ["step_output"],
                            "selector_data_kind": {
                                "step_output": ["object_detection_prediction"]
                            },
                        }
                    },
                    "outputs": {
                        "count": {
                            "type": "DynamicOutputDefinition",
                            "kind": ["integer"],
                        }
                    },
                },
                "code": {
                    "type": "PythonCode",
                    "run_function_code": (
                        "def run(self, predictions):\n"
                        "    raise RuntimeError('must never run during inspection')\n"
                    ),
                },
            }
        ],
    }


_CENSUS_CACHE: Dict[Tuple[str, str], List[dict]] = {}


def _run_census(tensor_mode: str, plugins: str) -> List[dict]:
    # A registry census is deterministic for a given configuration, and each one
    # costs a full interpreter start plus a registry import; cache them so the
    # file spends four subprocesses in total, not one per assertion.
    cached = _CENSUS_CACHE.get((tensor_mode, plugins))
    if cached is not None:
        return cached
    environment = dict(os.environ)
    if plugins:
        environment["WORKFLOWS_PLUGINS"] = plugins
    else:
        environment.pop("WORKFLOWS_PLUGINS", None)
    # the child installs the configuration itself; the variable must not race it
    environment.pop("ENABLE_TENSOR_DATA_REPRESENTATION", None)
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            CENSUS_CHILD,
            tensor_mode,
            json.dumps(DECLARATION_HOOKS),
        ],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, (
        f"census child failed (tensor={tensor_mode}, plugins={plugins or '-'})\n"
        f"{process.stdout[-2000:]}\n{process.stderr[-4000:]}"
    )
    payload = [
        line for line in process.stdout.splitlines() if line.startswith(CENSUS_SENTINEL)
    ]
    assert len(payload) == 1, process.stdout[-2000:]
    rows = json.loads(payload[0][len(CENSUS_SENTINEL) :])
    _CENSUS_CACHE[(tensor_mode, plugins)] = rows
    return rows


def _missing(rows: List[dict], hook: str) -> List[str]:
    return sorted(row["block_type"] for row in rows if not row["declared"][hook])


# --------------------------------------------------------------------------
# the reference example every artifact and the HTTP E2E test share
# --------------------------------------------------------------------------


def test_branched_reference_example_has_the_documented_shape() -> None:
    # when
    result = describe_workflow_workload(branched_definition())

    # then - graph: 1 input + 5 steps + 3 outputs
    body = result.model_dump(mode="json")
    assert [(node["id"], node["kind"]) for node in body["nodes"]] == [
        ("$inputs.image", "input"),
        ("$steps.detection", "step"),
        ("$steps.crop", "step"),
        ("$steps.crop_detection", "step"),
        ("$steps.counter", "step"),
        ("$steps.classification", "step"),
        ("$outputs.crop_detections", "output"),
        ("$outputs.counts", "output"),
        ("$outputs.classes", "output"),
    ]
    assert {
        (edge["source"], edge["target"], edge["kind"]) for edge in body["edges"]
    } == {
        ("$inputs.image", "$steps.detection", "data"),
        ("$inputs.image", "$steps.crop", "data"),
        ("$steps.detection", "$steps.crop", "data"),
        ("$steps.crop", "$steps.crop_detection", "data"),
        ("$steps.crop", "$steps.classification", "data"),
        ("$steps.crop_detection", "$steps.counter", "data"),
        ("$steps.crop_detection", "$outputs.crop_detections", "data"),
        ("$steps.counter", "$outputs.counts", "data"),
        ("$steps.classification", "$outputs.classes", "data"),
    }

    # then - dimensionality: the crop raises depth, everything after it is 2
    dimensions = {
        step["node_id"]: (step["input_dimensionality"], step["output_dimensionality"])
        for step in body["steps"]
    }
    assert dimensions == {
        "$steps.detection": (1, 1),
        "$steps.crop": (1, 2),
        "$steps.crop_detection": (2, 2),
        "$steps.counter": (2, 2),
        "$steps.classification": (2, 2),
    }
    assert body["summary"]["steps_by_dimensionality"] == {"1": 2, "2": 3}
    assert sum(body["summary"]["steps_by_dimensionality"].values()) == len(
        body["steps"]
    )
    assert body["summary"]["max_dimensionality"] == 2

    # then - one inventory entry per (provider, model id), both referring steps kept
    models = {model["model_id"]: model for model in body["summary"]["models"]["items"]}
    assert sorted(models) == ["my-other-project/1", "my-project/3"]
    assert models["my-project/3"]["used_by_steps"] == [
        "$steps.crop_detection",
        "$steps.detection",
    ]
    assert models["my-other-project/1"]["used_by_steps"] == ["$steps.classification"]
    # no provider was supplied to the standalone call: nothing is claimed
    assert {model["metadata_status"] for model in models.values()} == {"unavailable"}
    assert all(model["metadata"] is None for model in models.values())

    # then - the only unknown in the whole response comes from the custom block
    custom_step = next(
        step for step in body["steps"] if step["node_id"] == "$steps.counter"
    )
    assert custom_step["operations"]["items"] == ["custom_python"]
    assert custom_step["operations"]["complete"] is False
    assert custom_step["operations"]["unknown_reasons"] == [
        "custom_python_internal_operations_unknown:$steps.counter"
    ]
    assert body["summary"]["models"]["unknown_reasons"] == [
        "step_resources_unknown:$steps.counter"
    ]
    assert body["summary"]["models"]["complete"] is False
    for step in body["steps"]:
        if step["node_id"] == "$steps.counter":
            continue
        assert step["resources"]["complete"] is True, step["node_id"]
        assert step["operations"]["complete"] is True, step["node_id"]
        assert step["restrictions"]["complete"] is True, step["node_id"]

    # then - the wire format re-validates
    assert WorkflowIntrospection.model_validate_json(result.model_dump_json()) == result


def test_branched_reference_example_is_stable_and_leaves_the_definition_untouched() -> (
    None
):
    # given
    definition = branched_definition()
    snapshot = json.dumps(definition, sort_keys=True)

    # when
    first = describe_workflow_workload(definition)
    second = describe_workflow_workload(branched_definition())

    # then
    assert first == second
    assert json.dumps(definition, sort_keys=True) == snapshot


def test_introspection_and_executable_compilation_do_not_contaminate_each_other() -> (
    None
):
    """Both orders: the executable compiler must not see a structural artefact,
    and introspection must not return (or populate) an executable graph."""
    # given - the executable path needs the host-provided init parameters; a
    # stub is enough because nothing is executed, only compiled
    definition = branched_definition()
    init_parameters = {
        "model_manager": MagicMock(),
        "api_key": "fake-key",
        "workspace_resolver": StubResolver(workspace="my-workspace"),
        "execution_observer": MagicMock(),
    }

    # when - introspect first, then compile for execution
    cache_before = dict(COMPILATION_CACHE._cache)
    before = describe_workflow_workload(definition)
    cache_after_introspection = dict(COMPILATION_CACHE._cache)
    compiled = compile_workflow(
        workflow_definition=definition, init_parameters=init_parameters
    )
    # then - introspection added nothing to the executable compilation cache
    # (the dict may already hold entries from earlier tests in this session)
    assert cache_after_introspection == cache_before
    assert compiled.steps, "the executable compiler must still build runnable steps"
    assert all(
        step.step is not None for step in compiled.steps.values()
    ), "every executable step must carry an initialised block"

    # when - introspect again after an executable compilation happened
    after = describe_workflow_workload(definition)

    # then - identical answer, and the executable graph carries real step bodies
    assert after == before
    assert set(compiled.steps) == {
        "detection",
        "crop",
        "crop_detection",
        "counter",
        "classification",
    }


# --------------------------------------------------------------------------
# registry-wide declaration census (fresh subprocess per configuration)
# --------------------------------------------------------------------------


def test_every_registered_manifest_declares_the_hooks_in_the_loaded_registry() -> None:
    """In-process guard for the registry this test session actually loaded."""
    # given
    from roboflow_workflows.execution_engine.introspection.blocks_loader import (
        get_manifest_type_identifiers,
        load_workflow_blocks,
    )

    # when
    rows = []
    for block in load_workflow_blocks():
        manifest = block.manifest_class
        identifiers = get_manifest_type_identifiers(
            block_schema=manifest.model_json_schema(),
            block_source=block.block_source,
            block_identifier=block.identifier,
        )
        rows.append(
            {
                "block_type": identifiers[0],
                "declared": {
                    hook: hook in vars(manifest) for hook in DECLARATION_HOOKS
                },
            }
        )

    # then - counts are derived from the registry, never hardcoded
    assert len(rows) > 200, f"registry looks truncated: {len(rows)} blocks"
    assert _missing(rows, "discover_work_operations") == []
    assert _missing(rows, "discover_portable_restrictions") == []
    assert set(_missing(rows, "discover_dependent_resources")) <= (
        RESOURCES_INTENTIONALLY_UNKNOWN
    )
    # the base class still answers "unknown" for a third-party plugin that
    # declares nothing - the census rule is about REGISTERED blocks only
    assert WorkflowBlockManifest.discover_work_operations is not None


@pytest.mark.parametrize("tensor_mode", ["off", "on"])
@pytest.mark.parametrize(
    "plugins, label", [("", "core"), (ENTERPRISE_LOADER, "core+enterprise")]
)
def test_registry_census_in_a_fresh_subprocess(
    tensor_mode: str, plugins: str, label: str
) -> None:
    # when
    rows = _run_census(tensor_mode=tensor_mode, plugins=plugins)

    # then
    assert len(rows) > 200, f"{label}/{tensor_mode}: only {len(rows)} blocks loaded"
    assert _missing(rows, "discover_work_operations") == []
    assert _missing(rows, "discover_portable_restrictions") == []
    assert set(_missing(rows, "discover_dependent_resources")) <= (
        RESOURCES_INTENTIONALLY_UNKNOWN
    )
    # the flag-gated 3D block is only registered when the configuration enables
    # it; the child enables it, so a census that misses it is not exhaustive
    assert "roboflow_core/segment_anything3_3d_objects@v1" in {
        row["block_type"] for row in rows
    }
    if plugins:
        # the enterprise loader reports `workflows_core` as its block source, so
        # the plugin is identified by the module the manifest really lives in
        assert any(
            ".enterprise_blocks." in row["manifest_module"] for row in rows
        ), "the enterprise plugin contributed no block"


def test_tensor_mode_does_not_change_the_registered_block_types() -> None:
    # when
    numpy_rows = _run_census(tensor_mode="off", plugins=ENTERPRISE_LOADER)
    tensor_rows = _run_census(tensor_mode="on", plugins=ENTERPRISE_LOADER)

    # then - same public registry, only the implementation module may differ
    numpy_types = {row["block_type"] for row in numpy_rows}
    tensor_types = {row["block_type"] for row in tensor_rows}
    assert numpy_types == tensor_types
    swapped = {
        row["block_type"]
        for row in tensor_rows
        if row["manifest_module"].endswith("_tensor")
    }
    assert swapped, "no manifest was swapped for its tensor sibling"
    numpy_declarations: Dict[str, dict] = {
        row["block_type"]: row["declared"] for row in numpy_rows
    }
    for row in tensor_rows:
        assert row["declared"] == numpy_declarations[row["block_type"]], row[
            "block_type"
        ]


def test_a_tensor_sibling_declares_the_same_VALUES_as_its_numpy_twin() -> None:
    """Source-text parity is a proxy; this compares what the hooks return.

    A sibling that imports a different preset constant under the same name, or
    that reads a differently-named manifest field, is textually identical and
    still wrong. The child builds a placeholder manifest instance
    (`model_construct`, so no field validation and no block initialisation) and
    normalises each hook through the real `normalize_declaration`.
    """
    # when
    numpy_rows = _run_census(tensor_mode="off", plugins=ENTERPRISE_LOADER)
    tensor_rows = _run_census(tensor_mode="on", plugins=ENTERPRISE_LOADER)

    # then
    numpy_values = {row["block_type"]: row["values"] for row in numpy_rows}
    tensor_values = {row["block_type"]: row["values"] for row in tensor_rows}
    assert set(numpy_values) == set(tensor_values)
    with_sibling = {
        row["block_type"]
        for row in tensor_rows
        if row["manifest_module"].endswith("_tensor")
    }
    assert len(with_sibling) > 100, f"only {len(with_sibling)} tensor siblings loaded"

    mismatches = {
        block_type: {
            hook: (numpy_values[block_type][hook], tensor_values[block_type][hook])
            for hook in DECLARATION_HOOKS
            if numpy_values[block_type][hook] != tensor_values[block_type][hook]
        }
        for block_type in sorted(numpy_values)
    }
    mismatches = {key: value for key, value in mismatches.items() if value}
    assert not mismatches, f"tensor siblings declare different values: {mismatches}"

    # and the comparison is not vacuous: real declarations were compared, not a
    # registry of hooks that all raised
    answered = [
        block_type
        for block_type in with_sibling
        if "value" in numpy_values[block_type]["discover_work_operations"]
    ]
    assert len(answered) > 100, f"only {len(answered)} siblings answered by value"
