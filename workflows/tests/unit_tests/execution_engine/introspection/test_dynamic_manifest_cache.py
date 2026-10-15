import gc
import weakref
from copy import deepcopy
from unittest import mock

import pytest
from roboflow_workflows.core_steps.transformations.absolute_static_crop.v1 import (
    AbsoluteStaticCropBlockV1,
)
from roboflow_workflows.core_steps.transformations.absolute_static_crop.v1 import (
    BlockManifest as StaticManifest,
)
from roboflow_workflows.execution_engine.introspection import (
    blocks_loader,
    schema_parser,
)
from roboflow_workflows.execution_engine.v1.compiler.entities import BlockSpecification
from roboflow_workflows.execution_engine.v1.compiler.syntactic_parser import (
    WORKFLOW_DEFINITION_ENTITIES_CACHE,
    build_workflow_definition_entity,
)
from roboflow_workflows.execution_engine.v1.dynamic_blocks.block_assembler import (
    compile_dynamic_blocks,
)


def _definition():
    return {
        "type": "DynamicBlockDefinition",
        "manifest": {
            "type": "ManifestDescription",
            "block_type": "CacheLifetimeProbe",
            "inputs": {
                "value": {
                    "type": "DynamicInputDefinition",
                    "value_types": ["string"],
                    "selector_types": ["input_parameter", "step_output"],
                    "selector_data_kind": {
                        "input_parameter": ["string"],
                        "step_output": ["string"],
                    },
                }
            },
            "outputs": {
                "output": {"type": "DynamicOutputDefinition", "kind": ["string"]}
            },
        },
        "code": {
            "type": "PythonCode",
            "run_function_code": 'def run(self, value):\n    return {"output": value}',
        },
    }


@pytest.mark.parametrize("structural", [False, True])
@pytest.mark.parametrize(
    "inspect_manifest",
    [
        schema_parser.parse_block_manifest,
        blocks_loader._cached_model_json_schema,
        blocks_loader._cached_describe_outputs,
    ],
)
def test_dynamic_manifest_is_collectible_after_introspection(
    structural, inspect_manifest
):
    references = []
    for _ in range(3):
        blocks = compile_dynamic_blocks(
            dynamic_blocks_definitions=[_definition()], structural=structural
        )
        manifest = blocks[0].manifest_class
        references.append(weakref.ref(manifest))
        inspect_manifest(manifest)
        del manifest, blocks

    gc.collect()

    assert all(reference() is None for reference in references)


@pytest.mark.parametrize(
    "inspect_manifest,argument_name",
    [
        (schema_parser.parse_block_manifest, "manifest_type"),
        (blocks_loader._cached_model_json_schema, "manifest_class"),
        (blocks_loader._cached_describe_outputs, "manifest_class"),
    ],
)
@pytest.mark.parametrize("use_keyword", [False, True])
def test_static_manifest_still_uses_cache(inspect_manifest, argument_name, use_keyword):
    args = () if use_keyword else (StaticManifest,)
    kwargs = {argument_name: StaticManifest} if use_keyword else {}

    first = inspect_manifest(*args, **kwargs)
    second = inspect_manifest(*args, **kwargs)

    assert second is first
    info = inspect_manifest.cache_info()
    assert (info.hits, info.misses, info.currsize) == (1, 1, 1)
    inspect_manifest.cache_clear()
    assert inspect_manifest.cache_info().currsize == 0


def test_same_code_and_block_name_with_different_manifests_remain_distinct():
    first_definition = _definition()
    second_definition = deepcopy(first_definition)
    second_definition["manifest"]["inputs"]["value"]["value_types"] = ["integer"]
    second_definition["manifest"]["outputs"]["output"]["kind"] = ["integer"]

    first = compile_dynamic_blocks(dynamic_blocks_definitions=[first_definition])[0]
    second = compile_dynamic_blocks(dynamic_blocks_definitions=[second_definition])[0]
    first_schema = blocks_loader._cached_model_json_schema(
        manifest_class=first.manifest_class
    )
    second_schema = blocks_loader._cached_model_json_schema(
        manifest_class=second.manifest_class
    )
    first_metadata = schema_parser.parse_block_manifest(
        manifest_type=first.manifest_class
    )
    second_metadata = schema_parser.parse_block_manifest(
        manifest_type=second.manifest_class
    )

    assert first_metadata.primitive_types["value"].type_annotation == "str"
    assert second_metadata.primitive_types["value"].type_annotation == "int"
    assert first_schema["properties"]["value"] != second_schema["properties"]["value"]
    assert "__workflow_dynamic_manifest__" not in first_schema["properties"]
    assert blocks_loader._cached_describe_outputs(first.manifest_class) != (
        blocks_loader._cached_describe_outputs(second.manifest_class)
    )
    assert schema_parser.parse_block_manifest.cache_info().currsize == 0
    assert blocks_loader._cached_model_json_schema.cache_info().currsize == 0
    assert blocks_loader._cached_describe_outputs.cache_info().currsize == 0


def test_workflow_validation_model_does_not_cache_dynamic_manifests():
    blocks = compile_dynamic_blocks(dynamic_blocks_definitions=[_definition()])
    with mock.patch.object(
        WORKFLOW_DEFINITION_ENTITIES_CACHE, "get"
    ) as cache_get, mock.patch.object(
        WORKFLOW_DEFINITION_ENTITIES_CACHE, "cache"
    ) as cache_write:
        entity = build_workflow_definition_entity(available_blocks=blocks)

    cache_get.assert_not_called()
    cache_write.assert_not_called()
    workflow = entity.model_validate(
        {
            "version": "1.0",
            "inputs": [],
            "steps": [
                {"type": "CacheLifetimeProbe", "name": "custom", "value": "hello"}
            ],
            "outputs": [],
        }
    )
    assert workflow.steps[0].value == "hello"


def test_workflow_validation_model_still_caches_static_manifests():
    blocks = [
        BlockSpecification(
            block_source="workflows_core",
            identifier="test.static_crop",
            block_class=AbsoluteStaticCropBlockV1,
            manifest_class=StaticManifest,
        )
    ]

    first = build_workflow_definition_entity(available_blocks=blocks)
    second = build_workflow_definition_entity(available_blocks=blocks)

    assert second is first
