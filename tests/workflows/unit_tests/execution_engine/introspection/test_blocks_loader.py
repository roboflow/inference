from typing import Literal
from unittest import mock
from unittest.mock import MagicMock

import pytest
from packaging.version import Version
from pydantic import BaseModel

from inference.core.workflows.errors import (
    PluginInterfaceError,
    PluginLoadingError,
    WorkflowExecutionEngineVersionError,
)
from inference.core.workflows.execution_engine.introspection import blocks_loader
from inference.core.workflows.execution_engine.introspection.blocks_loader import (
    describe_available_blocks,
    get_manifest_type_identifiers,
    is_block_compatible_with_execution_engine,
    load_blocks_from_plugin,
    load_initializers,
    load_initializers_from_plugin,
    load_kinds_deserializers,
    load_kinds_serializers,
    load_workflow_blocks,
)
from tests.workflows.unit_tests.execution_engine.introspection import (
    plugin_with_multiple_versions_of_blocks,
    plugin_with_valid_blocks,
)


def test_get_manifest_type_identifiers_when_type_property_is_not_available() -> None:
    # given

    class Some(BaseModel):
        data: str

    # when
    with pytest.raises(PluginInterfaceError):
        _ = get_manifest_type_identifiers(
            block_schema=Some.model_json_schema(),
            block_source="test",
            block_identifier="test.Some",
        )


def test_get_manifest_type_identifiers_when_type_property_is_not_defined_as_literal() -> (
    None
):
    # given

    class Some(BaseModel):
        type: str

    # when
    with pytest.raises(PluginInterfaceError):
        _ = get_manifest_type_identifiers(
            block_schema=Some.model_json_schema(),
            block_source="test",
            block_identifier="test.Some",
        )


def test_get_manifest_type_identifiers_when_type_property_is_defined_as_literal() -> (
    None
):
    # given

    class Some(BaseModel):
        type: Literal["my_value"]

    # when
    result = get_manifest_type_identifiers(
        block_schema=Some.model_json_schema(),
        block_source="test",
        block_identifier="test.Some",
    )

    # then
    assert result == ["my_value"], "Single value defined in literal should be returned"


def test_get_manifest_type_identifiers_when_type_property_is_defined_as_literal_with_aliases() -> (
    None
):
    # given

    class Some(BaseModel):
        type: Literal["my_value", "my_alias"]

    # when
    result = get_manifest_type_identifiers(
        block_schema=Some.model_json_schema(),
        block_source="test",
        block_identifier="test.Some",
    )

    # then
    assert result == [
        "my_value",
        "my_alias",
    ], "Both main value of type literal and alias should be returned"


def test_load_blocks_from_plugin_when_plugin_does_not_exists() -> None:
    # when
    with pytest.raises(PluginLoadingError):
        _ = load_blocks_from_plugin("non_existing_dummy_plugin")


def test_load_blocks_from_plugin_when_plugin_does_not_implement_interface() -> None:
    # when
    with pytest.raises(PluginInterfaceError):
        _ = load_blocks_from_plugin(
            "tests.workflows.unit_tests.execution_engine.introspection.plugin_with_initializers"
        )


def test_load_blocks_from_plugin_when_plugin_implement_interface() -> None:
    # when
    result = load_blocks_from_plugin(
        "tests.workflows.unit_tests.execution_engine.introspection.plugin_with_valid_blocks"
    )

    # then
    assert len(result) == 2, "Expected 2 blocks to be loaded"
    assert result[0].block_class == plugin_with_valid_blocks.Block1
    assert result[0].manifest_class == plugin_with_valid_blocks.Block1Manifest
    assert result[1].block_class == plugin_with_valid_blocks.Block2
    assert result[1].manifest_class == plugin_with_valid_blocks.Block2Manifest


def test_load_initializers_from_plugin_when_plugin_does_not_exists() -> None:
    # when
    with pytest.raises(PluginLoadingError):
        _ = load_initializers_from_plugin("non_existing_dummy_plugin")


def test_load_initializers_from_plugin_when_plugin_exists_but_no_initializers_provided() -> (
    None
):
    # when
    result = load_initializers_from_plugin(
        "tests.workflows.unit_tests.execution_engine.introspection.plugin_with_valid_blocks"
    )

    # then
    assert len(result) == 0


def test_load_initializers_from_plugin_when_plugin_exists_and_initializers_provided() -> (
    None
):
    # when
    result = load_initializers_from_plugin(
        "tests.workflows.unit_tests.execution_engine.introspection.plugin_with_initializers"
    )

    # then
    assert len(result) == 2
    assert (
        result[
            "tests.workflows.unit_tests.execution_engine.introspection.plugin_with_initializers.a"
        ]
        == 38
    ), "This parameter is expected to be static value"
    assert (
        result[
            "tests.workflows.unit_tests.execution_engine.introspection.plugin_with_initializers.b"
        ]()
        == 7
    ), "This parameter is expected to be function returning static value"


@mock.patch.dict(
    blocks_loader.os.environ,
    {
        "WORKFLOWS_PLUGINS": "tests.workflows.unit_tests.execution_engine.introspection.plugin_with_initializers"
    },
    clear=True,
)
def test_load_initializers_when_plugin_exists_and_initializers_provided() -> None:
    # when
    result = load_initializers()

    # then
    assert len(result) > 0
    assert (
        result[
            "tests.workflows.unit_tests.execution_engine.introspection.plugin_with_initializers.a"
        ]
        == 38
    ), "This parameter is expected to be static value"
    assert (
        result[
            "tests.workflows.unit_tests.execution_engine.introspection.plugin_with_initializers.b"
        ]()
        == 7
    ), "This parameter is expected to be function returning static value"


@mock.patch.object(blocks_loader, "load_workflow_blocks")
def test_describe_available_blocks_when_valid_plugins_are_loaded(
    load_workflow_blocks_mock: MagicMock,
) -> None:
    # given
    load_workflow_blocks_mock.return_value = load_blocks_from_plugin(
        "tests.workflows.unit_tests.execution_engine.introspection.plugin_with_valid_blocks"
    )

    # when
    result = describe_available_blocks(
        dynamic_blocks=[],
        execution_engine_version=Version("1.0.0"),
    )

    # then
    assert len(result.blocks) == 2, "Expected 2 blocks to be loaded"
    assert result.blocks[0].block_class == plugin_with_valid_blocks.Block1
    assert result.blocks[0].manifest_class == plugin_with_valid_blocks.Block1Manifest
    assert result.blocks[1].block_class == plugin_with_valid_blocks.Block2
    assert result.blocks[1].manifest_class == plugin_with_valid_blocks.Block2Manifest
    assert len(result.declared_kinds) > 0


@mock.patch.object(blocks_loader, "load_workflow_blocks")
def test_describe_available_blocks_when_valid_plugins_are_loaded_and_multiple_versions_in_the_same_family(
    load_workflow_blocks_mock: MagicMock,
) -> None:
    # given
    load_workflow_blocks_mock.return_value = load_blocks_from_plugin(
        "tests.workflows.unit_tests.execution_engine.introspection.plugin_with_multiple_versions_of_blocks"
    )

    # when
    result = describe_available_blocks(
        dynamic_blocks=[],
        execution_engine_version=Version("1.0.0"),
    )

    # then
    assert len(result.blocks) == 3, "Expected 2 blocks to be loaded"
    assert (
        result.blocks[0].block_class == plugin_with_multiple_versions_of_blocks.Block1V1
    )
    assert (
        result.blocks[0].manifest_class
        == plugin_with_multiple_versions_of_blocks.Block1V1Manifest
    )
    assert (
        result.blocks[1].block_class == plugin_with_multiple_versions_of_blocks.Block1V2
    )
    assert (
        result.blocks[1].manifest_class
        == plugin_with_multiple_versions_of_blocks.Block1V2Manifest
    )
    assert (
        result.blocks[2].block_class == plugin_with_multiple_versions_of_blocks.Block2
    )
    assert (
        result.blocks[2].manifest_class
        == plugin_with_multiple_versions_of_blocks.Block2Manifest
    )
    assert len(result.declared_kinds) > 0


@mock.patch.object(blocks_loader, "load_workflow_blocks")
def test_describe_available_blocks_when_plugins_duplicate_class_names(
    load_workflow_blocks_mock: MagicMock,
) -> None:
    # given
    load_workflow_blocks_mock.return_value = (
        load_blocks_from_plugin(
            "tests.workflows.unit_tests.execution_engine.introspection.plugin_with_valid_blocks"
        )
        * 2
    )

    # when
    with pytest.raises(PluginLoadingError):
        _ = describe_available_blocks(dynamic_blocks=[])


@mock.patch.object(blocks_loader, "load_workflow_blocks")
def test_describe_available_blocks_when_plugins_duplicate_type_identifiers(
    load_workflow_blocks_mock: MagicMock,
) -> None:
    # given
    load_workflow_blocks_mock.return_value = load_blocks_from_plugin(
        "tests.workflows.unit_tests.execution_engine.introspection.plugin_duplicated_identifiers"
    )

    # when
    with pytest.raises(PluginLoadingError):
        _ = describe_available_blocks(dynamic_blocks=[])


def test_load_workflow_blocks_when_execution_engine_version_is_not_given() -> None:
    # when
    result = load_workflow_blocks(execution_engine_version=None)

    # then
    assert len(result) > 0, "Expected core blocks to be found"


def test_load_workflow_blocks_when_execution_engine_version_passed_as_string() -> None:
    # when
    result = load_workflow_blocks(execution_engine_version="1.0.0")

    # then
    assert len(result) > 0, "Expected core blocks to be found"


def test_load_workflow_blocks_when_execution_engine_version_passed_as_version_object() -> (
    None
):
    # when
    result = load_workflow_blocks(execution_engine_version=Version("1.0.0"))

    # then
    assert len(result) > 0, "Expected core blocks to be found"


def test_load_workflow_blocks_when_execution_engine_version_is_not_parsable() -> None:
    # when
    with pytest.raises(WorkflowExecutionEngineVersionError):
        _ = load_workflow_blocks(execution_engine_version="invalid")


def test_load_workflow_blocks_when_execution_engine_version_does_not_match_any_block() -> (
    None
):
    # when
    result = load_workflow_blocks(execution_engine_version="0.0.1")

    # then
    assert len(result) == 0, "Expected no blocks to be found"


def test_is_block_compatible_with_execution_engine_when_execution_engine_version_not_given() -> (
    None
):
    # when
    result = is_block_compatible_with_execution_engine(
        execution_engine_version=None,
        block_execution_engine_compatibility=">=1.0.0,<2.0.0",
        block_source="workflows_core",
        block_identifier="some",
    )

    # then
    assert result is True, "Expected None for EE version to be treated as *"


def test_is_block_compatible_with_execution_engine_when_block_compatibility_not_given() -> (
    None
):
    # when
    result = is_block_compatible_with_execution_engine(
        execution_engine_version=Version("1.3.0"),
        block_execution_engine_compatibility=None,
        block_source="workflows_core",
        block_identifier="some",
    )

    # then
    assert result is True, "Expected None for block compatibility to be treated as *"


@pytest.mark.parametrize(
    "execution_engine_version, block_execution_engine_compatibility",
    [
        (Version("1.0.0"), ">=1.0.0,<2.0.0"),
        (Version("1.3.0"), ">=1.0.0,<2.0.0"),
        (Version("2.3.0"), ">=2.0.0,<=2.3.0"),
        (Version("2.3.1"), "~=2.3.0"),
    ],
)
def test_is_block_compatible_with_execution_engine_when_compatible_setup_provided(
    execution_engine_version: Version,
    block_execution_engine_compatibility: str,
) -> None:
    # when
    result = is_block_compatible_with_execution_engine(
        execution_engine_version=execution_engine_version,
        block_execution_engine_compatibility=block_execution_engine_compatibility,
        block_source="workflows_core",
        block_identifier="some",
    )

    # then
    assert result is True, "Expected parameters to match"


@pytest.mark.parametrize(
    "execution_engine_version, block_execution_engine_compatibility",
    [
        (Version("2.0.0"), ">=1.0.0,<2.0.0"),
        (Version("1.3.0"), ">=1.0.0,<2.0.0,!=1.3.0"),
        (Version("2.3.1"), ">=2.0.0,<=2.3.0"),
        (Version("2.4.1"), "~=2.3.0"),
    ],
)
def test_is_block_compatible_with_execution_engine_when_non_compatible_setup_provided(
    execution_engine_version: Version,
    block_execution_engine_compatibility: str,
) -> None:
    # when
    result = is_block_compatible_with_execution_engine(
        execution_engine_version=execution_engine_version,
        block_execution_engine_compatibility=block_execution_engine_compatibility,
        block_source="workflows_core",
        block_identifier="some",
    )

    # then
    assert result is False, "Expected parameters not to match"


def test_is_block_compatible_with_execution_engine_when_block_execution_engine_compatibility_is_not_parsable() -> (
    None
):
    # when
    with pytest.raises(PluginInterfaceError):
        _ = is_block_compatible_with_execution_engine(
            execution_engine_version=Version("1.0.0"),
            block_execution_engine_compatibility="invalid",
            block_source="workflows_core",
            block_identifier="some",
        )


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_load_kinds_serializers(
    get_plugin_modules_mock: MagicMock,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.unit_tests.execution_engine.introspection.plugin_with_kinds_serializers"
    ]

    # when
    result = load_kinds_serializers()

    # then
    assert len(result) > 0
    assert result["1"]("some") == "1", "Expected hardcoded value from serializer"
    assert result["2"]("some") == "2", "Expected hardcoded value from serializer"
    assert result["3"]("some") == "3", "Expected hardcoded value from serializer"


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_load_kinds_deserializers(
    get_plugin_modules_mock: MagicMock,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.unit_tests.execution_engine.introspection.plugin_with_kinds_serializers"
    ]

    # when
    result = load_kinds_deserializers()

    # then
    assert len(result) > 0
    assert (
        result["1"]("some", "value") == "1"
    ), "Expected hardcoded value from deserializer"
    assert (
        result["2"]("some", "value") == "2"
    ), "Expected hardcoded value from deserializer"
    assert (
        result["3"]("some", "value") == "3"
    ), "Expected hardcoded value from deserializer"
