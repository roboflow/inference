from typing import Literal
from unittest import mock
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from inference.core.workflows.errors import PluginInterfaceError, PluginLoadingError
from inference.core.workflows.execution_engine.introspection import blocks_loader
from inference.core.workflows.execution_engine.introspection.blocks_loader import (
    describe_available_blocks,
    get_manifest_type_identifiers,
    load_blocks_from_plugin,
    load_initializers,
    load_initializers_from_plugin,
)
from tests.workflows.unit_tests.execution_engine.introspection import (
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


@mock.patch.object(blocks_loader, "load_workflow_blocks")
def test_describe_available_blocks_when_valid_plugins_are_loaded(
    load_workflow_blocks_mock: MagicMock,
) -> None:
    # given
    load_workflow_blocks_mock.return_value = load_blocks_from_plugin(
        "tests.workflows.unit_tests.execution_engine.introspection.plugin_with_valid_blocks"
    )

    # when
    result = describe_available_blocks()

    # then
    assert len(result.blocks) == 2, "Expected 2 blocks to be loaded"
    assert result.blocks[0].block_class == plugin_with_valid_blocks.Block1
    assert result.blocks[0].manifest_class == plugin_with_valid_blocks.Block1Manifest
    assert result.blocks[1].block_class == plugin_with_valid_blocks.Block2
    assert result.blocks[1].manifest_class == plugin_with_valid_blocks.Block2Manifest
    assert len(result.declared_kinds) == 3


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
        _ = describe_available_blocks()


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
        _ = describe_available_blocks()
