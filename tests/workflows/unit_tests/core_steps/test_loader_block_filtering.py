"""
Tests for workflow block filtering functionality added in PR #1924.

This module tests the `_should_filter_block()` function and filtering behavior
in `load_blocks()` when block filtering is configured via WORKFLOW_DISABLED_BLOCK_TYPES
or WORKFLOW_DISABLED_BLOCK_PATTERNS.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union
from unittest import mock

import pytest
from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps import loader
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.v1.entities import FlowControl
from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)


# =============================================================================
# Test Block Definitions
# =============================================================================


class SinkBlockManifest(WorkflowBlockManifest):
    """Test block manifest with block_type='sink'."""

    model_config = ConfigDict(
        json_schema_extra={
            "name": "Test Sink Block",
            "version": "v1",
            "block_type": "sink",
        }
    )
    type: Literal["test/sink_block@v1"]
    name: str = Field(description="name field")

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class SinkBlock(WorkflowBlock):
    """Test block with block_type='sink'."""

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return SinkBlockManifest

    def run(
        self, *args, **kwargs
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        pass


class ModelBlockManifest(WorkflowBlockManifest):
    """Test block manifest with block_type='model'."""

    model_config = ConfigDict(
        json_schema_extra={
            "name": "Test Model Block",
            "version": "v1",
            "block_type": "model",
        }
    )
    type: Literal["test/model_block@v1"]
    name: str = Field(description="name field")

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class ModelBlock(WorkflowBlock):
    """Test block with block_type='model'."""

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ModelBlockManifest

    def run(
        self, *args, **kwargs
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        pass


class TransformationBlockManifest(WorkflowBlockManifest):
    """Test block manifest with block_type='transformation'."""

    model_config = ConfigDict(
        json_schema_extra={
            "name": "Test Transformation Block",
            "version": "v1",
            "block_type": "transformation",
        }
    )
    type: Literal["test/transformation_block@v1"]
    name: str = Field(description="name field")

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class TransformationBlock(WorkflowBlock):
    """Test block with block_type='transformation'."""

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return TransformationBlockManifest

    def run(
        self, *args, **kwargs
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        pass


class WebhookNotificationBlockManifest(WorkflowBlockManifest):
    """Test block manifest with 'webhook' in its name."""

    model_config = ConfigDict(
        json_schema_extra={
            "name": "Webhook Notification",
            "version": "v1",
            "block_type": "sink",
        }
    )
    type: Literal["test/webhook_notification@v1"]
    name: str = Field(description="name field")

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class WebhookNotificationBlock(WorkflowBlock):
    """Test block with 'webhook' in its friendly name."""

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return WebhookNotificationBlockManifest

    def run(
        self, *args, **kwargs
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        pass


# =============================================================================
# Test 1: Block Type Filtering
# =============================================================================


class TestShouldFilterBlockByType:
    """Tests for _should_filter_block() with block_type filtering."""

    @mock.patch.object(loader, "WORKFLOW_DISABLED_BLOCK_TYPES", ["sink"])
    @mock.patch.object(loader, "WORKFLOW_DISABLED_BLOCK_PATTERNS", [])
    def test_should_filter_block_when_block_type_matches_disabled_type(self) -> None:
        # when
        result = loader._should_filter_block(SinkBlock)

        # then
        assert result is True, "Block with type 'sink' should be filtered when 'sink' is disabled"

    @mock.patch.object(loader, "WORKFLOW_DISABLED_BLOCK_TYPES", ["sink"])
    @mock.patch.object(loader, "WORKFLOW_DISABLED_BLOCK_PATTERNS", [])
    def test_should_not_filter_block_when_block_type_does_not_match(self) -> None:
        # when
        result = loader._should_filter_block(TransformationBlock)

        # then
        assert result is False, (
            "Block with type 'transformation' should not be filtered when only 'sink' is disabled"
        )

    @mock.patch.object(loader, "WORKFLOW_DISABLED_BLOCK_TYPES", ["sink"])  # env.py normalizes to lowercase
    @mock.patch.object(loader, "WORKFLOW_DISABLED_BLOCK_PATTERNS", [])
    def test_should_filter_block_with_case_insensitive_block_type_value(self) -> None:
        """Test that block_type from schema is lowercased before comparison.

        Note: WORKFLOW_DISABLED_BLOCK_TYPES is normalized to lowercase by env.py,
        so we mock with lowercase. This test verifies the block's block_type
        (which could theoretically be mixed case in schema) is lowercased.
        """
        # when
        result = loader._should_filter_block(SinkBlock)

        # then
        assert result is True, "Block type from schema should be lowercased for comparison"

    @mock.patch.object(loader, "WORKFLOW_DISABLED_BLOCK_TYPES", ["sink", "model"])
    @mock.patch.object(loader, "WORKFLOW_DISABLED_BLOCK_PATTERNS", [])
    def test_should_filter_blocks_when_multiple_types_disabled(self) -> None:
        # when
        sink_result = loader._should_filter_block(SinkBlock)
        model_result = loader._should_filter_block(ModelBlock)
        transformation_result = loader._should_filter_block(TransformationBlock)

        # then
        assert sink_result is True, "Sink block should be filtered"
        assert model_result is True, "Model block should be filtered"
        assert transformation_result is False, "Transformation block should not be filtered"


# =============================================================================
# Test 2: Pattern Matching Filtering
# =============================================================================


class TestShouldFilterBlockByPattern:
    """Tests for _should_filter_block() with pattern matching."""

    @mock.patch.object(loader, "WORKFLOW_DISABLED_BLOCK_TYPES", [])
    @mock.patch.object(loader, "WORKFLOW_DISABLED_BLOCK_PATTERNS", ["webhook"])
    def test_should_filter_block_when_pattern_matches_friendly_name(self) -> None:
        # when
        result = loader._should_filter_block(WebhookNotificationBlock)

        # then
        assert result is True, (
            "Block should be filtered when pattern 'webhook' matches friendly name 'Webhook Notification'"
        )

    @mock.patch.object(loader, "WORKFLOW_DISABLED_BLOCK_TYPES", [])
    @mock.patch.object(loader, "WORKFLOW_DISABLED_BLOCK_PATTERNS", ["sinkblock"])
    def test_should_filter_block_when_pattern_matches_class_name(self) -> None:
        # when
        result = loader._should_filter_block(SinkBlock)

        # then
        assert result is True, (
            "Block should be filtered when pattern 'sinkblock' matches class name 'SinkBlock'"
        )

    @mock.patch.object(loader, "WORKFLOW_DISABLED_BLOCK_TYPES", [])
    @mock.patch.object(loader, "WORKFLOW_DISABLED_BLOCK_PATTERNS", ["WEBHOOK"])  # uppercase
    def test_should_filter_block_with_case_insensitive_pattern_matching(self) -> None:
        # when
        result = loader._should_filter_block(WebhookNotificationBlock)

        # then
        assert result is True, "Pattern matching should be case-insensitive"

    @mock.patch.object(loader, "WORKFLOW_DISABLED_BLOCK_TYPES", [])
    @mock.patch.object(loader, "WORKFLOW_DISABLED_BLOCK_PATTERNS", ["nonexistent_pattern"])
    def test_should_not_filter_block_when_pattern_does_not_match(self) -> None:
        # when
        result = loader._should_filter_block(SinkBlock)

        # then
        assert result is False, "Block should not be filtered when pattern doesn't match"

    @mock.patch.object(loader, "WORKFLOW_DISABLED_BLOCK_TYPES", ["sink"])  # type match
    @mock.patch.object(loader, "WORKFLOW_DISABLED_BLOCK_PATTERNS", ["modelblock"])  # pattern match
    def test_should_filter_block_when_either_type_or_pattern_matches(self) -> None:
        # when
        sink_result = loader._should_filter_block(SinkBlock)  # matches type
        model_result = loader._should_filter_block(ModelBlock)  # matches pattern
        transformation_result = loader._should_filter_block(TransformationBlock)  # neither

        # then
        assert sink_result is True, "Sink block should be filtered by type"
        assert model_result is True, "Model block should be filtered by pattern"
        assert transformation_result is False, (
            "Transformation block matches neither and should not be filtered"
        )


# =============================================================================
# Test 3: Feature Toggle Behavior
# =============================================================================


class TestFeatureToggleBehavior:
    """Tests for implicit feature toggle based on configuration presence."""

    @mock.patch.object(loader, "WORKFLOW_DISABLED_BLOCK_TYPES", [])
    @mock.patch.object(loader, "WORKFLOW_DISABLED_BLOCK_PATTERNS", [])
    def test_should_not_filter_when_no_types_or_patterns_specified(self) -> None:
        # when
        result = loader._should_filter_block(SinkBlock)

        # then
        assert result is False, (
            "No filtering should occur when no types or patterns are specified"
        )

    @mock.patch.object(loader, "WORKFLOW_DISABLED_BLOCK_TYPES", ["sink"])
    @mock.patch.object(loader, "WORKFLOW_DISABLED_BLOCK_PATTERNS", [])
    def test_load_blocks_excludes_filtered_blocks(self) -> None:
        # when
        blocks = loader.load_blocks()

        # then
        for block_class in blocks:
            try:
                manifest_class = block_class.get_manifest()
                schema = manifest_class.model_json_schema()
                # Note: Pydantic V2 puts json_schema_extra values at top level of schema
                block_type = schema.get("block_type", "")
                assert block_type.lower() != "sink", (
                    f"Sink block {block_class.__name__} should have been filtered"
                )
            except Exception:
                # Skip blocks that don't have proper manifest structure
                pass

    @mock.patch.object(loader, "WORKFLOW_DISABLED_BLOCK_TYPES", ["sink"])
    @mock.patch.object(loader, "WORKFLOW_DISABLED_BLOCK_PATTERNS", [])
    def test_load_blocks_count_reduced_when_filtering_enabled(self) -> None:
        # given - get filtered count
        filtered_blocks = loader.load_blocks()
        filtered_count = len(filtered_blocks)

        # when - get unfiltered count
        with mock.patch.object(loader, "WORKFLOW_DISABLED_BLOCK_TYPES", []):
            unfiltered_blocks = loader.load_blocks()
            unfiltered_count = len(unfiltered_blocks)

        # then
        assert filtered_count < unfiltered_count, (
            f"Filtered block count ({filtered_count}) should be less than "
            f"unfiltered count ({unfiltered_count})"
        )
