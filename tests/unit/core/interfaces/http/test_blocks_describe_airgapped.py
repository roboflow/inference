"""Tests for air-gapped enrichment in /workflows/blocks/describe."""

import copy
import importlib
import sys
from typing import Any, Dict, List, Optional, Type
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.introspection.entities import (
    BlockDescription,
)
from inference.core.workflows.prototypes.block import WorkflowBlock


# ---------------------------------------------------------------------------
# Helpers: stub manifest classes
# ---------------------------------------------------------------------------


class _BaseManifest(BaseModel):
    type: str = "stub"

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return None

    @classmethod
    def get_input_dimensionality_offsets(cls) -> Dict[str, int]:
        return {}

    @classmethod
    def get_dimensionality_reference_property(cls) -> Optional[str]:
        return None

    @classmethod
    def get_output_dimensionality_offset(cls) -> int:
        return 0

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []


class PlainBlockManifest(_BaseManifest):
    """A block with no air-gapped classmethods -- should default to available."""

    type: str = "test/plain_block@v1"


class CloudOnlyManifest(_BaseManifest):
    """Simulates an OpenAI-style block that requires internet."""

    type: str = "test/cloud_only@v1"


# Attach the classmethod manually to avoid Pydantic treating it as a validator.
@classmethod  # type: ignore[misc]
def _cloud_only_air_gapped(cls) -> Dict[str, Any]:
    return {"available": False, "reason": "requires_internet"}


CloudOnlyManifest.get_air_gapped_availability = _cloud_only_air_gapped


class FoundationModelManifest(_BaseManifest):
    """Simulates a foundation model block with cache artifacts."""

    type: str = "test/foundation_model@v1"


@classmethod  # type: ignore[misc]
def _foundation_cache_artifacts(cls) -> Dict[str, Any]:
    return {
        "model_id": "yolov8n-640",
        "files": ["weights.pt", "config.json"],
    }


@classmethod  # type: ignore[misc]
def _foundation_task_types(cls) -> List[str]:
    return ["object-detection", "instance-segmentation"]


FoundationModelManifest.get_required_cache_artifacts = _foundation_cache_artifacts
FoundationModelManifest.get_compatible_task_types = _foundation_task_types


class ListFormatFoundationManifest(_BaseManifest):
    """Simulates a foundation model block with list-format cache artifacts."""

    type: str = "test/list_foundation@v1"


@classmethod  # type: ignore[misc]
def _list_cache_artifacts(cls) -> List[str]:
    return ["clip/RN50", "clip/ViT-B-32", "clip/ViT-L-14"]


@classmethod  # type: ignore[misc]
def _list_task_types(cls) -> List[str]:
    return ["embedding"]


ListFormatFoundationManifest.get_required_cache_artifacts = _list_cache_artifacts
ListFormatFoundationManifest.get_compatible_task_types = _list_task_types


class LocalNetworkManifest(_BaseManifest):
    """Simulates a local-network block (ONVIF, S3, etc.) -- should be available."""

    type: str = "test/local_network@v1"


class _StubBlock(WorkflowBlock):
    @classmethod
    def get_manifest(cls):
        return PlainBlockManifest

    async def run(self, *args, **kwargs):
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_block_description(
    manifest_cls: Type[BaseModel],
    block_type: str = "test/stub@v1",
) -> BlockDescription:
    schema = manifest_cls.model_json_schema()
    return BlockDescription(
        manifest_class=manifest_cls,
        block_class=_StubBlock,
        block_schema=schema,
        outputs_manifest=[],
        block_source="test",
        fully_qualified_block_class_name=f"test.{manifest_cls.__name__}",
        human_friendly_block_name=manifest_cls.__name__,
        manifest_type_identifier=block_type,
        manifest_type_identifier_aliases=[],
        execution_engine_compatibility=None,
        input_dimensionality_offsets={},
        dimensionality_reference_property=None,
        output_dimensionality_offset=0,
    )


def _make_result(blocks: List[BlockDescription]) -> "WorkflowsBlocksDescription":
    from inference.core.entities.responses.workflows import (
        UniversalQueryLanguageDescription,
        WorkflowsBlocksDescription,
    )

    return WorkflowsBlocksDescription(
        blocks=blocks,
        declared_kinds=[],
        kinds_connections={},
        primitives_connections=[],
        universal_query_language_description=UniversalQueryLanguageDescription(
            operations_description=[],
            operators_descriptions=[],
        ),
        dynamic_block_definition_schema={},
    )


def _import_enrichment():
    """Import only the enrichment helpers, avoiding the full block-loading chain."""
    # We need the module to be importable without triggering
    # inference.core.workflows.execution_engine.core (which loads all blocks).
    # The enrichment function itself does not depend on that import chain,
    # so we can import it directly once the module is loaded.
    import inference.core.interfaces.http.handlers.workflows as mod

    return mod.enrich_with_air_gapped_info, mod._get_air_gapped_info_for_block


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetAirGappedInfoForBlock:
    """Unit tests for _get_air_gapped_info_for_block (no full import chain needed)."""

    def test_plain_block_defaults_to_available(self):
        from inference.core.interfaces.http.handlers.workflows import (
            _get_air_gapped_info_for_block,
        )

        info = _get_air_gapped_info_for_block(PlainBlockManifest)
        assert info["available"] is True

    def test_cloud_only_block_requires_internet(self):
        from inference.core.interfaces.http.handlers.workflows import (
            _get_air_gapped_info_for_block,
        )

        info = _get_air_gapped_info_for_block(CloudOnlyManifest)
        assert info["available"] is False
        assert info["reason"] == "requires_internet"

    def test_local_network_block_is_available(self):
        from inference.core.interfaces.http.handlers.workflows import (
            _get_air_gapped_info_for_block,
        )

        info = _get_air_gapped_info_for_block(LocalNetworkManifest)
        assert info["available"] is True

    @patch(
        "inference.core.interfaces.http.handlers.workflows.is_block_cached",
        return_value=True,
    )
    def test_foundation_model_cached(self, mock_cache):
        from inference.core.interfaces.http.handlers.workflows import (
            _get_air_gapped_info_for_block,
        )

        info = _get_air_gapped_info_for_block(FoundationModelManifest)
        assert info["available"] is True
        assert info["model_id"] == "yolov8n-640"
        assert "object-detection" in info["compatible_task_types"]
        mock_cache.assert_called_once()

    @patch(
        "inference.core.interfaces.http.handlers.workflows.is_block_cached",
        return_value=False,
    )
    def test_foundation_model_not_cached(self, mock_cache):
        from inference.core.interfaces.http.handlers.workflows import (
            _get_air_gapped_info_for_block,
        )

        info = _get_air_gapped_info_for_block(FoundationModelManifest)
        assert info["available"] is False
        assert info["reason"] == "missing_cache_artifacts"
        assert info["model_id"] == "yolov8n-640"


class TestEnrichWithAirGappedInfo:
    def test_air_gapped_info_added_when_flag_set(self):
        """When enrich_with_air_gapped_info is called, blocks get air_gapped_info."""
        from inference.core.interfaces.http.handlers.workflows import (
            enrich_with_air_gapped_info,
        )

        blocks = [_make_block_description(PlainBlockManifest, "test/plain_block@v1")]
        result = _make_result(blocks)

        enriched = enrich_with_air_gapped_info(result)

        for block in enriched.blocks:
            assert "json_schema_extra" in block.block_schema
            info = block.block_schema["json_schema_extra"]["air_gapped_info"]
            assert "available" in info

    def test_air_gapped_info_absent_when_flag_not_set(self):
        """Without calling enrich, blocks should NOT have air_gapped_info."""
        blocks = [_make_block_description(PlainBlockManifest, "test/plain_block@v1")]
        result = _make_result(blocks)

        for block in result.blocks:
            extra = block.block_schema.get("json_schema_extra", {})
            assert "air_gapped_info" not in extra

    def test_cloud_only_blocks_marked_requires_internet(self):
        """Blocks with get_air_gapped_availability returning requires_internet
        should be marked as unavailable."""
        from inference.core.interfaces.http.handlers.workflows import (
            enrich_with_air_gapped_info,
        )

        blocks = [
            _make_block_description(CloudOnlyManifest, "test/cloud_only@v1"),
        ]
        result = _make_result(blocks)

        enriched = enrich_with_air_gapped_info(result)

        info = enriched.blocks[0].block_schema["json_schema_extra"]["air_gapped_info"]
        assert info["available"] is False
        assert info["reason"] == "requires_internet"

    def test_local_network_blocks_not_marked_unavailable(self):
        """Local network blocks with no special classmethods
        should default to available=True."""
        from inference.core.interfaces.http.handlers.workflows import (
            enrich_with_air_gapped_info,
        )

        blocks = [
            _make_block_description(LocalNetworkManifest, "test/local_network@v1"),
        ]
        result = _make_result(blocks)

        enriched = enrich_with_air_gapped_info(result)

        info = enriched.blocks[0].block_schema["json_schema_extra"]["air_gapped_info"]
        assert info["available"] is True

    @patch(
        "inference.core.interfaces.http.handlers.workflows.ENABLE_BUILDER",
        False,
    )
    def test_air_gapped_ignored_without_enable_builder(self):
        """When ENABLE_BUILDER is False, air_gapped=True should be silently ignored
        in handle_describe_workflows_blocks_request."""
        from inference.core.interfaces.http.handlers.workflows import (
            handle_describe_workflows_blocks_request,
        )

        with patch(
            "inference.core.interfaces.http.handlers.workflows.describe_available_blocks"
        ) as mock_describe, patch(
            "inference.core.interfaces.http.handlers.workflows.discover_blocks_connections"
        ) as mock_connections, patch(
            "inference.core.interfaces.http.handlers.workflows.compile_dynamic_blocks",
            return_value=[],
        ), patch(
            "inference.core.interfaces.http.handlers.workflows.prepare_operations_descriptions",
            return_value=[],
        ), patch(
            "inference.core.interfaces.http.handlers.workflows.prepare_operators_descriptions",
            return_value=[],
        ):
            blocks = [
                _make_block_description(PlainBlockManifest, "test/plain_block@v1")
            ]
            from inference.core.workflows.execution_engine.introspection.entities import (
                BlocksDescription,
            )

            mock_describe.return_value = BlocksDescription(
                blocks=blocks, declared_kinds=[]
            )
            mock_connections.return_value = MagicMock(
                kinds_connections={}, primitives_connections=[]
            )

            result = handle_describe_workflows_blocks_request(air_gapped=True)

            for block in result.blocks:
                extra = block.block_schema.get("json_schema_extra", {})
                assert "air_gapped_info" not in extra

    def test_enrichment_does_not_mutate_original(self):
        """enrich_with_air_gapped_info must not mutate the original result object."""
        from inference.core.interfaces.http.handlers.workflows import (
            enrich_with_air_gapped_info,
        )

        blocks = [_make_block_description(PlainBlockManifest, "test/plain_block@v1")]
        result = _make_result(blocks)
        original_schema = copy.deepcopy(result.blocks[0].block_schema)

        enrich_with_air_gapped_info(result)

        # The original result's block_schema should be untouched
        assert result.blocks[0].block_schema == original_schema

    @patch(
        "inference.core.interfaces.http.handlers.workflows.is_block_cached",
        return_value=True,
    )
    def test_foundation_model_cached_shows_available(self, mock_cache):
        """Foundation model blocks with all artifacts cached should show available=True."""
        from inference.core.interfaces.http.handlers.workflows import (
            enrich_with_air_gapped_info,
        )

        blocks = [
            _make_block_description(
                FoundationModelManifest, "test/foundation_model@v1"
            ),
        ]
        result = _make_result(blocks)

        enriched = enrich_with_air_gapped_info(result)

        info = enriched.blocks[0].block_schema["json_schema_extra"]["air_gapped_info"]
        assert info["available"] is True
        assert info["model_id"] == "yolov8n-640"
        assert "compatible_task_types" in info
        assert "object-detection" in info["compatible_task_types"]

    @patch(
        "inference.core.interfaces.http.handlers.workflows.is_block_cached",
        return_value=False,
    )
    def test_foundation_model_not_cached_shows_unavailable(self, mock_cache):
        """Foundation model blocks with missing artifacts should show available=False."""
        from inference.core.interfaces.http.handlers.workflows import (
            enrich_with_air_gapped_info,
        )

        blocks = [
            _make_block_description(
                FoundationModelManifest, "test/foundation_model@v1"
            ),
        ]
        result = _make_result(blocks)

        enriched = enrich_with_air_gapped_info(result)

        info = enriched.blocks[0].block_schema["json_schema_extra"]["air_gapped_info"]
        assert info["available"] is False
        assert info["reason"] == "missing_cache_artifacts"


class TestListFormatFoundationModel:
    """Tests for blocks using the new list-format get_required_cache_artifacts."""

    def test_list_format_cached_variant(self, tmp_path):
        """List-format block with a cached variant directory should be available."""
        import os

        from inference.core.interfaces.http.handlers.workflows import (
            _get_air_gapped_info_for_block,
        )

        cache = str(tmp_path)
        variant_dir = os.path.join(cache, "clip", "ViT-B-32")
        os.makedirs(variant_dir, exist_ok=True)
        open(os.path.join(variant_dir, "visual.onnx"), "w").close()

        with patch(
            "inference.core.cache.air_gapped.MODEL_CACHE_DIR",
            cache,
        ):
            info = _get_air_gapped_info_for_block(ListFormatFoundationManifest)

        assert info["available"] is True
        assert info["model_id"] == "clip/RN50"
        assert "embedding" in info["compatible_task_types"]

    def test_list_format_no_cached_variant(self, tmp_path):
        """List-format block with no cached variants should be unavailable."""
        from inference.core.interfaces.http.handlers.workflows import (
            _get_air_gapped_info_for_block,
        )

        cache = str(tmp_path)

        with patch(
            "inference.core.cache.air_gapped.MODEL_CACHE_DIR",
            cache,
        ):
            info = _get_air_gapped_info_for_block(ListFormatFoundationManifest)

        assert info["available"] is False
        assert info["reason"] == "missing_cache_artifacts"
