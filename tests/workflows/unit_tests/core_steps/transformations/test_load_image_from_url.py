import numpy as np
import pytest
from pydantic import ValidationError
from unittest.mock import patch

from inference.core.workflows.core_steps.transformations.load_image_from_url.v1 import (
    BlockManifest,
    LoadImageFromUrlBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize("type_alias", ["roboflow_core/load_image_from_url@v1"])
@pytest.mark.parametrize("url_input", ["https://example.com/image.jpg", "$inputs.image_url"])
@pytest.mark.parametrize("cache_input", [True, False, "$inputs.cache_enabled"])
def test_load_image_from_url_manifest_validation_when_valid_input_given(
    type_alias: str, url_input: str, cache_input
) -> None:
    # given
    raw_manifest = {
        "type": type_alias,
        "name": "load_image",
        "url": url_input,
        "cache": cache_input,
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result == BlockManifest(
        name="load_image",
        type=type_alias,
        url=url_input,
        cache=cache_input,
    )


@pytest.mark.parametrize("field_to_delete", ["type", "name", "url"])
def test_load_image_from_url_manifest_validation_when_required_field_missing(
    field_to_delete: str,
) -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/load_image_from_url@v1",
        "name": "load_image",
        "url": "https://example.com/image.jpg",
        "cache": True,
    }
    del raw_manifest[field_to_delete]

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(raw_manifest)


def test_load_image_from_url_manifest_validation_with_default_cache() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/load_image_from_url@v1",
        "name": "load_image",
        "url": "https://example.com/image.jpg",
        # cache field omitted - should default to True
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result.cache is True


@patch("inference.core.workflows.core_steps.transformations.load_image_from_url.v1.load_image_from_url")
def test_load_image_from_url_block_run_success(mock_load_image_from_url) -> None:
    # given
    test_url = "https://www.peta.org/wp-content/uploads/2023/05/wild-raccoon.jpg"
    mock_numpy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_load_image_from_url.return_value = mock_numpy_image
    
    block = LoadImageFromUrlBlockV1()

    # when
    result = block.run(url=test_url, cache=True)

    # then
    assert "image" in result
    assert isinstance(result["image"], WorkflowImageData)
    assert np.array_equal(result["image"].numpy_image, mock_numpy_image)
    assert isinstance(result["image"].parent_metadata, ImageParentMetadata)
    assert result["image"].parent_metadata.parent_id is not None
    
    # Verify the underlying function was called with correct parameters
    mock_load_image_from_url.assert_called_once_with(value=test_url)


@patch("inference.core.workflows.core_steps.transformations.load_image_from_url.v1.load_image_from_url")
def test_load_image_from_url_block_run_caching_behavior(mock_load_image_from_url) -> None:
    # given
    test_url = "https://example.com/cached-image.jpg"
    mock_numpy_image = np.zeros((50, 50, 3), dtype=np.uint8)
    mock_load_image_from_url.return_value = mock_numpy_image
    
    block = LoadImageFromUrlBlockV1()

    # when - first call should load the image
    result1 = block.run(url=test_url, cache=True)
    
    # when - second call with same URL should use cache
    result2 = block.run(url=test_url, cache=True)

    # then
    assert "image" in result1
    assert "image" in result2
    
    # Both results should have identical image data
    assert np.array_equal(result1["image"].numpy_image, result2["image"].numpy_image)
    
    # The underlying function should only be called once due to caching
    mock_load_image_from_url.assert_called_once_with(value=test_url)


@patch("inference.core.workflows.core_steps.transformations.load_image_from_url.v1.load_image_from_url")
def test_load_image_from_url_block_run_error_handling(mock_load_image_from_url) -> None:
    # given
    test_url = "https://nonexistent.example.com/image.jpg"
    mock_load_image_from_url.side_effect = Exception("Could not load image from url")
    
    block = LoadImageFromUrlBlockV1()

    # when/then
    with pytest.raises(RuntimeError) as exc_info:
        block.run(url=test_url, cache=False)
    
    assert "Failed to load image from URL" in str(exc_info.value)
    assert test_url in str(exc_info.value)
    mock_load_image_from_url.assert_called_once_with(value=test_url)


def test_load_image_from_url_block_manifest_outputs() -> None:
    # given/when
    outputs = BlockManifest.describe_outputs()
    
    # then
    assert len(outputs) == 1
    assert outputs[0].name == "image"
    assert "image" in [kind.name for kind in outputs[0].kind]


def test_load_image_from_url_block_compatibility() -> None:
    # given/when
    compatibility = BlockManifest.get_execution_engine_compatibility()
    
    # then
    assert compatibility == ">=1.0.0,<2.0.0"


# Tests for Requirement 4: URL validation at runtime
@patch("inference.core.workflows.core_steps.transformations.load_image_from_url.v1.load_image_from_url")
def test_load_image_from_url_block_validates_invalid_url_format_at_runtime(mock_load_image_from_url) -> None:
    # given
    invalid_url = "not-a-valid-url"
    mock_load_image_from_url.side_effect = Exception("Providing images via non https:// URL is not supported")
    
    block = LoadImageFromUrlBlockV1()

    # when/then
    with pytest.raises(RuntimeError) as exc_info:
        block.run(url=invalid_url, cache=False)
    
    assert "Failed to load image from URL" in str(exc_info.value)
    assert invalid_url in str(exc_info.value)
    mock_load_image_from_url.assert_called_once_with(value=invalid_url)


# Tests for Requirement 5: Image extension validation
@patch("inference.core.workflows.core_steps.transformations.load_image_from_url.v1.load_image_from_url")
def test_load_image_from_url_block_validates_non_image_extension_at_runtime(mock_load_image_from_url) -> None:
    # given
    non_image_url = "https://example.com/document.pdf"
    mock_load_image_from_url.side_effect = Exception("Could not decode bytes as image")
    
    block = LoadImageFromUrlBlockV1()

    # when/then
    with pytest.raises(RuntimeError) as exc_info:
        block.run(url=non_image_url, cache=False)
    
    assert "Failed to load image from URL" in str(exc_info.value)
    assert non_image_url in str(exc_info.value)
    mock_load_image_from_url.assert_called_once_with(value=non_image_url)
