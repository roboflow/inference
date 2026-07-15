"""Guards that the tensor-native VLM formatter blocks read image H/W without
forcing a device->host materialisation of a tensor-source image.

Previously these blocks called ``image.numpy_image.shape[:2]`` purely to read
(H, W). On a tensor-source ``WorkflowImageData`` that eagerly downloads the whole
frame to host (measured ~3-4 ms for 720p on a Jetson Orin Nano) and leaves a
numpy cache behind. They now use ``_read_shape_without_materialization()``, which
reads the shape straight off the already-present representation.

Each test drives a tensor-source image (CPU torch tensor - no CUDA required)
through a formatter and asserts BOTH:
  * the H/W-dependent output is correct (image_dimensions metadata, and for the
    coordinate-scaling parsers, the scaled xyxy), and
  * the image's internal numpy cache (``_numpy_image``) stays ``None`` - i.e. no
    materialisation happened as a side effect of the shape read.

Non-square dimensions (H=192, W=168) are used throughout so an accidental H/W
swap would flip both the dimensions metadata and the scaled coordinates.
"""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("inference_models")

import torch

from inference.core.workflows.core_steps.formatters.vlm_as_classifier.v1_tensor import (
    VLMAsClassifierBlockV1,
)
from inference.core.workflows.core_steps.formatters.vlm_as_classifier.v2_tensor import (
    VLMAsClassifierBlockV2,
)
from inference.core.workflows.core_steps.formatters.vlm_as_detector.v1_tensor import (
    VLMAsDetectorBlockV1,
)
from inference.core.workflows.core_steps.formatters.vlm_as_detector.v2_tensor import (
    VLMAsDetectorBlockV2,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)

IMAGE_HEIGHT = 192
IMAGE_WIDTH = 168


def _tensor_source_image() -> WorkflowImageData:
    # CHW RGB uint8 tensor source - constructing with `tensor_image` leaves the
    # numpy representation unmaterialised (`_numpy_image is None`).
    tensor_image = torch.zeros((3, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=torch.uint8)
    image = WorkflowImageData(
        tensor_image=tensor_image,
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )
    # Sanity: the source really is tensor-only before the block runs.
    assert image._numpy_image is None
    assert image.is_tensor_materialised() is True
    return image


def _assert_not_materialised(image: WorkflowImageData) -> None:
    assert image._numpy_image is None, (
        "Reading H/W must not materialise the numpy representation of a "
        "tensor-source image (forces a full-frame device->host download)."
    )


# --------------------------------------------------------------------------- #
# vlm_as_classifier
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("block_cls", [VLMAsClassifierBlockV1, VLMAsClassifierBlockV2])
def test_classifier_multi_class_reads_shape_without_materialization(
    block_cls,
) -> None:
    # given
    image = _tensor_source_image()
    vlm_output = '{"class_name": "cat", "confidence": 0.9}'

    # when
    result = block_cls().run(image=image, vlm_output=vlm_output, classes=["cat", "dog"])

    # then
    assert result["error_status"] is False
    dimensions = result["predictions"].images_metadata[0]["image_dimensions"]
    assert list(dimensions) == [IMAGE_HEIGHT, IMAGE_WIDTH]
    _assert_not_materialised(image)


@pytest.mark.parametrize("block_cls", [VLMAsClassifierBlockV1, VLMAsClassifierBlockV2])
def test_classifier_multi_label_reads_shape_without_materialization(
    block_cls,
) -> None:
    # given
    image = _tensor_source_image()
    vlm_output = (
        '{"predicted_classes": ['
        '{"class": "cat", "confidence": 0.8}, '
        '{"class": "dog", "confidence": 0.6}]}'
    )

    # when
    result = block_cls().run(image=image, vlm_output=vlm_output, classes=["cat", "dog"])

    # then
    assert result["error_status"] is False
    # MultiLabelClassificationPrediction exposes a single `image_metadata` dict
    # (multi-class uses a per-image `images_metadata` list).
    dimensions = result["predictions"].image_metadata["image_dimensions"]
    assert list(dimensions) == [IMAGE_HEIGHT, IMAGE_WIDTH]
    _assert_not_materialised(image)


# --------------------------------------------------------------------------- #
# vlm_as_detector
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("block_cls", [VLMAsDetectorBlockV1, VLMAsDetectorBlockV2])
def test_detector_gemini_scales_coords_without_materialization(block_cls) -> None:
    # given - gemini `box_2d` is [y_min, x_min, y_max, x_max] normalised to 1000,
    # so the resulting pixel box is a direct function of (H, W).
    image = _tensor_source_image()
    vlm_output = '{"detections": [{"box_2d": [100, 200, 500, 800], "label": "cat"}]}'

    # when
    result = block_cls().run(
        image=image,
        vlm_output=vlm_output,
        classes=["cat", "dog"],
        model_type="google-gemini",
        task_type="object-detection",
    )

    # then
    assert result["error_status"] is False
    xyxy = result["predictions"].xyxy.cpu().numpy()
    # x_min = 200/1000*168 = 33.6 -> 34 ; y_min = 100/1000*192 = 19.2 -> 19
    # x_max = 800/1000*168 = 134.4 -> 134 ; y_max = 500/1000*192 = 96.0 -> 96
    assert np.array_equal(xyxy, np.array([[34.0, 19.0, 134.0, 96.0]]))
    dimensions = result["predictions"].image_metadata["image_dimensions"]
    assert list(dimensions) == [IMAGE_HEIGHT, IMAGE_WIDTH]
    _assert_not_materialised(image)


def test_detector_v2_llm_scales_coords_without_materialization() -> None:
    # given - the openai/claude LLM parser (v2 only) multiplies normalised
    # [0, 1] coords by (W, H); exercises the second changed site in v2_tensor.
    image = _tensor_source_image()
    vlm_output = (
        '{"detections": [{"x_min": 0.1, "y_min": 0.2, "x_max": 0.5, '
        '"y_max": 0.8, "class_name": "cat"}]}'
    )

    # when
    result = VLMAsDetectorBlockV2().run(
        image=image,
        vlm_output=vlm_output,
        classes=["cat", "dog"],
        model_type="openai",
        task_type="object-detection",
    )

    # then
    assert result["error_status"] is False
    xyxy = result["predictions"].xyxy.cpu().numpy()
    # x_min = 0.1*168 = 16.8 -> 17 ; y_min = 0.2*192 = 38.4 -> 38
    # x_max = 0.5*168 = 84.0 -> 84 ; y_max = 0.8*192 = 153.6 -> 154
    assert np.array_equal(xyxy, np.array([[17.0, 38.0, 84.0, 154.0]]))
    dimensions = result["predictions"].image_metadata["image_dimensions"]
    assert list(dimensions) == [IMAGE_HEIGHT, IMAGE_WIDTH]
    _assert_not_materialised(image)


@pytest.mark.parametrize("block_cls", [VLMAsDetectorBlockV1, VLMAsDetectorBlockV2])
def test_detector_florence_reads_shape_without_materialization(block_cls) -> None:
    # given - the florence-2 parser passes (W, H) as resolution_wh to
    # sv.Detections.from_lmm; exercises the florence changed site in each block.
    image = _tensor_source_image()
    vlm_output = '{"bboxes": [[10.0, 20.0, 90.0, 120.0]], "bboxes_labels": ["cat"]}'

    # when
    result = block_cls().run(
        image=image,
        vlm_output=vlm_output,
        classes=["cat", "dog"],
        model_type="florence-2",
        task_type="open-vocabulary-object-detection",
    )

    # then
    assert result["error_status"] is False
    dimensions = result["predictions"].image_metadata["image_dimensions"]
    assert list(dimensions) == [IMAGE_HEIGHT, IMAGE_WIDTH]
    _assert_not_materialised(image)
