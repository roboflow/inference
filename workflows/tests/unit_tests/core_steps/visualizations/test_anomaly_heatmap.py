import numpy as np
import pytest
import torch
from pydantic import ValidationError
from roboflow_workflows.core_steps.visualizations.anomaly_heatmap.v1 import (
    AnomalyHeatmapManifest,
    AnomalyHeatmapVisualizationBlockV1,
)
from roboflow_workflows.core_steps.visualizations.anomaly_heatmap.v1_tensor import (
    AnomalyHeatmapManifest as TensorManifest,
)
from roboflow_workflows.core_steps.visualizations.anomaly_heatmap.v1_tensor import (
    AnomalyHeatmapVisualizationBlockV1 as TensorBlock,
)
from roboflow_workflows.core_steps.visualizations.common.annotators.anomaly_heatmap import (
    AnomalyHeatmapAnnotator,
    colorize_anomaly_map,
    normalize_anomaly_map,
)
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)

TYPE = "roboflow_core/anomaly_heatmap_visualization@v1"


@pytest.mark.parametrize("manifest_class", [AnomalyHeatmapManifest, TensorManifest])
@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_manifest_validation_when_valid_manifest_is_given(
    manifest_class, images_field_alias: str
) -> None:
    # given
    data = {
        "type": TYPE,
        "name": "heatmap",
        images_field_alias: "$inputs.image",
        "anomaly_map": "$steps.anomaly.anomaly_map",
        "anomaly_threshold": "$steps.anomaly.anomaly_threshold",
        "color_map": "JET",
        "opacity": 0.5,
    }

    # when
    result = manifest_class.model_validate(data)

    # then
    assert result == manifest_class(
        type=TYPE,
        name="heatmap",
        images="$inputs.image",
        anomaly_map="$steps.anomaly.anomaly_map",
        anomaly_threshold="$steps.anomaly.anomaly_threshold",
        color_map="JET",
        opacity=0.5,
    )
    assert result.copy_image is True


def test_manifest_defaults_and_literal_threshold() -> None:
    result = AnomalyHeatmapManifest.model_validate(
        {
            "type": TYPE,
            "name": "heatmap",
            "image": "$inputs.image",
            "anomaly_map": "$steps.anomaly.anomaly_map",
            "anomaly_threshold": 1.5,
        }
    )
    assert result.color_map == "ROBOFLOW"
    assert result.opacity == 0.75
    assert result.anomaly_threshold == 1.5


def test_sibling_manifests_declare_the_same_surface() -> None:
    assert set(AnomalyHeatmapManifest.model_fields) == set(TensorManifest.model_fields)
    assert [o.name for o in AnomalyHeatmapManifest.describe_outputs()] == [
        o.name for o in TensorManifest.describe_outputs()
    ]


@pytest.mark.parametrize(
    "field, value",
    [
        ("anomaly_map", "not a selector"),
        ("anomaly_threshold", "high"),
        ("color_map", "RAINBOW"),
        ("opacity", 1.5),
    ],
)
def test_manifest_validation_when_field_is_invalid(field: str, value) -> None:
    data = {
        "type": TYPE,
        "name": "heatmap",
        "image": "$inputs.image",
        "anomaly_map": "$steps.anomaly.anomaly_map",
        "anomaly_threshold": 1.5,
    }
    data[field] = value
    with pytest.raises(ValidationError):
        AnomalyHeatmapManifest.model_validate(data)


@pytest.mark.parametrize("field", ["anomaly_map", "anomaly_threshold"])
def test_manifest_validation_when_required_field_is_missing(field: str) -> None:
    data = {
        "type": TYPE,
        "name": "heatmap",
        "image": "$inputs.image",
        "anomaly_map": "$steps.anomaly.anomaly_map",
        "anomaly_threshold": 1.5,
    }
    del data[field]
    with pytest.raises(ValidationError):
        AnomalyHeatmapManifest.model_validate(data)


def test_normalize_anomaly_map_anchors_the_threshold_at_half() -> None:
    # given
    values = np.array([[-1.0, 0.0], [2.0, 6.0]], dtype=np.float32)

    # when
    normalized = normalize_anomaly_map(values, threshold=2.0)

    # then
    np.testing.assert_allclose(normalized, [[0.0, 0.0], [0.5, 0.75]])
    with pytest.raises(ValueError):
        normalize_anomaly_map(values, threshold=0.0)


def test_roboflow_ramp_hits_its_stops() -> None:
    # given
    normalized = np.array([[0.0, 0.25, 0.5, 1.0]], dtype=np.float32)

    # when
    bgr, alpha = colorize_anomaly_map(normalized, color_map="ROBOFLOW")

    # then: stops are (255,214,0,a0), (255,214,0,.55), (255,122,0,.85), (220,30,30,1)
    assert bgr.tolist() == [
        [[0, 214, 255], [0, 214, 255], [0, 122, 255], [30, 30, 220]]
    ]
    np.testing.assert_allclose(alpha, [[0.0, 0.55, 0.85, 1.0]])


def test_opencv_ramp_is_opaque() -> None:
    bgr, alpha = colorize_anomaly_map(np.full((2, 2), 0.5, dtype=np.float32), "JET")
    assert bgr.shape == (2, 2, 3) and bgr.dtype == np.uint8
    np.testing.assert_array_equal(alpha, np.ones((2, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        colorize_anomaly_map(np.zeros((2, 2), dtype=np.float32), "RAINBOW")


def _scene(height: int = 8, width: int = 8) -> np.ndarray:
    return np.full((height, width, 3), 100, dtype=np.uint8)


def test_annotator_blends_the_threshold_colour_at_the_ramp_midpoint() -> None:
    # given: a map at exactly the threshold everywhere, stretched 2x2 -> 8x8
    annotator = AnomalyHeatmapAnnotator(color_map="ROBOFLOW", opacity=1.0)
    scene = _scene()

    # when
    annotated = annotator.annotate(scene, np.full((2, 2), 2.0, dtype=np.float32), 2.0)

    # then: 100*(1-0.85) + (0,122,255)*0.85 per BGR channel
    assert annotated is scene
    assert annotated.shape == (8, 8, 3)
    expected = np.rint(100 * 0.15 + np.array([0, 122, 255]) * 0.85).astype(np.uint8)
    assert annotated[4, 4].tolist() == expected.tolist()


def test_annotator_leaves_zero_evidence_untouched_and_scales_by_opacity() -> None:
    # given
    annotator = AnomalyHeatmapAnnotator(color_map="ROBOFLOW", opacity=0.5)
    scene = _scene()
    anomaly_map = np.zeros((2, 2), dtype=np.float32)
    anomaly_map[1, 1] = 1e9  # saturates the ramp

    # when
    annotated = annotator.annotate(scene, anomaly_map, 2.0)

    # then
    assert annotated[0, 0].tolist() == [100, 100, 100]
    expected = np.rint(100 * 0.5 + np.array([30, 30, 220]) * 0.5).astype(np.uint8)
    assert annotated[7, 7].tolist() == expected.tolist()


def test_annotator_promotes_grayscale_scenes_and_rejects_bad_maps() -> None:
    annotator = AnomalyHeatmapAnnotator(color_map="HOT", opacity=0.5)
    annotated = annotator.annotate(
        np.zeros((4, 4), dtype=np.uint8), np.ones((2, 2), dtype=np.float32), 1.0
    )
    assert annotated.shape == (4, 4, 3)
    with pytest.raises(ValueError):
        annotator.annotate(_scene(), np.ones((2, 2, 2), dtype=np.float32), 1.0)
    with pytest.raises(ValueError):
        AnomalyHeatmapAnnotator(color_map="RAINBOW", opacity=0.5)


def _image(scene: np.ndarray) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="img"), numpy_image=scene
    )


@pytest.mark.parametrize(
    "block_class", [AnomalyHeatmapVisualizationBlockV1, TensorBlock]
)
def test_block_copies_the_input_by_default(block_class) -> None:
    # given
    block = block_class()
    scene = _scene()
    image = _image(scene)

    # when
    result = block.run(
        image=image,
        copy_image=True,
        anomaly_map=np.full((2, 2), 2.0, dtype=np.float32),
        anomaly_threshold=2.0,
        color_map="ROBOFLOW",
        opacity=1.0,
    )

    # then
    output = result["image"]
    assert output.numpy_image.shape == (8, 8, 3)
    assert output.parent_metadata.parent_id == "img"
    assert scene[4, 4].tolist() == [100, 100, 100]
    assert output.numpy_image[4, 4].tolist() != [100, 100, 100]
    assert output.numpy_image is not scene


@pytest.mark.parametrize(
    "block_class", [AnomalyHeatmapVisualizationBlockV1, TensorBlock]
)
def test_block_mutates_the_input_when_asked(block_class) -> None:
    # given
    block = block_class()
    scene = _scene()

    # when
    result = block.run(
        image=_image(scene),
        copy_image=False,
        anomaly_map=np.full((2, 2), 2.0, dtype=np.float32),
        anomaly_threshold=2.0,
        color_map="JET",
        opacity=0.75,
    )

    # then
    assert result["image"].numpy_image is scene
    assert scene[4, 4].tolist() != [100, 100, 100]


@pytest.mark.parametrize(
    "block_class", [AnomalyHeatmapVisualizationBlockV1, TensorBlock]
)
def test_block_requires_a_map(block_class) -> None:
    with pytest.raises(ValueError, match="include_anomaly_map"):
        block_class().run(
            image=_image(_scene()),
            copy_image=True,
            anomaly_map=None,
            anomaly_threshold=2.0,
            color_map="ROBOFLOW",
            opacity=1.0,
        )


def test_tensor_block_accepts_a_torch_map() -> None:
    # given
    scene = _scene()

    # when
    result = TensorBlock().run(
        image=_image(scene),
        copy_image=True,
        anomaly_map=torch.full((2, 2), 2.0),
        anomaly_threshold=2.0,
        color_map="ROBOFLOW",
        opacity=1.0,
    )

    # then
    expected = np.rint(100 * 0.15 + np.array([0, 122, 255]) * 0.85).astype(np.uint8)
    assert result["image"].numpy_image[4, 4].tolist() == expected.tolist()


def test_annotator_is_cached_per_style() -> None:
    block = AnomalyHeatmapVisualizationBlockV1()
    first = block.getAnnotator("ROBOFLOW", 0.5)
    assert block.getAnnotator("ROBOFLOW", 0.5) is first
    assert block.getAnnotator("JET", 0.5) is not first
