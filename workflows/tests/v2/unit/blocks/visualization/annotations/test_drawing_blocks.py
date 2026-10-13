"""Box, label, keypoint and classification-label drawing blocks.

Expected pixels come from the latest V1 block run on the same data (an exact
oracle: the ports promise V1 pixels for these four families) or from a fact
computed here without the code under test (grid cell counts, unchanged
pixels). V1 test modules are not imported.

    family                 V1 oracle
    bounding_box           roboflow_core/bounding_box_visualization@v1
    label                  roboflow_core/label_visualization@v2
    keypoint               roboflow_core/keypoint_visualization@v1
    classification_label   roboflow_core/classification_label_visualization@v1

``mask_quality`` follows decision 027: ``accurate`` (default) is the provider
reference, ``coarse`` the thresholded grid cells; dense masks draw the same
either way.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from roboflow_workflows.core_steps.visualizations.bounding_box.v1 import (
    BoundingBoxVisualizationBlockV1,
)
from roboflow_workflows.core_steps.visualizations.classification_label.v1 import (
    ClassificationLabelVisualizationBlockV1,
)
from roboflow_workflows.core_steps.visualizations.keypoint.v1 import (
    KeypointVisualizationBlockV1,
)
from roboflow_workflows.core_steps.visualizations.label.v2 import (
    LabelVisualizationBlockV2,
)
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.visualization.annotations import (
    BoundingBoxVisualizationBlock,
    ClassificationLabelVisualizationBlock,
    KeypointVisualizationBlock,
    LabelVisualizationBlock,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.row_identity import RowRef
from roboflow_workflows.v2.engine.declaration import spec_of
from roboflow_workflows.v2.engine.errors import ParamsValidationError

from inference_models.models.base.classification import (
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from tests.v2.unit.blocks.visualization.annotations.support import (
    CLASS_NAMES,
    OBJECTS,
    V1_COLOR_DEFAULTS,
    classification,
    detections,
    grid_view,
    image_of,
    instance_detections,
    key_points,
    metadata_for,
    multi_label,
    object_masks,
    pixels_of,
    rle_detections,
    run_v1,
    run_v2,
    scene,
    v1_detections,
)

SMALL_TEXT = {
    "text_color": "WHITE",
    "text_size_mode": "Manual",
    "text_scale": 0.4,
    "text_thickness": 1,
    "text_padding": 3,
    "border_radius": 2,
}
KEYPOINT_STYLE = {
    "color": "#A351FB",
    "text_color": "black",
    "text_scale": 0.5,
    "text_thickness": 1,
    "text_padding": 3,
    "thickness": 2,
    "radius": 3,
}
CLASSIFICATION_STYLE = {
    "text_color": "WHITE",
    "text_scale": 0.4,
    "text_thickness": 1,
    "text_padding": 2,
    "border_radius": 0,
}
# Box corners on quarter pixels: exact in float32, so crop offsets add exactly
# and the integer truncation V1 applies is reproduced bit for bit.
QUARTER_OBJECTS = (
    (4.5, 6.25, 30.5, 30.75, 0, 0.91),
    (20.0, 14.0, 52.75, 44.0, 1, 0.5),
    (40.25, 2.0, 60.0, 20.5, 2, 0.333),
)
CROP_XYXY = (10, 8, 50, 40)


# -- helpers ------------------------------------------------------------------


def _input_image(
    *, image_id: str = "frame", seed: int = 0
) -> Tuple[np.ndarray, ImageData]:
    pixels = scene(seed=seed)
    image = image_of(pixels, image_id=image_id)

    return pixels, image


def _assert_same_pixels(actual: np.ndarray, expected: np.ndarray) -> None:
    changed = np.abs(actual.astype(int) - expected.astype(int)).max(axis=-1) > 0
    assert not changed.any(), (
        f"{int(changed.sum())} pixels differ from the oracle; first at (y, x)="
        f"{tuple(int(v) for v in np.argwhere(changed)[0])}"
    )


def _v1_keypoints(points: KeyPoints, objects: Sequence[Tuple] = OBJECTS[:2]) -> Any:
    count = points.xy.shape[1]
    predictions = v1_detections(
        objects,
        data={
            "keypoints_xy": points.xy.numpy(),
            "keypoints_confidence": points.confidence.numpy(),
            "keypoints_class_name": np.array(
                [[str(index) for index in range(count)]] * len(objects), dtype=object
            ),
        },
    )

    return predictions


def _v1_single(width: int, height: int) -> Dict[str, Any]:
    # The V1 form of support.classification: person wins at 0.7.
    prediction = {
        "image": {"width": width, "height": height},
        "predictions": [
            {"class": "car", "class_id": 0, "confidence": 0.2},
            {"class": "person", "class_id": 1, "confidence": 0.7},
            {"class": "dog", "class_id": 2, "confidence": 0.1},
        ],
    }

    return prediction


def _v1_multi(width: int, height: int) -> Dict[str, Any]:
    # The V1 form of support.multi_label: car and dog predicted.
    prediction = {
        "image": {"width": width, "height": height},
        "predictions": {
            "car": {"class_id": 0, "confidence": 0.61},
            "person": {"class_id": 1, "confidence": 0.2},
            "dog": {"class_id": 2, "confidence": 0.83},
        },
        "predicted_classes": ["car", "dog"],
    }

    return prediction


def _shifted(objects: Sequence[Tuple], *, dx: float, dy: float) -> List[Tuple]:
    moved = [
        (x1 + dx, y1 + dy, x2 + dx, y2 + dy, *rest) for x1, y1, x2, y2, *rest in objects
    ]

    return moved


def _run_selected(
    block_class: type,
    *,
    image: ImageData,
    predictions: Any,
    selected: Dict[str, Any],
    **literals: Any,
) -> ImageData:
    # Validate with "$inputs.<name>" selectors, then run with the values, as
    # the engine does when the inputs arrive.
    selectors = {name: f"$inputs.{name}" for name in selected}
    params = spec_of(block_class).validate_params(
        {
            **literals,
            **selectors,
            "image": "$inputs.image",
            "predictions": "$inputs.predictions",
        }
    )
    arguments = {**dict(params), **selected, "image": image, "predictions": predictions}
    result = block_class().run(**arguments)

    return result["image"]


def _coarse_masks(stride: int = 4) -> np.ndarray:
    # support.grid_view scores a cell 16 * covered_fraction - 8 with threshold 0,
    # and the data layer documents foreground as score > threshold: a cell is
    # in the coarse mask when more than half its pixels are covered (two cells
    # of this fixture are exactly half covered and stay out).
    covered = F.avg_pool2d(
        torch.from_numpy(object_masks()).float().unsqueeze(1), stride
    )
    cells = covered.squeeze(1).numpy() > 0.5
    masks = np.kron(cells, np.ones((stride, stride), dtype=bool)).astype(bool)

    return masks


# -- bounding box -------------------------------------------------------------


@pytest.mark.parametrize("color_axis", ["CLASS", "INDEX"])
@pytest.mark.parametrize(
    ("thickness", "roundness"), [(2, 0.0), (1, 0.0), (4, 0.0), (2, 0.5), (3, 1.0)]
)
def test_box_outlines_equal_v1(
    thickness: int, roundness: float, color_axis: str
) -> None:
    pixels, image = _input_image()
    colors = {**V1_COLOR_DEFAULTS, "color_axis": color_axis}

    drawn = run_v2(
        BoundingBoxVisualizationBlock,
        image=image,
        predictions=detections(image),
        thickness=thickness,
        roundness=roundness,
        **colors,
    )

    expected = run_v1(
        BoundingBoxVisualizationBlockV1,
        pixels,
        predictions=v1_detections(),
        thickness=thickness,
        roundness=roundness,
        **colors,
    )
    _assert_same_pixels(pixels_of(drawn), expected)


@pytest.mark.parametrize("palette", ["ROBOFLOW", "Matplotlib Viridis", "CUSTOM"])
def test_box_palettes_equal_v1(palette: str) -> None:
    pixels, image = _input_image()
    colors = {
        **V1_COLOR_DEFAULTS,
        "color_palette": palette,
        "palette_size": 4,
        "custom_colors": ["#FF0000", "rgb(0, 255, 0)"] if palette == "CUSTOM" else [],
    }

    drawn = run_v2(
        BoundingBoxVisualizationBlock,
        image=image,
        predictions=detections(image),
        **colors,
    )

    expected = run_v1(
        BoundingBoxVisualizationBlockV1,
        pixels,
        predictions=v1_detections(),
        thickness=2,
        roundness=0.0,
        **colors,
    )
    _assert_same_pixels(pixels_of(drawn), expected)


def test_box_track_axis_colors_rows_by_tracker_id_like_v1() -> None:
    pixels, image = _input_image()
    rows = [{"tracker_id": 5}, {"tracker_id": 1}, {"tracker_id": 5}]
    colors = {**V1_COLOR_DEFAULTS, "color_axis": "TRACK"}

    drawn = run_v2(
        BoundingBoxVisualizationBlock,
        image=image,
        predictions=detections(image, rows=rows),
        **colors,
    )

    expected = run_v1(
        BoundingBoxVisualizationBlockV1,
        pixels,
        predictions=v1_detections(tracker_id=[5, 1, 5]),
        thickness=2,
        roundness=0.0,
        **colors,
    )
    _assert_same_pixels(pixels_of(drawn), expected)


def test_box_track_axis_with_a_row_missing_its_tracker_id_is_an_error() -> None:
    _, image = _input_image()
    rows = [{"tracker_id": 5}, {}, {"tracker_id": 2}]

    with pytest.raises(ValueError, match="(?i)track"):
        run_v2(
            BoundingBoxVisualizationBlock,
            image=image,
            predictions=detections(image, rows=rows),
            color_axis="TRACK",
        )


@pytest.mark.parametrize(
    "carrier", ["instance_dense", "instance_rle", "grid_view", "keypoints_with_boxes"]
)
def test_box_block_draws_the_boxes_of_every_box_carrier(carrier: str) -> None:
    _, image = _input_image()
    view = grid_view(image)
    points, boxes = key_points(image)
    carriers = {
        "instance_dense": instance_detections(image),
        "instance_rle": rle_detections(image),
        "grid_view": view,
        "keypoints_with_boxes": (points, boxes),
    }
    reference = detections(
        image, OBJECTS[:2] if carrier == "keypoints_with_boxes" else OBJECTS
    )

    drawn = run_v2(
        BoundingBoxVisualizationBlock, image=image, predictions=carriers[carrier]
    )

    expected = run_v2(BoundingBoxVisualizationBlock, image=image, predictions=reference)
    _assert_same_pixels(pixels_of(drawn), pixels_of(expected))
    assert (
        not view.is_full_res_materialized()
    ), "box drawing must not upscale a mask view"


# -- label: texts -------------------------------------------------------------


LABEL_ROWS = [
    {"tracker_id": 7, "time_in_zone": 3.14159, "area_px": 120.0, "area_converted": 1.5},
    {"tracker_id": 3, "area_px": None},
    {"tracker_id": 11, "time_in_zone": 0.5, "area_px": 88.25, "area_converted": 0.125},
]
V1_LABEL_DATA = {
    "time_in_zone": np.array([3.14159, None, 0.5], dtype=object),
    "area_px": np.array([120.0, None, 88.25], dtype=object),
    "area_converted": np.array([1.5, None, 0.125], dtype=object),
}


@pytest.mark.parametrize(
    "position", ["TOP_LEFT", "CENTER", "BOTTOM_RIGHT", "TOP_CENTER"]
)
@pytest.mark.parametrize(
    "text",
    [
        "Class",
        "Confidence",
        "Class and Confidence",
        "Index",
        "Dimensions",
        "Area",
        "Area (mask)",
        "Area (converted)",
        "Tracker Id",
        "Time In Zone",
    ],
)
def test_label_texts_on_boxes_equal_v1(text: str, position: str) -> None:
    # Row 1 lacks time_in_zone and area values: V1 prints its "N/A" forms.
    pixels, image = _input_image()
    style = {**SMALL_TEXT, "text": text, "text_position": position}

    drawn = run_v2(
        LabelVisualizationBlock,
        image=image,
        predictions=detections(image, rows=LABEL_ROWS),
        **style,
    )

    expected = run_v1(
        LabelVisualizationBlockV2,
        pixels,
        predictions=v1_detections(tracker_id=[7, 3, 11], data=V1_LABEL_DATA),
        **style,
        **V1_COLOR_DEFAULTS,
    )
    _assert_same_pixels(pixels_of(drawn), expected)


def test_label_tracker_id_text_without_tracker_ids_says_so_like_v1() -> None:
    pixels, image = _input_image()
    style = {**SMALL_TEXT, "text": "Tracker Id", "text_position": "TOP_LEFT"}

    drawn = run_v2(
        LabelVisualizationBlock, image=image, predictions=detections(image), **style
    )

    expected = run_v1(
        LabelVisualizationBlockV2,
        pixels,
        predictions=v1_detections(),
        **style,
        **V1_COLOR_DEFAULTS,
    )
    _assert_same_pixels(pixels_of(drawn), expected)


@pytest.mark.parametrize("scale", [1.0, 30.0])
def test_label_automatic_text_size_equals_v1(scale: float) -> None:
    # 48 px images: scale 1.0 clamps to V1's minimum, 30.0 lands inside the range.
    pixels, image = _input_image()
    style = {
        **SMALL_TEXT,
        "text": "Class",
        "text_position": "TOP_LEFT",
        "text_size_mode": "Automatic",
        "text_scale": scale,
    }

    drawn = run_v2(
        LabelVisualizationBlock, image=image, predictions=detections(image), **style
    )

    expected = run_v1(
        LabelVisualizationBlockV2,
        pixels,
        predictions=v1_detections(),
        **style,
        **V1_COLOR_DEFAULTS,
    )
    _assert_same_pixels(pixels_of(drawn), expected)


# -- label: class names (decision 025) ----------------------------------------


@pytest.mark.parametrize("text", ["Class", "Class and Confidence"])
def test_rows_of_one_class_id_keep_their_own_names_over_the_class_map(
    text: str,
) -> None:
    # Two rows share class id 0 ("car" in the class map) but carry their own
    # names; the third row has none and falls back to the map.
    pixels, image = _input_image()
    objects = [OBJECTS[0], (*OBJECTS[1][:4], 0, 0.5), OBJECTS[2]]
    rows = [{"class": "plate ABC"}, {"class": "plate XYZ"}, {}]
    style = {**SMALL_TEXT, "text": text, "text_position": "TOP_LEFT"}

    drawn = run_v2(
        LabelVisualizationBlock,
        image=image,
        predictions=detections(image, objects, rows=rows),
        **style,
    )

    expected = run_v1(
        LabelVisualizationBlockV2,
        pixels,
        predictions=v1_detections(
            objects, data={"class_name": np.array(["plate ABC", "plate XYZ", "dog"])}
        ),
        **style,
        **V1_COLOR_DEFAULTS,
    )
    _assert_same_pixels(pixels_of(drawn), expected)


def test_row_names_need_no_class_map() -> None:
    pixels, image = _input_image()
    predictions = detections(
        image, rows=[{"class": "a"}, {"class": "b"}, {"class": "c"}]
    )
    predictions.image_metadata.pop("class_names")
    style = {**SMALL_TEXT, "text": "Class", "text_position": "TOP_LEFT"}

    drawn = run_v2(
        LabelVisualizationBlock, image=image, predictions=predictions, **style
    )

    expected = run_v1(
        LabelVisualizationBlockV2,
        pixels,
        predictions=v1_detections(data={"class_name": np.array(["a", "b", "c"])}),
        **style,
        **V1_COLOR_DEFAULTS,
    )
    _assert_same_pixels(pixels_of(drawn), expected)


def test_a_row_without_any_class_name_is_an_error_naming_its_id() -> None:
    _, image = _input_image()
    objects = [OBJECTS[0], (*OBJECTS[1][:4], 9, 0.5)]

    with pytest.raises(ValueError, match="class id 9"):
        run_v2(
            LabelVisualizationBlock,
            image=image,
            predictions=detections(image, objects),
            text="Class",
        )


def test_confidence_text_needs_no_class_names() -> None:
    pixels, image = _input_image()
    predictions = detections(image)
    predictions.image_metadata.pop("class_names")
    style = {**SMALL_TEXT, "text": "Confidence", "text_position": "TOP_LEFT"}

    drawn = run_v2(
        LabelVisualizationBlock, image=image, predictions=predictions, **style
    )

    expected = run_v1(
        LabelVisualizationBlockV2,
        pixels,
        predictions=v1_detections(),
        **style,
        **V1_COLOR_DEFAULTS,
    )
    _assert_same_pixels(pixels_of(drawn), expected)


# -- label: metadata keys -----------------------------------------------------


def test_label_text_selector_naming_a_row_key_writes_its_values_like_v1() -> None:
    # Ragged rows: the second row has no "plate"; V1 writes an empty label.
    pixels, image = _input_image()
    rows = [{"plate": "KR 1234"}, {}, {"plate": "WA 77"}]
    style = {**SMALL_TEXT, "text_position": "TOP_LEFT"}

    drawn = _run_selected(
        LabelVisualizationBlock,
        image=image,
        predictions=detections(image, rows=rows),
        selected={"text": "plate"},
        **style,
    )

    expected = run_v1(
        LabelVisualizationBlockV2,
        pixels,
        predictions=v1_detections(data={"plate": np.array(["KR 1234", "", "WA 77"])}),
        text="plate",
        **style,
        **V1_COLOR_DEFAULTS,
    )
    _assert_same_pixels(pixels_of(drawn), expected)


def test_label_text_naming_an_unknown_key_is_an_error() -> None:
    _, image = _input_image()

    with pytest.raises(ValueError, match="'plate'"):
        _run_selected(
            LabelVisualizationBlock,
            image=image,
            predictions=detections(image, rows=[{"colour": "red"}] * 3),
            selected={"text": "plate"},
        )


# -- label: masks on demand and mask_quality (decision 027) -------------------


@pytest.mark.parametrize("quality", ["accurate", "coarse"])
@pytest.mark.parametrize(
    ("text", "position"),
    [("Area", "TOP_LEFT"), ("Class", "CENTER_OF_MASS"), ("Area", "CENTER_OF_MASS")],
)
@pytest.mark.parametrize("carrier", ["dense", "rle"])
def test_label_mask_area_and_centre_on_instance_masks_equal_v1(
    carrier: str, text: str, position: str, quality: str
) -> None:
    # Dense and RLE masks are exact either way: mask_quality changes nothing.
    pixels, image = _input_image()
    predictions = (
        instance_detections(image) if carrier == "dense" else rle_detections(image)
    )
    style = {**SMALL_TEXT, "text": text, "text_position": position}

    drawn = run_v2(
        LabelVisualizationBlock,
        image=image,
        predictions=predictions,
        mask_quality=quality,
        **style,
    )

    expected = run_v1(
        LabelVisualizationBlockV2,
        pixels,
        predictions=v1_detections(masks=object_masks()),
        **style,
        **V1_COLOR_DEFAULTS,
    )
    _assert_same_pixels(pixels_of(drawn), expected)


def test_label_area_without_masks_is_the_box_area_like_v1() -> None:
    pixels, image = _input_image()
    style = {**SMALL_TEXT, "text": "Area", "text_position": "TOP_LEFT"}

    drawn = run_v2(
        LabelVisualizationBlock, image=image, predictions=detections(image), **style
    )

    expected = run_v1(
        LabelVisualizationBlockV2,
        pixels,
        predictions=v1_detections(),
        **style,
        **V1_COLOR_DEFAULTS,
    )
    _assert_same_pixels(pixels_of(drawn), expected)


def test_label_center_of_mass_without_masks_is_an_error() -> None:
    _, image = _input_image()

    with pytest.raises(ValueError, match="CENTER_OF_MASS"):
        run_v2(
            LabelVisualizationBlock,
            image=image,
            predictions=detections(image),
            text_position="CENTER_OF_MASS",
        )


@pytest.mark.parametrize("quality", ["accurate", "coarse"])
def test_label_without_mask_demand_never_reads_view_masks(quality: str) -> None:
    _, image = _input_image()
    view = grid_view(image)
    style = {**SMALL_TEXT, "text": "Class and Confidence", "text_position": "CENTER"}

    drawn = run_v2(
        LabelVisualizationBlock,
        image=image,
        predictions=view,
        mask_quality=quality,
        **style,
    )

    expected = run_v2(
        LabelVisualizationBlock, image=image, predictions=detections(image), **style
    )
    _assert_same_pixels(pixels_of(drawn), pixels_of(expected))
    assert not view.is_full_res_materialized()


@pytest.mark.parametrize(
    ("text", "position"),
    [("Area", "TOP_LEFT"), ("Class", "CENTER_OF_MASS"), ("Area", "CENTER_OF_MASS")],
)
def test_accurate_label_masks_of_a_view_are_its_reference_masks(
    text: str, position: str
) -> None:
    pixels, image = _input_image()
    view = grid_view(image)
    style = {**SMALL_TEXT, "text": text, "text_position": position}

    drawn = run_v2(
        LabelVisualizationBlock,
        image=image,
        predictions=view,
        mask_quality="accurate",
        **style,
    )

    assert (
        view.is_full_res_materialized()
    ), "accurate reads the reference full_res masks"
    reference = view.full_res().mask.cpu().numpy()
    expected = run_v1(
        LabelVisualizationBlockV2,
        pixels,
        predictions=v1_detections(masks=reference),
        **style,
        **V1_COLOR_DEFAULTS,
    )
    _assert_same_pixels(pixels_of(drawn), expected)


def test_accurate_is_the_default_mask_quality() -> None:
    _, image = _input_image()
    style = {**SMALL_TEXT, "text": "Area", "text_position": "TOP_LEFT"}
    explicit_view, default_view = grid_view(image), grid_view(image)

    explicit = run_v2(
        LabelVisualizationBlock,
        image=image,
        predictions=explicit_view,
        mask_quality="accurate",
        **style,
    )
    default = run_v2(
        LabelVisualizationBlock, image=image, predictions=default_view, **style
    )

    _assert_same_pixels(pixels_of(default), pixels_of(explicit))
    assert default_view.is_full_res_materialized()


@pytest.mark.parametrize(
    ("text", "position"),
    [("Area", "TOP_LEFT"), ("Class", "CENTER_OF_MASS"), ("Area", "CENTER_OF_MASS")],
)
def test_coarse_label_masks_of_a_view_are_its_thresholded_grid_cells(
    text: str, position: str
) -> None:
    # Expected masks are built here from the fixture: every grid cell scoring
    # above the threshold, blown up to its 4 x 4 pixels.
    pixels, image = _input_image()
    view = grid_view(image)
    style = {**SMALL_TEXT, "text": text, "text_position": position}

    drawn = run_v2(
        LabelVisualizationBlock,
        image=image,
        predictions=view,
        mask_quality="coarse",
        **style,
    )

    assert not view.is_full_res_materialized(), "coarse must not take the accurate path"
    expected = run_v1(
        LabelVisualizationBlockV2,
        pixels,
        predictions=v1_detections(masks=_coarse_masks()),
        **style,
        **V1_COLOR_DEFAULTS,
    )
    _assert_same_pixels(pixels_of(drawn), expected)


def test_coarse_and_accurate_areas_differ_on_this_view() -> None:
    # Guards the two tests above: on this fixture the choices are visible.
    accurate = grid_view(image_of(scene())).full_res().mask.sum(dim=(1, 2)).tolist()
    coarse = _coarse_masks().sum(axis=(1, 2)).tolist()

    assert accurate != coarse
    assert all(area % 16 == 0 for area in coarse)


def test_unknown_mask_quality_literal_is_rejected_at_compile_time() -> None:
    with pytest.raises(ParamsValidationError, match="accurate"):
        spec_of(LabelVisualizationBlock).validate_params(
            {
                "image": "$inputs.image",
                "predictions": "$inputs.predictions",
                "mask_quality": "reference",
            }
        )


def test_unknown_mask_quality_selector_value_is_rejected_at_run_time() -> None:
    _, image = _input_image()

    with pytest.raises(ValueError, match="mask_quality") as caught:
        _run_selected(
            LabelVisualizationBlock,
            image=image,
            predictions=grid_view(image),
            selected={"mask_quality": "approximate"},
            text="Area",
        )

    assert "accurate" in str(caught.value) and "coarse" in str(caught.value)


# -- selectors ----------------------------------------------------------------


SELECTOR_CASES = [
    (BoundingBoxVisualizationBlock, "boxes", {"thickness": 3, "roundness": 0.5}),
    (
        BoundingBoxVisualizationBlock,
        "boxes",
        {"color_palette": "ROBOFLOW", "color_axis": "INDEX"},
    ),
    (
        LabelVisualizationBlock,
        "boxes",
        {"text": "Class and Confidence", "text_color": "#000000", "text_scale": 0.6},
    ),
    (
        LabelVisualizationBlock,
        "boxes",
        {"text_position": "BOTTOM_CENTER", "text_padding": 1},
    ),
    (KeypointVisualizationBlock, "keypoints", {"color": "GREEN", "thickness": 1}),
    (KeypointVisualizationBlock, "keypoints", {"edges": [[1, 2], [2, 3], [5, 17]]}),
    (
        ClassificationLabelVisualizationBlock,
        "multi",
        {"text_position": "CENTER", "text": "Confidence"},
    ),
]


@pytest.mark.parametrize(("block_class", "payload", "values"), SELECTOR_CASES)
def test_selector_values_draw_like_the_same_literals(
    block_class: type, payload: str, values: Dict[str, Any]
) -> None:
    _, image = _input_image()
    predictions = {
        "boxes": detections(image),
        "keypoints": key_points(image),
        "multi": multi_label(image),
    }[payload]

    selected = _run_selected(
        block_class, image=image, predictions=predictions, selected=values
    )

    literal = run_v2(block_class, image=image, predictions=predictions, **values)
    _assert_same_pixels(pixels_of(selected), pixels_of(literal))


INVALID_VALUE_CASES = [
    (
        BoundingBoxVisualizationBlock,
        "boxes",
        "thickness",
        0,
        "thickness must be an integer >= 1",
    ),
    (
        BoundingBoxVisualizationBlock,
        "boxes",
        "roundness",
        1.5,
        "roundness must be a number in [0.0, 1.0]",
    ),
    (
        LabelVisualizationBlock,
        "boxes",
        "text_scale",
        0,
        "text_scale must be a number > 0.0",
    ),
    (
        LabelVisualizationBlock,
        "boxes",
        "text_padding",
        -1,
        "text_padding must be an integer >= 0",
    ),
    (
        KeypointVisualizationBlock,
        "keypoints",
        "radius",
        -2,
        "radius must be an integer >= 0",
    ),
    (
        KeypointVisualizationBlock,
        "keypoints",
        "edges",
        [[0, 1]],
        "edges must be a list of 1-based",
    ),
    (
        ClassificationLabelVisualizationBlock,
        "multi",
        "text_thickness",
        0,
        "text_thickness must be an integer >= 1",
    ),
]


@pytest.mark.parametrize(
    ("block_class", "payload", "name", "value", "message"), INVALID_VALUE_CASES
)
def test_an_invalid_value_fails_with_one_message_as_literal_and_as_selector(
    block_class: type, payload: str, name: str, value: Any, message: str
) -> None:
    _, image = _input_image()
    predictions = {
        "boxes": detections(image),
        "keypoints": key_points(image),
        "multi": multi_label(image),
    }[payload]

    with pytest.raises(ParamsValidationError) as compile_error:
        spec_of(block_class).validate_params(
            {
                "image": "$inputs.image",
                "predictions": "$inputs.predictions",
                name: value,
            }
        )
    with pytest.raises(ValueError) as run_error:
        _run_selected(
            block_class, image=image, predictions=predictions, selected={name: value}
        )

    assert message in str(compile_error.value) + str(compile_error.value.__cause__)
    assert message in str(run_error.value)


@pytest.mark.parametrize(
    "block_class",
    [
        BoundingBoxVisualizationBlock,
        LabelVisualizationBlock,
        KeypointVisualizationBlock,
        ClassificationLabelVisualizationBlock,
    ],
)
@pytest.mark.parametrize("copy_image", [True, False])
def test_v1_copy_image_is_rejected_with_a_remedy(
    block_class: type, copy_image: bool
) -> None:
    with pytest.raises(ParamsValidationError, match="copy_image") as caught:
        spec_of(block_class).validate_params(
            {
                "image": "$inputs.image",
                "predictions": "$inputs.predictions",
                "copy_image": copy_image,
            }
        )

    assert "delete" in str(caught.value).lower()


# -- keypoints ----------------------------------------------------------------


@pytest.mark.parametrize("with_boxes", [True, False])
@pytest.mark.parametrize("annotator_type", ["edge", "vertex", "vertex_label"])
def test_keypoints_equal_v1(annotator_type: str, with_boxes: bool) -> None:
    # Keypoints at (0, 0) are below threshold and skipped, as in V1.
    pixels, image = _input_image()
    points, boxes = key_points(image, with_boxes=with_boxes)
    style = {**KEYPOINT_STYLE, "annotator_type": annotator_type}

    drawn = run_v2(
        KeypointVisualizationBlock, image=image, predictions=(points, boxes), **style
    )

    expected = run_v1(
        KeypointVisualizationBlockV1,
        pixels,
        predictions=_v1_keypoints(points),
        edges=None,
        **style,
    )
    _assert_same_pixels(pixels_of(drawn), expected)


def test_keypoint_custom_edges_equal_v1() -> None:
    pixels, image = _input_image()
    points, boxes = key_points(image)
    style = {
        **KEYPOINT_STYLE,
        "annotator_type": "edge",
        "color": "GREEN",
        "thickness": 1,
    }

    drawn = run_v2(
        KeypointVisualizationBlock,
        image=image,
        predictions=(points, boxes),
        edges=[[1, 2], [2, 3], [5, 17]],
        **style,
    )

    expected = run_v1(
        KeypointVisualizationBlockV1,
        pixels,
        predictions=_v1_keypoints(points),
        edges=[(1, 2), (2, 3), (5, 17)],
        **style,
    )
    _assert_same_pixels(pixels_of(drawn), expected)


def test_keypoint_edges_beyond_the_keypoint_count_are_an_error() -> None:
    _, image = _input_image()

    with pytest.raises(ValueError, match=r"\[1, 18\]"):
        run_v2(
            KeypointVisualizationBlock,
            image=image,
            predictions=key_points(image),
            edges=[[1, 18]],
        )


def test_keypoints_without_a_known_skeleton_draw_no_edges() -> None:
    # Four keypoints: supervision has no skeleton for that count.
    pixels, image = _input_image()
    points, _ = key_points(image)
    four = KeyPoints(
        xy=points.xy[:, :4],
        class_id=points.class_id,
        confidence=points.confidence[:, :4],
        image_metadata=points.image_metadata,
    )

    drawn = run_v2(KeypointVisualizationBlock, image=image, predictions=(four, None))

    _assert_same_pixels(pixels_of(drawn), pixels)


@pytest.mark.parametrize(
    "block_class", [BoundingBoxVisualizationBlock, LabelVisualizationBlock]
)
def test_keypoints_without_boxes_are_refused_by_box_blocks(block_class: type) -> None:
    _, image = _input_image()
    points, _ = key_points(image, with_boxes=False)

    with pytest.raises(ValueError, match="keypoint_visualization"):
        run_v2(block_class, image=image, predictions=(points, None))


# -- classification label -----------------------------------------------------


@pytest.mark.parametrize("color_axis", ["CLASS", "INDEX", "TRACK"])
@pytest.mark.parametrize(
    "position",
    [
        "TOP_LEFT",
        "TOP_CENTER",
        "BOTTOM_RIGHT",
        "BOTTOM_LEFT",
        "CENTER",
        "CENTER_LEFT",
        "CENTER_RIGHT",
    ],
)
@pytest.mark.parametrize("kind", ["single", "multi"])
def test_classification_labels_equal_v1(
    kind: str, position: str, color_axis: str
) -> None:
    pixels, image = _input_image()
    v2_predictions = classification(image) if kind == "single" else multi_label(image)
    v1_predictions = (_v1_single if kind == "single" else _v1_multi)(
        image.width, image.height
    )
    style = {
        **CLASSIFICATION_STYLE,
        **V1_COLOR_DEFAULTS,
        "text": "Class and Confidence",
        "text_position": position,
        "color_axis": color_axis,
    }

    drawn = run_v2(
        ClassificationLabelVisualizationBlock,
        image=image,
        predictions=v2_predictions,
        **style,
    )

    expected = run_v1(
        ClassificationLabelVisualizationBlockV1,
        pixels,
        predictions=v1_predictions,
        **style,
    )
    _assert_same_pixels(pixels_of(drawn), expected)


@pytest.mark.parametrize("text", ["Class", "Confidence"])
def test_classification_text_options_equal_v1(text: str) -> None:
    pixels, image = _input_image()
    style = {
        **CLASSIFICATION_STYLE,
        **V1_COLOR_DEFAULTS,
        "text": text,
        "text_position": "TOP_LEFT",
        "border_radius": 3,
    }

    drawn = run_v2(
        ClassificationLabelVisualizationBlock,
        image=image,
        predictions=multi_label(image),
        **style,
    )

    expected = run_v1(
        ClassificationLabelVisualizationBlockV1,
        pixels,
        predictions=_v1_multi(image.width, image.height),
        **style,
    )
    _assert_same_pixels(pixels_of(drawn), expected)


def test_classification_class_without_a_name_is_an_error() -> None:
    _, image = _input_image()
    prediction = MultiLabelClassificationPrediction(
        class_ids=torch.tensor([0, 3]),
        confidence=torch.tensor([0.6, 0.1, 0.1, 0.9]),
        image_metadata=metadata_for(image),
    )

    with pytest.raises(ValueError, match="class id 3"):
        run_v2(
            ClassificationLabelVisualizationBlock, image=image, predictions=prediction
        )


# -- every family: empty input, unchanged input, kept identity ----------------


def _empty_predictions(family: str, image: ImageData) -> Any:
    if family in ("bounding_box", "label"):
        return detections(image, [])
    if family == "keypoint":
        points = KeyPoints(
            xy=torch.zeros((0, 17, 2)),
            class_id=torch.zeros((0,), dtype=torch.int64),
            confidence=torch.zeros((0, 17)),
            image_metadata=metadata_for(image),
        )
        return points, None

    empty = MultiLabelClassificationPrediction(
        class_ids=torch.zeros((0,), dtype=torch.int64),
        confidence=torch.tensor([0.1, 0.2, 0.3]),
        image_metadata=metadata_for(image),
    )

    return empty


FAMILIES = {
    "bounding_box": (
        BoundingBoxVisualizationBlock,
        lambda image: detections(image),
        {},
    ),
    "label": (
        LabelVisualizationBlock,
        lambda image: instance_detections(image),
        {"text": "Area", "text_position": "CENTER_OF_MASS"},
    ),
    "keypoint": (KeypointVisualizationBlock, lambda image: key_points(image), {}),
    "classification_label": (
        ClassificationLabelVisualizationBlock,
        lambda image: classification(image),
        {},
    ),
}


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_empty_predictions_give_an_unchanged_new_image(family: str) -> None:
    block_class, _, parameters = FAMILIES[family]
    pixels, image = _input_image()
    if family == "label":
        parameters = {"text": "Class"}

    drawn = run_v2(
        block_class,
        image=image,
        predictions=_empty_predictions(family, image),
        **parameters,
    )

    _assert_same_pixels(pixels_of(drawn), pixels)
    assert drawn.tensor_image.data_ptr() != image.tensor_image.data_ptr()


def _tensors_of(value: Any) -> List[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in _tensors_of(item)]
    if value is None or isinstance(value, (str, int, float, dict)):
        return []

    fields = getattr(value, "__dict__", {})
    tensors = [tensor for item in fields.values() for tensor in _tensors_of(item)]

    return tensors


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_input_image_and_predictions_are_not_changed(family: str) -> None:
    block_class, make_predictions, parameters = FAMILIES[family]
    _, image = _input_image()
    predictions = make_predictions(image)
    image_before = image.tensor_image.clone()
    tensors_before = [tensor.clone() for tensor in _tensors_of(predictions)]

    drawn = run_v2(block_class, image=image, predictions=predictions, **parameters)

    assert torch.equal(image.tensor_image, image_before)
    for before, after in zip(tensors_before, _tensors_of(predictions)):
        assert torch.equal(before, after)
    assert not torch.equal(
        drawn.tensor_image, image_before
    ), "the case must draw something"


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_output_keeps_identity_frames_video_metadata_and_parent_row(
    family: str,
) -> None:
    # A crop of a video frame made for a detection row: every provenance field
    # must survive drawing, and the pixels must be a new contiguous tensor.
    block_class, make_predictions, parameters = FAMILIES[family]
    video = VideoMetadata(
        video_identifier="camera-1",
        frame_number=42,
        frame_timestamp=datetime(2026, 10, 12, tzinfo=timezone.utc),
        fps=25.0,
    )
    frame = ImageData.from_numpy_rgb(
        scene(64, 80), image_id="video-frame", video_metadata=video
    )
    image = frame.crop((8, 4, 72, 52), image_id="crop").with_parent_row(
        RowRef("detections", 3)
    )

    drawn = run_v2(
        block_class, image=image, predictions=make_predictions(image), **parameters
    )

    assert drawn.image_id == image.image_id
    assert drawn.parent == image.parent and drawn.root == image.root
    assert drawn.video_metadata == image.video_metadata
    assert drawn.parent_row == RowRef("detections", 3)
    assert drawn.composite_sources is None
    assert (
        drawn.tensor_image.dtype == torch.uint8 and drawn.tensor_image.is_contiguous()
    )
    assert drawn.tensor_image.data_ptr() != image.tensor_image.data_ptr()


def test_one_block_instance_draws_images_with_different_row_counts() -> None:
    # Ragged rows across images: no state carries over between invocations.
    block = LabelVisualizationBlock()
    params = dict(
        spec_of(LabelVisualizationBlock).validate_params(
            {
                "image": "$inputs.image",
                "predictions": "$inputs.predictions",
                **SMALL_TEXT,
            }
        )
    )
    first_pixels, first = _input_image(image_id="first", seed=1)
    second_pixels, second = _input_image(image_id="second", seed=2)

    drawn_first = block.run(
        **{**params, "image": first, "predictions": detections(first)}
    )
    drawn_second = block.run(
        **{**params, "image": second, "predictions": detections(second, OBJECTS[1:2])}
    )

    for drawn, pixels, objects in (
        (drawn_first, first_pixels, OBJECTS),
        (drawn_second, second_pixels, OBJECTS[1:2]),
    ):
        expected = run_v1(
            LabelVisualizationBlockV2,
            pixels,
            predictions=v1_detections(objects),
            text="Class",
            text_position="TOP_LEFT",
            **SMALL_TEXT,
            **V1_COLOR_DEFAULTS,
        )
        _assert_same_pixels(pixels_of(drawn["image"]), expected)


# -- frames: crop -> parent and parent -> crop --------------------------------


def _crop_of(image: ImageData) -> ImageData:
    crop = image.crop(CROP_XYXY, image_id="crop")

    return crop


@pytest.mark.parametrize(
    ("block_class", "v1_class", "parameters"),
    [
        (
            BoundingBoxVisualizationBlock,
            BoundingBoxVisualizationBlockV1,
            {"thickness": 2, "roundness": 0.0},
        ),
        (
            BoundingBoxVisualizationBlock,
            BoundingBoxVisualizationBlockV1,
            {"thickness": 3, "roundness": 0.5},
        ),
        (
            LabelVisualizationBlock,
            LabelVisualizationBlockV2,
            {**SMALL_TEXT, "text": "Class", "text_position": "TOP_LEFT"},
        ),
        (
            LabelVisualizationBlock,
            LabelVisualizationBlockV2,
            {**SMALL_TEXT, "text": "Dimensions", "text_position": "CENTER"},
        ),
    ],
)
def test_crop_predictions_are_drawn_where_they_lie_in_the_parent(
    block_class: type, v1_class: type, parameters: Dict[str, Any]
) -> None:
    # V1 oracle: the same rows given to V1 already shifted by the crop offset.
    pixels, image = _input_image()
    crop = _crop_of(image)
    local = [(2.5, 3.25, 20.75, 18.5, 0, 0.9), (15.0, 10.5, 38.25, 31.0, 2, 0.4)]

    drawn = run_v2(
        block_class, image=image, predictions=detections(crop, local), **parameters
    )

    x0, y0 = CROP_XYXY[:2]
    expected = run_v1(
        v1_class,
        pixels,
        predictions=v1_detections(_shifted(local, dx=x0, dy=y0)),
        **parameters,
        **V1_COLOR_DEFAULTS,
    )
    _assert_same_pixels(pixels_of(drawn), expected)


@pytest.mark.parametrize(
    ("block_class", "v1_class", "parameters"),
    [
        (
            BoundingBoxVisualizationBlock,
            BoundingBoxVisualizationBlockV1,
            {"thickness": 2, "roundness": 0.0},
        ),
        (
            BoundingBoxVisualizationBlock,
            BoundingBoxVisualizationBlockV1,
            {"thickness": 4, "roundness": 0.3},
        ),
        (
            LabelVisualizationBlock,
            LabelVisualizationBlockV2,
            {**SMALL_TEXT, "text": "Class", "text_position": "TOP_LEFT"},
        ),
        (
            LabelVisualizationBlock,
            LabelVisualizationBlockV2,
            {**SMALL_TEXT, "text": "Index", "text_position": "BOTTOM_RIGHT"},
        ),
    ],
)
def test_parent_predictions_drawn_on_a_crop_are_clipped_to_it(
    block_class: type, v1_class: type, parameters: Dict[str, Any]
) -> None:
    # Every box crosses the crop border; parts outside the crop are dropped.
    pixels, image = _input_image()
    crop = _crop_of(image)
    crop_pixels = pixels_of(crop)

    drawn = run_v2(
        block_class,
        image=crop,
        predictions=detections(image, QUARTER_OBJECTS),
        **parameters,
    )

    x0, y0 = CROP_XYXY[:2]
    expected = run_v1(
        v1_class,
        crop_pixels,
        predictions=v1_detections(_shifted(QUARTER_OBJECTS, dx=-x0, dy=-y0)),
        **parameters,
        **V1_COLOR_DEFAULTS,
    )
    _assert_same_pixels(pixels_of(drawn), expected)
    assert drawn.size_hw == crop.size_hw


@pytest.mark.parametrize("annotator_type", ["edge", "vertex", "vertex_label"])
def test_crop_keypoints_are_drawn_where_they_lie_in_the_parent(
    annotator_type: str,
) -> None:
    # Hidden keypoints sit at the crop's (0, 0); mapped to the parent they must
    # stay hidden, together with the edges that touch them.
    pixels, image = _input_image()
    crop = _crop_of(image)
    points, boxes = key_points(crop)
    style = {**KEYPOINT_STYLE, "annotator_type": annotator_type}

    drawn = run_v2(
        KeypointVisualizationBlock, image=image, predictions=(points, boxes), **style
    )

    x0, y0 = CROP_XYXY[:2]
    # Points at (0, 0) stay "not detected": keep them at the origin for V1.
    xy = points.xy.clone()
    detected = (xy != 0).any(dim=-1, keepdim=True)
    xy = torch.where(detected, xy + torch.tensor([x0, y0], dtype=xy.dtype), xy)
    shifted = KeyPoints(xy=xy, class_id=points.class_id, confidence=points.confidence)
    expected = run_v1(
        KeypointVisualizationBlockV1,
        pixels,
        predictions=_v1_keypoints(shifted, _shifted(OBJECTS[:2], dx=x0, dy=y0)),
        edges=None,
        **style,
    )
    _assert_same_pixels(pixels_of(drawn), expected)


@pytest.mark.parametrize(
    "block_class",
    [
        BoundingBoxVisualizationBlock,
        LabelVisualizationBlock,
        KeypointVisualizationBlock,
    ],
)
def test_predictions_of_another_input_are_an_error_naming_both(
    block_class: type,
) -> None:
    _, image = _input_image(image_id="frame")
    _, other = _input_image(image_id="other-input")
    predictions = (
        key_points(other)
        if block_class is KeypointVisualizationBlock
        else detections(other)
    )

    with pytest.raises(ValueError, match="other-input") as caught:
        run_v2(block_class, image=image, predictions=predictions)

    assert "'frame'" in str(caught.value)


def test_predictions_claiming_the_image_with_another_size_are_an_error() -> None:
    _, image = _input_image(image_id="frame")
    larger = image_of(scene(60, 80), image_id="frame")

    with pytest.raises(ValueError, match=r"\(48, 64\)"):
        run_v2(
            BoundingBoxVisualizationBlock, image=image, predictions=detections(larger)
        )


def test_predictions_without_a_frame_record_are_drawn_as_given() -> None:
    # V1-style predictions carry no frame: coordinates are the image's own.
    pixels, image = _input_image()
    predictions = detections(image)
    plain = Detections(
        xyxy=predictions.xyxy,
        class_id=predictions.class_id,
        confidence=predictions.confidence,
        image_metadata={"class_names": dict(CLASS_NAMES)},
    )

    drawn = run_v2(BoundingBoxVisualizationBlock, image=image, predictions=plain)

    expected = run_v1(
        BoundingBoxVisualizationBlockV1,
        pixels,
        predictions=v1_detections(),
        thickness=2,
        roundness=0.0,
        **V1_COLOR_DEFAULTS,
    )
    _assert_same_pixels(pixels_of(drawn), expected)
