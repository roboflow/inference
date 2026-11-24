"""Device runs of box, label, keypoint and classification-label drawing.

The CPU run of the same block is the oracle: the CPU path is checked against
V1 in ``test_drawing_blocks.py``, and these blocks promise identical pixels on
every device (outlines and text are rasterized on the host, then written into
the device canvas). Each case runs on every available accelerator::

    cuda   device qualification (skipped without CUDA)
    mps    optional local check on Apple GPUs (skipped without MPS)

CUDA cases also check stream behavior: no device-wide synchronization, and a
complete drawing when the caller runs the block under a side stream.
"""

from typing import Any, Callable, Dict, List, Tuple

import pytest
import torch
from roboflow_workflows.v2.blocks.visualization.annotations import (
    BoundingBoxVisualizationBlock,
    ClassificationLabelVisualizationBlock,
    KeypointVisualizationBlock,
    LabelVisualizationBlock,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.masks import MaskGridGeometry, SegmentationView
from torch.nn import functional as F

from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from tests.v2.unit.blocks.visualization.annotations.support import (
    OBJECTS,
    classification,
    detections,
    image_of,
    instance_detections,
    key_points,
    multi_label,
    pixels_of,
    run_v2,
    scene,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
MPS = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
DEVICES = [pytest.param("cuda", marks=CUDA), pytest.param("mps", marks=MPS)]
SMALL_TEXT = {
    "text_color": "WHITE",
    "text_scale": 0.4,
    "text_thickness": 1,
    "text_padding": 3,
    "border_radius": 2,
}
CROP_XYXY = (10, 8, 50, 40)


# -- predictions on a device --------------------------------------------------


def _keypoints_on(
    image: ImageData, device: str, *, with_boxes: bool
) -> Tuple[Any, Any]:
    points, _ = key_points(image, with_boxes=False)
    moved = KeyPoints(
        xy=points.xy.to(device),
        class_id=points.class_id.to(device),
        confidence=points.confidence.to(device),
        image_metadata=points.image_metadata,
    )
    boxes = detections(image, OBJECTS[:2], device=device) if with_boxes else None

    return moved, boxes


def _grid_view_on(
    image: ImageData, device: str, *, stride: int = 4
) -> SegmentationView:
    # support.grid_view with every tensor on ``device``.
    dense = instance_detections(image, device=device)
    height, width = image.size_hw
    covered = F.avg_pool2d(dense.mask.float().unsqueeze(1), stride).squeeze(1)
    view = SegmentationView(
        detections=Detections(
            xyxy=dense.xyxy,
            class_id=dense.class_id,
            confidence=dense.confidence,
            image_metadata=dense.image_metadata,
        ),
        scores=covered * 16.0 - 8.0,
        score_type="logits",
        threshold=0.0,
        geometry=MaskGridGeometry(
            grid_size_hw=(height // stride, width // stride),
            inference_size_hw=(height, width),
            padding_ltrb=(0, 0, 0, 0),
            scale_xy=(1.0, 1.0),
            crop_size_hw=(height, width),
            original_size_hw=(height, width),
            static_crop_xywh=(0, 0, width, height),
        ),
        take_ownership=True,
    )

    return view


def _single_on(image: ImageData, device: str) -> ClassificationPrediction:
    prediction = classification(image)
    moved = ClassificationPrediction(
        class_id=prediction.class_id.to(device),
        confidence=prediction.confidence.to(device),
        images_metadata=prediction.images_metadata,
    )

    return moved


def _multi_on(image: ImageData, device: str) -> MultiLabelClassificationPrediction:
    prediction = multi_label(image)
    moved = MultiLabelClassificationPrediction(
        class_ids=prediction.class_ids.to(device),
        confidence=prediction.confidence.to(device),
        image_metadata=prediction.image_metadata,
    )

    return moved


PAYLOADS: Dict[str, Callable[[ImageData, str], Any]] = {
    "boxes": lambda image, device: detections(image, device=device),
    "tracked_boxes": lambda image, device: detections(
        image,
        rows=[
            {"tracker_id": 4, "class": "plate A"},
            {"tracker_id": 9},
            {"tracker_id": 4},
        ],
        device=device,
    ),
    "dense_masks": lambda image, device: instance_detections(image, device=device),
    "grid_view": _grid_view_on,
    "keypoints": lambda image, device: _keypoints_on(image, device, with_boxes=True),
    "keypoints_only": lambda image, device: _keypoints_on(
        image, device, with_boxes=False
    ),
    "single_label": _single_on,
    "multi_label": _multi_on,
    "empty_boxes": lambda image, device: detections(image, [], device=device),
}

# (case id, block, payload, parameters): every drawing path of the four families.
CASES: List[Tuple[str, type, str, Dict[str, Any]]] = [
    ("box-square", BoundingBoxVisualizationBlock, "boxes", {"thickness": 2}),
    (
        "box-round",
        BoundingBoxVisualizationBlock,
        "boxes",
        {"thickness": 3, "roundness": 0.6},
    ),
    (
        "box-track",
        BoundingBoxVisualizationBlock,
        "tracked_boxes",
        {"color_axis": "TRACK"},
    ),
    ("box-view", BoundingBoxVisualizationBlock, "grid_view", {}),
    ("box-keypoint-boxes", BoundingBoxVisualizationBlock, "keypoints", {}),
    ("box-empty", BoundingBoxVisualizationBlock, "empty_boxes", {}),
    (
        "label-class-confidence",
        LabelVisualizationBlock,
        "tracked_boxes",
        {**SMALL_TEXT, "text": "Class and Confidence", "text_position": "CENTER"},
    ),
    (
        "label-dimensions",
        LabelVisualizationBlock,
        "boxes",
        {**SMALL_TEXT, "text": "Dimensions", "text_position": "BOTTOM_RIGHT"},
    ),
    (
        "label-dense-area-mass",
        LabelVisualizationBlock,
        "dense_masks",
        {**SMALL_TEXT, "text": "Area", "text_position": "CENTER_OF_MASS"},
    ),
    (
        "label-view-accurate",
        LabelVisualizationBlock,
        "grid_view",
        {
            **SMALL_TEXT,
            "text": "Area",
            "text_position": "CENTER_OF_MASS",
            "mask_quality": "accurate",
        },
    ),
    (
        "label-view-coarse",
        LabelVisualizationBlock,
        "grid_view",
        {
            **SMALL_TEXT,
            "text": "Area",
            "text_position": "CENTER_OF_MASS",
            "mask_quality": "coarse",
        },
    ),
    (
        "keypoint-edge",
        KeypointVisualizationBlock,
        "keypoints",
        {"annotator_type": "edge"},
    ),
    (
        "keypoint-vertex",
        KeypointVisualizationBlock,
        "keypoints_only",
        {"annotator_type": "vertex", "radius": 3},
    ),
    (
        "keypoint-vertex-label",
        KeypointVisualizationBlock,
        "keypoints",
        {"annotator_type": "vertex_label", "text_scale": 0.4, "text_padding": 2},
    ),
    (
        "classification-single",
        ClassificationLabelVisualizationBlock,
        "single_label",
        {**SMALL_TEXT, "text": "Class and Confidence", "text_position": "TOP_LEFT"},
    ),
    (
        "classification-multi",
        ClassificationLabelVisualizationBlock,
        "multi_label",
        {
            **SMALL_TEXT,
            "text": "Class",
            "text_position": "BOTTOM_RIGHT",
            "color_axis": "INDEX",
        },
    ),
]
CASE_PARAMS = [pytest.param(*case[1:], id=case[0]) for case in CASES]


def _image_on(device: str, *, image_id: str = "frame") -> ImageData:
    cpu_image = image_of(scene(), image_id=image_id)
    image = ImageData.from_tensor(cpu_image.tensor_image.to(device), image_id=image_id)

    return image


def _draw(
    block_class: type, payload: str, parameters: Dict[str, Any], device: str
) -> ImageData:
    image = _image_on(device)
    drawn = run_v2(
        block_class,
        image=image,
        predictions=PAYLOADS[payload](image, device),
        **parameters,
    )

    return drawn


# -- pixels -------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("block_class", "payload", "parameters"), CASE_PARAMS)
def test_device_pixels_equal_cpu_pixels(
    block_class: type, payload: str, parameters: Dict[str, Any], device: str
) -> None:
    expected = _draw(block_class, payload, parameters, "cpu")

    drawn = _draw(block_class, payload, parameters, device)

    assert drawn.device.type == device, "the output stays on the input's device"
    changed = (pixels_of(drawn) != pixels_of(expected)).any(axis=-1)
    assert not changed.any(), f"{int(changed.sum())} pixels differ from the CPU run"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("block_class", "payload", "parameters"), CASE_PARAMS)
def test_device_input_image_is_not_changed(
    block_class: type, payload: str, parameters: Dict[str, Any], device: str
) -> None:
    image = _image_on(device)
    before = image.tensor_image.clone()

    run_v2(
        block_class,
        image=image,
        predictions=PAYLOADS[payload](image, device),
        **parameters,
    )

    assert torch.equal(image.tensor_image, before)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    ("quality", "materialized"), [("accurate", True), ("coarse", False)]
)
def test_device_mask_quality_takes_the_accurate_path_only_when_asked(
    quality: str, materialized: bool, device: str
) -> None:
    image = _image_on(device)
    view = _grid_view_on(image, device)

    run_v2(
        LabelVisualizationBlock,
        image=image,
        predictions=view,
        text="Area",
        mask_quality=quality,
    )

    assert view.is_full_res_materialized() is materialized


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    ("block_class", "parameters"),
    [
        (BoundingBoxVisualizationBlock, {"thickness": 2}),
        (
            LabelVisualizationBlock,
            {**SMALL_TEXT, "text": "Class", "text_position": "TOP_LEFT"},
        ),
    ],
)
@pytest.mark.parametrize("direction", ["crop_to_parent", "parent_to_crop"])
def test_device_frame_mapping_equals_cpu(
    direction: str, block_class: type, parameters: Dict[str, Any], device: str
) -> None:
    local = [(2.5, 3.25, 20.75, 18.5, 0, 0.9), (15.0, 10.5, 38.25, 31.0, 2, 0.4)]

    def draw(on: str) -> ImageData:
        image = _image_on(on)
        crop = image.crop(CROP_XYXY, image_id="crop")
        if direction == "crop_to_parent":
            target, predictions = image, detections(crop, local, device=on)
        else:
            target, predictions = crop, detections(image, device=on)

        drawn = run_v2(block_class, image=target, predictions=predictions, **parameters)

        return drawn

    expected = draw("cpu")

    drawn = draw(device)

    assert drawn.device.type == device
    assert (pixels_of(drawn) == pixels_of(expected)).all()


# -- CUDA streams -------------------------------------------------------------


@CUDA
@pytest.mark.parametrize(("block_class", "payload", "parameters"), CASE_PARAMS)
def test_cuda_drawing_never_synchronizes_the_whole_device(
    block_class: type, payload: str, parameters: Dict[str, Any], monkeypatch
) -> None:
    # Host transfers of a few rows may wait for their own stream; a
    # device-wide wait would stall every other stream of a pipeline.
    image = _image_on("cuda")
    predictions = PAYLOADS[payload](image, "cuda")
    torch.cuda.synchronize()

    def forbidden(*_: Any, **__: Any) -> None:
        raise AssertionError("torch.cuda.synchronize was called")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)

    drawn = run_v2(block_class, image=image, predictions=predictions, **parameters)

    assert drawn.device.type == "cuda"


@CUDA
@pytest.mark.parametrize(("block_class", "payload", "parameters"), CASE_PARAMS)
def test_cuda_drawing_on_a_side_stream_equals_cpu(
    block_class: type, payload: str, parameters: Dict[str, Any]
) -> None:
    # A pipeline may call the block under its own stream. The drawing must be
    # complete once that stream finishes; this does not prove which stream
    # ran each kernel.
    expected = _draw(block_class, payload, parameters, "cpu")
    image = _image_on("cuda")
    predictions = PAYLOADS[payload](image, "cuda")
    torch.cuda.synchronize()
    side = torch.cuda.Stream()

    with torch.cuda.stream(side):
        drawn = run_v2(block_class, image=image, predictions=predictions, **parameters)
    side.synchronize()

    assert (pixels_of(drawn) == pixels_of(expected)).all()
