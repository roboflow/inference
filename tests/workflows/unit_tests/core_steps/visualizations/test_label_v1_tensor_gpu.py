import numpy as np
import pytest
import supervision as sv
import torch

from inference.core.workflows.core_steps.visualizations.label.v1_tensor import (
    LabelVisualizationBlockV1,
    gpu_draw_labels,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from inference_models.models.base.object_detection import Detections

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

SCENE_HEIGHT = 180
SCENE_WIDTH = 260
LABELS = ["person 0.91", "forklift"]
BOXES = np.asarray([[30, 50, 130, 130], [120, 25, 230, 150]], dtype=np.float32)


def _label_properties_for_boxes(
    boxes: np.ndarray,
    labels=LABELS,
    position: str = "TOP_LEFT",
    text_scale: float = 0.5,
    text_thickness: int = 1,
    text_padding: int = 4,
) -> np.ndarray:
    detections = sv.Detections(xyxy=boxes)
    annotator = sv.LabelAnnotator(
        text_position=getattr(sv.Position, position),
        text_scale=text_scale,
        text_thickness=text_thickness,
        text_padding=text_padding,
    )
    return annotator._get_label_properties(
        detections=detections,
        labels=list(labels),
    )


def _draw_on_tensor(
    device: str = "cpu",
    position: str = "TOP_LEFT",
    border_radius: int = 3,
) -> torch.Tensor:
    scene = torch.zeros(
        (3, SCENE_HEIGHT, SCENE_WIDTH), dtype=torch.uint8, device=device
    )
    colors = np.asarray([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    return gpu_draw_labels(
        scene_chw=scene,
        labels=LABELS,
        label_properties=_label_properties_for_boxes(BOXES, position=position),
        background_colors_rgb=colors,
        text_color_rgb=(255, 255, 255),
        text_scale=0.5,
        text_thickness=1,
        text_padding=4,
        border_radius=border_radius,
    )


@pytest.mark.parametrize(
    "position",
    [
        "TOP_LEFT",
        "TOP_RIGHT",
        "TOP_CENTER",
        "CENTER",
        "BOTTOM_LEFT",
        "BOTTOM_RIGHT",
        "BOTTOM_CENTER",
        "CENTER_LEFT",
        "CENTER_RIGHT",
    ],
)
@pytest.mark.parametrize("border_radius", [0, 3])
def test_tensor_label_renderer_matches_supervision_pixels(
    position: str, border_radius: int
) -> None:
    tensor_output = (
        _draw_on_tensor(position=position, border_radius=border_radius)
        .permute(1, 2, 0)
        .numpy()
    )
    supervision_scene = np.zeros((SCENE_HEIGHT, SCENE_WIDTH, 3), dtype=np.uint8)
    annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(["#FF0000", "#00FF00"]),
        color_lookup=sv.ColorLookup.INDEX,
        text_position=getattr(sv.Position, position),
        text_color=sv.Color.WHITE,
        text_scale=0.5,
        text_thickness=1,
        text_padding=4,
        border_radius=border_radius,
    )
    supervision_output_bgr = annotator.annotate(
        scene=supervision_scene,
        detections=sv.Detections(xyxy=BOXES),
        labels=LABELS,
    )
    supervision_output_rgb = supervision_output_bgr[:, :, ::-1]
    different = np.any(tensor_output != supervision_output_rgb, axis=2)
    if not different.any():
        return
    # Rasterising the complete patch and then clipping it is pixel-identical
    # except for OpenCV's anti-aliasing exactly on a clipped frame boundary.
    # The two CENTER_* cases above expose at most a handful of those pixels.
    rows, columns = np.nonzero(different)
    on_frame_boundary = (
        (rows == 0)
        | (rows == SCENE_HEIGHT - 1)
        | (columns == 0)
        | (columns == SCENE_WIDTH - 1)
    )
    assert different.sum() <= 8
    assert on_frame_boundary.all()


def test_tensor_label_renderer_is_in_place_and_clips_to_frame() -> None:
    scene = torch.zeros((3, 64, 64), dtype=torch.uint8)
    output = gpu_draw_labels(
        scene_chw=scene,
        labels=["outside", "edge"],
        label_properties=np.asarray(
            [[-100, -100, -20, -80, 10], [50, 50, 100, 70, 10]],
            dtype=np.int32,
        ),
        background_colors_rgb=np.asarray([[255, 0, 0], [0, 255, 0]], dtype=np.uint8),
        text_color_rgb=(255, 255, 255),
        text_scale=0.5,
        text_thickness=1,
        text_padding=4,
        border_radius=0,
    )
    assert output.data_ptr() == scene.data_ptr()
    assert torch.count_nonzero(output[:, :50, :50]) == 0
    assert torch.count_nonzero(output[:, 50:, 50:]) > 0


@requires_cuda
def test_cuda_label_renderer_matches_cpu() -> None:
    cpu_output = _draw_on_tensor()
    cuda_output = _draw_on_tensor(device="cuda")
    assert torch.equal(cpu_output, cuda_output.cpu())


def _tensor_detections(device: str = "cpu") -> Detections:
    return Detections(
        xyxy=torch.tensor(BOXES, device=device),
        class_id=torch.tensor([0, 1], dtype=torch.int32, device=device),
        confidence=torch.tensor([0.91, 0.87], device=device),
        image_metadata={"class_names": {0: "person", 1: "forklift"}},
    )


def _run_block(
    image: WorkflowImageData, predictions: Detections, copy_image: bool
) -> WorkflowImageData:
    return LabelVisualizationBlockV1().run(
        image=image,
        predictions=predictions,
        copy_image=copy_image,
        color_palette="DEFAULT",
        palette_size=10,
        custom_colors=None,
        color_axis="CLASS",
        text="Class and Confidence",
        text_position="TOP_LEFT",
        text_color="WHITE",
        text_scale=0.5,
        text_thickness=1,
        text_padding=4,
        border_radius=3,
    )["image"]


def test_block_keeps_full_frame_tensor_backed() -> None:
    source_tensor = torch.zeros((3, SCENE_HEIGHT, SCENE_WIDTH), dtype=torch.uint8)
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="p"),
        tensor_image=source_tensor,
    )
    output = _run_block(image, _tensor_detections(), copy_image=True)
    assert output._tensor_image is not None
    assert output._numpy_image is None
    assert output._tensor_image.data_ptr() != image._tensor_image.data_ptr()
    assert torch.count_nonzero(output._tensor_image) > 0
    assert torch.count_nonzero(image._tensor_image) == 0


def test_block_mutates_tensor_source_when_copy_disabled() -> None:
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="p"),
        tensor_image=torch.zeros((3, SCENE_HEIGHT, SCENE_WIDTH), dtype=torch.uint8),
    )
    source_pointer = image._tensor_image.data_ptr()
    output = _run_block(image, _tensor_detections(), copy_image=False)
    assert output._tensor_image.data_ptr() == source_pointer
    assert output._numpy_image is None
    assert torch.count_nonzero(output._tensor_image) > 0


def test_empty_predictions_do_not_materialise_numpy_frame() -> None:
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="p"),
        tensor_image=torch.zeros((3, SCENE_HEIGHT, SCENE_WIDTH), dtype=torch.uint8),
    )
    empty = Detections(
        xyxy=torch.empty((0, 4)),
        class_id=torch.empty((0,), dtype=torch.int32),
        confidence=torch.empty((0,)),
        image_metadata={"class_names": {}},
    )
    output = _run_block(image, empty, copy_image=True)
    assert output._tensor_image is not None
    assert output._numpy_image is None
