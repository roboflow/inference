"""Visualization-phase tests for the per-box host mirror
(``tensor_native.HOST_MIRROR_KEYS``).

Proves that ``to_supervision_for_annotation`` builds the sv view with ZERO
device reads when every box carries the mirror (TorchDispatchMode audit), that
the mirror and tensor-read paths produce identical sv views, that mirror-less /
partially-mirrored predictions keep today's tensor-read behaviour, and that the
label and bounding-box blocks render identically (and without detection-tensor
reads) end to end.
"""

import numpy as np
import pytest
import supervision as sv
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from inference.core.workflows.core_steps.common.tensor_native import (
    HOST_MIRROR_KEYS,
    attach_native_detection_metadata,
    strip_host_mirror_metadata,
)
from inference.core.workflows.core_steps.visualizations.bounding_box.v1_tensor import (
    BoundingBoxVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.common.base_tensor import (
    to_supervision_for_annotation,
)
from inference.core.workflows.core_steps.visualizations.label.v1_tensor import (
    LabelVisualizationBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.representation_boundary import (
    native_detections_to_sv,
    sv_detections_to_native,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections

CLASS_NAMES = {0: "cat", 1: "dog", 2: "goggles"}
SCENE_H, SCENE_W = 480, 640
N_BOXES = 3


def _storage_ptr(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


class _OpAudit(TorchDispatchMode):
    """Record every dispatched aten op with the storage pointers of its tensor
    arguments — lets a test assert that SPECIFIC tensors (the detection
    xyxy/class_id/confidence) were never touched, while the annotator's own
    tensors (scene, sprites, paste indices) stay unconstrained."""

    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        pointers = []

        def _collect(value):
            if isinstance(value, torch.Tensor):
                try:
                    pointers.append(_storage_ptr(value))
                except RuntimeError:
                    pass
            elif isinstance(value, (list, tuple)):
                for item in value:
                    _collect(item)

        _collect(args)
        _collect(list((kwargs or {}).values()))
        self.ops.append((str(func), pointers))
        return func(*args, **(kwargs or {}))

    def op_names(self):
        return [name for name, _ in self.ops]

    def assert_never_touched(self, *tensors: torch.Tensor) -> None:
        assert not any("_local_scalar_dense" in name for name in self.op_names())
        assert not any("nonzero" in name for name in self.op_names())
        forbidden = {_storage_ptr(tensor) for tensor in tensors}
        touched = [
            name
            for name, pointers in self.ops
            if any(pointer in forbidden for pointer in pointers)
        ]
        assert touched == [], f"ops read the detection tensors: {touched}"


def _image_data() -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((SCENE_H, SCENE_W, 3), dtype=np.uint8),
    )


def _mirrored_od_detections(n: int = N_BOXES) -> Detections:
    boxes = np.array(
        [
            [40.5, 60.25, 200.75, 200.125],
            [250.0, 90.0, 420.0, 260.0],
            [90.0, 260.0, 300.0, 430.0],
            [10.0, 10.0, 30.0, 30.0],
        ],
        dtype=np.float32,
    )[:n]
    detections = Detections(
        xyxy=torch.tensor(boxes, dtype=torch.float32),
        class_id=torch.tensor([index % 3 for index in range(n)], dtype=torch.long),
        confidence=torch.tensor(
            np.linspace(0.42, 0.99, max(n, 1))[:n], dtype=torch.float32
        ),
        image_metadata=None,
        bboxes_metadata=None,
    )
    return attach_native_detection_metadata(
        detections=detections,
        image=_image_data(),
        class_names=CLASS_NAMES,
        prediction_type="object-detection",
    )


def _stripped_twin(detections: Detections) -> Detections:
    return Detections(
        xyxy=detections.xyxy,
        class_id=detections.class_id,
        confidence=detections.confidence,
        image_metadata=detections.image_metadata,
        bboxes_metadata=strip_host_mirror_metadata(detections.bboxes_metadata),
    )


def _assert_same_sv_view(first: sv.Detections, second: sv.Detections) -> None:
    assert np.array_equal(first.xyxy, second.xyxy)
    assert first.xyxy.dtype == second.xyxy.dtype
    assert np.array_equal(first.class_id, second.class_id)
    assert first.class_id.dtype == second.class_id.dtype
    assert np.array_equal(first.confidence, second.confidence)
    assert first.confidence.dtype == second.confidence.dtype
    assert (first.tracker_id is None) == (second.tracker_id is None)
    if first.tracker_id is not None:
        assert np.array_equal(first.tracker_id, second.tracker_id)
    assert set(first.data.keys()) == set(second.data.keys())
    for key in first.data:
        assert np.array_equal(first.data[key], second.data[key]), key


# --------------------------------------------------------------------------- #
# to_supervision_for_annotation: zero device reads on the mirror path
# --------------------------------------------------------------------------- #


def test_to_supervision_with_full_mirror_dispatches_zero_tensor_ops() -> None:
    # given
    detections = _mirrored_od_detections()

    # when
    audit = _OpAudit()
    with audit:
        sv_view = to_supervision_for_annotation(detections)

    # then - the mirror path never touches a tensor: no aten op is dispatched
    # at all (in particular no `_to_copy` D2H, no `_local_scalar_dense`, no
    # `nonzero`).
    assert audit.ops == []
    assert len(sv_view) == N_BOXES


def test_to_supervision_mirror_and_tensor_paths_are_bit_identical() -> None:
    # given
    mirrored = _mirrored_od_detections()
    stripped = _stripped_twin(mirrored)

    # when
    from_mirror = to_supervision_for_annotation(mirrored)
    from_tensors = to_supervision_for_annotation(stripped)

    # then
    _assert_same_sv_view(from_mirror, from_tensors)


def test_to_supervision_mirror_keys_never_surface_in_sv_data() -> None:
    # when
    sv_view = to_supervision_for_annotation(_mirrored_od_detections())

    # then
    assert not any(key in sv_view.data for key in HOST_MIRROR_KEYS)


@pytest.mark.parametrize("missing_key", HOST_MIRROR_KEYS)
def test_to_supervision_partial_mirror_falls_back_to_tensor_reads(
    missing_key,
) -> None:
    # given - one box lost one mirror key (e.g. a transformed row)
    mirrored = _mirrored_od_detections()
    reference = to_supervision_for_annotation(_stripped_twin(mirrored))
    del mirrored.bboxes_metadata[1][missing_key]

    # when
    audit = _OpAudit()
    with audit:
        partial_view = to_supervision_for_annotation(mirrored)

    # then - identical output via the tensor-read path (detach ops dispatched)
    assert any("detach" in name for name in audit.op_names())
    assert np.array_equal(partial_view.xyxy, reference.xyxy)
    assert np.array_equal(partial_view.class_id, reference.class_id)
    assert np.array_equal(partial_view.confidence, reference.confidence)
    # the surviving mirror keys of other boxes still never surface in .data
    assert not any(key in partial_view.data for key in HOST_MIRROR_KEYS)


def test_to_supervision_tracker_id_and_extra_keys_unaffected_by_mirror() -> None:
    # given
    mirrored = _mirrored_od_detections()
    for index, entry in enumerate(mirrored.bboxes_metadata):
        entry["tracker_id"] = 7 + index
        entry["time_in_zone"] = float(index)

    # when
    audit = _OpAudit()
    with audit:
        sv_view = to_supervision_for_annotation(mirrored)

    # then
    assert audit.ops == []
    assert np.array_equal(sv_view.tracker_id, np.asarray([7, 8, 9]))
    assert np.array_equal(sv_view.data["time_in_zone"], np.asarray([0.0, 1.0, 2.0]))


def test_to_supervision_keypoint_tuple_with_mirrored_bbox_is_read_free() -> None:
    # given
    detections = _mirrored_od_detections()
    key_points = KeyPoints(
        xy=torch.zeros((N_BOXES, 4, 2), dtype=torch.float32),
        class_id=detections.class_id.clone(),
        confidence=torch.zeros((N_BOXES, 4), dtype=torch.float32),
        image_metadata=detections.image_metadata,
    )

    # when
    audit = _OpAudit()
    with audit:
        sv_view = to_supervision_for_annotation((key_points, detections))

    # then
    assert audit.ops == []
    assert len(sv_view) == N_BOXES


def test_to_supervision_instance_segmentation_mirror_reads_only_the_mask() -> None:
    # given - dense-mask instance segmentation with a full mirror
    base = _mirrored_od_detections(2)
    masks = torch.zeros((2, SCENE_H, SCENE_W), dtype=torch.bool)
    masks[0, 5:15, 10:30] = True
    masks[1, 20:35, 40:55] = True
    detections = InstanceDetections(
        xyxy=base.xyxy,
        class_id=base.class_id,
        confidence=base.confidence,
        mask=masks,
        image_metadata=base.image_metadata,
        bboxes_metadata=base.bboxes_metadata,
    )

    # when
    audit = _OpAudit()
    with audit:
        sv_view = to_supervision_for_annotation(detections, materialise_masks=True)

    # then - the mask materialisation is untouched (its bulk transfer remains),
    # but xyxy/class_id/confidence never touch the device
    assert sv_view.mask is not None
    audit.assert_never_touched(
        detections.xyxy, detections.class_id, detections.confidence
    )


# --------------------------------------------------------------------------- #
# custom-python boundary: the mirror is internal transport only
# --------------------------------------------------------------------------- #


def test_representation_boundary_excludes_mirror_from_sv_data_and_roundtrip() -> None:
    # given
    mirrored = _mirrored_od_detections()

    # when
    sv_view = native_detections_to_sv(mirrored)
    rebuilt = sv_detections_to_native(sv_view)

    # then - legacy user code never sees the mirror, and the round-trip does
    # not re-attach one (user code may have edited the boxes)
    assert not any(key in sv_view.data for key in HOST_MIRROR_KEYS)
    assert rebuilt.bboxes_metadata is not None
    for entry in rebuilt.bboxes_metadata:
        assert not any(key in entry for key in HOST_MIRROR_KEYS)


# --------------------------------------------------------------------------- #
# label block end to end (GPU sprite path)
# --------------------------------------------------------------------------- #

_LABEL_RUN_KWARGS = dict(
    copy_image=True,
    color_palette="DEFAULT",
    palette_size=10,
    custom_colors=None,
    color_axis="CLASS",
    text="Class and Confidence",
    text_position="TOP_LEFT",
    text_color="WHITE",
    text_scale=1.0,
    text_thickness=1,
    text_padding=10,
    border_radius=0,
)


def _tensor_image(seed: int = 7) -> WorkflowImageData:
    rng = np.random.default_rng(seed)
    scene = rng.integers(0, 255, (3, SCENE_H, SCENE_W)).astype(np.uint8)
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        tensor_image=torch.from_numpy(scene),
    )


def test_label_block_run_with_mirror_never_reads_detection_tensors() -> None:
    # given
    block = LabelVisualizationBlockV1()
    detections = _mirrored_od_detections()

    # warm the sprite cache so the audited run measures the steady state
    block.run(image=_tensor_image(), predictions=detections, **_LABEL_RUN_KWARGS)

    # when
    audit = _OpAudit()
    with audit:
        result = block.run(
            image=_tensor_image(), predictions=detections, **_LABEL_RUN_KWARGS
        )

    # then - the sprite compositor works on the scene / sprites only; the
    # detection tensors are never read (no D2H outside the sprite uploads)
    audit.assert_never_touched(
        detections.xyxy, detections.class_id, detections.confidence
    )
    assert result["image"]._tensor_image is not None


def test_label_block_output_identical_with_and_without_mirror() -> None:
    # given
    mirrored = _mirrored_od_detections()
    stripped = _stripped_twin(mirrored)

    # when
    out_mirrored = LabelVisualizationBlockV1().run(
        image=_tensor_image(), predictions=mirrored, **_LABEL_RUN_KWARGS
    )["image"]
    out_stripped = LabelVisualizationBlockV1().run(
        image=_tensor_image(), predictions=stripped, **_LABEL_RUN_KWARGS
    )["image"]

    # then
    assert torch.equal(out_mirrored._tensor_image, out_stripped._tensor_image)


# --------------------------------------------------------------------------- #
# bounding-box block (GPU painter path)
# --------------------------------------------------------------------------- #

_BBOX_RUN_KWARGS = dict(
    copy_image=True,
    color_palette="DEFAULT",
    palette_size=10,
    custom_colors=None,
    color_axis="CLASS",
    thickness=2,
    roundness=0.0,
)


def test_bounding_box_block_run_with_mirror_never_reads_detection_tensors() -> None:
    # given
    block = BoundingBoxVisualizationBlockV1()
    detections = _mirrored_od_detections()

    # when
    audit = _OpAudit()
    with audit:
        result = block.run(
            image=_tensor_image(), predictions=detections, **_BBOX_RUN_KWARGS
        )

    # then
    audit.assert_never_touched(
        detections.xyxy, detections.class_id, detections.confidence
    )
    assert result["image"]._tensor_image is not None


def test_bounding_box_block_output_identical_with_and_without_mirror() -> None:
    # given
    mirrored = _mirrored_od_detections()
    stripped = _stripped_twin(mirrored)

    # when
    out_mirrored = BoundingBoxVisualizationBlockV1().run(
        image=_tensor_image(), predictions=mirrored, **_BBOX_RUN_KWARGS
    )["image"]
    out_stripped = BoundingBoxVisualizationBlockV1().run(
        image=_tensor_image(), predictions=stripped, **_BBOX_RUN_KWARGS
    )["image"]

    # then
    assert torch.equal(out_mirrored._tensor_image, out_stripped._tensor_image)
