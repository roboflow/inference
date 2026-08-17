"""Whole-viz-phase host<->device transfer audit.

Runs the tensor-native mask compositor followed by the label sprite painter —
the full viz phase of the tensor pipeline — under ``TorchDispatchMode`` and
asserts the combined op trace is transfer-clean:

* no ``aten.nonzero`` and no ``aten._local_scalar_dense`` (both force a
  device→host sync),
* no device→host ``_to_copy`` / ``copy_`` at all,
* every host→device ``_to_copy`` / ``copy_`` carries ``non_blocking=True``.

The dispatcher exposes the flag directly (``copy_`` passes it as the third
positional argument, ``_to_copy`` in kwargs). On the CPU test host every
tensor lives on the CPU, so cross-device classification is vacuous there —
the flag is asserted anyway via the structural fact that on the label tensor
path EVERY ``aten.copy_`` is a staged host→device upload (the ring table
upload plus, cold, the sprite pixel/index payloads), while the mask phase
dispatches only same-device copies. The identical assertions become the real
H2D/D2H checks when this suite runs on a CUDA host.

Also pins the warm-path dispatch budget: the label paste enqueues a fixed,
label-count-independent number of ops.
"""

from collections import Counter

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from inference.core.workflows.core_steps.common.tensor_native import (
    attach_native_detection_metadata,
)
from inference.core.workflows.core_steps.visualizations.label.v1_tensor import (
    LabelVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.mask.v1_tensor import (
    MaskVisualizationBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections

CLASS_NAMES = {0: "cat", 1: "dog", 2: "goggles", 3: "bird"}
SCENE_H, SCENE_W = 480, 640
N_BOXES = 3

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

_MASK_RUN_KWARGS = dict(
    copy_image=True,
    color_palette="DEFAULT",
    palette_size=10,
    custom_colors=None,
    color_axis="CLASS",
    opacity=0.5,
)


class _TransferAudit(TorchDispatchMode):
    """Record every dispatched aten op; for ``copy_`` / ``_to_copy`` also the
    source/target devices and the ``non_blocking`` flag exactly as the
    dispatcher exposes them."""

    def __init__(self):
        super().__init__()
        self.ops = []
        self.copies = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        self.ops.append(name)
        if name == "aten.copy_.default":
            self.copies.append(
                {
                    "op": name,
                    "dst": args[0].device,
                    "src": args[1].device,
                    "non_blocking": (
                        bool(args[2])
                        if len(args) > 2
                        else bool(kwargs.get("non_blocking", False))
                    ),
                }
            )
        elif name == "aten._to_copy.default":
            source = args[0]
            target = kwargs.get("device")
            self.copies.append(
                {
                    "op": name,
                    "dst": (
                        torch.device(target) if target is not None else source.device
                    ),
                    "src": source.device,
                    "non_blocking": bool(kwargs.get("non_blocking", False)),
                }
            )
        return func(*args, **kwargs)


def _assert_no_forbidden_ops(ops) -> None:
    offenders = [
        name for name in ops if "nonzero" in name or "_local_scalar_dense" in name
    ]
    assert offenders == [], f"sync-forcing ops dispatched: {offenders}"


def _assert_no_device_to_host(copies) -> None:
    downloads = [
        record
        for record in copies
        if record["src"].type == "cuda" and record["dst"].type == "cpu"
    ]
    assert downloads == [], f"device->host transfers dispatched: {downloads}"


def _assert_uploads_non_blocking(copies) -> None:
    blocking = [
        record
        for record in copies
        if record["src"].type == "cpu"
        and record["dst"].type == "cuda"
        and not record["non_blocking"]
    ]
    assert blocking == [], f"blocking host->device transfers dispatched: {blocking}"


def _tensor_image(seed: int = 7) -> WorkflowImageData:
    rng = np.random.default_rng(seed)
    scene = rng.integers(0, 255, (3, SCENE_H, SCENE_W)).astype(np.uint8)
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        tensor_image=torch.from_numpy(scene),
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
        image=_tensor_image(),
        class_names=CLASS_NAMES,
        prediction_type="object-detection",
    )


def _disjoint_od_detections(n: int) -> Detections:
    """Vertically separated boxes whose TOP_LEFT labels can never overlap, so
    the paste's owner-resolution branch state is identical for every ``n``."""
    boxes = np.asarray(
        [
            [60.0, 60.0 + 110.0 * index, 240.0, 130.0 + 110.0 * index]
            for index in range(n)
        ],
        dtype=np.float32,
    )
    detections = Detections(
        xyxy=torch.tensor(boxes, dtype=torch.float32),
        class_id=torch.tensor(list(range(n)), dtype=torch.long),
        confidence=torch.tensor(np.linspace(0.5, 0.9, n), dtype=torch.float32),
        image_metadata=None,
        bboxes_metadata=None,
    )
    return attach_native_detection_metadata(
        detections=detections,
        image=_tensor_image(),
        class_names=CLASS_NAMES,
        prediction_type="object-detection",
    )


def _mirrored_is_detections(n: int = 2) -> InstanceDetections:
    base = _mirrored_od_detections(n)
    masks = torch.zeros((n, SCENE_H, SCENE_W), dtype=torch.bool)
    masks[0, 70:180, 50:190] = True
    if n > 1:
        masks[1, 100:250, 260:410] = True
    return InstanceDetections(
        xyxy=base.xyxy,
        class_id=base.class_id,
        confidence=base.confidence,
        mask=masks,
        image_metadata=base.image_metadata,
        bboxes_metadata=base.bboxes_metadata,
    )


# --------------------------------------------------------------------------- #
# whole viz phase: mask composite -> label paste, warm caches
# --------------------------------------------------------------------------- #


def test_viz_phase_mask_then_label_combined_trace_is_transfer_clean() -> None:
    # given
    mask_block = MaskVisualizationBlockV1()
    label_block = LabelVisualizationBlockV1()
    segmentation = _mirrored_is_detections()
    detections = _mirrored_od_detections()
    # warm every cache: the mask palette LUT, the label sprite cache + flat
    # templates + the pinned table ring slabs
    mask_block.run(image=_tensor_image(), predictions=segmentation, **_MASK_RUN_KWARGS)
    label_block.run(image=_tensor_image(), predictions=detections, **_LABEL_RUN_KWARGS)

    # when - the audited steady-state viz phase, label consuming mask's output
    mask_audit = _TransferAudit()
    with mask_audit:
        masked = mask_block.run(
            image=_tensor_image(), predictions=segmentation, **_MASK_RUN_KWARGS
        )
    label_audit = _TransferAudit()
    with label_audit:
        result = label_block.run(
            image=masked["image"], predictions=detections, **_LABEL_RUN_KWARGS
        )

    # then - the combined trace never syncs and never crosses the bus blocking
    combined_ops = mask_audit.ops + label_audit.ops
    combined_copies = mask_audit.copies + label_audit.copies
    _assert_no_forbidden_ops(combined_ops)
    _assert_no_device_to_host(combined_copies)
    _assert_uploads_non_blocking(combined_copies)
    # mask phase: everything is device-resident — its only in-place copy is
    # the same-device staged scene write, no host->device staging at all
    for record in mask_audit.copies:
        if record["op"] == "aten.copy_.default":
            assert record["src"] == record["dst"]
    # label phase: with a warm sprite cache EVERY `copy_` is a staged
    # host->device upload — exactly one (the packed table through the pinned
    # ring) and it must carry non_blocking=True, also on the CPU host where
    # the copy itself is trivial
    label_copies = [
        record for record in label_audit.copies if record["op"] == "aten.copy_.default"
    ]
    assert len(label_copies) == 1, f"expected only the table upload: {label_copies}"
    assert all(record["non_blocking"] for record in label_copies)
    assert result["image"]._tensor_image is not None


def test_cold_cache_label_run_stages_every_upload_non_blocking() -> None:
    # given - a fresh block: sprite cache, flat templates and ring all cold
    block = LabelVisualizationBlockV1()
    detections = _mirrored_od_detections()
    image = _tensor_image()

    # when
    audit = _TransferAudit()
    with audit:
        result = block.run(image=image, predictions=detections, **_LABEL_RUN_KWARGS)

    # then - even the cache-miss payloads (sprite colors + flat-index
    # templates) and the first table upload are staged non-blocking
    _assert_no_forbidden_ops(audit.ops)
    _assert_no_device_to_host(audit.copies)
    _assert_uploads_non_blocking(audit.copies)
    cold_copies = [
        record for record in audit.copies if record["op"] == "aten.copy_.default"
    ]
    # one colors payload + one flat-index template per distinct sprite, plus
    # the packed table
    assert len(cold_copies) == 2 * N_BOXES + 1, f"unexpected copies: {cold_copies}"
    assert all(record["non_blocking"] for record in cold_copies), cold_copies
    assert result["image"]._tensor_image is not None


# --------------------------------------------------------------------------- #
# micro-check: warm paste dispatch budget is flat in the label count
# --------------------------------------------------------------------------- #


def test_warm_label_paste_dispatch_count_is_flat_in_label_count() -> None:
    # given / when
    counts = {}
    traces = {}
    for n in (2, 3, 4):
        block = LabelVisualizationBlockV1()
        detections = _disjoint_od_detections(n)
        block.run(
            image=_tensor_image(), predictions=detections, **_LABEL_RUN_KWARGS
        )  # warm sprites, flat templates, ring slabs
        audit = _TransferAudit()
        audited_image = _tensor_image()
        with audit:
            block.run(image=audited_image, predictions=detections, **_LABEL_RUN_KWARGS)
        counts[n] = len(audit.ops)
        traces[n] = Counter(audit.ops)

    # then - flat in N, and no growth against the pre-ring baseline: the warm
    # paste measured 15 dispatched ops before this change (its pageable packed
    # upload was a from_numpy lift + `.to`); the ring swaps that for the one
    # staged `copy_`, keeping the budget at 15 (clone, copy_, 3 slices,
    # arange, repeat_interleave, index_select, 2 cats, index, add, t, view,
    # index_put_).
    assert counts[2] == counts[3] == counts[4], counts
    assert counts[4] <= 15, traces[4]
