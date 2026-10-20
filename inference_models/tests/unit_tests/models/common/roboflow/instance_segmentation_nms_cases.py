"""Finite cases and a frozen reference for ``run_nms_for_instance_segmentation``.

Pytest-independent so CPU tests and device runners share one case table.
Values are built deterministically on CPU, converted to the requested dtype,
moved to ``device`` and only then given their layout, so strides and storage
offsets are the intended ones on every device.
"""

import contextlib
import hashlib
from typing import Any, Callable, Dict, Iterator, List, Literal, NamedTuple, Union

import torch
import torchvision

# ``reference_run_nms_for_instance_segmentation`` below is the verbatim body of
# ``run_nms_for_instance_segmentation`` before ordered-index compaction: public
# commit 4ecfcbeccd366576fac2f9dd8b387453582687a7,
# inference_models/models/common/roboflow/post_processing.py lines 112-168,
# file sha256 254d638d04254196b67457224e0bce3a034eba7a2ddf864dc0669e4943489931.
REFERENCE_SOURCE_SHA256 = (
    "254d638d04254196b67457224e0bce3a034eba7a2ddf864dc0669e4943489931"
)
REFERENCE_SOURCE_LINES = (112, 168)


def reference_run_nms_for_instance_segmentation(
    output: torch.Tensor,
    conf_thresh: Union[float, torch.Tensor] = 0.25,
    iou_thresh: float = 0.45,
    max_detections: int = 100,
    class_agnostic: bool = False,
    box_format: Literal["xywh", "xyxy"] = "xywh",
) -> List[torch.Tensor]:
    """
    `conf_thresh`: scalar applies to all classes; 1-D tensor of shape
    (num_classes,) indexed by class_id for per-class thresholds.
    """
    bs = output.shape[0]
    boxes = output[:, :4, :]  # (N, 4, 8400)
    scores = output[:, 4:-32, :]  # (N, 80, 8400)
    masks = output[:, -32:, :]
    results = []

    for b in range(bs):
        bboxes = boxes[b].T  # (8400, 4)
        class_scores = scores[b].T  # (8400, 80)
        box_masks = masks[b].T
        class_conf, class_ids = class_scores.max(1)  # (8400,), (8400,)
        if isinstance(conf_thresh, torch.Tensor):
            mask = class_conf > conf_thresh.to(output.device)[class_ids]
        else:
            mask = class_conf > conf_thresh
        if mask.sum() == 0:
            results.append(torch.zeros((0, 38), device=output.device))
            continue
        bboxes = bboxes[mask]
        class_conf = class_conf[mask]
        class_ids = class_ids[mask]
        box_masks = box_masks[mask]
        if box_format == "xywh":
            # Vectorized [x, y, w, h] -> [x1, y1, x2, y2]
            xy = bboxes[:, :2]
            wh = bboxes[:, 2:]
            half_wh = wh / 2
            xyxy = torch.cat((xy - half_wh, xy + half_wh), 1)
        else:
            xyxy = bboxes
        # Class-agnostic NMS -> use dummy class ids
        nms_class_ids = torch.zeros_like(class_ids) if class_agnostic else class_ids
        keep = torchvision.ops.batched_nms(xyxy, class_conf, nms_class_ids, iou_thresh)
        keep = keep[:max_detections]
        detections = torch.cat(
            [
                xyxy[keep],
                class_conf[keep].unsqueeze(1),
                class_ids[keep].unsqueeze(1).float(),
                box_masks[keep],
            ],
            dim=1,
        )  # [x1, y1, x2, y2, conf, cls]
        results.append(detections)
    return results


class ParityCase(NamedTuple):
    """One finite input: a raw output tensor and the keyword options for NMS."""

    name: str
    raw: torch.Tensor
    options: Dict[str, Any]


class Outcome(NamedTuple):
    """``("ok", [tensors])`` or ``("error", (type name, message))``."""

    kind: Literal["ok", "error"]
    value: Any


def make_raw(
    batch: int,
    anchors: int,
    classes: int,
    seed: int,
    *,
    layout: str = "contiguous",
    dtype: torch.dtype = torch.float32,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Build raw ``(batch, 4 + classes + 32, anchors)`` output with edge values.

    With ``anchors >= 8`` every image holds: anchor 0 isolated at score 0.99
    with a NaN and a -0.0 coefficient; anchors 1-3 tied boxes and scores;
    anchor 4 with x = -0.0; anchor 5 at exactly 0.25 and anchor 6 at exactly
    0.5; anchor 7 with a NaN score. With ``batch > 3`` image 3 scores are all
    0.1 and image ``5 % batch`` scores are all 0.0.

    Layouts: ``contiguous``; ``transposed`` (channel stride 1, anchor stride
    4 + classes + 32); ``strided`` (anchor stride 2, storage offset 3 inside a
    buffer padded with 123.0);
    ``expanded`` (image 0 broadcast over the batch with batch stride 0).
    """
    generator = torch.Generator().manual_seed(seed)
    built_batch = 1 if layout == "expanded" else batch
    xy = torch.rand((built_batch, 2, anchors), generator=generator) * 100
    wh = torch.rand((built_batch, 2, anchors), generator=generator) * 60 + 1
    scores = torch.rand((built_batch, classes, anchors), generator=generator)
    coefficients = torch.randn((built_batch, 32, anchors), generator=generator)
    raw = torch.cat([xy, wh, scores, coefficients], dim=1)
    if anchors >= 8:
        for b in range(built_batch):
            raw[b, 4 : 4 + classes, 0:8] = 0.0
            raw[b, :4, 0] = torch.tensor([900.0, 900.0, 10.0, 10.0])
            raw[b, 4, 0] = 0.99
            raw[b, 4 + classes, 0] = float("nan")
            raw[b, 4 + classes + 1, 0] = -0.0
            raw[b, :4, 1:4] = torch.tensor([50.0, 50.0, 20.0, 20.0]).unsqueeze(1)
            raw[b, 4 + (classes - 1), 1:4] = 0.6
            raw[b, 0, 4] = -0.0
            raw[b, 4, 4] = 0.7
            raw[b, 4, 5] = 0.25
            raw[b, 4 + (classes - 1), 6] = 0.5
            raw[b, 4, 7] = float("nan")
        if built_batch > 3:
            raw[3, 4 : 4 + classes, :] = 0.1
            raw[5 % built_batch, 4 : 4 + classes, :] = 0.0
    raw = raw.to(dtype).to(device)

    if layout == "contiguous":
        return raw
    if layout == "transposed":
        return raw.transpose(1, 2).contiguous().transpose(1, 2)
    if layout == "strided":
        padded = torch.full(
            (batch, raw.shape[1], 2 * anchors + 3),
            123.0,
            dtype=dtype,
            device=raw.device,
        )
        padded[:, :, 3::2] = raw
        return padded[:, :, 3::2]
    if layout == "expanded":
        return raw.expand(batch, -1, -1)
    raise ValueError(f"unknown layout {layout!r}")


def _cycled_thresholds(classes: int) -> torch.Tensor:
    # Same shape and placement as ConfidenceFilter.get_threshold: 1-D CPU float32.
    return torch.tensor([(0.25, 0.5, 0.75)[i % 3] for i in range(classes)])


def _inference_tensor(raw: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        frozen = raw.clone()
    return frozen


def parity_cases(device: Union[str, torch.device] = "cpu") -> List[ParityCase]:
    """Finite table: original and new code must give identical outcomes.

    Names are identical for every device. Some entries fail inside the backend
    on some devices (for example CPU NMS rejects half precision); parity then
    means the same error.
    """
    b1 = make_raw(1, 64, 3, seed=1, device=device)
    # Dtype variants are cast once on CPU, so every device receives identical bytes.
    b1_double = make_raw(1, 64, 3, seed=1, dtype=torch.float64, device=device)
    b1_half = make_raw(1, 64, 3, seed=1, dtype=torch.float16, device=device)
    b1_bfloat16 = make_raw(1, 64, 3, seed=1, dtype=torch.bfloat16, device=device)
    b8 = make_raw(8, 48, 3, seed=2, layout="transposed", device=device)
    strided_c1 = make_raw(2, 40, 1, seed=3, layout="strided", device=device)
    strided_c3 = make_raw(2, 40, 3, seed=6, layout="strided", device=device)
    c80 = make_raw(8, 200, 80, seed=5, device=device)
    block = {"conf_thresh": 0.3, "iou_thresh": 0.3, "max_detections": 300}
    cases = [
        ("B1_C3_defaults", b1, {}),
        (
            "B1_C3_exact_0.25_no_suppression",
            b1,
            {"iou_thresh": 1.0, "max_detections": 1000},
        ),
        ("B1_C3_exact_0.5_xyxy", b1, {"conf_thresh": 0.5, "box_format": "xyxy"}),
        ("B1_C3_unknown_box_format_as_xyxy", b1, {"box_format": "cxcywh"}),
        ("B1_C3_all_below_threshold", b1, {"conf_thresh": 1.0}),
        ("B1_C3_max_detections_0", b1, {"max_detections": 0}),
        ("B1_C3_max_detections_1", b1, {"max_detections": 1}),
        ("B1_C3_max_detections_above_count", b1, {"max_detections": 1000}),
        ("B1_C3_max_detections_negative", b1, {"max_detections": -1}),
        ("B1_C3_float64", b1_double, {}),
        ("B1_C3_float16", b1_half, {}),
        ("B1_C3_bfloat16", b1_bfloat16, {}),
        # Empty images keep the default dtype, not the input dtype.
        ("B1_C3_float64_all_below_threshold", b1_double, {"conf_thresh": 1.0}),
        ("B1_C3_float16_all_below_threshold", b1_half, {"conf_thresh": 1.0}),
        ("B1_C3_inference_tensor", _inference_tensor(b1), {}),
        ("B8_C3_transposed_block_options", b8, dict(block)),
        ("B8_C3_transposed_block_agnostic", b8, {**block, "class_agnostic": True}),
        ("B8_C3_transposed_threshold_0", b8, {"conf_thresh": 0.0}),
        ("B2_C1_strided_per_class", strided_c1, {"conf_thresh": torch.tensor([0.25])}),
        (
            "B2_C3_strided_per_class_agnostic_xyxy",
            strided_c3,
            {
                "conf_thresh": _cycled_thresholds(3),
                "class_agnostic": True,
                "box_format": "xyxy",
            },
        ),
        (
            "B2_C3_strided_per_class_uniform",
            strided_c3,
            {"conf_thresh": torch.full((3,), 0.3)},
        ),
        ("B1_C3_zero_anchors", make_raw(1, 0, 3, seed=4, device=device), {}),
        ("B0_C3", make_raw(0, 16, 3, seed=7, device=device), {}),
        ("B8_C80_defaults", c80, {}),
        ("B8_C80_per_class", c80, {"conf_thresh": _cycled_thresholds(80)}),
        (
            "B4_C3_expanded",
            make_raw(4, 32, 3, seed=8, layout="expanded", device=device),
            {},
        ),
        # Out-of-contract (3, 1) thresholds make a 2-D mask; all-false takes the empty branch.
        (
            "B1_C3_rank2_threshold_all_false",
            b1,
            {"conf_thresh": torch.full((3, 1), 2.0)},
        ),
    ]
    parity = [ParityCase(name, raw, options) for name, raw, options in cases]

    return parity


def error_cases(device: Union[str, torch.device] = "cpu") -> List[ParityCase]:
    """Preexisting failures; both versions must raise the same error.

    Every entry fails in host-side shape checks, so none poisons a CUDA context.
    """
    b1 = make_raw(1, 64, 3, seed=1, device=device)
    cases = [
        ("rank2_threshold_mixed", b1, {"conf_thresh": torch.full((3, 1), 0.25)}),
        (
            "rank2_threshold_mixed_4_anchors",
            make_raw(1, 4, 3, seed=9, device=device),
            {"conf_thresh": torch.full((3, 1), 0.25)},
        ),
        ("scalar_tensor_threshold", b1, {"conf_thresh": torch.tensor(0.25)}),
        ("raw_2d", b1[0], {}),
        ("raw_without_class_channels", b1[:, :36], {}),
    ]
    errors = [ParityCase(name, raw, options) for name, raw, options in cases]

    return errors


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Raw bytes in logical order; NaN payloads and signed zeros survive."""
    # A fresh buffer: size-1 dimensions may keep any stride under contiguous().
    flat = torch.empty(tensor.numel(), dtype=tensor.dtype)
    flat.copy_(tensor.detach().to("cpu").reshape(-1))
    data = flat.view(torch.uint8).numpy().tobytes()

    return data


def call_outcome(
    function: Callable[..., List[torch.Tensor]],
    raw: torch.Tensor,
    options: Dict[str, Any],
) -> Outcome:
    """Call ``function(raw, **options)`` and capture its result or its error."""
    try:
        result = function(raw, **options)
    except Exception as error:  # the outcome itself is the compared value
        return Outcome("error", (type(error).__name__, str(error)))

    return Outcome("ok", result)


def _describe(tensor: torch.Tensor) -> tuple:
    return (
        tensor.dtype,
        tensor.device,
        tuple(tensor.shape),
        tensor.stride(),
        tensor_bytes(tensor),
    )


def assert_same_outcome(left: Outcome, right: Outcome) -> None:
    """Require identical errors, or identical tensors bit for bit and in layout."""
    assert left.kind == right.kind, (left.kind, right.kind, left.value, right.value)
    if left.kind == "error":
        assert left.value == right.value
        return

    assert len(left.value) == len(right.value)
    for index, (a, b) in enumerate(zip(left.value, right.value)):
        assert _describe(a) == _describe(b), f"image {index} differs"


def shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Whether two tensors use one storage; zero-byte storages never alias."""
    a, b = left.untyped_storage(), right.untyped_storage()
    shared = a.nbytes() > 0 and b.nbytes() > 0 and a.data_ptr() == b.data_ptr()

    return shared


@contextlib.contextmanager
def record_nms_inputs() -> Iterator[List[Dict[str, Any]]]:
    """Record digests of every ``torchvision.ops.batched_nms`` call.

    Records keep no tensor references. The real function is always restored,
    including when the body raises. Digests copy to host, so do not time this.
    """
    real = torchvision.ops.batched_nms
    records: List[Dict[str, Any]] = []

    def recorder(boxes, scores, idxs, iou_threshold):
        records.append(
            {
                name: (
                    str(value.dtype),
                    tuple(value.shape),
                    hashlib.sha256(tensor_bytes(value)).hexdigest(),
                )
                for name, value in (
                    ("boxes", boxes),
                    ("scores", scores),
                    ("idxs", idxs),
                )
            }
            | {"iou_threshold": float(iou_threshold)}
        )
        keep = real(boxes, scores, idxs, iou_threshold)
        return keep

    torchvision.ops.batched_nms = recorder
    try:
        yield records
    finally:
        torchvision.ops.batched_nms = real
