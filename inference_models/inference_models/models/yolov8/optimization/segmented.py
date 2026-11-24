"""Opt-in segmented YOLOv8 decoding with per-call output and scratch storage.

The base decoder runs ``torchvision.ops.batched_nms`` once per image. This stage
runs the same per-image computation for a whole batch::

    threshold, compaction       the base operations over (B, N); one count wait
    segmented_nms               B maxima, B stable sorts, one greedy kernel pass
    rescale, round, int         the base operation order on padded (B, cap) rows
    completion                  kept counts copied; one event; wait() slices

Triton is imported only when the first supported request initializes the
kernels on a device. That initialization calibrates the IoU arithmetic against
the installed ``torchvision.ops.nms`` (see ``calibration.calibrate``).
"""

import importlib.util
import math
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from numbers import Integral, Real
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torchvision

from inference_models import Detections
from inference_models.errors import ModelRuntimeError
from inference_models.models.optimization.contracts import (
    CompatibilityResult,
    DeviceCompatibility,
    ExecutionContext,
    InputCompatibility,
    OptimizationMetadata,
    OptimizationStage,
    immutable_mapping,
)

from .contracts import (
    PostprocessRequest,
    PostprocessResult,
    _completion_event,
    _run_on_stream,
)
from .segmented_nms import DEFAULT_SCRATCH_LIMIT_BYTES, NmsKernels, segmented_nms

SEGMENTED_IMPLEMENTATION_ID = "segmented"
# The coordinate-trick routing limit and the stable sort are read from this tag;
# arithmetic calibration alone cannot qualify another torchvision version.
QUALIFIED_TORCHVISION_VERSION = "0.28.0"
TRITON_COMPONENT = "triton"
TORCHVISION_COMPONENT = f"torchvision=={QUALIFIED_TORCHVISION_VERSION}"
NONZERO_STATIC_COMPONENT = "torch.nonzero_static"
# False for subclasses of the qualified model class (YOLOv9, YOLOv11, ...).
MODEL_COMPONENT = "yolov8_trt"
_METADATA_FIELDS = (
    ("pad_left",),
    ("pad_top",),
    ("scale_width",),
    ("scale_height",),
    ("static_crop_offset", "offset_x"),
    ("static_crop_offset", "offset_y"),
    ("original_size", "width"),
    ("original_size", "height"),
)


@lru_cache(maxsize=1)
def segmented_runtime_components() -> Mapping[str, bool]:
    """Report package facts that qualify the segmented stage, without importing Triton.

    Returns:
        Read-only mapping of ``TRITON_COMPONENT``, ``TORCHVISION_COMPONENT`` and
        ``NONZERO_STATIC_COMPONENT`` to availability.
    """
    torchvision_release = torchvision.__version__.split("+", 1)[0]
    components = immutable_mapping(
        {
            TRITON_COMPONENT: importlib.util.find_spec("triton") is not None,
            TORCHVISION_COMPONENT: torchvision_release == QUALIFIED_TORCHVISION_VERSION,
            NONZERO_STATIC_COMPONENT: hasattr(torch, "nonzero_static"),
        }
    )

    return components


@dataclass(frozen=True)
class PaddedDecode:
    """Decoded rows of every image, padded to ``cap``; counts readable after the wait.

    ``kept`` is a host copy enqueued on the current stream; read it only after
    that stream's work is complete.

    Attributes:
        xyxy: (B, cap, 4) int32 boxes in original-image coordinates.
        class_id: (B, cap) int32 class ids.
        confidence: (B, cap) float32 scores.
        kept: (B,) int64 host copy of kept counts per image.
        routes: Images per NMS route of this call.
        scratch_bytes: IoU bitmask bytes of this call.
    """

    xyxy: torch.Tensor
    class_id: torch.Tensor
    confidence: torch.Tensor
    kept: torch.Tensor
    routes: Dict[str, int]
    scratch_bytes: int

    def per_image(self) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Each image's (xyxy, class_id, confidence) prefix, as views.

        Returns:
            One triple per image, in batch order.
        """
        result = [
            (
                self.xyxy[image, :count],
                self.class_id[image, :count],
                self.confidence[image, :count],
            )
            for image, count in enumerate(self.kept.tolist())
        ]

        return result

    def to_detections(self) -> List[Detections]:
        """Wrap each image's prefix views as detections.

        Returns:
            One ``Detections`` per image, in batch order.
        """
        detections = [
            Detections(xyxy=xyxy, class_id=class_id, confidence=confidence)
            for xyxy, class_id, confidence in self.per_image()
        ]

        return detections


def decode_padded(
    output: torch.Tensor,
    images_metadata: List[Any],
    *,
    conf_thresh: Union[float, torch.Tensor],
    iou_thresh: float,
    max_detections: int,
    class_agnostic: bool,
    kernels: NmsKernels,
    scratch_limit_bytes: int = DEFAULT_SCRATCH_LIMIT_BYTES,
) -> PaddedDecode:
    """Threshold, segmented NMS, rescale and integer boxes, padded per image.

    The values equal the base decoder's for a float32 output; only the NMS loop
    and the result layout differ.

    Args:
        output: Raw float32 (B, 4 + C, N) xywh output. Not modified.
        images_metadata: One ``PreProcessingMetadata`` per image.
        conf_thresh: Scalar threshold, or a (C,) tensor indexed by class id.
        iou_thresh: IoU threshold for NMS.
        max_detections: Maximum detections kept per image, ``>= 0``.
        class_agnostic: Run NMS with a single dummy class.
        kernels: The NMS backend.
        scratch_limit_bytes: See ``segmented_nms``.

    Returns:
        Padded int32 boxes and class ids, float32 confidences, and kept counts
        whose host copy is valid once the current stream's work is complete.
    """
    batch_size = output.shape[0]
    xyxy, scores, ids, image_index, counts, counts_device = _candidates(
        output, conf_thresh=conf_thresh
    )
    nms_ids = torch.zeros_like(ids) if class_agnostic else ids

    nms = segmented_nms(
        xyxy,
        scores,
        nms_ids,
        image_index=image_index,
        counts=counts,
        counts_device=counts_device,
        iou_threshold=iou_thresh,
        max_detections=max_detections,
        kernels=kernels,
        scratch_limit_bytes=scratch_limit_bytes,
    )
    cap = nms.rows.shape[1]
    rows = nms.rows.flatten()
    detections = torch.cat((xyxy[rows], scores[rows, None], ids[rows, None].float()), 1)
    row_image = torch.arange(batch_size, device=output.device).repeat_interleave(cap)
    _rescale_rows_in_place(
        detections, row_image=row_image, images_metadata=images_metadata
    )

    padded = PaddedDecode(
        xyxy=detections[:, :4].round().int().view(batch_size, cap, 4),
        class_id=detections[:, 5].int().view(batch_size, cap),
        confidence=detections[:, 4].view(batch_size, cap),
        kept=nms.kept.to("cpu", non_blocking=True),
        routes=nms.routes,
        scratch_bytes=nms.scratch_bytes,
    )

    return padded


@dataclass(frozen=True)
class KernelInitialization:
    """One-time outcome of building the NMS kernels for one device.

    Attributes:
        device: Device the kernels were built and calibrated on.
        status: ``calibrated``, ``calibration_unqualified`` or ``injected``.
        seconds: Wall time of the build and calibration.
        kernels: Usable kernels, or None when calibration did not qualify.
        identity: Kernel identity facts; empty without kernels.
        report: Calibration report; empty for injected kernels.
        reason: Why the kernels are unusable, or None.
    """

    device: str
    status: str
    seconds: float
    kernels: Optional[NmsKernels] = None
    identity: Mapping[str, Any] = field(default_factory=immutable_mapping)
    report: Mapping[str, Any] = field(default_factory=immutable_mapping)
    reason: Optional[str] = None

    def summary(self) -> Dict[str, Any]:
        """Bounded facts for per-call diagnostics.

        Returns:
            Device, status, seconds, matching arithmetic variants and reason.
        """
        summary = {
            "device": self.device,
            "status": self.status,
            "seconds": self.seconds,
            "matching": self.report.get("matching"),
            "reason": self.reason,
        }

        return summary


class SegmentedPostprocessor:
    """Decode a whole batch with one segmented NMS pass; opt-in, CUDA only.

    Production selection rejects CPU contexts. CPU numerical tests may inject
    ``kernels`` and call ``decode_padded`` or ``postprocess`` directly.

    Args:
        kernels: NMS backend. None builds and calibrates the Triton kernels
            once per device, on the first supported request.
        scratch_limit_bytes: Bitmask bytes per call above which every image of
            the call takes the ``torchvision.ops.batched_nms`` route (counted).

    Raises:
        ValueError: If ``scratch_limit_bytes`` is negative.
    """

    metadata = OptimizationMetadata(
        implementation_id=SEGMENTED_IMPLEMENTATION_ID,
        stage=OptimizationStage.POSTPROCESS,
        version="1",
        target=DeviceCompatibility(device_kind="gpu", device_types=("cuda",)),
        inputs=InputCompatibility(
            scenarios=("yolov8-object-detection",),
            axis_constraints={"batch": ">= 2"},
            dtypes=("float32",),
            layouts=("(B, 4 + C, N) raw, non-fused",),
        ),
        dependencies=(
            TRITON_COMPONENT,
            TORCHVISION_COMPONENT,
            NONZERO_STATIC_COMPONENT,
            MODEL_COMPONENT,
        ),
        fallback_id="base",
        changes_numerics=False,
        supports_concurrency=True,
        supports_cuda_graphs=False,
        output_contract={
            "xyxy": "int32 (n, 4), rounded original-image pixels",
            "class_id": "int32 (n,)",
            "confidence": "float32 (n,)",
            "order": "per-image batched_nms order, cut to max_detections",
            "storage": "views of per-call padded (B, cap) tensors",
        },
        numerical_behavior=(
            "Base thresholding, per-image coordinate-trick offsets, stable "
            "descending sort, float32 IoU suppression and rescale order. The IoU "
            "arithmetic variant is chosen by empirical per-device calibration "
            "against the installed torchvision.ops.nms. Images above torchvision's "
            "vanilla-branch size, and calls above the scratch limit, run "
            "torchvision.ops.batched_nms; both are counted in routes."
        ),
        stream_behavior=(
            "Caller ExecutionContext.current_stream; awaits the request "
            "ready_event. Host waits: candidate counts, rescale constants upload "
            "and library-route images. One completion event; no stage "
            "synchronize."
        ),
    )

    def __init__(
        self,
        *,
        kernels: Optional[NmsKernels] = None,
        scratch_limit_bytes: int = DEFAULT_SCRATCH_LIMIT_BYTES,
    ) -> None:
        if scratch_limit_bytes < 0:
            raise ValueError(
                f"scratch_limit_bytes must be >= 0, got {scratch_limit_bytes}"
            )

        self._injected_kernels = kernels
        self._scratch_limit_bytes = scratch_limit_bytes
        self._initializations: Dict[str, KernelInitialization] = {}
        self._lock = threading.Lock()

    @property
    def scratch_limit_bytes(self) -> int:
        """Return the bitmask bytes per call above which the library route runs.

        Returns:
            Scratch limit in bytes.
        """
        return self._scratch_limit_bytes

    def is_compatible(self, context: ExecutionContext) -> bool:
        """Check package and model facts without importing Triton.

        The registry reports missing components named by the context; this
        check also covers contexts that omit them.

        Args:
            context: Runtime facts.

        Returns:
            Whether the installed packages qualify and the context does not
            mark the model as unqualified.
        """
        packages = dict(segmented_runtime_components())
        if self._injected_kernels is not None:
            packages.pop(TRITON_COMPONENT)
        compatible = (
            all(packages.values())
            and context.runtime_components.get(MODEL_COMPONENT) is not False
        )

        return compatible

    def check_request_compatibility(
        self, *, request: PostprocessRequest, context: ExecutionContext
    ) -> CompatibilityResult:
        """Check shapes, dtypes, options and metadata without reading GPU values.

        Args:
            request: Resolved output, metadata and options.
            context: Target and caller stream.

        Returns:
            Compatible, or the first reason the base path must run instead.
        """
        reason = _request_incompatibility(request=request, context=context)
        if reason is not None:
            result = CompatibilityResult.incompatible(reason)
        else:
            result = CompatibilityResult.compatible()

        return result

    def check_runtime_compatibility(
        self, *, request: PostprocessRequest, context: ExecutionContext
    ) -> CompatibilityResult:
        """Build and calibrate the kernels once per device, on the caller stream.

        Call only for a request accepted by ``check_request_compatibility``.
        Triton import, compilation and CUDA errors propagate and leave the
        device uninitialized.

        Args:
            request: A supported request; its output device selects the kernels.
            context: Target and caller stream.

        Returns:
            Compatible, or the recorded calibration reason for this device.
        """
        initialization = self._initialization(
            device=request.model_results.device, context=context
        )
        if initialization.kernels is None:
            result = CompatibilityResult.incompatible(initialization.reason or "")
        else:
            result = CompatibilityResult.compatible()

        return result

    def initialization_status(
        self, *, device: Union[str, torch.device]
    ) -> Optional[KernelInitialization]:
        """Read the one-time kernel initialization of a device.

        Args:
            device: Device to inspect.

        Returns:
            The recorded initialization, or None before the first supported
            request on that device or after a propagated failure.
        """
        status = self._initializations.get(str(torch.device(device)))

        return status

    def postprocess(
        self, request: PostprocessRequest, context: ExecutionContext
    ) -> PostprocessResult:
        """Enqueue segmented decoding on the caller's stream.

        Args:
            request: A request accepted by ``check_request_compatibility``.
            context: Target and caller stream.

        Returns:
            Per-call result holding its own padded buffers and host counts;
            call wait before consuming detections.

        Raises:
            ModelRuntimeError: If calibration did not qualify kernels for the
                request's device.
        """
        initialization = self._initialization(
            device=request.model_results.device, context=context
        )
        kernels = initialization.kernels
        if kernels is None:
            raise ModelRuntimeError(
                message=(
                    f"Segmented postprocessing is unavailable on "
                    f"{initialization.device}: {initialization.reason}"
                ),
                help_url="https://inference-models.roboflow.com/errors/models-runtime/#modelruntimeerror",
            )

        with _run_on_stream(request, context):
            padded = decode_padded(
                request.model_results,
                request.pre_processing_meta,
                conf_thresh=request.threshold,
                iou_thresh=float(request.iou_threshold),
                max_detections=int(request.max_detections),
                class_agnostic=request.class_agnostic_nms,
                kernels=kernels,
                scratch_limit_bytes=self._scratch_limit_bytes,
            )
            event = _completion_event(context)

        details = {
            "routes": dict(sorted(padded.routes.items())),
            "scratch_bytes": padded.scratch_bytes,
            "scratch_limit_bytes": self._scratch_limit_bytes,
            "kernels": initialization.identity,
            "calibration": initialization.summary(),
        }
        result = PostprocessResult(
            ready_event=event, _materialize=padded.to_detections, details=details
        )

        return result

    def _initialization(
        self, *, device: torch.device, context: ExecutionContext
    ) -> KernelInitialization:
        key = str(device)
        initialization = self._initializations.get(key)
        if initialization is not None:
            return initialization

        with self._lock:
            initialization = self._initializations.get(key)
            if initialization is None:
                initialization = self._initialize(device=device, context=context)
                self._initializations[key] = initialization

        return initialization

    def _initialize(
        self, *, device: torch.device, context: ExecutionContext
    ) -> KernelInitialization:
        if self._injected_kernels is not None:
            initialization = KernelInitialization(
                device=str(device),
                status="injected",
                seconds=0.0,
                kernels=self._injected_kernels,
                identity=immutable_mapping(self._injected_kernels.identity()),
            )
            return initialization

        stream = context.current_stream
        if stream is None:
            raise ValueError(
                "Kernel calibration requires ExecutionContext.current_stream"
            )

        # Imports Triton; static selection has already found it installed.
        from .calibration import CalibrationError, build_kernels

        started = time.perf_counter()
        with torch.cuda.device(device), torch.cuda.stream(stream):
            try:
                kernels, report = build_kernels(device)
            except CalibrationError as error:
                initialization = KernelInitialization(
                    device=str(device),
                    status="calibration_unqualified",
                    seconds=round(time.perf_counter() - started, 3),
                    report=immutable_mapping(error.report),
                    reason=f"IoU arithmetic calibration was inconclusive: {error}",
                )
                return initialization

        initialization = KernelInitialization(
            device=str(device),
            status="calibrated",
            seconds=round(time.perf_counter() - started, 3),
            kernels=kernels,
            identity=immutable_mapping(kernels.identity()),
            report=immutable_mapping(report),
        )

        return initialization


def _request_incompatibility(
    *, request: PostprocessRequest, context: ExecutionContext
) -> Optional[str]:
    # Reads only shapes, dtypes, devices and host values: never GPU contents.
    if request.fused:
        return "fused export output already applied NMS"

    output = request.model_results
    if not isinstance(output, torch.Tensor):
        return "model output is not a tensor"
    if not output.is_cuda:
        return f"model output is on {output.device}, not CUDA"
    if output.dtype != torch.float32:
        return f"model output dtype {output.dtype} is not torch.float32"
    if torch.get_default_dtype() != torch.float32:
        return "torch default dtype is not float32; empty results would differ"
    if output.dim() != 3 or output.shape[1] < 5:
        return f"model output shape {tuple(output.shape)} is not (B, 4 + C, N), C >= 1"

    batch_size = output.shape[0]
    if batch_size < 2:
        return f"batch size {batch_size} is below 2"
    metadata = request.pre_processing_meta
    if not isinstance(metadata, Sequence) or len(metadata) != batch_size:
        return f"pre_processing_meta does not hold one entry per image of {batch_size}"

    stream = context.current_stream
    if stream is None:
        return "ExecutionContext.current_stream is missing"
    if stream.device != output.device or context.device != str(output.device):
        return (
            f"model output device {output.device} differs from context device "
            f"{context.device} or its stream"
        )

    option_reason = _options_incompatibility(
        request=request, class_count=output.shape[1] - 4
    )
    if option_reason is not None:
        return option_reason

    for index, image_metadata in enumerate(metadata):
        field_name = _non_numeric_metadata_field(image_metadata)
        if field_name is not None:
            return f"pre_processing_meta[{index}].{field_name} is not a host number"

    return None


def _options_incompatibility(
    *, request: PostprocessRequest, class_count: int
) -> Optional[str]:
    max_detections = request.max_detections
    if (
        isinstance(max_detections, bool)
        or not isinstance(max_detections, Integral)
        or max_detections < 0
    ):
        return "max_detections must be a non-negative host integer"

    iou_threshold = request.iou_threshold
    if (
        isinstance(iou_threshold, bool)
        or not isinstance(iou_threshold, Real)
        or not math.isfinite(iou_threshold)
        or not 0 <= iou_threshold <= 1
    ):
        return "iou_threshold must be a finite host number in [0, 1]"

    threshold = request.threshold
    if isinstance(threshold, torch.Tensor):
        if threshold.dim() != 1 or threshold.shape[0] != class_count:
            return (
                f"per-class threshold shape {tuple(threshold.shape)} is not "
                f"({class_count},)"
            )
        if threshold.device not in (torch.device("cpu"), request.model_results.device):
            return f"per-class threshold is on {threshold.device}"
    elif isinstance(threshold, bool) or not isinstance(threshold, Real):
        return f"confidence threshold {type(threshold).__name__} is not a host number"

    return None


def _non_numeric_metadata_field(metadata: Any) -> Optional[str]:
    for path in _METADATA_FIELDS:
        value = metadata
        for name in path:
            value = getattr(value, name, None)
        if isinstance(value, bool) or not isinstance(value, Real):
            field_name = ".".join(path)
            return field_name

    return None


def _candidates(
    output: torch.Tensor, *, conf_thresh: Union[float, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    # run_nms_for_object_detection up to its NMS call, op for op, over the whole
    # batch, plus the device counts reduced to a host list (one wait). Rows are
    # image-major with each image's anchors ascending, as its per-image nonzero.
    class_conf, class_ids = output[:, 4:, :].max(1)  # (B, N) each
    if isinstance(conf_thresh, torch.Tensor):
        mask = class_conf > conf_thresh.to(output.device)[class_ids]
    else:
        mask = class_conf > conf_thresh
    counts_device = mask.sum(1)
    counts = counts_device.tolist()
    total = sum(counts)
    if total == 0:
        no_rows = torch.zeros((0,), dtype=torch.long, device=output.device)
        empty = (output.new_zeros((0, 4)), output.new_zeros((0,)), no_rows, no_rows)
        return (*empty, counts, counts_device)

    image_index, anchor_index = torch.nonzero_static(mask, size=total).unbind(1)
    boxes = output[image_index, :4, anchor_index]  # (K, 4)
    scores = class_conf[image_index, anchor_index]
    ids = class_ids[image_index, anchor_index]
    half_wh = boxes[:, 2:] / 2
    xyxy = torch.cat((boxes[:, :2] - half_wh, boxes[:, :2] + half_wh), 1)

    return xyxy, scores, ids, image_index, counts, counts_device


def _rescale_rows_in_place(
    detections: torch.Tensor, *, row_image: torch.Tensor, images_metadata: Sequence[Any]
) -> None:
    # The library's rescale_image_detections op sequence (sub pad, div scale,
    # add crop, clamp to the original size), one upload for the whole batch.
    constants = torch.tensor(
        [_rescale_constants(metadata) for metadata in images_metadata],
        dtype=detections.dtype,
    ).to(detections.device)
    row_constants = constants[row_image]  # (K, 4, 4)

    boxes = detections[:, :4]
    boxes.sub_(row_constants[:, 0])
    boxes.div_(row_constants[:, 1])
    if any(_has_static_crop(metadata) for metadata in images_metadata):
        # Rows of uncropped images get +0: only -0.0 becomes +0.0, which
        # round().int() maps to the same 0.
        boxes.add_(row_constants[:, 2])
    original_size = row_constants[:, 3]
    boxes.clamp_(min=torch.zeros_like(original_size), max=original_size)


def _rescale_constants(metadata: Any) -> List[List[float]]:
    crop = metadata.static_crop_offset
    size = metadata.original_size
    constants = [
        [metadata.pad_left, metadata.pad_top] * 2,
        [metadata.scale_width, metadata.scale_height] * 2,
        [crop.offset_x, crop.offset_y] * 2,
        [size.width, size.height] * 2,
    ]

    return constants


def _has_static_crop(metadata: Any) -> bool:
    crop = metadata.static_crop_offset
    has_crop = crop.offset_x != 0 or crop.offset_y != 0

    return has_crop
