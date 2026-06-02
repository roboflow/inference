from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    List,
    Literal,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np
import supervision as sv
import torch

from inference_models.models.base.types import (
    InstancesRLEMasks,
    PreprocessedInputs,
    PreprocessingMetadata,
    RawPrediction,
)
from inference_models.models.common.rle_utils import coco_rle_masks_to_numpy_mask

InstanceSegmentationMaskFormat = Literal["dense", "rle"]


_MISSING = object()


@runtime_checkable
class InferenceFuture(Protocol):
    """Future-like handle over an in-flight inference request.

    The returned object lets a caller start a subsequent ``infer_async`` call
    while the GPU is still executing the previous one. Calling ``result()``
    blocks on a single GPU event, then runs CPU-side post-processing and
    returns the decoded detections. ``done()`` is a non-blocking probe.
    """

    def result(self) -> List["InstanceDetections"]: ...

    def done(self) -> bool: ...


class _DirectInferenceFuture:
    """Concrete ``InferenceFuture`` backed by a single ``torch.cuda.Event``.

    Holds the raw forward output plus the preprocessing metadata needed by
    ``post_process``. The event is recorded on the stream that produced the
    raw output; ``result()`` synchronizes on it before running CPU decode.
    Post-process output is memoised so ``result()`` may be called repeatedly.
    """

    # No __slots__: adapters stash per-request context on the future
    # (e.g. pipeline-depth-2 stashes `_adapter_kwargs` so `postprocess`
    # can rebuild the decode call for the PREVIOUS frame even when the
    # submit site passed `meta=None`). The Future is short-lived so the
    # per-instance dict overhead is negligible.

    def __init__(
        self,
        model: "InstanceSegmentationModel",
        raw: Any,
        meta: Any,
        evt: Optional[torch.cuda.Event],
        kwargs: dict,
    ) -> None:
        self._model = model
        self._raw = raw
        self._meta = meta
        self._evt = evt
        self._kwargs = kwargs
        self._cached: Any = _MISSING

    @property
    def preprocess_metadata(self) -> Any:
        """The metadata captured at ``pre_process`` time for this request."""
        return self._meta

    def done(self) -> bool:
        if self._cached is not _MISSING:
            return True
        if self._evt is None:
            return True
        return self._evt.query()

    def submit_gpu_work(self, meta: Any = None) -> None:
        """Enqueue the ``post_process`` GPU work eagerly.

        Under depth>=2 pipelining ``result()`` is intentionally delayed so
        the source loop can prepare later frames. Without eager submission,
        the postproc kernels are also delayed until that future is finalized,
        leaving a bubble between the TensorRT produce event and postproc.

        Calling ``submit_gpu_work`` from the adapter's ``postprocess`` step
        enqueues the postproc stream wait immediately after the corresponding
        TensorRT graph has been submitted. The host still does not block, and
        ``result()`` later reuses the enqueued postproc result.

        Idempotent: calling it once is enough; subsequent calls to
        ``result()`` reuse the enqueued postproc result.
        """
        if self._cached is not _MISSING:
            return
        if meta is None:
            meta = self._meta
        else:
            self._meta = meta
        # `post_process` is expected to be non-blocking: it enqueues its
        # CUDA kernels on a private stream and returns a handle/structure
        # that the caller reads later. The host does NOT block here.
        self._cached = self._model.post_process(self._raw, meta, **self._kwargs)

    def result(self) -> List["InstanceDetections"]:
        # No host sync here: post_process() enqueues its GPU work on a
        # dedicated stream and uses stream.wait_event() internally to order
        # itself after the forward stream. The final host sync happens where
        # CPU-visible results are actually needed (DtoH copies in the adapter).
        if self._cached is _MISSING:
            self._cached = self._model.post_process(
                self._raw, self._meta, **self._kwargs
            )
        return self._cached


@dataclass
class InstanceDetections:
    xyxy: torch.Tensor  # (n_boxes, 4)
    class_id: torch.Tensor  # (n_boxes, )
    confidence: torch.Tensor  # (n_boxes, )
    mask: Union[
        torch.Tensor, InstancesRLEMasks
    ]  # for dense representation (n_boxes, mask_height, mask_width)
    image_metadata: Optional[dict] = None
    bboxes_metadata: Optional[List[dict]] = (
        None  # if given, list of size equal to # of bboxes
    )

    def to_supervision(self) -> sv.Detections:
        """Convert instance segmentation detections to Supervision Detections format.

        Converts the PyTorch tensor-based instance segmentation results to Supervision's
        NumPy-based format. This includes both bounding boxes and segmentation masks,
        enabling use of Supervision's mask annotators and analysis tools.

        Returns:
            sv.Detections: Supervision Detections object with:

                - xyxy: Bounding boxes as NumPy array (N, 4) in [x1, y1, x2, y2] format

                - class_id: Class IDs as NumPy array (N,)

                - confidence: Confidence scores as NumPy array (N,)

                - mask: Segmentation masks as NumPy array (N, H, W) with boolean values

        Examples:
            Convert and visualize instance segmentation:

            >>> import cv2
            >>> import supervision as sv
            >>> from inference_models import AutoModel
            >>>
            >>> model = AutoModel.from_pretrained("yolov8n-seg-640")
            >>> image = cv2.imread("image.jpg")
            >>> predictions = model(image)
            >>>
            >>> # Convert to Supervision format
            >>> detections = predictions[0].to_supervision()
            >>>
            >>> # Use Supervision mask annotator
            >>> mask_annotator = sv.MaskAnnotator()
            >>> annotated = mask_annotator.annotate(image.copy(), detections)

            Access masks:

            >>> detections = predictions[0].to_supervision()
            >>> print(f"Masks shape: {detections.mask.shape}")  # (N, H, W)
            >>> print(f"First mask: {detections.mask[0]}")  # Boolean array

        See Also:
            - Supervision documentation: https://supervision.roboflow.com
        """
        if isinstance(self.mask, torch.Tensor):
            mask = self.mask.cpu().numpy()
        else:
            mask = coco_rle_masks_to_numpy_mask(self.mask)
        return sv.Detections(
            xyxy=self.xyxy.cpu().numpy(),
            class_id=self.class_id.cpu().numpy(),
            confidence=self.confidence.cpu().numpy(),
            mask=mask,
        )


class InstanceSegmentationModel(
    ABC, Generic[PreprocessedInputs, PreprocessingMetadata, RawPrediction]
):

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls, model_name_or_path: str, **kwargs
    ) -> "InstanceSegmentationModel":
        pass

    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def supported_mask_formats(self) -> Set[InstanceSegmentationMaskFormat]:
        pass

    def infer(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> List[InstanceDetections]:
        # Synchronous direct path: pre_process → forward → post_process in
        # sequence, with no per-call output cloning. The async variant
        # (``infer_async``) exists for pipelined callers that need to
        # submit frame N+1 before frame N's output buffers have been read
        # — cloning makes those callers safe. Here, ``post_process``
        # consumes the raw forward output immediately, so no clone is
        # needed and we avoid the ~80µs of DtoD copies on the inference
        # stream. This keeps the ``infer()`` entry point at maximum
        # throughput for single-thread, single-model users.
        pre_processed_images, pre_processing_meta = self.pre_process(images, **kwargs)
        model_results = self.forward(pre_processed_images, **kwargs)
        return self.post_process(model_results, pre_processing_meta, **kwargs)

    def infer_async(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> InferenceFuture:
        """Submit an inference request and return a future.

        The default implementation performs ``pre_process`` and ``forward``
        synchronously, records a CUDA event on the current stream, and defers
        ``post_process`` until ``result()`` is called on the returned future.
        Subclasses that run ``forward`` on a dedicated stream should override
        this to record the event on that stream (see the TRT model).
        """
        pre_processed_images, pre_processing_meta = self.pre_process(images, **kwargs)
        return self.forward_async(pre_processed_images, pre_processing_meta, **kwargs)

    def forward_async(
        self,
        pre_processed_images: PreprocessedInputs,
        pre_processing_meta: PreprocessingMetadata,
        **kwargs,
    ) -> InferenceFuture:
        """Run ``forward`` only and return a future pinned to that launch.

        Separating this from ``infer_async`` lets the adapter interleave
        preprocessing for frame N+1 with the forward pass for frame N on a
        dedicated stream while holding a future whose ``result()`` will
        decode frame N once its outputs are ready.
        """
        model_results = self.forward(pre_processed_images, **kwargs)
        # Prefer a produce-event already recorded on the forward stream (eg.
        # the TRT graph stream) so `done()` reflects true GPU completion
        # without straddling a stream boundary. Fall back to recording on
        # the current stream for models that don't expose one.
        evt: Optional[torch.cuda.Event] = None
        first = (
            model_results[0]
            if isinstance(model_results, (tuple, list))
            else model_results
        )
        existing = getattr(first, "_trt_produce_event", None)
        if existing is not None:
            evt = existing
        elif torch.cuda.is_available():
            evt = torch.cuda.Event()
            evt.record()
        return _DirectInferenceFuture(
            self, model_results, pre_processing_meta, evt, kwargs
        )

    @abstractmethod
    def pre_process(
        self, images: Union[torch.Tensor, List[torch.Tensor]], **kwargs
    ) -> Tuple[PreprocessedInputs, PreprocessingMetadata]:
        pass

    @abstractmethod
    def forward(
        self, pre_processed_images: PreprocessedInputs, **kwargs
    ) -> RawPrediction:
        pass

    @abstractmethod
    def post_process(
        self,
        model_results: RawPrediction,
        pre_processing_meta: PreprocessedInputs,
        **kwargs,
    ) -> List[InstanceDetections]:
        pass

    def __call__(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> List[InstanceDetections]:
        return self.infer(images, **kwargs)
