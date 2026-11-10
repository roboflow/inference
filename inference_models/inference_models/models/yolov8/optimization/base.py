"""The original YOLOv8 TensorRT decoder, with explicit stream completion."""

from inference_models import Detections
from inference_models.models.common.roboflow.post_processing import (
    post_process_nms_fused_model_output,
    rescale_detections,
    run_nms_for_object_detection,
)
from inference_models.models.optimization.contracts import (
    CompatibilityResult,
    DeviceCompatibility,
    ExecutionContext,
    InputCompatibility,
    OptimizationMetadata,
    OptimizationStage,
)

from .contracts import (
    PostprocessRequest,
    PostprocessResult,
    _completion_event,
    _run_on_stream,
)


class BasePostprocessor:
    """Preserve the library's original fused and unfused numerical operations."""

    metadata = OptimizationMetadata(
        implementation_id="base",
        stage=OptimizationStage.POSTPROCESS,
        version="1",
        target=DeviceCompatibility(device_kind="any"),
        inputs=InputCompatibility(scenarios=("yolov8-object-detection",)),
        dependencies=(),
        fallback_id="base",
        changes_numerics=False,
        supports_concurrency=True,
        supports_cuda_graphs=False,
        numerical_behavior="Original NMS, rescaling, round and int conversion.",
        stream_behavior="Caller stream; explicit completion event; no final stage wait.",
    )

    def is_compatible(self, context: ExecutionContext) -> bool:
        """Accept every context supported by the existing model.

        Args:
            context: Runtime facts.

        Returns:
            True; original model validation remains authoritative.
        """
        return True

    def check_request_compatibility(
        self, *, request: PostprocessRequest, context: ExecutionContext
    ) -> CompatibilityResult:
        """Delegate request validation to the original numerical path.

        Args:
            request: Original model arguments.
            context: Execution stream and target.

        Returns:
            Compatible; original exceptions propagate during execution.
        """
        result = CompatibilityResult.compatible()

        return result

    def check_runtime_compatibility(
        self, *, request: PostprocessRequest, context: ExecutionContext
    ) -> CompatibilityResult:
        """Accept every runtime; the original path needs no initialization.

        Args:
            request: Original model arguments.
            context: Execution stream and target.

        Returns:
            Compatible.
        """
        result = CompatibilityResult.compatible()

        return result

    def postprocess(
        self, request: PostprocessRequest, context: ExecutionContext
    ) -> PostprocessResult:
        """Enqueue the original decoder on the caller's stream.

        Args:
            request: Resolved output, metadata and options.
            context: Target and caller stream.

        Returns:
            Per-call result; call wait before consuming detections.
        """
        with _run_on_stream(request, context):
            if request.fused:
                nms_results = post_process_nms_fused_model_output(
                    output=request.model_results, conf_thresh=request.threshold
                )
            else:
                nms_results = run_nms_for_object_detection(
                    output=request.model_results,
                    conf_thresh=request.threshold,
                    iou_thresh=request.iou_threshold,
                    max_detections=request.max_detections,
                    class_agnostic=request.class_agnostic_nms,
                )
            rescaled = rescale_detections(
                detections=nms_results, images_metadata=request.pre_processing_meta
            )
            detections = [
                Detections(
                    xyxy=result[:, :4].round().int(),
                    class_id=result[:, 5].int(),
                    confidence=result[:, 4],
                )
                for result in rescaled
            ]
            event = _completion_event(context)

        result = PostprocessResult(ready_event=event, _materialize=lambda: detections)

        return result
