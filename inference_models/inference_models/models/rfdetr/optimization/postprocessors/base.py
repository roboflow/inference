"""Preserved base RF-DETR object-detection postprocessing choice."""

from typing import List

from inference_models import Detections
from inference_models.models.optimization.contracts import (
    CompatibilityResult,
    DeviceCompatibility,
    ExecutionContext,
    InputCompatibility,
    OptimizationMetadata,
    OptimizationStage,
    immutable_mapping,
    metadata_supports_context,
)
from inference_models.models.rfdetr.common import post_process_object_detection_results
from inference_models.models.rfdetr.optimization.contracts import PostprocessRequest
from inference_models.models.rfdetr.optimization.ids import RFDETR_POSTPROCESSOR_BASE


class BasePostprocessor:
    """Preserve the original RF-DETR PyTorch postprocessing path."""

    metadata = OptimizationMetadata(
        implementation_id=RFDETR_POSTPROCESSOR_BASE,
        stage=OptimizationStage.POSTPROCESS,
        version="1",
        target=DeviceCompatibility(device_kind="gpu"),
        inputs=InputCompatibility(
            scenarios=("*",),
            axis_constraints=immutable_mapping({"batch": ">=1", "queries": ">=1"}),
            dtypes=("float32",),
            layouts=("contiguous BQ4", "contiguous BQC"),
        ),
        dependencies=("torch",),
        fallback_id=RFDETR_POSTPROCESSOR_BASE,
        changes_numerics=False,
        supports_concurrency=True,
        supports_cuda_graphs=False,
        output_contract=immutable_mapping(
            {
                "type": "list[Detections]",
                "ownership": "per-call tensors owned by returned Detections",
            }
        ),
        numerical_behavior="reference RF-DETR PyTorch pipeline",
        stream_behavior="runs on the caller postprocessing stream",
    )

    def is_compatible(self, context: ExecutionContext) -> bool:
        """Return whether the base path supports the runtime context.

        Args:
            context: Runtime target and request context.

        Returns:
            Whether the target is compatible.
        """
        return metadata_supports_context(self.metadata, context)

    def check_request_compatibility(
        self,
        *,
        request: PostprocessRequest,
        context: ExecutionContext,
    ) -> CompatibilityResult:
        """Accept requests handled by the preserved postprocessing path.

        Args:
            request: Typed postprocessing request.
            context: Runtime target and request context.

        Returns:
            Compatible result; detailed validation remains in the base path.
        """
        del request, context
        result = CompatibilityResult.compatible()

        return result

    def postprocess(
        self,
        request: PostprocessRequest,
        context: ExecutionContext,
    ) -> List[Detections]:
        """Run preserved RF-DETR postprocessing.

        Args:
            request: Typed postprocessing request.
            context: Runtime context containing the postprocessing stream.

        Returns:
            Per-image detections.
        """
        results = post_process_object_detection_results(
            bboxes=request.bboxes,
            logits=request.logits,
            pre_processing_meta=request.pre_processing_meta,
            threshold=request.threshold,
            num_classes=request.num_classes,
            classes_re_mapping=request.classes_re_mapping,
            device=request.bboxes.device,
        )

        return results
