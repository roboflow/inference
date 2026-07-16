"""Fused Triton RF-DETR object-detection postprocessing choice."""

from typing import List

import torch

from inference_models import Detections
from inference_models.errors import ModelRuntimeError
from inference_models.models.optimization.contracts import (
    DeviceCompatibility,
    ExecutionContext,
    InputCompatibility,
    OptimizationMetadata,
    OptimizationStage,
    immutable_mapping,
    metadata_supports_context,
)
from inference_models.models.rfdetr.optimization.contracts import PostprocessRequest
from inference_models.models.rfdetr.optimization.ids import (
    RFDETR_POSTPROCESSOR_BASE,
    RFDETR_POSTPROCESSOR_TRITON_FUSED_V1,
)
from inference_models.models.rfdetr.triton_object_detection_postprocess import (
    FusedObjectDetectionPostprocessor,
)


class TritonFusedPostprocessor:
    """Run the explicit fused Triton object-detection postprocessor."""

    metadata = OptimizationMetadata(
        implementation_id=RFDETR_POSTPROCESSOR_TRITON_FUSED_V1,
        stage=OptimizationStage.POSTPROCESS,
        version="1",
        target=DeviceCompatibility(
            device_kind="gpu",
            device_families=("nvidia_jetson", "nvidia_discrete_gpu"),
        ),
        inputs=InputCompatibility(
            scenarios=("*",),
            axis_constraints=immutable_mapping({"batch": ">=1", "queries": "1..1024"}),
            dtypes=("float32",),
            layouts=("contiguous BQ4", "contiguous BQC"),
        ),
        dependencies=("torch", "triton"),
        fallback_id=RFDETR_POSTPROCESSOR_BASE,
        changes_numerics=False,
        supports_concurrency=True,
        supports_cuda_graphs=False,
        output_contract=immutable_mapping(
            {
                "type": "list[Detections]",
                "boxes_dtype": "int32",
                "confidence_dtype": "float32",
                "class_dtype": "int32",
                "ownership": "per-call tensors owned by returned Detections",
            }
        ),
        numerical_behavior=(
            "same sigmoid, sorted flat top-k, strict threshold, metadata rescaling, "
            "clipping, and round-to-int semantics as base"
        ),
        stream_behavior=(
            "runs on the caller stream with one batched count synchronization"
        ),
    )

    def __init__(self, *, device: torch.device) -> None:
        self._runtime = FusedObjectDetectionPostprocessor(device=device)

    def is_compatible(self, context: ExecutionContext) -> bool:
        """Return whether the Triton path supports the runtime context.

        Args:
            context: Runtime target and request context.

        Returns:
            Whether the target is compatible.
        """
        return metadata_supports_context(self.metadata, context)

    def postprocess(
        self,
        request: PostprocessRequest,
        context: ExecutionContext,
    ) -> List[Detections]:
        """Run fused Triton postprocessing.

        Args:
            request: Typed postprocessing request.
            context: Runtime context containing the postprocessing stream.

        Returns:
            Per-image detections.

        Raises:
            ModelRuntimeError: If the execution context has no CUDA stream.
        """
        stream = context.current_stream
        if stream is None:
            raise ModelRuntimeError(
                message="triton-fused-v1 requires a postprocessing CUDA stream.",
                help_url=(
                    "https://inference-models.roboflow.com/errors/models-runtime/"
                    "#modelruntimeerror"
                ),
            )
        results = self._runtime.postprocess(
            bboxes=request.bboxes,
            logits=request.logits,
            pre_processing_meta=request.pre_processing_meta,
            threshold=request.threshold,
            num_classes=request.num_classes,
            classes_re_mapping=request.classes_re_mapping,
            stream=stream,
        )

        return results
