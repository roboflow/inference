"""Preserved RF-DETR TensorRT engine-boundary implementation."""

from typing import Tuple

import torch

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
from inference_models.models.rfdetr.optimization.contracts import (
    EngineExecutionRequest,
)
from inference_models.models.rfdetr.optimization.ids import RFDETR_ENGINE_PLUGIN_BASE


class BaseEngineAdjacentPlugin:
    """Execute the existing TensorRT boundary without an engine plugin."""

    metadata = OptimizationMetadata(
        implementation_id=RFDETR_ENGINE_PLUGIN_BASE,
        stage=OptimizationStage.ENGINE_PLUGIN,
        version="1",
        target=DeviceCompatibility(device_kind="gpu"),
        inputs=InputCompatibility(
            scenarios=("*",),
            axis_constraints=immutable_mapping({"batch": ">=1"}),
            dtypes=("float32",),
            layouts=("contiguous NCHW",),
        ),
        dependencies=("TensorRT", "torch"),
        fallback_id=RFDETR_ENGINE_PLUGIN_BASE,
        changes_numerics=False,
        supports_concurrency=False,
        supports_cuda_graphs=True,
        output_contract=immutable_mapping(
            {
                "outputs": ("dets", "labels"),
                "ownership": "TensorRT output tensors owned by caller",
                "semantic_forward": "unchanged TensorRT engine",
            }
        ),
        numerical_behavior="delegates to the existing TensorRT execution helper",
        stream_behavior="executes on the scheduler-provided CUDA stream",
    )

    def is_compatible(self, context: ExecutionContext) -> bool:
        """Return whether the base TensorRT boundary supports the runtime context.

        Args:
            context: Runtime target and request context.

        Returns:
            Whether the target is compatible.
        """
        compatible = metadata_supports_context(self.metadata, context)

        return compatible

    def execute(
        self,
        request: EngineExecutionRequest,
        context: ExecutionContext,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute the protected TensorRT forward boundary.

        Args:
            request: TensorRT input and engine runtime objects.
            context: Runtime context containing TensorRT's execution stream.

        Returns:
            Raw TensorRT detection boxes and logits.

        Raises:
            ModelRuntimeError: If the scheduler did not provide an execution stream.
        """
        if context.current_stream is None:
            raise ModelRuntimeError(
                message=(
                    "RF-DETR base engine plugin requires a scheduler-provided "
                    "CUDA stream."
                ),
                help_url=(
                    "https://inference-models.roboflow.com/errors/models-runtime/"
                    "#modelruntimeerror"
                ),
            )

        from inference_models.models.common.trt import infer_from_trt_engine

        detections, labels = infer_from_trt_engine(
            pre_processed_images=request.pre_processed_images,
            trt_config=request.trt_config,
            engine=request.engine,
            context=request.execution_context,
            device=request.device,
            input_name=request.input_name,
            outputs=request.output_names,
            stream=context.current_stream,
            trt_cuda_graph_cache=request.trt_cuda_graph_cache,
        )

        return detections, labels
