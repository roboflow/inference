"""Shared adapter for reference RF-DETR preprocessing implementations."""

import torch

from inference_models.errors import ModelRuntimeError
from inference_models.models.optimization.contracts import ExecutionContext
from inference_models.models.rfdetr.optimization.contracts import (
    PreprocessRequest,
    PreprocessResult,
)
from inference_models.models.rfdetr.pre_processing import pre_process_network_input


def run_reference_preprocessor(
    request: PreprocessRequest,
    context: ExecutionContext,
    *,
    implementation_id: str,
    max_workers: int,
) -> PreprocessResult:
    """Run the existing RF-DETR preprocessor on the context stream.

    Args:
        request: Typed preprocessing request.
        context: Runtime context containing the CUDA stream.
        implementation_id: Base or threaded implementation ID.
        max_workers: Bounded threaded worker limit.

    Returns:
        Typed preprocessing result.

    Raises:
        ModelRuntimeError: If the execution context has no CUDA stream.
    """
    stream = context.current_stream
    if stream is None:
        raise ModelRuntimeError(
            message=f"{implementation_id!r} requires a preprocessing CUDA stream.",
            help_url=(
                "https://inference-models.roboflow.com/errors/models-runtime/"
                "#modelruntimeerror"
            ),
        )

    with torch.cuda.stream(stream):
        tensor, metadata = pre_process_network_input(
            images=request.images,
            image_pre_processing=request.image_pre_processing,
            network_input=request.network_input,
            target_device=torch.device(context.device),
            input_color_format=request.input_color_format,
            pre_processing_overrides=request.pre_processing_overrides,
            preprocessor_implementation_id=implementation_id,
            preprocessor_max_workers=max_workers,
        )

    result = PreprocessResult(
        tensor=tensor,
        metadata=metadata,
        input_kind="reference",
        implementation_id=implementation_id,
    )

    return result
