"""Universal Triton RF-DETR preprocessing choice."""

import torch

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
    PreprocessRequest,
    PreprocessResult,
)
from inference_models.models.rfdetr.optimization.ids import (
    RFDETR_PREPROCESSOR_BASE,
    RFDETR_PREPROCESSOR_TRITON_UNIVERSAL_V1,
)
from inference_models.models.rfdetr.triton_universal_preprocess_runtime import (
    UniversalFastPreprocessRuntime,
)


class TritonUniversalPreprocessor:
    """Run the explicit universal CUDA/Triton preprocessing path."""

    metadata = OptimizationMetadata(
        implementation_id=RFDETR_PREPROCESSOR_TRITON_UNIVERSAL_V1,
        stage=OptimizationStage.PREPROCESS,
        version="1",
        target=DeviceCompatibility(
            device_kind="gpu",
            device_families=("nvidia_jetson", "nvidia_discrete_gpu"),
        ),
        inputs=InputCompatibility(
            scenarios=("*",),
            axis_constraints=immutable_mapping(
                {
                    "batch": ">=1",
                    "channels": 3,
                    "source_dimensions": "homogeneous",
                    "resize_mode": "stretch",
                }
            ),
            dtypes=("uint8", "floating"),
            layouts=("HWC", "NHWC", "CHW", "NCHW"),
        ),
        dependencies=("torch", "torchvision", "triton"),
        fallback_id=RFDETR_PREPROCESSOR_BASE,
        changes_numerics=False,
        supports_concurrency=True,
        supports_cuda_graphs=False,
        output_contract=immutable_mapping(
            {
                "device": "selected CUDA device",
                "dtype": "float32",
                "layout": "contiguous NCHW",
                "ownership": "per-call tensor from PyTorch CUDA allocator",
            }
        ),
        numerical_behavior=(
            "PIL byte-exact fixed-point resize for uint8; floating tensors preserve "
            "RF-DETR tensor-input CUDA resize semantics"
        ),
        stream_behavior=(
            "submits preprocessing to the caller stream and returns a completion event"
        ),
    )

    def __init__(self, *, device: torch.device) -> None:
        self._runtime = UniversalFastPreprocessRuntime(device=device)

    def is_compatible(self, context: ExecutionContext) -> bool:
        """Return whether the Triton path supports the runtime context.

        Args:
            context: Runtime target and request context.

        Returns:
            Whether the target is compatible.
        """
        return metadata_supports_context(self.metadata, context)

    def preprocess(
        self,
        request: PreprocessRequest,
        context: ExecutionContext,
    ) -> PreprocessResult:
        """Run universal Triton preprocessing.

        Args:
            request: Typed preprocessing request.
            context: Runtime context containing the preprocessing stream.

        Returns:
            Typed preprocessing result and completion event.
        """
        stream = context.current_stream
        if stream is None:
            from inference_models.errors import ModelRuntimeError

            raise ModelRuntimeError(
                message="triton-universal-v1 requires a preprocessing CUDA stream.",
                help_url=(
                    "https://inference-models.roboflow.com/errors/models-runtime/"
                    "#modelruntimeerror"
                ),
            )
        runtime_result = self._runtime.preprocess(
            images=request.images,
            input_color_format=request.input_color_format,
            image_pre_processing=request.image_pre_processing,
            network_input=request.network_input,
            pre_processing_overrides=request.pre_processing_overrides,
            stream=stream,
        )
        result = PreprocessResult(
            tensor=runtime_result.tensor,
            metadata=runtime_result.metadata,
            ready_event=runtime_result.ready_event,
            input_kind=runtime_result.input_kind,
        )

        return result
