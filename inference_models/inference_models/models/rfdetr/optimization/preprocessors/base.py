"""Preserved base RF-DETR preprocessing choice."""

from inference_models.models.rfdetr.optimization.contracts import (
    DeviceCompatibility,
    ExecutionContext,
    InputCompatibility,
    OptimizationMetadata,
    OptimizationStage,
    PreprocessRequest,
    PreprocessResult,
    immutable_mapping,
    metadata_supports_context,
)
from inference_models.models.rfdetr.optimization.ids import RFDETR_PREPROCESSOR_BASE
from inference_models.models.rfdetr.optimization.preprocessors.common import (
    run_reference_preprocessor,
)


class BasePreprocessor:
    """Preserve the original RF-DETR PIL/tensor preprocessing path."""

    metadata = OptimizationMetadata(
        implementation_id=RFDETR_PREPROCESSOR_BASE,
        stage=OptimizationStage.PREPROCESS,
        version="1",
        target=DeviceCompatibility(device_kind="gpu"),
        inputs=InputCompatibility(
            scenarios=("*",),
            axis_constraints=immutable_mapping({"batch": ">=1"}),
            dtypes=("uint8", "floating"),
            layouts=("HWC", "NHWC", "CHW", "NCHW"),
        ),
        dependencies=("Pillow", "torch", "torchvision"),
        fallback_id=RFDETR_PREPROCESSOR_BASE,
        changes_numerics=False,
        supports_concurrency=True,
        supports_cuda_graphs=False,
        output_contract=immutable_mapping(
            {
                "device": "selected CUDA device",
                "dtype": "float32",
                "layout": "contiguous NCHW",
                "ownership": "new tensor owned by caller",
            }
        ),
        numerical_behavior="reference RF-DETR PIL/torch pipeline",
        stream_behavior="submits H2D work to the caller preprocessing stream",
    )

    def __init__(self, *, max_workers: int) -> None:
        self._max_workers = max_workers

    def is_compatible(self, context: ExecutionContext) -> bool:
        """Return whether the base path supports the runtime context.

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
        """Run the preserved base preprocessor.

        Args:
            request: Typed preprocessing request.
            context: Runtime context containing the preprocessing stream.

        Returns:
            Typed preprocessing result.
        """
        result = run_reference_preprocessor(
            request,
            context,
            implementation_id=self.metadata.implementation_id,
            max_workers=self._max_workers,
        )

        return result
