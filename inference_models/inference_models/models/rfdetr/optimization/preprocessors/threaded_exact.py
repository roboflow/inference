"""Threaded exact RF-DETR preprocessing choice."""

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
from inference_models.models.rfdetr.optimization.ids import (
    RFDETR_PREPROCESSOR_BASE,
    RFDETR_PREPROCESSOR_THREADED_EXACT_V1,
)
from inference_models.models.rfdetr.optimization.preprocessors.common import (
    run_reference_preprocessor,
)


class ThreadedExactPreprocessor:
    """Parallelize exact per-image CPU preprocessing for NumPy batches."""

    metadata = OptimizationMetadata(
        implementation_id=RFDETR_PREPROCESSOR_THREADED_EXACT_V1,
        stage=OptimizationStage.PREPROCESS,
        version="1",
        target=DeviceCompatibility(
            device_kind="gpu",
            device_families=("nvidia_jetson", "nvidia_discrete_gpu"),
        ),
        inputs=InputCompatibility(
            scenarios=("*",),
            axis_constraints=immutable_mapping(
                {"batch": ">=1", "channels": 3, "source_dimensions": "per image"}
            ),
            dtypes=("uint8",),
            layouts=("HWC", "NHWC"),
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
        numerical_behavior="byte-identical per-image reference pipeline",
        stream_behavior=(
            "CPU work completes before ordered H2D copies are submitted to the "
            "caller preprocessing stream"
        ),
    )

    def __init__(self, *, max_workers: int) -> None:
        self._max_workers = max_workers

    def is_compatible(self, context: ExecutionContext) -> bool:
        """Return whether the threaded path supports the runtime context.

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
        """Run threaded exact preprocessing.

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
