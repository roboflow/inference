"""Threaded exact RF-DETR preprocessing choice."""

from inference_models.models.common.roboflow.model_packages import (
    ImagePreProcessing,
    NetworkInputDefinition,
)
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
from inference_models.models.rfdetr.optimization.contracts import (
    PreprocessRequest,
    PreprocessResult,
)
from inference_models.models.rfdetr.optimization.ids import (
    RFDETR_PREPROCESSOR_BASE,
    RFDETR_PREPROCESSOR_THREADED_EXACT_V1,
)
from inference_models.models.rfdetr.optimization.preprocessors.common import (
    run_reference_preprocessor,
)
from inference_models.models.rfdetr.optimization.preprocessors.compatibility import (
    check_threaded_request_compatibility,
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

    def check_model_compatibility(
        self,
        *,
        image_pre_processing: ImagePreProcessing,
        network_input: NetworkInputDefinition,
    ) -> CompatibilityResult:
        """Accept model transformations delegated to the reference path.

        Args:
            image_pre_processing: Model-package image transformations.
            network_input: Model-package network input definition.

        Returns:
            Compatible result.
        """
        del image_pre_processing, network_input
        result = CompatibilityResult.compatible()

        return result

    def check_request_compatibility(
        self,
        *,
        request: PreprocessRequest,
        context: ExecutionContext,
    ) -> CompatibilityResult:
        """Check the threaded implementation's NumPy request constraints.

        Args:
            request: Typed preprocessing request.
            context: Runtime target and request context.

        Returns:
            Compatibility result with actionable reasons.
        """
        del context
        result = check_threaded_request_compatibility(request=request)

        return result

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
