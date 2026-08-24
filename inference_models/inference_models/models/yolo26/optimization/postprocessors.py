"""Selectable YOLO26 depth-estimation postprocessing implementations."""

from typing import List

import torch

from inference_models.models.common.roboflow.model_packages import PreProcessingMetadata
from inference_models.models.common.roboflow.post_processing import (
    post_process_depth_estimation_map,
)
from inference_models.models.optimization.contracts import (
    DeviceCompatibility,
    ExecutionContext,
    InputCompatibility,
    OptimizationMetadata,
    OptimizationStage,
    immutable_mapping,
    metadata_supports_context,
)
from inference_models.models.optimization.registry import ImplementationRegistry
from inference_models.models.yolo26.optimization.ids import (
    YOLO26_DEPTH_POSTPROCESSOR_BASE,
    YOLO26_DEPTH_POSTPROCESSOR_TRITON_AA_RESIZE_EXACT_FUSED_V3,
    YOLO26_DEPTH_POSTPROCESSOR_TRITON_AA_RESIZE_EXACT_V2,
    YOLO26_DEPTH_POSTPROCESSOR_TRITON_AA_RESIZE_V1,
)
from inference_models.models.yolo26.optimization.preprocessors import (
    BaseYOLO26DepthPreprocessor,
    TritonCV2ResizeFusedConvertYOLO26DepthPreprocessor,
    TritonCV2ResizePinnedFusedConvertYOLO26DepthPreprocessor,
)
from inference_models.models.yolo26.optimization.schedulers import (
    BaseYOLO26DepthExecutionScheduler,
    CUDAEventHandoffYOLO26DepthExecutionScheduler,
)
from inference_models.models.yolo26.triton_depth_postprocess import (
    ExactFusedTritonDepthMapResizer,
    ExactSeparableTritonDepthMapResizer,
    TritonDepthMapResizer,
)


class BaseYOLO26DepthPostprocessor:
    """Preserve the original torchvision depth-map postprocessing path."""

    metadata = OptimizationMetadata(
        implementation_id=YOLO26_DEPTH_POSTPROCESSOR_BASE,
        stage=OptimizationStage.POSTPROCESS,
        version="1",
        target=DeviceCompatibility(device_kind="gpu"),
        inputs=InputCompatibility(
            scenarios=("*",),
            axis_constraints=immutable_mapping({"batch": ">=1"}),
            dtypes=("float32",),
            layouts=("strided B1HW", "strided BHW"),
        ),
        dependencies=("torch", "torchvision"),
        fallback_id=YOLO26_DEPTH_POSTPROCESSOR_BASE,
        changes_numerics=False,
        supports_concurrency=True,
        supports_cuda_graphs=False,
        output_contract=immutable_mapping(
            {
                "type": "list[torch.Tensor]",
                "dtype": "float32",
                "shape": "per-image original spatial dimensions",
                "ownership": "per-call tensors",
            }
        ),
        numerical_behavior="reference torchvision bilinear-antialias resize",
        stream_behavior="runs on the caller postprocessing stream",
    )

    def is_compatible(self, context: ExecutionContext) -> bool:
        """Return whether the base implementation supports the target.

        Args:
            context: Runtime target context.

        Returns:
            Whether the target satisfies the metadata contract.
        """
        return metadata_supports_context(self.metadata, context)

    def postprocess(
        self,
        *,
        model_results: torch.Tensor,
        pre_processing_meta: List[PreProcessingMetadata],
        context: ExecutionContext,
    ) -> List[torch.Tensor]:
        """Run the preserved depth-map postprocessor.

        Args:
            model_results: Batched raw depth maps.
            pre_processing_meta: Per-image preprocessing geometry.
            context: Runtime target context.

        Returns:
            Per-image float32 depth maps.
        """
        with torch.cuda.nvtx.range("yolo26-depth.postprocess[effective=base]"):
            results = post_process_depth_estimation_map(
                model_results=model_results,
                pre_processing_meta=pre_processing_meta,
                device=torch.device(context.device),
            )

        return results


class TritonAAYOLO26DepthPostprocessor:
    """Run the explicit Triton antialiased depth-map resize candidate."""

    metadata = OptimizationMetadata(
        implementation_id=YOLO26_DEPTH_POSTPROCESSOR_TRITON_AA_RESIZE_V1,
        stage=OptimizationStage.POSTPROCESS,
        version="1",
        target=DeviceCompatibility(
            device_kind="gpu",
            minimum_compute_capability=(7, 0),
        ),
        inputs=InputCompatibility(
            scenarios=(
                "camera_640x480_batch_1_base",
                "camera_3840x2160_batch_1_high",
            ),
            axis_constraints=immutable_mapping(
                {
                    "batch": 1,
                    "channels": 1,
                    "source_width_stride": 1,
                    "maximum_antialias_filter_size": 5,
                }
            ),
            dtypes=("float32",),
            layouts=("strided B1HW", "strided BHW"),
        ),
        dependencies=("torch", "triton"),
        fallback_id=YOLO26_DEPTH_POSTPROCESSOR_BASE,
        changes_numerics=True,
        supports_concurrency=True,
        supports_cuda_graphs=False,
        output_contract=immutable_mapping(
            {
                "type": "list[torch.Tensor]",
                "dtype": "float32",
                "shape": "per-image original spatial dimensions",
                "ownership": "per-call tensors; immutable cached axis tables",
                "per_call_allocations": "one output depth map per resized image",
                "aliasing": "none",
            }
        ),
        numerical_behavior=(
            "approximates PyTorch 2.6 CUDA bilinear-antialias float32 behavior; "
            "target validation observed non-bitwise output differences"
        ),
        stream_behavior=(
            "launches on the active caller stream without a private stream or "
            "additional synchronization"
        ),
    )

    def __init__(self, *, device: torch.device) -> None:
        self._resizer = TritonDepthMapResizer(device=device)

    def is_compatible(self, context: ExecutionContext) -> bool:
        """Return whether the Triton implementation supports the target.

        Args:
            context: Runtime target and installed-component context.

        Returns:
            Whether the target satisfies the metadata contract.
        """
        return metadata_supports_context(self.metadata, context)

    def postprocess(
        self,
        *,
        model_results: torch.Tensor,
        pre_processing_meta: List[PreProcessingMetadata],
        context: ExecutionContext,
    ) -> List[torch.Tensor]:
        """Run Triton resize inside the shared depth-map geometry pipeline.

        Args:
            model_results: Batched raw CUDA float32 depth maps.
            pre_processing_meta: Per-image preprocessing geometry.
            context: Runtime context containing the active CUDA stream.

        Returns:
            Per-image float32 depth maps.
        """
        with torch.cuda.nvtx.range(
            "yolo26-depth.postprocess[effective=triton-aa-resize-v1]"
        ):
            results = post_process_depth_estimation_map(
                model_results=model_results,
                pre_processing_meta=pre_processing_meta,
                device=torch.device(context.device),
                resize_function=self._resizer.resize,
            )

        return results


class ExactTritonAAYOLO26DepthPostprocessor:
    """Run the target-weighted, separable Triton resize candidate."""

    metadata = OptimizationMetadata(
        implementation_id=YOLO26_DEPTH_POSTPROCESSOR_TRITON_AA_RESIZE_EXACT_V2,
        stage=OptimizationStage.POSTPROCESS,
        version="2",
        target=DeviceCompatibility(
            device_kind="gpu",
            minimum_compute_capability=(7, 0),
        ),
        inputs=InputCompatibility(
            scenarios=(
                "camera_640x480_batch_1_base",
                "camera_3840x2160_batch_1_high",
            ),
            axis_constraints=immutable_mapping(
                {
                    "batch": 1,
                    "channels": 1,
                    "source_width_stride": 1,
                    "maximum_antialias_filter_size": 5,
                }
            ),
            dtypes=("float32",),
            layouts=("strided B1HW", "strided BHW"),
        ),
        dependencies=("torch", "torchvision", "triton"),
        fallback_id=YOLO26_DEPTH_POSTPROCESSOR_BASE,
        changes_numerics=False,
        supports_concurrency=True,
        supports_cuda_graphs=False,
        output_contract=immutable_mapping(
            {
                "type": "list[torch.Tensor]",
                "dtype": "float32",
                "shape": "per-image original spatial dimensions",
                "ownership": (
                    "per-call output and horizontal workspace; immutable cached "
                    "target-derived axis tables"
                ),
                "per_call_allocations": (
                    "one horizontal workspace and one output depth map per image"
                ),
                "aliasing": "none",
            }
        ),
        numerical_behavior=(
            "uses interpolation weights generated by the preserved torchvision "
            "CUDA operation and reproduces its horizontal-then-vertical float32 "
            "accumulation order; exact target snapshot validation is required"
        ),
        stream_behavior=(
            "launches two ordered kernels on the active caller stream without "
            "private synchronization"
        ),
    )

    def __init__(self, *, device: torch.device) -> None:
        self._resizer = ExactSeparableTritonDepthMapResizer(device=device)

    def is_compatible(self, context: ExecutionContext) -> bool:
        """Return whether the exact Triton implementation supports the target."""
        return metadata_supports_context(self.metadata, context)

    def postprocess(
        self,
        *,
        model_results: torch.Tensor,
        pre_processing_meta: List[PreProcessingMetadata],
        context: ExecutionContext,
    ) -> List[torch.Tensor]:
        """Run the exact candidate inside the shared depth geometry pipeline."""
        with torch.cuda.nvtx.range(
            "yolo26-depth.postprocess[effective=triton-aa-resize-exact-v2]"
        ):
            results = post_process_depth_estimation_map(
                model_results=model_results,
                pre_processing_meta=pre_processing_meta,
                device=torch.device(context.device),
                resize_function=self._resizer.resize,
            )

        return results


class ExactFusedTritonAAYOLO26DepthPostprocessor:
    """Run the shape-aware, exact fused resize candidate."""

    metadata = OptimizationMetadata(
        implementation_id=(YOLO26_DEPTH_POSTPROCESSOR_TRITON_AA_RESIZE_EXACT_FUSED_V3),
        stage=OptimizationStage.POSTPROCESS,
        version="3",
        target=DeviceCompatibility(
            device_kind="gpu",
            minimum_compute_capability=(7, 0),
        ),
        inputs=InputCompatibility(
            scenarios=(
                "camera_640x480_batch_1_base",
                "camera_3840x2160_batch_1_high",
            ),
            axis_constraints=immutable_mapping(
                {
                    "batch": 1,
                    "channels": 1,
                    "source_width_stride": 1,
                    "maximum_antialias_filter_size": 5,
                    "torchvision_dispatch_max_output_elements": 640 * 480,
                }
            ),
            dtypes=("float32",),
            layouts=("strided B1HW", "strided BHW"),
        ),
        dependencies=("torch", "torchvision", "triton"),
        fallback_id=YOLO26_DEPTH_POSTPROCESSOR_BASE,
        changes_numerics=False,
        supports_concurrency=True,
        supports_cuda_graphs=False,
        output_contract=immutable_mapping(
            {
                "type": "list[torch.Tensor]",
                "dtype": "float32",
                "shape": "per-image original spatial dimensions",
                "ownership": (
                    "per-call output; immutable cached compact target-derived "
                    "axis tables"
                ),
                "per_call_allocations": (
                    "torchvision-managed output for small maps; one output depth "
                    "map and no horizontal workspace for fused maps"
                ),
                "first_use_allocations": (
                    "one compact starts, sizes, and weights table per cached axis"
                ),
                "aliasing": "none",
            }
        ),
        numerical_behavior=(
            "small maps retain the exact torchvision CUDA primitive; larger maps "
            "generate compact weights with the target CUDA float32 formula and "
            "reproduce its horizontal-then-vertical accumulation order in one "
            "fused launch; exact target snapshot validation is required"
        ),
        stream_behavior=(
            "runs on the active caller stream; small maps use torchvision and "
            "large maps launch one Triton kernel; immutable table readiness is "
            "established once per consuming stream"
        ),
    )

    def __init__(self, *, device: torch.device) -> None:
        self._resizer = ExactFusedTritonDepthMapResizer(device=device)

    def is_compatible(self, context: ExecutionContext) -> bool:
        """Return whether the fused exact implementation supports the target."""
        return metadata_supports_context(self.metadata, context)

    def postprocess(
        self,
        *,
        model_results: torch.Tensor,
        pre_processing_meta: List[PreProcessingMetadata],
        context: ExecutionContext,
    ) -> List[torch.Tensor]:
        """Run shape-aware exact resize inside the shared geometry pipeline."""
        with torch.cuda.nvtx.range(
            "yolo26-depth.postprocess[" "effective=triton-aa-resize-exact-fused-v3]"
        ):
            results = post_process_depth_estimation_map(
                model_results=model_results,
                pre_processing_meta=pre_processing_meta,
                device=torch.device(context.device),
                resize_function=self._resizer.resize,
            )

        return results


def build_yolo26_depth_implementation_registry(
    *,
    device: torch.device,
) -> ImplementationRegistry:
    """Build the YOLO26 depth-estimation implementation registry.

    Args:
        device: CUDA target selected for the TensorRT model.

    Returns:
        Registry containing preserved and explicit preprocessing/postprocessing paths.
    """
    registry = ImplementationRegistry(scope_name="YOLO26 depth")
    registry.register_factory(
        metadata=BaseYOLO26DepthPreprocessor.metadata,
        factory=BaseYOLO26DepthPreprocessor,
    )
    registry.register_factory(
        metadata=TritonCV2ResizeFusedConvertYOLO26DepthPreprocessor.metadata,
        factory=lambda: TritonCV2ResizeFusedConvertYOLO26DepthPreprocessor(
            device=device
        ),
    )
    registry.register_factory(
        metadata=TritonCV2ResizePinnedFusedConvertYOLO26DepthPreprocessor.metadata,
        factory=lambda: TritonCV2ResizePinnedFusedConvertYOLO26DepthPreprocessor(
            device=device
        ),
    )
    registry.register_factory(
        metadata=BaseYOLO26DepthExecutionScheduler.metadata,
        factory=lambda: BaseYOLO26DepthExecutionScheduler(device=device),
    )
    registry.register_factory(
        metadata=CUDAEventHandoffYOLO26DepthExecutionScheduler.metadata,
        factory=lambda: CUDAEventHandoffYOLO26DepthExecutionScheduler(device=device),
    )
    registry.register_factory(
        metadata=BaseYOLO26DepthPostprocessor.metadata,
        factory=BaseYOLO26DepthPostprocessor,
    )
    registry.register_factory(
        metadata=TritonAAYOLO26DepthPostprocessor.metadata,
        factory=lambda: TritonAAYOLO26DepthPostprocessor(device=device),
    )
    registry.register_factory(
        metadata=ExactTritonAAYOLO26DepthPostprocessor.metadata,
        factory=lambda: ExactTritonAAYOLO26DepthPostprocessor(device=device),
    )
    registry.register_factory(
        metadata=ExactFusedTritonAAYOLO26DepthPostprocessor.metadata,
        factory=lambda: ExactFusedTritonAAYOLO26DepthPostprocessor(device=device),
    )
    registry.set_auto_preferences(
        stage=OptimizationStage.PREPROCESS,
        implementation_ids=(),
    )
    registry.set_auto_preferences(
        stage=OptimizationStage.SCHEDULER,
        implementation_ids=(),
    )
    registry.set_auto_preferences(
        stage=OptimizationStage.POSTPROCESS,
        implementation_ids=(),
    )

    return registry
