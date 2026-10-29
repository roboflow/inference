"""Bounded preprocessing selection for the existing segmentation TRT adapter."""

import os
from dataclasses import replace

import torch
from inference_models.models.optimization.contracts import (
    CompatibilityResult,
    OptimizationStage,
)
from inference_models.models.optimization.registry import ImplementationRegistry
from inference_models.models.rfdetr.optimization.execution_plan import (
    RFDetrExecutionPlan,
)
from inference_models.models.rfdetr.optimization.ids import RFDETR_PREPROCESSOR_ENV_NAME
from inference_models.models.rfdetr.optimization.preprocessors.base import (
    BasePreprocessor,
)
from inference_models.models.rfdetr.optimization.preprocessors.pillow_simd import (
    PillowSIMDPreprocessor,
)
from inference_models.models.rfdetr.optimization.preprocessors.triton_universal import (
    TritonUniversalPreprocessor,
)
from inference_models.models.rfdetr.triton_universal_preprocess_runtime import (
    UniversalFastPreprocessRuntime,
)


class _SegmentationUniversalPreprocessor(TritonUniversalPreprocessor):
    """Use exact uint8 normalization and preserve float dtype/device semantics."""

    def __init__(self, *, device):
        self._runtime = UniversalFastPreprocessRuntime(
            device=device, exact_uint8_normalization=True
        )

    def check_request_compatibility(self, *, request, context):
        result = super().check_request_compatibility(request=request, context=context)
        if not result.supported:
            return result

        images = (
            request.images if isinstance(request.images, list) else [request.images]
        )
        for image in images:
            if (
                not isinstance(image, torch.Tensor) or not image.is_floating_point()
            ) and torch.get_default_dtype() != torch.float32:
                result = CompatibilityResult.incompatible(
                    "uint8 normalization requires the reference FP32 default dtype"
                )
                break

            if (
                isinstance(image, torch.Tensor)
                and image.is_cuda
                and image.device != torch.device(context.device)
            ):
                result = CompatibilityResult.incompatible(
                    "CUDA input must be on the model device"
                )
                break

            if isinstance(image, torch.Tensor) and image.is_floating_point():
                if image.dtype != torch.float32 or image.device != torch.device(
                    context.device
                ):
                    result = CompatibilityResult.incompatible(
                        "segmentation universal float resize requires FP32 tensors "
                        "already on the model device; preserve reference dtype/device semantics"
                    )
                    break

        return result


def _resolve_segmentation_plan(execution_plan, *, legacy_enabled):
    if execution_plan is None:
        requested = os.getenv(RFDETR_PREPROCESSOR_ENV_NAME)
        plan = RFDetrExecutionPlan(
            preprocessor_id=(
                requested
                if requested is not None
                else ("triton-universal-v1" if legacy_enabled else "base")
            ),
            postprocessor_id="base",
        )
    else:
        plan = RFDetrExecutionPlan.resolve(execution_plan=execution_plan)

    for stage in ("buffer_strategy", "scheduler", "engine_plugin", "postprocessor"):
        selected = getattr(plan, f"{stage}_id")
        if selected not in ("base", "auto"):
            raise ValueError(
                f"Segmentation TRT supports only its existing base {stage}, got {selected!r}"
            )

    plan = replace(
        plan,
        buffer_strategy_id="base",
        scheduler_id="base",
        engine_plugin_id="base",
        postprocessor_id="base",
    )

    return plan


def _build_segmentation_preprocessor_registry(*, device):
    registry = ImplementationRegistry(scope_name="RF-DETR segmentation preprocessing")
    for implementation, factory in (
        (BasePreprocessor, BasePreprocessor),
        (PillowSIMDPreprocessor, PillowSIMDPreprocessor),
        (
            _SegmentationUniversalPreprocessor,
            lambda: _SegmentationUniversalPreprocessor(device=device),
        ),
    ):
        registry.register_factory(metadata=implementation.metadata, factory=factory)
    registry.set_auto_preferences(
        stage=OptimizationStage.PREPROCESS,
        implementation_ids=("triton-universal-v1", "pillow-simd-v1"),
    )

    return registry
