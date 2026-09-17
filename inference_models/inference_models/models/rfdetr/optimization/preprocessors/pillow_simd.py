"""Explicit, numerically approximate CPU resize for RF-DETR NumPy images."""

from dataclasses import replace

from inference_models.models.common.pillow_simd import load_pillow_simd_image
from inference_models.models.optimization.contracts import (
    CompatibilityResult,
    DeviceCompatibility,
    immutable_mapping,
)
from inference_models.models.rfdetr.optimization.ids import (
    RFDETR_PREPROCESSOR_PILLOW_SIMD_V1,
)
from inference_models.models.rfdetr.optimization.preprocessors.base import (
    BasePreprocessor,
)
from inference_models.models.rfdetr.optimization.preprocessors.common import (
    run_reference_preprocessor,
)
from inference_models.models.rfdetr.optimization.preprocessors.compatibility import (
    check_threaded_request_compatibility,
)


class PillowSIMDPreprocessor(BasePreprocessor):
    """Opt-in Pillow-SIMD; regular Pillow remains the exact base/fallback."""

    metadata = replace(
        BasePreprocessor.metadata,
        implementation_id=RFDETR_PREPROCESSOR_PILLOW_SIMD_V1,
        target=DeviceCompatibility(
            device_kind="any", host_architectures=("x86_64", "amd64")
        ),
        inputs=replace(
            BasePreprocessor.metadata.inputs, dtypes=("uint8",), layouts=("HWC", "NHWC")
        ),
        dependencies=("Pillow", "Pillow-SIMD>=12.3.0.post0", "torch", "torchvision"),
        changes_numerics=True,
        numerical_behavior=(
            "Not byte-exact with standard Pillow: bilinear downscales can differ "
            "by one uint8 intensity level before normalization. The approximately "
            "0.1% differing pixels reported in PR #2989 are workload-specific, "
            "not a guarantee. No-op resize is exact."
        ),
        output_contract=immutable_mapping(
            {
                "device": "selected target device (resize runs on CPU)",
                "dtype": "float32",
                "layout": "contiguous NCHW",
                "ownership": "new tensor owned by caller",
            }
        ),
        stream_behavior="SSE4.1 CPU resize followed by transfer on the caller stream",
    )

    def __init__(self, *, max_workers=1):
        super().__init__(max_workers=max_workers)
        self._image = None
        self._unavailable_reason = None
        try:
            self._image = load_pillow_simd_image()
        except ImportError as error:
            self._unavailable_reason = str(error)

    def check_model_compatibility(self, *, image_pre_processing, network_input):
        if self._unavailable_reason:
            return CompatibilityResult.incompatible(self._unavailable_reason)
        return CompatibilityResult.compatible()

    def check_request_compatibility(self, *, request, context):
        return check_threaded_request_compatibility(request)

    def preprocess(self, request, context):
        # Selection validates availability before execution; never silently change
        # the effective implementation ID by falling back inside this stage.
        if self._image is None:
            raise ImportError(self._unavailable_reason)
        result = run_reference_preprocessor(
            request,
            context,
            implementation_id="base",
            max_workers=1,
            image_module=self._image,
        )
        return replace(
            result,
            implementation_id=self.metadata.implementation_id,
            input_kind="numpy-pillow-simd",
        )
