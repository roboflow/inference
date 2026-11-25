"""Native RF-DETR OBB ONNX consumer. Angles are clockwise image-space radians."""

from typing import List

import numpy as np

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.models.common.onnx import (
    create_onnx_inference_session,
    set_onnx_execution_provider_defaults,
)
from inference_models.models.common.roboflow.oriented_box_metadata import (
    OrientedBoxMetadata,
)
from inference_models.models.rfdetr.rfdetr_oriented_object_detection_common import (  # noqa: F401
    OUTPUT_NAMES,
    OrientedDetections,
    OrientedPreprocessingMetadata,
    RFDetrForOrientedObjectDetectionBase,
    _corners,
    check_model_input,
    check_output_shapes,
    load_oriented_package,
)
from inference_models.utils.onnx_introspection import (
    get_selected_onnx_execution_providers,
)

# Execution providers the native OBB graph runs on. OpenVINO and CoreML have not been validated with it.
SUPPORTED_EXECUTION_PROVIDERS = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
)


class RFDetrForOrientedObjectDetectionONNX(RFDetrForOrientedObjectDetectionBase):
    """Run the three-output native OBB artifact with ONNX Runtime on TensorRT, CUDA or CPU.

    Preprocessing and decoding are shared with the TRT backend (``RFDetrForOrientedObjectDetectionBase``).
    """

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """Load a standard RF-DETR package with Core-owned frozen class semantics.

        Args:
            model_name_or_path: Local package directory with standard RF-DETR ONNX files.
            **kwargs: Shared loader options. onnx_execution_providers (TensorRT, CUDA and/or CPU; default: the
                environment's ONNXRUNTIME_EXECUTION_PROVIDERS among those), device and default_onnx_trt_options
                as for other ONNX models.

        Returns:
            Initialized native model.

        Raises:
            ValueError: For unsupported backend, malformed outputs or missing metadata.
        """
        maximum = kwargs.get("rf_detr_max_input_resolution")
        package = load_oriented_package(
            model_name_or_path,
            raw_metadata=kwargs.get("oriented_box_metadata"),
            weights_files=["weights.onnx"],
            max_input_resolution=maximum,
        )
        providers = kwargs.get("onnx_execution_providers")
        if providers is None:
            providers = get_selected_onnx_execution_providers()
        providers = [
            provider
            for provider in providers
            if (provider[0] if isinstance(provider, tuple) else provider)
            in SUPPORTED_EXECUTION_PROVIDERS
        ]
        if not providers:
            raise ValueError(
                "Native OBB runs on TensorrtExecutionProvider, CUDAExecutionProvider or CPUExecutionProvider; "
                "none of them is selected"
            )
        providers = set_onnx_execution_provider_defaults(
            providers=providers,
            model_package_path=model_name_or_path,
            device=kwargs.get("device") or DEFAULT_DEVICE,
            default_onnx_trt_options=kwargs.get("default_onnx_trt_options", True),
        )
        session = create_onnx_inference_session(
            model_path=package.files["weights.onnx"], providers=providers
        )
        model = cls(
            session=session,
            metadata=package.metadata,
            class_names=package.class_names,
            output_indices=package.output_indices,
            image_pre_processing=package.inference_config.image_pre_processing,
            network_input=package.inference_config.network_input,
        )
        check_model_input(package, model.resolution, maximum)
        return model

    def __init__(
        self,
        *,
        session,
        metadata: OrientedBoxMetadata,
        class_names: List[str],
        output_indices: List[int],
        image_pre_processing,
        network_input,
    ):
        inputs = session.get_inputs()
        if len(inputs) != 1 or len(inputs[0].shape) != 4:
            raise ValueError("OBB ONNX must have one BCHW image input")
        shape = inputs[0].shape
        if not isinstance(shape[2], int) or shape[2] != shape[3]:
            raise ValueError("OBB ONNX requires a static square input size")
        check_output_shapes(
            {o.name: tuple(o.shape) for o in session.get_outputs()}, output_indices
        )
        super().__init__(
            metadata=metadata,
            class_names=class_names,
            output_indices=output_indices,
            image_pre_processing=image_pre_processing,
            network_input=network_input,
            resolution=shape[2],
        )
        self._session = session
        self._input_name = inputs[0].name
        self._input_batch_size = shape[0]

    def forward(self, pre_processed_images, **kwargs):
        """Execute the exported graph, respecting a fixed model batch dimension.

        Args:
            pre_processed_images: Normalized float32 BCHW array.
            **kwargs: Reserved shared inference options.

        Returns:
            Named raw dets, labels and heading arrays before selection.
        """
        batch_limit = (
            self._input_batch_size
            if isinstance(self._input_batch_size, int)
            else len(pre_processed_images)
        )
        outputs = []
        for start in range(0, len(pre_processed_images), batch_limit):
            batch = pre_processed_images[start : start + batch_limit]
            count = len(batch)
            if count < batch_limit:
                batch = np.concatenate(
                    [batch, np.repeat(batch[-1:], batch_limit - count, axis=0)]
                )
            values = self._session.run(list(OUTPUT_NAMES), {self._input_name: batch})
            outputs.append([v[:count] for v in values])
        result = tuple(np.concatenate([o[i] for o in outputs]) for i in range(3))
        return result
