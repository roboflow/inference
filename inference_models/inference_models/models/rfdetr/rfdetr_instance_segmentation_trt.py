import threading
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from inference_models import InstanceDetections, InstanceSegmentationModel
from inference_models.configuration import (
    DEFAULT_DEVICE,
    INFERENCE_MODELS_RFDETR_DEFAULT_CONFIDENCE,
)
from inference_models.entities import ColorFormat
from inference_models.errors import (
    CorruptedModelPackageError,
    MissingDependencyError,
    ModelRuntimeError,
)
from inference_models.models.common.cuda import (
    use_cuda_context,
    use_primary_cuda_context,
)
from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.models.common.roboflow.model_packages import (
    InferenceConfig,
    PreProcessingMetadata,
    ResizeMode,
    TRTConfig,
    parse_class_names_file,
    parse_inference_config,
    parse_trt_config,
)
from inference_models.models.common.roboflow.pre_processing import (
    pre_process_network_input,
)
from inference_models.models.common.trt import (
    get_trt_engine_inputs_and_outputs,
    infer_from_trt_engine,
    load_trt_model,
)
from inference_models.models.rfdetr.class_remapping import (
    ClassesReMapping,
    prepare_class_remapping,
)
from inference_models.models.rfdetr.common import (
    post_process_instance_segmentation_results,
)

try:
    import tensorrt as trt
except ImportError as import_error:
    raise MissingDependencyError(
        message=f"Could not import RFDetr model with TRT backend - this error means that some additional dependencies "
        f"are not installed in the environment.  If you see this error running locally, "
        "please follow our installation guide: https://inference-models.roboflow.com/getting-started/installation/"
        " If you see this error using Roboflow infrastructure, make sure the service you use does support the "
        f"model, You can also contact Roboflow to get support. "
        "Additionally - if AutoModel.from_pretrained(...) "
        f"automatically selects model package which does not match your environment - that's a serious problem and "
        f"we will really appreciate letting us know - https://github.com/roboflow/inference/issues",
        help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror",
    ) from import_error

try:
    import pycuda.driver as cuda
except ImportError as import_error:
    raise MissingDependencyError(
        message="Running model RFDETR with TRT backend on GPU requires pycuda installation, which is brought with "
        "`trt-*` extras of `inference-models` library. If you see this error running locally, "
        "please follow our installation guide: https://inference-models.roboflow.com/getting-started/installation/"
        " If you see this error using Roboflow infrastructure, make sure the service you use does support the "
        f"model, You can also contact Roboflow to get support."
        "Additionally - if AutoModel.from_pretrained(...) "
        f"automatically selects model package which does not match your environment - that's a serious problem and "
        f"we will really appreciate letting us know - https://github.com/roboflow/inference/issues",
        help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror"
    ) from import_error


class RFDetrForInstanceSegmentationTRT(
    InstanceSegmentationModel[
        torch.Tensor,
        PreProcessingMetadata,
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]
):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        engine_host_code_allowed: bool = False,
        **kwargs,
    ) -> "RFDetrForInstanceSegmentationTRT":
        if device.type != "cuda":
            raise ModelRuntimeError(
                message="TRT engine only runs on CUDA device - {device} device detected.",
                help_url="https://todo",
            )
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=[
                "class_names.txt",
                "inference_config.json",
                "trt_config.json",
                "engine.plan",
            ],
        )
        class_names = parse_class_names_file(
            class_names_path=model_package_content["class_names.txt"]
        )
        inference_config = parse_inference_config(
            config_path=model_package_content["inference_config.json"],
            allowed_resize_modes={
                ResizeMode.STRETCH_TO,
                ResizeMode.LETTERBOX,
                ResizeMode.CENTER_CROP,
                ResizeMode.LETTERBOX_REFLECT_EDGES,
            },
            implicit_resize_mode_substitutions={
                ResizeMode.FIT_LONGER_EDGE: (
                    ResizeMode.STRETCH_TO,
                    None,
                    "RFDetr Instance Segmentation model running with TRT backend was trained with "
                    "`fit-longer-edge` input resize mode. This transform cannot be applied properly for "
                    "RFDetr models. To ensure interoperability, `stretch` "
                    "resize mode will be used instead. If model was trained on Roboflow platform, "
                    "we recommend using preprocessing method different that `fit-longer-edge`.",
                )
            },
        )
        classes_re_mapping = None
        if inference_config.class_names_operations:
            class_names, classes_re_mapping = prepare_class_remapping(
                class_names=class_names,
                class_names_operations=inference_config.class_names_operations,
                device=device,
            )
        trt_config = parse_trt_config(
            config_path=model_package_content["trt_config.json"]
        )
        cuda.init()
        cuda_device = cuda.Device(device.index or 0)
        with use_primary_cuda_context(cuda_device=cuda_device) as cuda_context:
            engine = load_trt_model(
                model_path=model_package_content["engine.plan"],
                engine_host_code_allowed=engine_host_code_allowed,
            )
            execution_context = engine.create_execution_context()
        inputs, outputs = get_trt_engine_inputs_and_outputs(engine=engine)
        if len(inputs) != 1:
            raise CorruptedModelPackageError(
                message=f"Implementation assume single model input, found: {len(inputs)}.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        if len(outputs) != 3:
            raise CorruptedModelPackageError(
                message=f"Implementation assume 3 model outputs, found: {len(outputs)}.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        return cls(
            engine=engine,
            input_name=inputs[0],
            output_names=outputs,
            class_names=class_names,
            classes_re_mapping=classes_re_mapping,
            inference_config=inference_config,
            trt_config=trt_config,
            device=device,
            cuda_context=cuda_context,
            execution_context=execution_context,
        )

    def __init__(
        self,
        engine: trt.ICudaEngine,
        input_name: str,
        output_names: List[str],
        class_names: List[str],
        classes_re_mapping: Optional[ClassesReMapping],
        inference_config: InferenceConfig,
        trt_config: TRTConfig,
        device: torch.device,
        cuda_context: cuda.Context,
        execution_context: trt.IExecutionContext,
    ):
        self._engine = engine
        self._input_name = input_name
        self._output_names = output_names
        self._inference_config = inference_config
        self._class_names = class_names
        self._classes_re_mapping = classes_re_mapping
        self._device = device
        self._cuda_context = cuda_context
        self._execution_context = execution_context
        self._trt_config = trt_config
        self._lock = threading.Lock()

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        image_size: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
        return pre_process_network_input(
            images=images,
            image_pre_processing=self._inference_config.image_pre_processing,
            network_input=self._inference_config.network_input,
            target_device=self._device,
            input_color_format=input_color_format,
            image_size_wh=image_size,
        )

    def forward(
        self, pre_processed_images: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with self._lock:
            with use_cuda_context(context=self._cuda_context):
                detections, labels, masks = infer_from_trt_engine(
                    pre_processed_images=pre_processed_images,
                    trt_config=self._trt_config,
                    engine=self._engine,
                    context=self._execution_context,
                    device=self._device,
                    input_name=self._input_name,
                    outputs=self._output_names,
                )
                return detections, labels, masks

    def post_process(
        self,
        model_results: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        pre_processing_meta: List[PreProcessingMetadata],
        confidence: float = INFERENCE_MODELS_RFDETR_DEFAULT_CONFIDENCE,
        **kwargs,
    ) -> List[InstanceDetections]:
        bboxes, logits, masks = model_results
        return post_process_instance_segmentation_results(
            bboxes=bboxes,
            logits=logits,
            masks=masks,
            pre_processing_meta=pre_processing_meta,
            threshold=confidence,
            classes_re_mapping=self._classes_re_mapping,
        )
