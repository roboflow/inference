import threading
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from inference_exp import Detections, ObjectDetectionModel
from inference_exp.configuration import DEFAULT_DEVICE
from inference_exp.entities import ColorFormat
from inference_exp.errors import (
    CorruptedModelPackageError,
    MissingDependencyError,
    ModelRuntimeError,
)
from inference_exp.models.common.cuda import use_cuda_context, use_primary_cuda_context
from inference_exp.models.common.model_packages import get_model_package_contents
from inference_exp.models.common.roboflow.model_packages import (
    InferenceConfig,
    PreProcessingMetadata,
    ResizeMode,
    TRTConfig,
    parse_class_names_file,
    parse_inference_config,
    parse_trt_config,
)
from inference_exp.models.common.roboflow.post_processing import (
    rescale_image_detections,
)
from inference_exp.models.common.roboflow.pre_processing import (
    pre_process_network_input,
)
from inference_exp.models.common.trt import (
    get_engine_inputs_and_outputs,
    infer_from_trt_engine,
    load_model,
)
from inference_exp.models.rfdetr.class_remapping import (
    ClassesReMapping,
    prepare_class_remapping,
)

try:
    import tensorrt as trt
except ImportError as import_error:
    raise MissingDependencyError(
        message=f"Could not import RFDetr model with TRT backend - this error means that some additional dependencies "
        f"are not installed in the environment. If you run the `inference-exp` library directly in your Python "
        f"program, make sure the following extras of the package are installed: `trt10` - installation can only "
        f"succeed for Linux and Windows machines with Cuda 12 installed. Jetson devices, should have TRT 10.x "
        f"installed for all builds with Jetpack 6. "
        f"If you see this error using Roboflow infrastructure, make sure the service you use does support the model. "
        f"You can also contact Roboflow to get support.",
        help_url="https://todo",
    ) from import_error

try:
    import pycuda.driver as cuda
except ImportError as import_error:
    raise MissingDependencyError(
        message="TODO", help_url="https://todo"
    ) from import_error


class RFDetrForObjectDetectionTRT(
    (
        ObjectDetectionModel[
            torch.Tensor, PreProcessingMetadata, Tuple[torch.Tensor, torch.Tensor]
        ]
    )
):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        engine_host_code_allowed: bool = False,
        **kwargs,
    ) -> "RFDetrForObjectDetectionTRT":
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
            engine = load_model(
                model_path=model_package_content["engine.plan"],
                engine_host_code_allowed=engine_host_code_allowed,
            )
            execution_context = engine.create_execution_context()
        inputs, outputs = get_engine_inputs_and_outputs(engine=engine)
        if len(inputs) != 1:
            raise CorruptedModelPackageError(
                message=f"Implementation assume single model input, found: {len(inputs)}.",
                help_url="https://todo",
            )
        if len(outputs) != 2:
            raise CorruptedModelPackageError(
                message=f"Implementation assume 2 model outputs, found: {len(outputs)}.",
                help_url="https://todo",
            )
        if "dets" not in outputs or "labels" not in outputs:
            raise CorruptedModelPackageError(
                message=f"Expected model outputs to be named `output0` and `output1`, but found: {outputs}.",
                help_url="https://todo",
            )
        return cls(
            engine=engine,
            input_name=inputs[0],
            output_names=["dets", "labels"],
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
        **kwargs,
    ) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
        return pre_process_network_input(
            images=images,
            image_pre_processing=self._inference_config.image_pre_processing,
            network_input=self._inference_config.network_input,
            target_device=self._device,
            input_color_format=input_color_format,
        )

    def forward(
        self, pre_processed_images: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with self._lock:
            with use_cuda_context(context=self._cuda_context):
                detections, labels = infer_from_trt_engine(
                    pre_processed_images=pre_processed_images,
                    trt_config=self._trt_config,
                    engine=self._engine,
                    context=self._execution_context,
                    device=self._device,
                    input_name=self._input_name,
                    outputs=self._output_names,
                )
                return detections, labels

    def post_process(
        self,
        model_results: Tuple[torch.Tensor, torch.Tensor],
        pre_processing_meta: List[PreProcessingMetadata],
        threshold: float = 0.5,
        **kwargs,
    ) -> List[Detections]:
        bboxes, logits = model_results
        logits_sigmoid = torch.nn.functional.sigmoid(logits)
        results = []
        for image_bboxes, image_logits, image_meta in zip(
            bboxes, logits_sigmoid, pre_processing_meta
        ):
            confidence, top_classes = image_logits.max(dim=1)
            confidence_mask = confidence > threshold
            confidence = confidence[confidence_mask]
            top_classes = top_classes[confidence_mask]
            selected_boxes = image_bboxes[confidence_mask]
            confidence, sorted_indices = torch.sort(confidence, descending=True)
            top_classes = top_classes[sorted_indices]
            selected_boxes = selected_boxes[sorted_indices]
            if self._classes_re_mapping is not None:
                remapping_mask = torch.isin(
                    top_classes, self._classes_re_mapping.remaining_class_ids
                )
                top_classes = self._classes_re_mapping.class_mapping[
                    top_classes[remapping_mask]
                ]
                selected_boxes = selected_boxes[remapping_mask]
                confidence = confidence[remapping_mask]
            cxcy = selected_boxes[:, :2]
            wh = selected_boxes[:, 2:]
            xy_min = cxcy - 0.5 * wh
            xy_max = cxcy + 0.5 * wh
            selected_boxes_xyxy_pct = torch.cat([xy_min, xy_max], dim=-1)
            inference_size_hwhw = torch.tensor(
                [
                    image_meta.inference_size.height,
                    image_meta.inference_size.width,
                    image_meta.inference_size.height,
                    image_meta.inference_size.width,
                ],
                device=self._device,
            )
            selected_boxes_xyxy = selected_boxes_xyxy_pct * inference_size_hwhw
            selected_boxes_xyxy = rescale_image_detections(
                image_detections=selected_boxes_xyxy,
                image_metadata=image_meta,
            )
            detections = Detections(
                xyxy=selected_boxes_xyxy.round().int(),
                confidence=confidence,
                class_id=top_classes.int(),
            )
            results.append(detections)
        return results
