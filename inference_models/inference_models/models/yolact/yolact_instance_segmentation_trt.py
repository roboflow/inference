import threading
from threading import Lock
from typing import List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torchvision

from inference_models import (
    InstanceDetections,
    InstanceSegmentationMaskFormat,
    InstanceSegmentationModel,
    PreProcessingOverrides,
)
from inference_models.configuration import (
    DEFAULT_DEVICE,
    INFERENCE_MODELS_YOLACT_DEFAULT_CLASS_AGNOSTIC_NMS,
    INFERENCE_MODELS_YOLACT_DEFAULT_CONFIDENCE,
    INFERENCE_MODELS_YOLACT_DEFAULT_IOU_THRESHOLD,
    INFERENCE_MODELS_YOLACT_DEFAULT_MAX_DETECTIONS,
)
from inference_models.entities import ColorFormat, Confidence
from inference_models.errors import (
    CorruptedModelPackageError,
    MissingDependencyError,
    ModelInputError,
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
from inference_models.models.common.roboflow.post_processing import (
    ConfidenceFilter,
    align_instance_segmentation_results,
    crop_masks_to_boxes,
)
from inference_models.models.common.roboflow.pre_processing import (
    pre_process_network_input,
)
from inference_models.models.common.trt import (
    TRTCudaGraphCache,
    establish_trt_cuda_graph_cache,
    get_trt_engine_inputs_and_outputs,
    infer_from_trt_engine,
    load_trt_model,
)
from inference_models.models.yolact.common import prepare_dense_masks, prepare_rle_masks
from inference_models.weights_providers.entities import RecommendedParameters

try:
    import tensorrt as trt
except ImportError as import_error:
    raise MissingDependencyError(
        message="Running YOLA-CT model with TRT backend on GPU requires pycuda installation, which is brought with "
        "`trt-*` extras of `inference-models` library. If you see this error running locally, "
        "please follow our installation guide: https://inference-models.roboflow.com/getting-started/installation/"
        " If you see this error using Roboflow infrastructure, make sure the service you use does support the "
        f"model, You can also contact Roboflow to get support."
        "Additionally - if AutoModel.from_pretrained(...) "
        f"automatically selects model package which does not match your environment - that's a serious problem and "
        f"we will really appreciate letting us know - https://github.com/roboflow/inference/issues",
        help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror",
    ) from import_error

try:
    import pycuda.driver as cuda
except ImportError as import_error:
    raise MissingDependencyError(
        message="Running TRT backend for YOLA-CT requires pycuda installation, which is brought with "
        "relevant `trt-*` extras of `inference-models` library. If you see this error running locally, "
        "please follow our installation guide: https://inference-models.roboflow.com/getting-started/installation/"
        " If you see this error using Roboflow infrastructure, make sure the service you use does support the "
        f"model, You can also contact Roboflow to get support.",
        help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror",
    ) from import_error


class YOLOACTForInstanceSegmentationTRT(
    InstanceSegmentationModel[
        torch.Tensor,
        PreProcessingMetadata,
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]
):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        engine_host_code_allowed: bool = False,
        trt_cuda_graph_cache: Optional[TRTCudaGraphCache] = None,
        default_trt_cuda_graph_cache_size: int = 8,
        recommended_parameters: Optional[RecommendedParameters] = None,
        **kwargs,
    ) -> "YOLOACTForInstanceSegmentationTRT":
        if device.type != "cuda":
            raise ModelRuntimeError(
                message=f"TRT engine only runs on CUDA device - {device} device detected.",
                help_url="https://inference-models.roboflow.com/errors/models-runtime/#modelruntimeerror",
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
                    ResizeMode.LETTERBOX,
                    127,
                    "YOLACT model running with TRT backend was trained with "
                    "`fit-longer-edge` input resize mode. This transform cannot be applied properly for "
                    "models with input dimensions fixed during weights export. To ensure interoperability, `letterbox` "
                    "resize mode with gray edges will be used instead. If model was trained on Roboflow platform, "
                    "we recommend using preprocessing method different that `fit-longer-edge`.",
                )
            },
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
        if len(outputs) != 5:
            raise CorruptedModelPackageError(
                message=f"Implementation assume 5 model outputs, found: {len(outputs)}.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        trt_cuda_graph_cache = establish_trt_cuda_graph_cache(
            default_cuda_graph_cache_size=default_trt_cuda_graph_cache_size,
            cuda_graph_cache=trt_cuda_graph_cache,
        )
        return cls(
            engine=engine,
            input_name=inputs[0],
            output_names=outputs,
            class_names=class_names,
            inference_config=inference_config,
            trt_config=trt_config,
            device=device,
            cuda_context=cuda_context,
            execution_context=execution_context,
            trt_cuda_graph_cache=trt_cuda_graph_cache,
            recommended_parameters=recommended_parameters,
        )

    def __init__(
        self,
        engine: trt.ICudaEngine,
        input_name: str,
        output_names: List[str],
        class_names: List[str],
        inference_config: InferenceConfig,
        trt_config: TRTConfig,
        device: torch.device,
        cuda_context: cuda.Context,
        execution_context: trt.IExecutionContext,
        trt_cuda_graph_cache: Optional[TRTCudaGraphCache],
        recommended_parameters=None,
    ):
        self._engine = engine
        self._input_name = input_name
        self._output_names = output_names
        self._class_names = class_names
        self._inference_config = inference_config
        self._trt_config = trt_config
        self._device = device
        self._cuda_context = cuda_context
        self._execution_context = execution_context
        self._trt_cuda_graph_cache = trt_cuda_graph_cache
        self._lock = Lock()
        self._inference_stream = torch.cuda.Stream(device=self._device)
        self._thread_local_storage = threading.local()
        self.recommended_parameters = recommended_parameters

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    @property
    def supported_mask_formats(self) -> Set[InstanceSegmentationMaskFormat]:
        return {"dense", "rle"}

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        pre_processing_overrides: Optional[PreProcessingOverrides] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
        with torch.cuda.stream(self._pre_process_stream):
            pre_processed_images, pre_processing_meta = pre_process_network_input(
                images=images,
                image_pre_processing=self._inference_config.image_pre_processing,
                network_input=self._inference_config.network_input,
                target_device=self._device,
                input_color_format=input_color_format,
                pre_processing_overrides=pre_processing_overrides,
            )
        self._pre_process_stream.synchronize()
        return pre_processed_images, pre_processing_meta

    def forward(
        self,
        pre_processed_images: torch.Tensor,
        disable_cuda_graphs: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cache = self._trt_cuda_graph_cache if not disable_cuda_graphs else None
        with self._lock:
            with use_cuda_context(context=self._cuda_context):
                (
                    all_loc_data,
                    all_conf_data,
                    all_mask_data,
                    all_prior_data,
                    all_proto_data,
                ) = ([], [], [], [], [])
                for image in pre_processed_images:
                    loc_data, conf_data, mask_data, prior_data, proto_data = (
                        infer_from_trt_engine(
                            pre_processed_images=image.unsqueeze(0).contiguous(),
                            trt_config=self._trt_config,
                            engine=self._engine,
                            context=self._execution_context,
                            device=self._device,
                            input_name=self._input_name,
                            outputs=self._output_names,
                            stream=self._inference_stream,
                            trt_cuda_graph_cache=cache,
                        )
                    )
                    all_loc_data.append(loc_data)
                    all_conf_data.append(conf_data)
                    all_mask_data.append(mask_data)
                    all_prior_data.append(prior_data)
                    all_proto_data.append(proto_data)
                results = (
                    torch.cat(all_loc_data, dim=0),
                    torch.cat(all_conf_data, dim=0),
                    torch.cat(all_mask_data, dim=0),
                    torch.stack(all_prior_data, dim=0),
                    torch.cat(all_proto_data, dim=0),
                )
                return results

    def post_process(
        self,
        model_results: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
        pre_processing_meta: List[PreProcessingMetadata],
        confidence: Confidence = "default",
        iou_threshold: float = INFERENCE_MODELS_YOLACT_DEFAULT_IOU_THRESHOLD,
        max_detections: int = INFERENCE_MODELS_YOLACT_DEFAULT_MAX_DETECTIONS,
        class_agnostic_nms: bool = INFERENCE_MODELS_YOLACT_DEFAULT_CLASS_AGNOSTIC_NMS,
        mask_format: InstanceSegmentationMaskFormat = "dense",
        **kwargs,
    ) -> List[InstanceDetections]:
        if mask_format not in self.supported_mask_formats:
            raise ModelInputError(
                message=f"YOLA-CT Instance Segmentation models support the following mask "
                f"formats: {self.supported_mask_formats}. Requested format: {mask_format} "
                f"is not supported. If you see this error while running on Roboflow platform, "
                f"contact support or raise an issue at https://github.com/roboflow/inference/issues. "
                f"When running locally - please verify your integration to make sure that appropriate "
                f"value of `mask_format` parameter is set.",
                help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
            )
        confidence_filter = ConfidenceFilter(
            confidence=confidence,
            recommended_parameters=self.recommended_parameters,
            default_confidence=INFERENCE_MODELS_YOLACT_DEFAULT_CONFIDENCE,
        )
        confidence = confidence_filter.get_threshold(self.class_names)
        with torch.cuda.stream(self._post_process_stream):
            for result_element in model_results:
                result_element.record_stream(self._post_process_stream)
            (
                all_loc_data,
                all_conf_data,
                all_mask_data,
                all_prior_data,
                all_proto_data,
            ) = model_results
            batch_size = all_loc_data.shape[0]
            num_priors = all_loc_data.shape[1]
            boxes = torch.zeros((batch_size, num_priors, 4), device=self._device)
            for batch_element_id, (
                batch_element_loc_data,
                batch_element_priors,
                image_prep_meta,
            ) in enumerate(zip(all_loc_data, all_prior_data, pre_processing_meta)):
                image_boxes = decode_predicted_bboxes(
                    loc_data=batch_element_loc_data,
                    priors=batch_element_priors,
                )
                inference_height, inference_width = (
                    image_prep_meta.inference_size.height,
                    image_prep_meta.inference_size.width,
                )
                scale = torch.tensor(
                    [
                        inference_width,
                        inference_height,
                        inference_width,
                        inference_height,
                    ],
                    device=self._device,
                )
                image_boxes = image_boxes.mul_(scale)
                boxes[batch_element_id, :, :] = image_boxes
            all_conf_data = all_conf_data[:, :, 1:]  # remove background class
            instances = torch.cat([boxes, all_conf_data, all_mask_data], dim=2)
            nms_results = run_nms_for_instance_segmentation(
                output=instances,
                conf_thresh=confidence,
                iou_thresh=iou_threshold,
                max_detections=max_detections,
                class_agnostic=class_agnostic_nms,
            )
            if mask_format == "dense":
                final_results = prepare_dense_masks(
                    nms_results=nms_results,
                    all_proto_data=all_proto_data,
                    pre_processing_meta=pre_processing_meta,
                )
            else:
                final_results = prepare_rle_masks(
                    nms_results=nms_results,
                    all_proto_data=all_proto_data,
                    pre_processing_meta=pre_processing_meta,
                )
        self._post_process_stream.synchronize()
        return final_results

    @property
    def _pre_process_stream(self) -> torch.cuda.Stream:
        if not hasattr(self._thread_local_storage, "pre_process_stream"):
            self._thread_local_storage.pre_process_stream = torch.cuda.Stream(
                device=self._device
            )
        return self._thread_local_storage.pre_process_stream

    @property
    def _post_process_stream(self) -> torch.cuda.Stream:
        if not hasattr(self._thread_local_storage, "post_process_stream"):
            self._thread_local_storage.post_process_stream = torch.cuda.Stream(
                device=self._device
            )
        return self._thread_local_storage.post_process_stream


def decode_predicted_bboxes(
    loc_data: torch.Tensor, priors: torch.Tensor
) -> torch.Tensor:
    variances = torch.tensor([0.1, 0.2], device=loc_data.device)
    boxes = torch.cat(
        [
            priors[:, :2] + loc_data[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc_data[:, 2:] * variances[1]),
        ],
        dim=1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def run_nms_for_instance_segmentation(
    output: torch.Tensor,
    conf_thresh: Union[float, torch.Tensor] = 0.25,
    iou_thresh: float = 0.45,
    max_detections: int = 100,
    class_agnostic: bool = False,
) -> List[torch.Tensor]:
    """
    `conf_thresh`: scalar applies to all classes; 1-D tensor of shape
    (num_classes,) indexed by class_id for per-class thresholds.
    """
    bs = output.shape[0]
    boxes = output[:, :, :4]  # (N, 19248, 4)
    scores = output[:, :, 4:-32]  # (N, 19248, num_classes)
    masks = output[:, :, -32:]
    results = []
    for b in range(bs):
        bboxes = boxes[b]  # (19248, 4)
        class_scores = scores[b]  # (19248, 80)
        box_masks = masks[b]
        class_conf, class_ids = class_scores.max(1)  # (8400,), (8400,)
        if isinstance(conf_thresh, torch.Tensor):
            mask = class_conf > conf_thresh.to(output.device)[class_ids]
        else:
            mask = class_conf > conf_thresh
        if mask.sum() == 0:
            results.append(torch.zeros((0, 38), device=output.device))
            continue
        bboxes = bboxes[mask]
        class_conf = class_conf[mask]
        class_ids = class_ids[mask]
        box_masks = box_masks[mask]
        # Class-agnostic NMS -> use dummy class ids
        nms_class_ids = torch.zeros_like(class_ids) if class_agnostic else class_ids
        keep = torchvision.ops.batched_nms(
            bboxes, class_conf, nms_class_ids, iou_thresh
        )
        keep = keep[:max_detections]
        detections = torch.cat(
            [
                bboxes[keep],
                class_conf[keep].unsqueeze(1),
                class_ids[keep].unsqueeze(1).float(),
                box_masks[keep],
            ],
            dim=1,
        )  # [x1, y1, x2, y2, conf, cls]
        results.append(detections)
    return results
