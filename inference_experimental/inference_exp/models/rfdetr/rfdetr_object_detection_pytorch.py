from copy import deepcopy
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from inference_exp import Detections, ObjectDetectionModel
from inference_exp.configuration import DEFAULT_DEVICE
from inference_exp.entities import ColorFormat
from inference_exp.errors import CorruptedModelPackageError, ModelRuntimeError
from inference_exp.logger import LOGGER
from inference_exp.models.common.model_packages import get_model_package_contents
from inference_exp.models.common.roboflow.model_packages import (
    PreProcessingConfig,
    PreProcessingMetadata,
    parse_class_names_file,
    parse_model_characteristics,
    parse_pre_processing_config,
)
from inference_exp.models.common.roboflow.pre_processing import (
    pre_process_network_input,
)
from inference_exp.models.rfdetr.post_processor import PostProcess
from inference_exp.models.rfdetr.rfdetr_base_pytorch import (
    LWDETR,
    RFDETRBaseConfig,
    RFDETRLargeConfig,
    build_model,
)

try:
    torch.set_float32_matmul_precision("high")
except:
    pass

CONFIG_FOR_MODEL_TYPE = {
    "rfdetr-base": RFDETRBaseConfig,
    "rfdetr-large": RFDETRLargeConfig,
}


class RFDetrForObjectDetectionTorch(
    (ObjectDetectionModel[torch.Tensor, PreProcessingMetadata, dict])
):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "RFDetrForObjectDetectionTorch":
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=[
                "class_names.txt",
                "environment.json",
                "model_type.json",
                "weights.pth",
            ],
        )
        class_names = parse_class_names_file(
            class_names_path=model_package_content["class_names.txt"]
        )
        pre_processing_config = parse_pre_processing_config(
            config_path=model_package_content["environment.json"],
        )
        weights_dict = torch.load(
            model_package_content["weights.pth"],
            map_location=device,
            weights_only=False,
        )["model"]
        model_characteristics = parse_model_characteristics(
            config_path=model_package_content["model_type.json"]
        )
        if model_characteristics.model_type not in CONFIG_FOR_MODEL_TYPE:
            raise CorruptedModelPackageError(
                message=f"Model package describes model_type as '{model_characteristics.model_type}' which is not supported. "
                f"Supported model types: {list(CONFIG_FOR_MODEL_TYPE.keys())}.",
                help_url="https://todo",
            )
        model_config = CONFIG_FOR_MODEL_TYPE[model_characteristics.model_type](
            device=device
        )

        model = build_model(config=model_config)
        model.load_state_dict(weights_dict)
        model = model.eval().to(device)
        post_processor = PostProcess()
        return cls(
            model=model,
            class_names=class_names,
            device=device,
            pre_processing_config=pre_processing_config,
            post_processor=post_processor,
            resolution=model_config.resolution,
        )

    def __init__(
        self,
        model: LWDETR,
        pre_processing_config: PreProcessingConfig,
        class_names: List[str],
        device: torch.device,
        post_processor: PostProcess,
        resolution: int,
    ):
        self._model = model
        self._pre_processing_config = pre_processing_config
        self._class_names = class_names
        self._post_processor = post_processor
        self._device = device
        self._resolution = resolution
        self._has_warned_about_not_being_optimized_for_inference = False
        self._inference_model: Optional[LWDETR] = None
        self._optimized_has_been_compiled = False
        self._optimized_batch_size = None
        self._optimized_dtype = None

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    def optimize_for_inference(
        self,
        compile: bool = True,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.remove_optimized_model()
        self._inference_model = deepcopy(self._model)
        self._inference_model.eval()
        self._inference_model.export()
        self._inference_model = self._inference_model.to(dtype=dtype)
        self._optimized_dtype = dtype
        if compile:
            self._inference_model = torch.jit.trace(
                self._inference_model,
                torch.randn(
                    batch_size,
                    3,
                    self._resolution,
                    self._resolution,
                    device=self._device,
                    dtype=dtype,
                ),
            )
            self._optimized_has_been_compiled = True
            self._optimized_batch_size = batch_size

    def remove_optimized_model(self) -> None:
        self._has_warned_about_not_being_optimized_for_inference = False
        self._inference_model = None
        self._optimized_has_been_compiled = False
        self._optimized_batch_size = None

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
        return pre_process_network_input(
            images=images,
            pre_processing_config=self._pre_processing_config,
            expected_network_color_format="rgb",
            target_device=self._device,
            input_color_format=input_color_format,
            normalization=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        )

    def forward(self, pre_processed_images: torch.Tensor, **kwargs) -> dict:
        if (
            self._inference_model is None
            and not self._has_warned_about_not_being_optimized_for_inference
        ):
            LOGGER.warning(
                "Model is not optimized for inference. "
                "Latency may be higher than expected. "
                "You can optimize the model for inference by calling model.optimize_for_inference()."
            )
            self._has_warned_about_not_being_optimized_for_inference = True
        if self._inference_model is not None:
            if (self._resolution, self._resolution) != tuple(
                pre_processed_images.shape[2:]
            ):
                raise ModelRuntimeError(
                    message=f"Resolution mismatch. Model was optimized for resolution {self._resolution}, "
                    f"but got {tuple(pre_processed_images.shape[2:])}. "
                    "You can explicitly remove the optimized model by calling model.remove_optimized_model().",
                    help_url="https://todo",
                )
            if self._optimized_has_been_compiled:
                if self._optimized_batch_size != pre_processed_images.shape[0]:
                    raise ModelRuntimeError(
                        message="Batch size mismatch. Optimized model was compiled for batch size "
                        f"{self._optimized_batch_size}, but got {pre_processed_images.shape[0]}. "
                        "You can explicitly remove the optimized model by calling model.remove_optimized_model(). "
                        "Alternatively, you can recompile the optimized model for a different batch size "
                        "by calling model.optimize_for_inference(batch_size=<new_batch_size>).",
                        help_url="https://todo",
                    )
        with torch.inference_mode():
            if self._inference_model:
                predictions = self._inference_model(
                    pre_processed_images.to(dtype=self._optimized_dtype)
                )
            else:
                predictions = self._model(pre_processed_images)
            if isinstance(predictions, tuple):
                predictions = {
                    "pred_logits": predictions[1],
                    "pred_boxes": predictions[0],
                }
            return predictions

    def post_process(
        self,
        model_results: dict,
        pre_processing_meta: List[PreProcessingMetadata],
        threshold: float = 0.5,
        **kwargs,
    ) -> List[Detections]:
        orig_sizes = [
            (e.original_size.height, e.original_size.width) for e in pre_processing_meta
        ]
        target_sizes = torch.tensor(orig_sizes, device=self._device)
        results = self._post_processor(model_results, target_sizes=target_sizes)
        detections_list = []
        for result in results:
            scores = result["scores"]
            labels = result["labels"]
            boxes = result["boxes"]

            keep = scores > threshold
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]

            detections = Detections(
                xyxy=boxes.round().int(),
                confidence=scores,
                class_id=labels.int(),
            )
            detections_list.append(detections)
        return detections_list
