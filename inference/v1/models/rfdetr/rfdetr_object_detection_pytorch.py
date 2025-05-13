from typing import Any, List, Union, Tuple

import torch

from inference.v1 import ObjectDetectionModel, Detections
from inference.v1.configuration import DEFAULT_DEVICE
from inference.v1.entities import ColorFormat
from inference.v1.errors import CorruptedModelPackageError
from inference.v1.models.rfdetr.post_processor import PostProcess
from inference.v1.models.rfdetr.rfdetr_base_pytorch import (
    RFDETRBaseConfig,
    RFDETRLargeConfig,
    build_model,
    LWDETR,
)
from inference.v1.models.yolov8.common import (
    parse_class_names_file,
    parse_pre_processing_config,
    parse_model_characteristics,
    PreProcessingConfig,
    pre_process_images_list,
    pre_process_images_tensor,
    PreProcessingMetadata,
)
from inference.v1.utils.model_packages import get_model_package_contents


CONFIG_FOR_MODEL_TYPE = {
    "rfdetr-base": RFDETRBaseConfig,
    "rfdetr-large": RFDETRLargeConfig,
}


class RFDetrForObjectDetectionTorch(ObjectDetectionModel):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "ObjectDetectionModel":
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
            environment_file_path=model_package_content["environment.json"],
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
                f"Model package describes model_type as '{model_characteristics.model_type}' which is not supported. "
                f"Supported model types: {list(CONFIG_FOR_MODEL_TYPE.keys())}."
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
        )

    def __init__(
        self,
        model: LWDETR,
        pre_processing_config: PreProcessingConfig,
        class_names: List[str],
        device: torch.device,
        post_processor: PostProcess,
    ):
        self._model = model
        self._pre_processing_config = pre_processing_config
        self._class_names = class_names
        self._post_processor = post_processor
        self._device = device

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor]],
        input_color_format: ColorFormat = "bgr",
        **kwargs,
    ) -> Tuple[torch.Tensor, Any]:
        if isinstance(images, list):
            return pre_process_images_list(
                images=images,
                pre_processing_config=self._pre_processing_config,
                input_color_format=input_color_format,
                target_device=self._device,
            )
        return pre_process_images_tensor(
            images=images,
            pre_processing_config=self._pre_processing_config,
            input_color_format=input_color_format,
            target_device=self._device,
        )

    def forward(self, pre_processed_images: torch.Tensor, **kwargs) -> torch.Tensor:
        with torch.inference_mode():
            return self._model(pre_processed_images)

    def post_process(
        self,
        model_results: torch.Tensor,
        pre_processing_meta: List[PreProcessingMetadata],
        threshold: float = 0.5,
        **kwargs,
    ) -> List[Detections]:
        # Faster construction with list comprehension, direct tensor creation
        orig_sizes = [
            (e.original_size.height, e.original_size.width) for e in pre_processing_meta
        ]
        # Minor: for small lists, torch.as_tensor a little faster, and pin_memory False
        target_sizes = torch.as_tensor(orig_sizes, device=self._device)

        results = self._post_processor(model_results, target_sizes=target_sizes)

        detections_list = []
        append = detections_list.append  # Minor optimization: localize method
        for result in results:
            scores = result["scores"]
            labels = result["labels"]
            boxes = result["boxes"]
            # --------- Optimization: combine mask and indexing, no repeated slice eval ---------
            keep = (scores > threshold).nonzero(as_tuple=False).squeeze(-1)
            if keep.numel() == 0:
                # No object kept, skip to next. Save object construction/allocation.
                append(
                    Detections(
                        xyxy=boxes.new_zeros((0, boxes.size(-1))),
                        confidence=scores.new_zeros((0,)),
                        class_ids=labels.new_zeros((0,), dtype=labels.dtype),
                    )
                )
                continue

            # Use indexing once for all
            filtered_boxes = boxes.index_select(0, keep)
            filtered_scores = scores.index_select(0, keep)
            filtered_labels = labels.index_select(0, keep)

            detections = Detections(
                xyxy=filtered_boxes,
                confidence=filtered_scores,
                class_ids=filtered_labels,
            )
            append(detections)
        return detections_list
