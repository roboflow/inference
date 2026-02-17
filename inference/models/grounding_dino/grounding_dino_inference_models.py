from time import perf_counter
from typing import Any, List

import torch

from inference.core.entities.requests.groundingdino import GroundingDINOInferenceRequest
from inference.core.entities.requests.inference import InferenceRequestImage
from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
)
from inference.core.env import (
    ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
    ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
    API_KEY,
    CLASS_AGNOSTIC_NMS,
)
from inference.core.models.base import Model
from inference.core.roboflow_api import get_extra_weights_provider_headers
from inference.core.utils.image_utils import load_image_bgr, xyxy_to_xywh
from inference_models import AutoModel
from inference_models.models.grounding_dino.grounding_dino_torch import (
    GroundingDinoForObjectDetectionTorch,
)


class InferenceModelsGroundingDINOAdapter(Model):
    """GroundingDINO class for zero-shot object detection.

    Attributes:
        model: The GroundingDINO model.
    """

    def __init__(
        self,
        model_id: str = "grounding_dino/groundingdino_swint_ogc",
        api_key: str = None,
        **kwargs
    ):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY

        self.task_type = "object-detection"

        extra_weights_provider_headers = get_extra_weights_provider_headers()

        self._model: GroundingDinoForObjectDetectionTorch = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            weights_provider_extra_headers=extra_weights_provider_headers,
            **kwargs,
        )

    def preproc_image(self, image: Any):
        """Preprocesses an image.

        Args:
            image (InferenceRequestImage): The image to preprocess.

        Returns:
            np.array: The preprocessed image.
        """
        return load_image_bgr(image)

    def infer_from_request(
        self,
        request: GroundingDINOInferenceRequest,
    ) -> ObjectDetectionInferenceResponse:
        """
        Perform inference based on the details provided in the request, and return the associated responses.
        """
        result = self.infer(**request.dict())
        return result

    def infer(
        self,
        image: InferenceRequestImage,
        text: List[str] = None,
        class_filter: list = None,
        box_threshold=0.5,
        text_threshold=0.5,
        class_agnostic_nms=CLASS_AGNOSTIC_NMS,
        **kwargs
    ):
        """
        Run inference on a provided image.
            - image: can be a BGR numpy array, filepath, InferenceRequestImage, PIL Image, byte-string, etc.

        Args:
            request (CVInferenceRequest): The inference request.
            class_filter (Optional[List[str]]): A list of class names to filter, if provided.

        Returns:
            GroundingDINOInferenceRequest: The inference response.
        """
        if text is None:
            raise ValueError(
                "`text` parameter is required for GroundingDINO inference."
            )
        t1 = perf_counter()
        image = self.preproc_image(image)
        img_dims = image.shape

        detections = self._model.infer(
            images=image,
            classes=text,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            class_agnostic_nms=class_agnostic_nms,
        )[0]
        t2 = perf_counter() - t1
        predictions_for_image = []
        for instance_id in range(detections.xyxy.shape[0]):
            x_min, y_min, x_max, y_max = detections.xyxy[instance_id].tolist()
            width = x_max - x_min
            height = y_max - y_min
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            class_id = detections.class_id[instance_id].item()
            confidence = detections.confidence[instance_id].item()
            class_name = text[class_id]
            if class_filter and class_name not in class_filter:
                continue
            predictions_for_image.append(
                ObjectDetectionPrediction(
                    # Passing args as a dictionary here since one of the args is 'class' (a protected term in Python)
                    **{
                        "x": center_x,
                        "y": center_y,
                        "width": width,
                        "height": height,
                        "confidence": confidence,
                        "class": text[class_id],
                        "class_id": class_id,  # you can only prompt for one object at once
                    }
                )
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return ObjectDetectionInferenceResponse(
            predictions=predictions_for_image,
            image=InferenceResponseImage(width=img_dims[1], height=img_dims[0]),
            time=t2,
        )

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        """Clears any cache if necessary. TODO: Implement this to delete the cache from the experimental model.

        Args:
            delete_from_disk (bool, optional): Whether to delete cached files from disk. Defaults to True.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
