from typing import Any, List, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
)
from inference.core.env import (
    ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
    ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
    API_KEY,
)
from inference.core.models.base import Model, PreprocessReturnMetadata
from inference.core.models.inference_models_adapters import (
    get_extra_weights_provider_headers,
)
from inference.core.utils.image_utils import load_image_bgr
from inference_models import AutoModel, Detections
from inference_models.models.moondream2.moondream2_hf import MoonDream2HF


class InferenceModelsMoondream2Adapter(Model):
    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY

        self.task_type = "llm"

        extra_weights_provider_headers = get_extra_weights_provider_headers()

        self._model: MoonDream2HF = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            extra_weights_provider_headers=extra_weights_provider_headers,
            **kwargs,
        )

    def preprocess(self, image: Any, prompt: str, **kwargs):
        is_batch = isinstance(image, list)
        if is_batch:
            raise ValueError("This model does not support batched-inference.")
        np_image = load_image_bgr(
            image,
            disable_preproc_auto_orient=kwargs.get(
                "disable_preproc_auto_orient", False
            ),
        )
        input_shape = PreprocessReturnMetadata({"image_dims": np_image.shape[:2][::-1]})
        return (
            np_image,
            input_shape,
        )

    def predict(
        self,
        image_in: Union[Image.Image, np.array],
        prompt: Union[str, List[str]] = "",
        **kwargs,
    ):
        return self.detect(image_in, prompt=prompt, **kwargs)

    def postprocess(
        self,
        predictions: List[ObjectDetectionInferenceResponse],
        preprocess_return_metadata: PreprocessReturnMetadata,
        **kwargs,
    ) -> List[ObjectDetectionInferenceResponse]:
        return predictions

    def caption(self, image_in: Union[Image.Image, np.array], **kwargs):
        if not isinstance(image_in, np.ndarray):
            np_img = np.array(image_in)  # RGB
            bgr_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        else:
            bgr_img = image_in
        return self._model.caption(bgr_img)[0]

    def query(self, image_in: Union[Image.Image, np.array], prompt="", **kwargs):
        if not isinstance(image_in, np.ndarray):
            np_img = np.array(image_in)  # RGB
            bgr_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        else:
            bgr_img = image_in
        return self._model.query(bgr_img, question=prompt)[0]

    def detect(
        self,
        image_in: Union[Image.Image, np.array],
        prompt: str = "",
        **kwargs,
    ):
        if not isinstance(image_in, np.ndarray):
            np_img = np.array(image_in)  # RGB
            bgr_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        else:
            bgr_img = image_in
        detections = self._model.detect(bgr_img, classes=[prompt])
        return self.make_response(detections, [bgr_img.shape[:2]], prompt=prompt)

    def make_response(
        self, predictions: List[Detections], image_sizes, prompt: str,
    ):
        responses = []

        for ind, image_detections in enumerate(predictions):
            predictions_for_image = []
            for instance_id in range(image_detections.xyxy.shape[0]):
                x_min, y_min, x_max, y_max = image_detections.xyxy[instance_id].tolist()
                width = x_max - x_min
                height = y_max - y_min
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                predictions_for_image.append(
                    ObjectDetectionPrediction(
                        # Passing args as a dictionary here since one of the args is 'class' (a protected term in Python)
                        **{
                            "x": center_x,
                            "y": center_y,
                            "width": width,
                            "height": height,
                            "confidence": 1.0,  # confidence is not returned by the model
                            "class": prompt if prompt is not None else "",
                            "class_id": 0,  # you can only prompt for one object at once
                        }
                    )
                )
            responses.append(
                ObjectDetectionInferenceResponse(
                    predictions=predictions_for_image,
                    image=InferenceResponseImage(
                        width=image_sizes[ind][1], height=image_sizes[ind][0]
                    ),
                )
            )
        return responses

    def postprocess(
        self,
        predictions: Tuple[str],
        preprocess_return_metadata: PreprocessReturnMetadata,
        **kwargs,
    ) -> List[ObjectDetectionInferenceResponse]:
        return predictions
