from typing import List, Optional, Tuple

from inference.core.entities.requests import ObjectDetectionInferenceRequest
from inference.core.entities.responses import (
    InferenceResponseImage,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
)
from inference.core.env import (
    ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
    ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
    API_KEY,
    MAX_DETECTIONS,
    OWLV2_VERSION_ID,
)
from inference.core.models.base import Model
from inference.core.models.inference_models_adapters import (
    get_extra_weights_provider_headers,
)
from inference.core.models.roboflow import DEFAULT_COLOR_PALETTE
from inference.core.utils.image_utils import load_image_bgr
from inference.core.utils.visualisation import draw_detection_predictions
from inference.models.owlv2.owlv2_inference_models import Owlv2AdapterSingleton
from inference_models import AnyModel, AutoModel, Detections
from inference_models.models.auto_loaders.access_manager import (
    LiberalModelAccessManager,
)
from inference_models.models.roboflow_instant.roboflow_instant_hf import (
    RoboflowInstantHF,
)


class RFInstantSpecificLiberalModelAccessManager(LiberalModelAccessManager):

    def retrieve_model_instance(
        self,
        model_id: str,
        package_id: Optional[str],
        api_key: Optional[str],
        loading_parameter_digest: Optional[str],
    ) -> Optional[AnyModel]:
        if (
            model_id != f"owlv2/{OWLV2_VERSION_ID}"
            and model_id != f"google/{OWLV2_VERSION_ID}"
        ):
            return None
        print(f"Intercepted call to dependent model init, {model_id}")
        return Owlv2AdapterSingleton(
            model_id,
            api_key=api_key,
        ).model


class InferenceModelsRFInstantModelAdapter(Model):
    def __init__(
        self,
        model_id: str,
        api_key: str = None,
        **kwargs,
    ):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY

        self.task_type = "object-detection"
        model_access_manager = RFInstantSpecificLiberalModelAccessManager()
        extra_weights_provider_headers = get_extra_weights_provider_headers()
        self._model: RoboflowInstantHF = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            model_access_manager=model_access_manager,
            extra_weights_provider_headers=extra_weights_provider_headers,
            **kwargs,
        )

    def infer(
        self,
        image,
        confidence: float = 0.95,
        iou_threshold: float = 0.3,
        max_detections: int = MAX_DETECTIONS,
        **kwargs,
    ):
        if isinstance(image, list):
            decoded_images = [load_image_bgr(i) for i in image]
        else:
            decoded_images = [load_image_bgr(image)]
        image_sizes: List[Tuple[int, int]] = [i.shape[:2][::-1] for i in decoded_images]  # type: ignore
        results = self._model.infer(
            images=decoded_images,
            confidence=confidence,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )
        return self.make_response(
            predictions=results,
            image_sizes=image_sizes,
            class_names=self._model.class_names,
        )

    def draw_predictions(
        self,
        inference_request,
        inference_response,
    ) -> bytes:
        """Draw predictions from an inference response onto the original image provided by an inference request

        Args:
            inference_request (ObjectDetectionInferenceRequest): The inference request containing the image on which to draw predictions
            inference_response (ObjectDetectionInferenceResponse): The inference response containing predictions to be drawn

        Returns:
            str: A base64 encoded image string
        """
        all_class_names = [x.class_name for x in inference_response.predictions]
        all_class_names = sorted(list(set(all_class_names)))
        return draw_detection_predictions(
            inference_request=inference_request,
            inference_response=inference_response,
            colors={
                class_name: DEFAULT_COLOR_PALETTE[i % len(DEFAULT_COLOR_PALETTE)]
                for (i, class_name) in enumerate(all_class_names)
            },
        )

    def make_response(
        self,
        predictions: List[Detections],
        image_sizes: List[Tuple[int, int]],
        class_names: List[str],
        **kwargs,
    ) -> List[ObjectDetectionInferenceResponse]:
        responses = []
        for image_prediction, image_size in zip(predictions, image_sizes):
            image_instances = []
            for instance_id in range(image_prediction.xyxy.shape[0]):
                x_min, y_min, x_max, y_max = image_prediction.xyxy[instance_id].tolist()
                width = x_max - x_min
                height = y_max - y_min
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                class_id = image_prediction.class_id[instance_id].item()
                confidence = image_prediction.confidence[instance_id].item()
                class_name = class_names[class_id]
                image_instances.append(
                    ObjectDetectionPrediction(
                        # Passing args as a dictionary here since one of the args is 'class' (a protected term in Python)
                        **{
                            "x": center_x,
                            "y": center_y,
                            "width": width,
                            "height": height,
                            "confidence": confidence,
                            "class": class_name,
                            "class_id": class_id,
                        }
                    )
                )
            responses.append(
                ObjectDetectionInferenceResponse(
                    predictions=image_instances,
                    image=InferenceResponseImage(
                        width=image_size[0], height=image_size[1]
                    ),
                )
            )
        return responses
