from typing import Any, List, Tuple

import numpy as np

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
    OWLV2_IMAGE_CACHE_SIZE,
    OWLV2_MODEL_CACHE_SIZE,
)
from inference.core.models.base import Model
from inference.core.models.roboflow import DEFAULT_COLOR_PALETTE
from inference.core.utils.image_utils import load_image_bgr
from inference.core.utils.visualisation import draw_detection_predictions
from inference_models import AutoModel, Detections
from inference_models.models.owlv2.cache import (
    InMemoryOwlV2ClassEmbeddingsCache,
    InMemoryOwlV2ImageEmbeddingsCache,
)
from inference_models.models.owlv2.entities import (
    ReferenceBoundingBox,
    ReferenceExample,
)
from inference_models.models.owlv2.owlv2_hf import OWLv2HF


class InferenceModelsOwlV2Adapter(Model):
    def __init__(
        self,
        model_id: str = "owlv2/owlv2-large-patch14-ensemble",
        api_key: str = None,
        **kwargs,
    ):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY

        self.task_type = "object-detection"
        owlv2_class_embeddings_cache = InMemoryOwlV2ClassEmbeddingsCache.init(
            size_limit=OWLV2_MODEL_CACHE_SIZE,
            send_to_cpu=True,
        )
        owlv2_images_embeddings_cache = InMemoryOwlV2ImageEmbeddingsCache.init(
            size_limit=OWLV2_IMAGE_CACHE_SIZE,
            send_to_cpu=True,
        )
        self._model: OWLv2HF = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            owlv2_class_embeddings_cache=owlv2_class_embeddings_cache,
            owlv2_images_embeddings_cache=owlv2_images_embeddings_cache,
            **kwargs,
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

    def embed_image(self, image: np.ndarray) -> Tuple[str, tuple]:
        image_embeddings = self._model.embed_image(image=image)
        image_embeds = (
            image_embeddings.objectness,
            image_embeddings.boxes,
            image_embeddings.image_class_embeddings,
            image_embeddings.logit_shift,
            image_embeddings.logit_scale,
        )
        return image_embeddings.image_hash, image_embeds

    def infer(
        self,
        image: Any,
        training_data: List[dict],
        confidence: float = 0.99,
        iou_threshold: float = 0.3,
        max_detections: int = MAX_DETECTIONS,
        **kwargs,
    ):
        reference_examples = []
        for example in training_data:
            boxes = []
            for box in example["boxes"]:
                boxes.append(ReferenceBoundingBox.model_validate(box))
            reference_examples.append(
                ReferenceExample(
                    image=example["image"]["value"],
                    boxes=boxes,
                )
            )
        if isinstance(image, list):
            image_decoded = [load_image_bgr(i) for i in image]
        else:
            image_decoded = [load_image_bgr(image)]
        image_sizes: List[Tuple[int, int]] = [i.shape[:2][::-1] for i in image_decoded]  # type: ignore
        results = self._model.infer_with_reference_examples(
            images=image_decoded,
            reference_examples=reference_examples,
            confidence=confidence,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )
        return self.make_response(predictions=results, image_sizes=image_sizes)

    def make_response(
        self,
        predictions: List[Detections],
        image_sizes: List[Tuple[int, int]],
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
                class_name = image_prediction.image_metadata["class_names"][class_id]
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

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        """Clears any cache if necessary. TODO: Implement this to delete the cache from the experimental model.

        Args:
            delete_from_disk (bool, optional): Whether to delete cached files from disk. Defaults to True.
        """
        pass
