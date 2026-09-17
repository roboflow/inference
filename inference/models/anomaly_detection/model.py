from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

from inference.core.entities.responses.inference import (
    AnomalyDetectionResponse,
    ClassificationPrediction,
    InferenceResponseImage,
)
from inference.core.models.roboflow import RoboflowInferenceModel
from inference.core.utils.image_utils import load_image_rgb
from inference.usage_tracking.collector import usage_collector

CLASS_NAMES = ("normal", "anomalous")


def make_anomaly_response(result, width, height, elapsed):
    top_index = int(result["is_anomalous"])
    confidence = result["anomalous_confidence"]
    # The runtime maps the saved threshold to exactly 0.5 (sigmoid of the
    # scaled score margin), so the decision and the argmax of the two class
    # confidences must agree; a disagreement means a broken artifact.
    if (confidence >= 0.5) != bool(result["is_anomalous"]):
        raise ValueError(
            "Anomaly decision disagrees with the anomalous confidence: "
            f"is_anomalous={result['is_anomalous']} confidence={confidence}"
        )
    scores = [1 - confidence, confidence]
    predictions = [
        ClassificationPrediction(
            **{
                "class": CLASS_NAMES[index],
                "class_id": index,
                "confidence": scores[index],
            }
        )
        for index in (top_index, 1 - top_index)
    ]
    return AnomalyDetectionResponse(
        predictions=predictions,
        top=CLASS_NAMES[top_index],
        confidence=scores[top_index],
        image=InferenceResponseImage(width=width, height=height),
        time=elapsed,
        anomaly_score=result["anomaly_score"],
        anomaly_threshold=result["anomaly_threshold"],
        is_anomalous=result["is_anomalous"],
        anomaly_map=result.get("anomaly_map"),
    )


class AnomalyDetectionModel(RoboflowInferenceModel):
    task_type = "classification"
    architecture = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            from roboflow_anomaly import AnomalyDetector
        except ImportError as error:
            raise ImportError(
                "FoundAD and PatchCore require the roboflow-anomaly runtime wheel. "
                "Install it with pip install /path/to/roboflow_anomaly-0.1.0-py3-none-any.whl."
            ) from error
        import torch

        self.get_model_artifacts(**kwargs)
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.detector = AnomalyDetector.load(
            Path(self.cache_file(self.weights_file)), device=device
        )
        if self.detector.config.architecture != self.architecture:
            raise ValueError(
                "Anomaly artifact architecture does not match the model registry"
            )
        self.class_names = list(CLASS_NAMES)
        self.num_classes = 2

    @property
    def weights_file(self):
        return "best.pt"

    def get_infer_bucket_file_list(self):
        return ["environment.json"]

    @usage_collector("model")
    def infer(
        self, image: Any, include_anomaly_map=False, return_image_dims=False, **kwargs
    ):
        started = perf_counter()
        images = image if isinstance(image, list) else [image]
        pil_images = [Image.fromarray(load_image_rgb(item)) for item in images]
        results = self.detector.predict(pil_images, include_maps=include_anomaly_map)
        responses = [
            make_anomaly_response(
                result, pil.width, pil.height, perf_counter() - started
            )
            for result, pil in zip(results, pil_images)
        ]
        if return_image_dims:
            return responses, [(pil.height, pil.width) for pil in pil_images]
        return responses

    def draw_predictions(self, inference_request, inference_response):
        from inference.core.models.classification_base import (
            ClassificationBaseOnnxRoboflowInferenceModel,
        )

        return ClassificationBaseOnnxRoboflowInferenceModel.draw_predictions(
            self, inference_request, inference_response
        )


class PatchCoreAnomalyDetection(AnomalyDetectionModel):
    architecture = "patchcore"


class FoundADAnomalyDetection(AnomalyDetectionModel):
    architecture = "foundad"
