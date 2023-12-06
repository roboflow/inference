from abc import abstractmethod
from time import perf_counter
from typing import Any, List, Tuple, Union

import numpy as np

from inference.core.cache.model_artifacts import clear_cache, initialise_cache
from inference.core.entities.requests.inference import InferenceRequest
from inference.core.entities.responses.inference import InferenceResponse, StubResponse
from inference.core.models.base import Model
from inference.core.models.types import PreprocessReturnMetadata
from inference.core.utils.image_utils import np_image_to_base64


class ModelStub(Model):
    def __init__(self, model_id: str, api_key: str):
        super().__init__()
        self.model_id = model_id
        self.api_key = api_key
        self.dataset_id, self.version_id = model_id.split("/")
        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}
        initialise_cache(model_id=model_id)

    def infer_from_request(
        self, request: InferenceRequest
    ) -> Union[InferenceResponse, List[InferenceResponse]]:
        t1 = perf_counter()
        stub_prediction = self.infer(**request.dict())
        response = self.make_response(request=request, prediction=stub_prediction)
        response.time = perf_counter() - t1
        return response

    def infer(self, *args, **kwargs) -> Any:
        _ = self.preprocess()
        dummy_prediction = self.predict()
        return self.postprocess(dummy_prediction)

    def preprocess(
        self, *args, **kwargs
    ) -> Tuple[np.ndarray, PreprocessReturnMetadata]:
        return np.zeros((128, 128, 3), dtype=np.uint8), {}  # type: ignore

    def predict(self, *args, **kwargs) -> Tuple[np.ndarray, ...]:
        return (np.zeros((1, 8)),)

    def postprocess(self, predictions: Tuple[np.ndarray, ...], *args, **kwargs) -> Any:
        return {
            "is_stub": True,
            "model_id": self.model_id,
        }

    def clear_cache(self) -> None:
        clear_cache(model_id=self.model_id)

    @abstractmethod
    def make_response(
        self, request: InferenceRequest, prediction: dict, **kwargs
    ) -> Union[InferenceResponse, List[InferenceResponse]]:
        pass


class ClassificationModelStub(ModelStub):
    task_type = "classification"

    def make_response(
        self, request: InferenceRequest, prediction: dict, **kwargs
    ) -> Union[InferenceResponse, List[InferenceResponse]]:
        stub_visualisation = None
        if getattr(request, "visualize_predictions", False):
            stub_visualisation = np_image_to_base64(
                np.zeros((128, 128, 3), dtype=np.uint8)
            )
        return StubResponse(
            is_stub=prediction["is_stub"],
            model_id=prediction["model_id"],
            task_type=self.task_type,
            visualization=stub_visualisation,
        )


class ObjectDetectionModelStub(ModelStub):
    task_type = "object-detection"

    def make_response(
        self, request: InferenceRequest, prediction: dict, **kwargs
    ) -> Union[InferenceResponse, List[InferenceResponse]]:
        stub_visualisation = None
        if getattr(request, "visualize_predictions", False):
            stub_visualisation = np_image_to_base64(
                np.zeros((128, 128, 3), dtype=np.uint8)
            )
        return StubResponse(
            is_stub=prediction["is_stub"],
            model_id=prediction["model_id"],
            task_type=self.task_type,
            visualization=stub_visualisation,
        )


class InstanceSegmentationModelStub(ModelStub):
    task_type = "instance-segmentation"

    def make_response(
        self, request: InferenceRequest, prediction: dict, **kwargs
    ) -> Union[InferenceResponse, List[InferenceResponse]]:
        stub_visualisation = None
        if getattr(request, "visualize_predictions", False):
            stub_visualisation = np_image_to_base64(
                np.zeros((128, 128, 3), dtype=np.uint8)
            )
        return StubResponse(
            is_stub=prediction["is_stub"],
            model_id=prediction["model_id"],
            task_type=self.task_type,
            visualization=stub_visualisation,
        )


class KeypointsDetectionModelStub(ModelStub):
    task_type = "keypoint-detection"

    def make_response(
        self, request: InferenceRequest, prediction: dict, **kwargs
    ) -> Union[InferenceResponse, List[InferenceResponse]]:
        stub_visualisation = None
        if getattr(request, "visualize_predictions", False):
            stub_visualisation = np_image_to_base64(
                np.zeros((128, 128, 3), dtype=np.uint8)
            )
        return StubResponse(
            is_stub=prediction["is_stub"],
            model_id=prediction["model_id"],
            task_type=self.task_type,
            visualization=stub_visualisation,
        )
