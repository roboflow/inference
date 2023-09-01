from typing import Any, List, Union

from inference.core.data_models import (
    ClassificationInferenceRequest,
    ClassificationInferenceResponse,
    InferenceRequest,
    InferenceResponse,
    InstanceSegmentationInferenceRequest,
    InstanceSegmentationInferenceResponse,
    MultiLabelClassificationInferenceResponse,
    ObjectDetectionInferenceRequest,
    ObjectDetectionInferenceResponse,
)


class InferenceMixin:
    """General inference mixin class.

    This mixin provides a basic interface for inference tasks.
    """

    def infer_from_request(self, request: InferenceRequest) -> InferenceResponse:
        """Runs inference on a given request.

        Args:
            request (InferenceRequest): The inference request object.

        Returns:
            InferenceResponse: The response object.

        """
        raise NotImplementedError

    def infer(self, *args, **kwargs) -> Any:
        """Runs inference on given data."""
        raise NotImplementedError


class ObjectDetectionMixin:
    """Object detection inference mixin class.

    This mixin provides the interface for object detection inference tasks.
    """

    task_type = "object-detection"

    def infer(self, *args, **kwargs) -> Any:
        """Runs inference on given data.

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """
        raise NotImplementedError

    def infer_from_request(
        self, request: ObjectDetectionInferenceRequest
    ) -> Union[
        ObjectDetectionInferenceResponse, List[ObjectDetectionInferenceResponse]
    ]:
        """Runs inference on an object detection request.

        Args:
            request (ObjectDetectionInferenceRequest): The request object.

        Returns:
            Union[ObjectDetectionInferenceResponse, List[ObjectDetectionInferenceResponse]]: The response object(s).

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """
        raise NotImplementedError

    def make_response(
        self, *args, **kwargs
    ) -> Union[
        ObjectDetectionInferenceResponse, List[ObjectDetectionInferenceResponse]
    ]:
        """Constructs an object detection response.

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """
        raise NotImplementedError

    def postprocess(self, *args, **kwargs) -> Any:
        """Postprocesses the object detection inference results.

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """
        raise NotImplementedError

    def predict(self, *args, **kwargs) -> Any:
        """Runs the prediction for object detection.

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """
        raise NotImplementedError

    def preprocess(self, request: ObjectDetectionInferenceRequest) -> Any:
        """Preprocesses an object detection inference request.

        Args:
            request (ObjectDetectionInferenceRequest): The request object.

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """
        raise NotImplementedError


class InstanceSegmentationMixin:
    """Instance segmentation inference mixin class.

    This mixin provides the interface for instance segmentation inference tasks.
    """

    task_type = "instance-segmentation"

    def infer(
        self, request: InstanceSegmentationInferenceRequest
    ) -> InstanceSegmentationInferenceResponse:
        """Runs inference on an instance segmentation request.

        Args:
            request (InstanceSegmentationInferenceRequest): The request object.

        Returns:
            InstanceSegmentationInferenceResponse: The response object.

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """
        raise NotImplementedError


class ClassificationMixin:
    """Classification inference mixin class.

    This mixin provides the interface for classification inference tasks.
    """

    task_type = "classification"

    def infer(
        self, request: ClassificationInferenceRequest
    ) -> Union[
        ClassificationInferenceResponse, MultiLabelClassificationInferenceResponse
    ]:
        """Runs inference on a classification request.

        Args:
            request (ClassificationInferenceRequest): The request object.

        Returns:
            Union[ClassificationInferenceResponse, MultiLabelClassificationInferenceResponse]: The response object.

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """
        raise NotImplementedError
