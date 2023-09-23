from typing import Any, List, Union, TYPE_CHECKING, Tuple, Dict

from inference.core.data_models import (
    InferenceRequest,
    InferenceResponse,
    ObjectDetectionInferenceRequest,
    ObjectDetectionInferenceResponse,
)
from inference.core.models.types import PreprocessReturnMetadata

if TYPE_CHECKING:
    import numpy as np

class InferenceMixin:
    """General inference mixin class.

    This mixin provides a basic interface for inference tasks.
    """


    def infer(self, image: Any, **kwargs) -> Any:
        """Runs inference on given data."""
        preproc_image, returned_metadata = self.preprocess(image, **kwargs)
        predicted_arrays = self.predict(preproc_image, **kwargs)
        postprocessed = self.postprocess(predicted_arrays, returned_metadata, **kwargs)

        return postprocessed

    
    def preprocess(self, image: Any, **kwargs) -> Tuple[np.ndarray, PreprocessReturnMetadata]:
        raise NotImplementedError

    def predict(self, img_in: np.ndarray, **kwargs) -> Tuple[np.ndarray, ...]:
        raise NotImplementedError


    def postprocess(
        self,
        predictions: Tuple[np.ndarray, ...],
        preprocess_return_metadata: PreprocessReturnMetadata,
        **kwargs
    ) -> Any:
        raise NotImplementedError

    def infer_from_request(
        self, request: InferenceRequest
    ) -> Union[
        InferenceResponse, List[InferenceResponse]
    ]:
        """Runs inference on a request

        Args:
            request (CVInferenceRequest): The request object.

        Returns:
            Union[CVInferenceResponse, List[CVInferenceResponse]]: The response object(s).

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """
        raise NotImplementedError


    def make_response(
        self, *args, **kwargs
    ) -> Union[
        InferenceResponse, List[InferenceResponse]
    ]:
        """Constructs an object detection response.

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """
        raise NotImplementedError
