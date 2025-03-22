from time import perf_counter
from typing import Any, List, Tuple, Union

import numpy as np

from inference.core import logger
from inference.core.entities.requests.inference import InferenceRequest
from inference.core.entities.responses.inference import InferenceResponse
from inference.core.models.types import PreprocessReturnMetadata
from inference.usage_tracking.collector import usage_collector


class BaseInference:
    """General inference class.

    This class provides a basic interface for inference tasks.
    """

    @usage_collector("model")
    def infer(self, image: Any, **kwargs) -> Any:
        """Runs inference on given data.
        - image:
            can be a BGR numpy array, filepath, InferenceRequestImage, PIL Image, byte-string, etc.
        """
        preproc_image, returned_metadata = self.preprocess(image, **kwargs)
        logger.debug(
            f"Preprocessed input shape: {getattr(preproc_image, 'shape', None)}"
        )
        predicted_arrays = self.predict(preproc_image, **kwargs)
        postprocessed = self.postprocess(predicted_arrays, returned_metadata, **kwargs)

        return postprocessed

    def preprocess(
        self, image: Any, **kwargs
    ) -> Tuple[np.ndarray, PreprocessReturnMetadata]:
        raise NotImplementedError

    def predict(self, img_in: np.ndarray, **kwargs) -> Tuple[np.ndarray, ...]:
        raise NotImplementedError

    def postprocess(
        self,
        predictions: Tuple[np.ndarray, ...],
        preprocess_return_metadata: PreprocessReturnMetadata,
        **kwargs,
    ) -> Any:
        raise NotImplementedError

    def infer_from_request(
        self, request: InferenceRequest
    ) -> Union[InferenceResponse, List[InferenceResponse]]:
        """Runs inference on a request

        Args:
            request (InferenceRequest): The request object.

        Returns:
            Union[CVInferenceResponse, List[CVInferenceResponse]]: The response object(s).

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """
        raise NotImplementedError

    def make_response(
        self, *args, **kwargs
    ) -> Union[InferenceResponse, List[InferenceResponse]]:
        """Constructs an object detection response.

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """
        raise NotImplementedError


class Model(BaseInference):
    """Base Inference Model (Inherits from BaseInference to define the needed methods)

    This class provides the foundational methods for inference and logging, and can be extended by specific models.

    Methods:
        log(m): Print the given message.
        clear_cache(): Clears any cache if necessary.
    """

    def log(self, m):
        """Prints the given message.

        Args:
            m (str): The message to print.
        """
        print(m)

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        """Clears any cache if necessary. This method should be implemented in derived classes as needed.

        Args:
            delete_from_disk (bool, optional): Whether to delete cached files from disk. Defaults to True.
        """
        pass

    def infer_from_request(
        self,
        request: InferenceRequest,
    ) -> Union[List[InferenceResponse], InferenceResponse]:
        """
        Perform inference based on the details provided in the request, and return the associated responses.
        The function can handle both single and multiple image inference requests. Optionally, it also provides
        a visualization of the predictions if requested.

        Args:
            request (InferenceRequest): The request object containing details for inference, such as the image or
                images to process, any classes to filter by, and whether or not to visualize the predictions.

        Returns:
            Union[List[InferenceResponse], InferenceResponse]: A list of response objects if the request contains
            multiple images, or a single response object if the request contains one image. Each response object
            contains details about the segmented instances, the time taken for inference, and optionally, a visualization.

        Examples:
            >>> request = InferenceRequest(image=my_image, visualize_predictions=True)
            >>> response = infer_from_request(request)
            >>> print(response.time)  # Prints the time taken for inference
            0.125
            >>> print(response.visualization)  # Accesses the visualization of the prediction if available

        Notes:
            - The processing time for each response is included within the response itself.
            - If `visualize_predictions` is set to True in the request, a visualization of the prediction
              is also included in the response.
        """
        t1 = perf_counter()
        responses = self.infer(**request.dict(), return_image_dims=False)
        for response in responses:
            response.time = perf_counter() - t1
            if request.id:
                response.inference_id = request.id

        if request.visualize_predictions:
            for response in responses:
                response.visualization = self.draw_predictions(request, response)

        if not isinstance(request.image, list) and len(responses) > 0:
            responses = responses[0]

        return responses

    def make_response(
        self, *args, **kwargs
    ) -> Union[InferenceResponse, List[InferenceResponse]]:
        """Makes an inference response from the given arguments.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            InferenceResponse: The inference response.
        """
        raise NotImplementedError(self.__class__.__name__ + ".make_response")
