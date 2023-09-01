from time import perf_counter
from typing import List, Union

from inference.core.data_models import InferenceRequest, InferenceResponse
from inference.core.models.mixins import InferenceMixin


class Model(InferenceMixin):
    """Base Inference Model (Inherits from InferenceMixin to define the needed methods)

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

    def clear_cache(self):
        """Clears any cache if necessary. This method should be implemented in derived classes as needed."""
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
        predictions_data = self.infer(**request.dict(), return_image_dims=True)
        responses = self.make_response(
            *predictions_data, class_filter=request.class_filter
        )
        for response in responses:
            response.time = perf_counter() - t1

        if request.visualize_predictions:
            for response in responses:
                response.visualization = self.draw_predictions(request, response)

        if not isinstance(request.image, list):
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
