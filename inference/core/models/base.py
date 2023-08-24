from time import perf_counter
from typing import List, Union

from inference.core.data_models import (
    InferenceRequest,
    InferenceResponse,
)
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
