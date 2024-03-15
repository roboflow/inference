class ContentTypeInvalid(Exception):
    """Raised when the content type is invalid.

    Attributes:
        message (str): Optional message describing the error.
    """


class ContentTypeMissing(Exception):
    """Raised when the content type is missing.

    Attributes:
        message (str): Optional message describing the error.
    """


class EngineIgnitionFailure(Exception):
    """Raised when the engine fails to ignite.

    Attributes:
        message (str): Optional message describing the error.
    """


class InferenceModelNotFound(Exception):
    """Raised when the inference model is not found.

    Attributes:
        message (str): Optional message describing the error.
    """


class InvalidEnvironmentVariableError(Exception):
    """Raised when an environment variable is invalid.

    Attributes:
        message (str): Optional message describing the error.
    """


class InvalidMaskDecodeArgument(Exception):
    """Raised when an invalid argument is provided for mask decoding.

    Attributes:
        message (str): Optional message describing the error.
    """


class MissingApiKeyError(Exception):
    """Raised when the API key is missing.

    Attributes:
        message (str): Optional message describing the error.
    """


class MissingServiceSecretError(Exception):
    """Raised when the service secret is missing.

    Attributes:
        message (str): Optional message describing the error.
    """


class OnnxProviderNotAvailable(Exception):
    """Raised when the ONNX provider is not available.

    Attributes:
        message (str): Optional message describing the error.
    """


class WorkspaceLoadError(Exception):
    """Raised when there is an error loading the workspace.

    Attributes:
        message (str): Optional message describing the error.
    """


class InputImageLoadError(Exception):

    def __init__(self, message: str, public_message: str):
        super().__init__(message)
        self._public_message = public_message

    def get_public_error_details(self) -> str:
        return self._public_message


class InvalidNumpyInput(InputImageLoadError):
    """Raised when the input is an invalid NumPy array.

    Attributes:
        message (str): Optional message describing the error.
    """


class InvalidImageTypeDeclared(InputImageLoadError):
    pass


class InputFormatInferenceFailed(InputImageLoadError):
    pass


class PreProcessingError(Exception):
    pass


class PostProcessingError(Exception):
    pass


class InvalidModelIDError(Exception):
    pass


class MalformedRoboflowAPIResponseError(Exception):
    pass


class ServiceConfigurationError(Exception):
    pass


class MissingDefaultModelError(ServiceConfigurationError):
    pass


class ModelNotRecognisedError(ServiceConfigurationError):
    pass


class RoboflowAPIRequestError(Exception):
    pass


class RoboflowAPIUnsuccessfulRequestError(RoboflowAPIRequestError):
    pass


class RoboflowAPINotAuthorizedError(RoboflowAPIUnsuccessfulRequestError):
    pass


class RoboflowAPINotNotFoundError(RoboflowAPIUnsuccessfulRequestError):
    pass


class RoboflowAPIConnectionError(RoboflowAPIRequestError):
    pass


class RoboflowAPIImageUploadRejectionError(RoboflowAPIRequestError):
    pass


class RoboflowAPIIAnnotationRejectionError(RoboflowAPIRequestError):
    pass


class MalformedWorkflowResponseError(RoboflowAPIRequestError):
    pass


class RoboflowAPIIAlreadyAnnotatedError(RoboflowAPIIAnnotationRejectionError):
    pass


class ModelArtefactError(Exception):
    pass


class ActiveLearningError(Exception):
    pass


class PredictionFormatNotSupported(ActiveLearningError):
    pass


class ActiveLearningConfigurationDecodingError(ActiveLearningError):
    pass


class ActiveLearningConfigurationError(ActiveLearningError):
    pass


class CannotInitialiseModelError(Exception):
    pass
