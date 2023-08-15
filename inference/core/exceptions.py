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


class DatasetLoadError(Exception):
    """Raised when there is an error loading the dataset.

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


class InputMethodNotAllowed(Exception):
    """Raised when the input method is not allowed.

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


class InvalidNumpyInput(Exception):
    """Raised when the input is an invalid NumPy array.

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


class ModelArtifactsRetrievalError(Exception):
    """Raised when there is an error retrieving the model artifacts.

    Attributes:
        message (str): Optional message describing the error.
    """


class ModelCompilationFailure(Exception):
    """Raised when the model fails to compile.

    Attributes:
        message (str): Optional message describing the error.
    """


class OnnxProviderNotAvailable(Exception):
    """Raised when the ONNX provider is not available.

    Attributes:
        message (str): Optional message describing the error.
    """


class TensorrtRoboflowAPIError(Exception):
    """Raised when there is an error with the TensorRT Roboflow API.

    Attributes:
        message (str): Optional message describing the error.
    """


class WorkspaceLoadError(Exception):
    """Raised when there is an error loading the workspace.

    Attributes:
        message (str): Optional message describing the error.
    """
