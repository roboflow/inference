from typing import Optional


class BaseInferenceError(Exception):

    def __init__(self, message: str, help_url: Optional[str] = None):
        super().__init__(message)
        self._help_url = help_url

    @property
    def help_url(self) -> Optional[str]:
        return self._help_url

    def __str__(self) -> str:
        if self._help_url is None:
            return super().__str__()
        return f"{super().__str__()} - VISIT {self._help_url} FOR FURTHER SUPPORT"


class AssumptionError(BaseInferenceError):
    pass


class EnvironmentConfigurationError(BaseInferenceError):
    pass


class ModelRuntimeError(BaseInferenceError):
    pass


class RetryError(BaseInferenceError):
    pass


class ModelRetrievalError(BaseInferenceError):
    pass


class UntrustedFileError(BaseInferenceError):
    pass


class FileHashSumMissmatch(BaseInferenceError):
    pass


class UnauthorizedModelAccessError(ModelRetrievalError):
    pass


class ModelMetadataConsistencyError(ModelRetrievalError):
    pass


class ModelMetadataHandlerNotImplementedError(ModelRetrievalError):
    pass


class InvalidEnvVariable(BaseInferenceError):
    pass


class ModelPackageNegotiationError(BaseInferenceError):
    pass


class UnknownBackendTypeError(ModelPackageNegotiationError):
    pass


class UnknownQuantizationError(ModelPackageNegotiationError):
    pass


class InvalidRequestedBatchSizeError(ModelPackageNegotiationError):
    pass


class RuntimeIntrospectionError(ModelPackageNegotiationError):
    pass


class JetsonTypeResolutionError(RuntimeIntrospectionError):
    pass


class NoModelPackagesAvailableError(ModelPackageNegotiationError):
    pass


class AmbiguousModelPackageResolutionError(ModelPackageNegotiationError):
    pass


class ModelLoadingError(BaseInferenceError):
    pass


class InsecureModelIdentifierError(ModelLoadingError):
    pass


class DirectLocalStorageAccessError(ModelLoadingError):
    pass


class ModelImplementationLoaderError(ModelLoadingError):
    pass


class CorruptedModelPackageError(ModelLoadingError):
    pass


class MissingDependencyError(BaseInferenceError):
    pass


class InvalidParameterError(BaseInferenceError):
    pass


class DependencyModelParametersValidationError(ModelLoadingError):
    pass


class ModelPipelineInitializationError(ModelLoadingError):
    pass


class ModelPipelineNotFound(ModelPipelineInitializationError):
    pass
