from typing import Optional


class BaseInferenceModelsError(Exception):

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


class AssumptionError(BaseInferenceModelsError):
    pass


class EnvironmentConfigurationError(BaseInferenceModelsError):
    pass


class ModelRuntimeError(BaseInferenceModelsError):
    pass


class ModelInputError(BaseInferenceModelsError):
    pass


class RetryError(BaseInferenceModelsError):
    pass


class ModelRetrievalError(BaseInferenceModelsError):
    pass


class UntrustedFileError(BaseInferenceModelsError):
    pass


class FileHashSumMissmatch(BaseInferenceModelsError):
    pass


class ModelNotFoundError(ModelRetrievalError):
    pass


class UnauthorizedModelAccessError(ModelRetrievalError):
    pass


class ModelMetadataConsistencyError(ModelRetrievalError):
    pass


class ModelMetadataHandlerNotImplementedError(ModelRetrievalError):
    pass


class InvalidEnvVariable(BaseInferenceModelsError):
    pass


class ModelPackageNegotiationError(BaseInferenceModelsError):
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


class ModelLoadingError(BaseInferenceModelsError):
    pass


class ModelPackageAlternativesExhaustedError(ModelLoadingError):
    pass


class MissingModelInitParameterError(ModelLoadingError):
    pass


class InvalidModelInitParameterError(ModelLoadingError):
    pass


class InsecureModelIdentifierError(ModelLoadingError):
    pass


class DirectLocalStorageAccessError(ModelLoadingError):
    pass


class ForbiddenLocalCodePackageAccessError(ModelLoadingError):
    pass


class ModelImplementationNotFoundError(ModelLoadingError):
    pass


class CorruptedModelPackageError(ModelLoadingError):
    pass


class MissingDependencyError(BaseInferenceModelsError):
    pass


class InvalidParameterError(BaseInferenceModelsError):
    pass


class DependencyModelParametersValidationError(ModelLoadingError):
    pass


class ModelPipelineInitializationError(ModelLoadingError):
    pass


class ModelPipelineNotFound(ModelPipelineInitializationError):
    pass
