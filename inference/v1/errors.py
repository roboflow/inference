class BaseInferenceError(Exception):
    pass


class CorruptedModelPackageError(BaseInferenceError):
    pass


class EnvironmentConfigurationError(BaseInferenceError):
    pass


class ModelRuntimeError(BaseInferenceError):
    pass


class RetryError(BaseInferenceError):
    pass


class ModelRetrievalError(BaseInferenceError):
    pass


class UnauthorizedModelAccessError(ModelRetrievalError):
    pass


class ModelMetadataConsistencyError(ModelRetrievalError):
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
