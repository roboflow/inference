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


class RuntimeIntrospectionError(BaseInferenceError):
    pass


class JetsonTypeResolutionError(RuntimeIntrospectionError):
    pass
