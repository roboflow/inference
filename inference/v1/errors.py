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
