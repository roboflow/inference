from inference.core.interfaces.stream.warnings import (
    InferenceExperimentalFeatureWarning,
)


class InferenceDeprecationWarning(Warning):
    pass


class InferenceConfigurationWarning(Warning):
    pass


class ModelDependencyMissing(Warning):
    pass


class InferenceModelsStackMissing(Warning):
    pass
