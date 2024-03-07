class CLIError(Exception):
    pass


class DockerConnectionErrorException(CLIError):
    pass


class DatasetLoadingError(CLIError):
    pass


class InferencePackageMissingError(CLIError):
    pass
