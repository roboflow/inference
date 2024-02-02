class CLIError(Exception):
    pass


class DockerConnectionErrorException(CLIError):
    pass


class DatasetLoadingError(CLIError):
    pass
