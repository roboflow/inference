from inference_cli.lib.exceptions import CLIError


class RemoteAPICallError(CLIError):
    pass


class RetryError(RemoteAPICallError):
    pass


class RequestError(RemoteAPICallError):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class InferenceCompilerError(CLIError):
    pass


class RuntimeConfigurationError(CLIError):
    pass


class LackOfSourcePackageError(CLIError):
    pass


class PackageNegotiationError(CLIError):
    pass


class PackageDownloadError(CLIError):
    pass


class CorruptedPackageError(CLIError):
    pass


class TRTCompilerError(CLIError):
    pass


class QuantizationNotSupportedError(TRTCompilerError):
    pass


class InvalidNetworkInputsError(TRTCompilerError):
    pass


class NetworkParsingError(TRTCompilerError):
    pass


class TRTModelCompilationError(TRTCompilerError):
    pass


class AlreadyCompiledError(CLIError):
    pass


class CompiledPackageRegistrationError(CLIError):
    pass


class ModelArchitectureNotSupportedError(CLIError):
    pass


class ModelVerificationError(CLIError):
    pass


class ModelLoadingError(ModelVerificationError):
    pass


class ModelInferenceError(ModelVerificationError):
    pass
