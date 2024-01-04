class DeploymentCompilerError(Exception):
    pass


class ValidationError(DeploymentCompilerError):
    pass


class DuplicatedSymbolError(ValidationError):
    pass


class InvalidReferenceError(ValidationError):
    pass


class VariableNotBounderError(ValidationError):
    pass


class ExecutionGraphError(DeploymentCompilerError):
    pass


class NotAcyclicGraphError(ExecutionGraphError):
    pass


class NodesNotReachingOutputError(ExecutionGraphError):
    pass


class AmbiguousPathDetected(ExecutionGraphError):
    pass


class DeploymentCompilerRuntimeError(DeploymentCompilerError):
    pass


class RuntimeParameterMissingError(DeploymentCompilerRuntimeError):
    pass
