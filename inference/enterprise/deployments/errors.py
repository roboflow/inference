class DeploymentCompilerError(Exception):
    pass


class ValidationError(DeploymentCompilerError):
    pass


class InvalidSpecificationVersionError(ValidationError):
    pass


class DuplicatedSymbolError(ValidationError):
    pass


class InvalidReferenceError(ValidationError):
    pass


class ExecutionGraphError(DeploymentCompilerError):
    pass


class SelectorToUndefinedNodeError(ExecutionGraphError):
    pass


class NotAcyclicGraphError(ExecutionGraphError):
    pass


class NodesNotReachingOutputError(ExecutionGraphError):
    pass


class AmbiguousPathDetected(ExecutionGraphError):
    pass


class InvalidStepInputDetected(ExecutionGraphError):
    pass


class DeploymentCompilerRuntimeError(DeploymentCompilerError):
    pass


class RuntimeParameterMissingError(DeploymentCompilerRuntimeError):
    pass


class VariableTypeError(DeploymentCompilerRuntimeError):
    pass
