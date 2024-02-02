class WorkflowsCompilerError(Exception):
    pass


class ValidationError(WorkflowsCompilerError):
    pass


class InvalidSpecificationVersionError(ValidationError):
    pass


class DuplicatedSymbolError(ValidationError):
    pass


class InvalidReferenceError(ValidationError):
    pass


class ExecutionGraphError(WorkflowsCompilerError):
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


class WorkflowsCompilerRuntimeError(WorkflowsCompilerError):
    pass


class RuntimePayloadError(WorkflowsCompilerRuntimeError):
    pass


class RuntimeParameterMissingError(RuntimePayloadError):
    pass


class VariableTypeError(RuntimePayloadError):
    pass


class ExecutionEngineError(WorkflowsCompilerRuntimeError):
    pass
