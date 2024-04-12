from typing import Optional


class WorkflowsCompilerError(Exception):
    # Message of error must be prepared to be revealed in any API response.

    def get_public_message(self) -> str:
        return str(self)


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


###### NEW COMPILER ERRORS ######


class WorkflowError(Exception):

    def __init__(
        self,
        public_message: str,
        context: str,
        inner_error: Optional[Exception] = None,
    ):
        super().__init__(public_message)
        self._public_message = public_message
        self._context = context
        self._inner_error = inner_error

    @property
    def public_message(self) -> str:
        return self._public_message

    @property
    def context(self) -> str:
        return self._context

    @property
    def inner_error_type(self) -> Optional[str]:
        if self._inner_error is None:
            return None
        return self._inner_error.__class__.__name__

    @property
    def inner_error(self) -> Optional[Exception]:
        return self._inner_error


class WorkflowCompilerError(WorkflowError):
    pass


class PluginLoadingError(WorkflowCompilerError):
    pass


class PluginInterfaceError(WorkflowCompilerError):
    pass


class BlockInterfaceError(WorkflowCompilerError):
    pass


class WorkflowDefinitionError(WorkflowCompilerError):
    pass


class WorkflowSyntaxError(WorkflowDefinitionError):
    pass


class DuplicatedNameError(WorkflowDefinitionError):
    pass


class ExecutionGraphStructureError(WorkflowCompilerError):
    pass


class ReferenceTypeError(WorkflowCompilerError):
    pass


class InvalidReferenceTargetError(WorkflowCompilerError):
    pass


class DanglingExecutionBranchError(ExecutionGraphStructureError):
    pass


class ConditionalBranchesClashError(ExecutionGraphStructureError):
    pass


class UnknownManifestType(WorkflowCompilerError):
    pass


class BlockInitParameterNotProvidedError(WorkflowCompilerError):
    pass


class WorkflowExecutionEngineError(WorkflowError):
    pass


class InvalidBlockBehaviourError(WorkflowExecutionEngineError):
    pass


class StepExecutionError(WorkflowExecutionEngineError):
    pass


class ExecutionEngineRuntimeError(WorkflowExecutionEngineError):
    pass


class ExecutionEngineNotImplementedError(WorkflowExecutionEngineError):
    pass


class RuntimeInputError(WorkflowExecutionEngineError):
    pass
