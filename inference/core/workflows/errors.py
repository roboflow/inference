from typing import Optional


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


class WorkflowEnvironmentConfigurationError(WorkflowError):
    pass


class WorkflowCompilerError(WorkflowError):
    pass


class AssumptionError(WorkflowError):
    pass


class PluginLoadingError(WorkflowCompilerError):
    pass


class PluginInterfaceError(WorkflowCompilerError):
    pass


class BlockInterfaceError(WorkflowCompilerError):
    pass


class DynamicBlockError(WorkflowCompilerError):
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


class UnknownManifestType(WorkflowCompilerError):
    pass


class BlockInitParameterNotProvidedError(WorkflowCompilerError):
    pass


class StepInputDimensionalityError(WorkflowCompilerError):
    pass


class StepInputLineageError(WorkflowCompilerError):
    pass


class StepOutputLineageError(WorkflowCompilerError):
    pass


class ControlFlowDefinitionError(WorkflowCompilerError):
    pass


class WorkflowExecutionEngineError(WorkflowError):
    pass


class NotSupportedExecutionEngineError(WorkflowExecutionEngineError):
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


class WorkflowExecutionEngineVersionError(WorkflowError):
    pass
