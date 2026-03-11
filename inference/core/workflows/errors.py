from typing import List, Optional

from pydantic import BaseModel, Field


class BlockTraceback(BaseModel):
    traceback: Optional[str] = None
    error_line: Optional[int] = None
    code_snippet: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None


class WorkflowBlockError(BaseModel):
    block_id: Optional[str] = None
    block_type: Optional[str] = None
    block_details: Optional[str] = None
    property_name: Optional[str] = None
    property_details: Optional[str] = None
    block_traceback: Optional[BlockTraceback] = None


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
    def __init__(
        self,
        blocks_errors: Optional[List[WorkflowBlockError]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.blocks_errors = blocks_errors


class DuplicatedNameError(WorkflowDefinitionError):
    pass


class ExecutionGraphStructureError(WorkflowCompilerError):
    def __init__(
        self,
        *args,
        blocks_errors: Optional[List[WorkflowBlockError]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.blocks_errors = blocks_errors


class ReferenceTypeError(WorkflowCompilerError):
    pass


class InvalidReferenceTargetError(WorkflowCompilerError):
    def __init__(
        self,
        *args,
        blocks_errors: Optional[List[WorkflowBlockError]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.blocks_errors = blocks_errors


class UnknownManifestType(WorkflowCompilerError):
    pass


class BlockInitParameterNotProvidedError(WorkflowCompilerError):
    pass


class StepInputDimensionalityError(WorkflowCompilerError):

    def __init__(
        self,
        *args,
        blocks_errors: Optional[List[WorkflowBlockError]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.blocks_errors = blocks_errors


class StepInputLineageError(WorkflowCompilerError):
    pass


class StepOutputLineageError(WorkflowCompilerError):
    pass


class ControlFlowDefinitionError(WorkflowCompilerError):
    pass


class WorkflowExecutionEngineError(WorkflowError):
    pass


class DynamicBlockCodeError(WorkflowExecutionEngineError):
    """Exception for dynamic block code execution errors (errors provoked by user's code)."""

    def __init__(
        self,
        public_message: str,
        context: str = "dynamic_block_code_execution",
        inner_error: Optional[Exception] = None,
        block_type_name: Optional[str] = None,
        error_line: Optional[int] = None,
        code_snippet: Optional[str] = None,
        traceback_str: Optional[str] = None,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
    ):
        super().__init__(
            public_message=public_message, context=context, inner_error=inner_error
        )
        self.block_type_name = block_type_name
        self.error_line = error_line
        self.code_snippet = code_snippet
        self.traceback_str = traceback_str
        self.stdout = stdout
        self.stderr = stderr


class NotSupportedExecutionEngineError(WorkflowExecutionEngineError):
    pass


class InvalidBlockBehaviourError(WorkflowExecutionEngineError):
    pass


class StepExecutionError(WorkflowExecutionEngineError):
    def __init__(
        self,
        block_id: str,
        block_type: str,
        block_traceback: Optional[BlockTraceback] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.block_id = block_id
        self.block_type = block_type
        self.block_traceback = block_traceback


class ClientCausedStepExecutionError(WorkflowExecutionEngineError):
    def __init__(
        self,
        block_id: str,
        status_code: int,
        public_message: str,
        context: str,
        inner_error: Optional[Exception] = None,
    ):
        super().__init__(
            public_message=public_message, context=context, inner_error=inner_error
        )
        self.block_id = block_id
        self.status_code = status_code


class ExecutionEngineRuntimeError(WorkflowExecutionEngineError):
    pass


class ExecutionEngineNotImplementedError(WorkflowExecutionEngineError):
    pass


class RuntimeInputError(WorkflowExecutionEngineError):
    pass


class WorkflowExecutionEngineVersionError(WorkflowError):
    pass
