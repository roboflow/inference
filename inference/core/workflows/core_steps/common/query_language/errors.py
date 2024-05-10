from inference.core.workflows.errors import WorkflowExecutionEngineError


class RoboflowQueryLanguageError(WorkflowExecutionEngineError):
    pass


class InvalidInputTypeError(RoboflowQueryLanguageError):
    pass

