from inference.core.workflows.errors import WorkflowExecutionEngineError


class RoboflowQueryLanguageError(WorkflowExecutionEngineError):
    pass


class UndeclaredSymbolError(RoboflowQueryLanguageError):
    pass


class InvalidInputTypeError(RoboflowQueryLanguageError):
    pass


class OperationTypeNotRecognisedError(RoboflowQueryLanguageError):
    pass


class OperationError(RoboflowQueryLanguageError):
    pass


class EvaluationEngineError(RoboflowQueryLanguageError):
    pass
