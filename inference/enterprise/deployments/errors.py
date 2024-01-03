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
