"""Error types shared by the V2 execution engine.

The V2 runtime distinguishes three failure families:

* ``ContractError``: a data or block contract was violated. Raised by the data
  API (malformed batches, layouts or metadata), by the registry (duplicate or
  unknown names, failed kind validation) and by block contracts.
* ``WorkflowCompileError``: a workflow definition cannot be compiled.
* ``WorkflowExecutionError``: a compiled workflow failed while running.

The engine attaches step/port/index context when it re-raises lower level
errors; the original exception is preserved as ``__cause__``.
"""


class ContractError(ValueError):
    """A V2 data, block or registry contract was violated."""


class WorkflowCompileError(ValueError):
    """A V2 workflow definition could not be compiled."""


class WorkflowExecutionError(RuntimeError):
    """A compiled V2 workflow failed during execution."""
