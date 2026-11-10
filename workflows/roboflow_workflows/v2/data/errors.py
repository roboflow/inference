"""The error raised when a V2 data contract is violated.

Data modules raise it for malformed batches, layouts, kinds or metadata. Block
declarations, the catalogue and execution sessions raise it and its engine
subclasses too, so ``roboflow_workflows.v2.engine.errors`` re-exports the same
class.
"""


class ContractError(ValueError):
    """A V2 data, block or registry contract was violated."""
