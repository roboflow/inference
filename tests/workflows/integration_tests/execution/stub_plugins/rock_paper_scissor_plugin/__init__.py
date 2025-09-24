from typing import List, Type

from inference.core.workflows.prototypes.block import WorkflowBlock
from tests.workflows.integration_tests.execution.stub_plugins.rock_paper_scissor_plugin.expression import (
    ExpressionBlock,
)
from tests.workflows.integration_tests.execution.stub_plugins.rock_paper_scissor_plugin.take_first_non_empty import (
    TakeFirstNonEmptyBlock,
)


def load_blocks() -> List[Type[WorkflowBlock]]:
    return [
        ExpressionBlock,
        TakeFirstNonEmptyBlock,
    ]
