"""Return the first input that is not ``None``, else ``default``.

Canonical type: ``roboflow_core/first_non_empty_or_default@v1``.
"""

from typing import Any, Dict, List, Literal
from pydantic import Field
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

import roboflow_workflows.v2.blocks.formatters._shared.values as _helpers


class FirstNonEmptyOrDefaultBlock(Block):
    """Return the first input that is not ``None``, else ``default``.

    Example: ``data=["$steps.branch_a.output", "$steps.branch_b.output"]``
    outputs branch B's value when the gate skipped branch A.
    """

    type = "roboflow_core/first_non_empty_or_default@v1"
    metadata = {
        "name": "First Non Empty Or Default",
        "port_of": "roboflow_core/first_non_empty_or_default@v1",
        "section": "advanced",
    }
    accepts_empty = True
    outputs = {
        "output": Output(
            description="The first value that is not None, or default; any kind."
        ),
    }

    class Params(BlockParams):
        data: List[Ref(allow_none=True)] = Field(
            min_length=1,
            description=(
                "Selectors in priority order. Missing, filtered or gated values "
                "count as None."
            ),
            examples=[["$steps.branch_a.output", "$steps.branch_b.output"]],
        )
        default: Any = Field(
            default=None,
            description="Literal output when every data value is None.",
            examples=["N/A", 0, []],
        )

    def run(self, *, data: List[Any], default: Any) -> Dict[str, Any]:
        """Pick the first value that is not ``None``.

        Args:
            data: Resolved values in priority order; ``None`` where missing.
            default: Literal fallback.

        Returns:
            ``output``: the first value that is not ``None``, else ``default``.
        """
        for value in data:
            if value is not None:
                return {"output": value}

        return {"output": default}
