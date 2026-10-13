"""Collect the children of each parent into one list: ``v2/dimension_collapse@v1``.

Port of V1 ``roboflow_core/dimension_collapse@v1``. The engine's ``Group``
selector already removes the last nesting axis (decision 007), so the block
only turns that group into a plain list. As in V1, a child that was filtered
or skipped keeps its position as ``None`` (decision 016)::

    known children 0..3, children 1 and 3 skipped   ->  [v0, None, v2, None]
    two children, both skipped                      ->  [None, None]
    no children at all                              ->  []
    parent itself absent                            ->  absent (None)

The extent comes from ``Batch.known_indices``: every child the producer
emitted, skipped ones included. A producer that never emitted a slot cannot be
padded; ``v2/dynamic_crop@v1`` therefore emits a ``None`` slot per empty box.
"""

from typing import Any, Dict, Optional

from pydantic import Field
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.kinds import LIST_OF_VALUES_KIND
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    Group,
    Output,
)


class DimensionCollapseBlock(Block):
    """Turn the children of one parent into a list, one slot per child.

    Example: image 0 has crops ``(0, 0)``, ``(0, 1)`` and ``(0, 2)``; crop 1
    was filtered. Image 0 gets ``[value_0, None, value_2]``, so list position
    ``k`` is child ``k``. An image without children gets ``[]``. Values are not
    copied or converted.
    """

    type = "v2/dimension_collapse@v1"
    accepts_empty = True
    outputs = {
        "output": Output(
            LIST_OF_VALUES_KIND,
            source="data",
            description="Children of one parent by position; None where skipped.",
        ),
    }

    class Params(BlockParams):
        data: Group() = Field(
            description="Values with one more nesting axis than the result."
        )

    def run(self, *, data: Optional[Batch]) -> Dict[str, Any]:
        """Collect ``data`` into a list with one slot per known child.

        Args:
            data: The surviving children of one parent, possibly none;
                ``None`` when the parent itself is absent.

        Returns:
            ``output``: the children's values by position, ``None`` at skipped
            children; ``None`` for an absent parent.
        """
        if data is None:
            return {"output": None}

        by_position = {index[-1]: value for index, value in data.iter_with_indices()}
        extent = max((index[-1] for index in data.known_indices), default=-1) + 1
        output = [by_position.get(position) for position in range(extent)]
        result = {"output": output}

        return result
