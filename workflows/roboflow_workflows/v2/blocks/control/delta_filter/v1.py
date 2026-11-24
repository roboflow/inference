"""Continue to ``next_steps`` only when the stream's value changed.

Canonical type: ``roboflow_core/delta_filter@v1``.
"""

import threading
from typing import Any, Dict, List
from pydantic import Field, StrictInt
from roboflow_workflows.v2.blocks.control._shared.routing import (
    DEFAULT_MAX_SOURCES,
    Lane,
    _check_capacity,
    _lane_of,
    _values_equal,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    Ref,
    Select,
    StepRef,
    Stop,
)


class DeltaFilterBlock(Block):
    """Continue to ``next_steps`` only when the stream's value changed.

    A stream (lane) is the image's ``video_identifier``, else its source, else
    its batch position; crops use their frame's lane.

    Example: bound to a per-frame object count of one camera, ``3, 3, 4, 4``
    continues on the first and third frame. Equality is Python ``==``, so
    ``1``, ``1.0`` and ``True`` are one value. Tensors, arrays, lists, dicts
    and prediction dataclasses compare element by element.

    The lane's last ``value`` is kept by reference, not copied
    (``retains_inputs``). A workflow step that mutates that payload in place
    changes the remembered value; the compiler adds a warning to
    ``CompiledWorkflow.warnings``, and
    ``CompileOptions(mutation_conflicts="error")`` rejects the workflow
    instead. The engine does not copy caller inputs either: a caller that
    changes an object in place and passes it again compares it with itself,
    which is no change. Pass a new object per run instead.
    """

    type = "roboflow_core/delta_filter@v1"
    metadata = {
        "name": "Delta Filter",
        "port_of": "roboflow_core/delta_filter@v1",
        "section": "flow_control",
    }
    retains_inputs = ("value",)

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(
            description=(
                "Frame whose video_metadata.video_identifier names the lane. "
                "Without video metadata: this frame's source, else its batch "
                "position. The value's own source is not read."
            ),
            examples=["$inputs.image"],
        )
        value: Ref(allow_none=True) = Field(
            description=(
                "Value watched for changes. The first value of a lane passes, "
                "unless it is None. After a value, None counts as a change."
            ),
            examples=["$steps.count.output"],
        )
        next_steps: List[StepRef] = Field(
            description="Steps that run when the value changed.",
            examples=[["$steps.notify"]],
        )
        max_sources: StrictInt = Field(
            default=DEFAULT_MAX_SOURCES,
            ge=1,
            description=(
                "Most lanes (video streams, sources or batch positions) the "
                "block remembers a value for. One more fails with "
                "SourceCapacityError until a processing reset."
            ),
            examples=[DEFAULT_MAX_SOURCES],
        )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: Dict[Lane, Any] = {}

    def run(
        self,
        *,
        image: ImageData,
        value: Any,
        next_steps: List[str],
        max_sources: int,
    ) -> Select:
        """Select ``next_steps`` when ``value`` differs from the lane's last one.

        Args:
            image: Frame that names the lane.
            value: Current value; ``None`` is a value too.
            next_steps: Step selectors as written in the step parameters.
            max_sources: Most lanes with a remembered value.

        Returns:
            ``Select(next_steps)`` on a change, else ``Stop()``.

        Raises:
            TypeError: When ``value`` and the remembered value cannot be
                compared to a single bool.
            SourceCapacityError: When a new lane would exceed ``max_sources``.
        """
        lane = _lane_of(image, block=self, field="image")
        with self._lock:
            if lane in self._values:
                unchanged = _values_equal(self._values[lane], value)
            else:
                # V1 compares with dict.get's None: only a first None is no change.
                unchanged = value is None
            if unchanged:
                return Stop()

            if lane not in self._values:
                _check_capacity(
                    self, records=self._values, lane=lane, limit=max_sources
                )
            self._values[lane] = value

        return Select(next_steps)
