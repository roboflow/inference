"""``self`` state of a custom Python block across remote execution.

In local mode a dynamic block's ``run`` receives the real block instance, so
attributes it sets (``self.has_first_frame = True``) survive between frames.
Under ``WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE=modal`` the function runs in a
sandbox that cannot hold that instance, and the sandbox has no durable
identity to key per-run state on (containers are recycled, websocket
reconnects are not container-affine, and one executor serves every
concurrent run of a workspace). The engine-side instance is the only object
that lives exactly as long as the run, so it owns the state: every remote
call ships the block's user attributes, the sandbox seeds ``self`` with them
and echoes the mutated set back, and the engine writes it onto the instance.

State therefore has to cross a serialization boundary. The transportable
types are the ones the existing input/result serializers already understand
(scalars, strings, bytes, lists, string-keyed dicts, ``datetime``, numpy
arrays and scalars, ``sv.Detections``, ``WorkflowImageData``,
``VideoMetadata``). Anything else — a tracker object, an open file, a model —
is refused with the attribute path, instead of being stringified by the
generic encoder and silently corrupting the block's next frame.
"""

from datetime import datetime
from typing import Any, Dict, FrozenSet

import numpy as np
from roboflow_workflows.errors import DynamicBlockError

# Attributes the engine itself sets on a dynamic block instance. Everything
# else in ``__dict__`` is the user's — including underscore-prefixed names,
# which custom code uses freely (``self._calls``), so a leading underscore is
# not the marker.
ENGINE_ATTRIBUTE_NAMES: FrozenSet[str] = frozenset(
    {
        "_api_key",
        "_workspace_resolver",
        "_execution_observer",
        "_manifest_description",
        "_init_results",
        "_workflow_step_name",
        "_workflow_step_type",
        "_workflow_step_selector",
    }
)

# Serialized ceiling per call. State rides every frame in both directions,
# so a block keeping frame buffers on ``self`` would pay for them twice per
# frame; refusing early keeps that a visible error instead of a slow stream.
BLOCK_STATE_MAX_BYTES = 1024 * 1024


def collect_block_state(block: Any) -> Dict[str, Any]:
    """Return the user attributes of a dynamic block instance.

    Args:
        block: The engine-side dynamic block instance.

    Returns:
        A new dict of every instance attribute the engine did not set itself.
    """
    state = {
        name: value
        for name, value in vars(block).items()
        if name not in ENGINE_ATTRIBUTE_NAMES
    }

    return state


def apply_block_state(block: Any, state: Dict[str, Any]) -> None:
    """Replace the user attributes of a dynamic block instance with ``state``.

    Attributes absent from ``state`` are removed, so a ``del self.x`` in the
    sandbox takes effect on the engine side too. Engine attributes are left
    untouched.

    Args:
        block: The engine-side dynamic block instance.
        state: The user attributes echoed back by the sandbox.
    """
    for name in list(vars(block)):
        if name not in ENGINE_ATTRIBUTE_NAMES:
            delattr(block, name)

    for name, value in state.items():
        setattr(block, name, value)


def validate_block_state(state: Dict[str, Any]) -> None:
    """Raise unless every value in ``state`` can cross the remote boundary.

    Args:
        state: User attributes about to be serialized.

    Raises:
        DynamicBlockError: Naming the first attribute path whose value is not
            transportable, and that value's type.
    """
    for name, value in state.items():
        _validate_value(value, path=name)


def _validate_value(value: Any, *, path: str) -> None:
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return
    if isinstance(value, (datetime, np.ndarray, np.generic)):
        return
    if _is_transportable_workflow_object(value):
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise DynamicBlockError(
                    public_message=(
                        f"Custom Python block state `self.{path}` has a "
                        f"non-string key {key!r}; only string keys can be "
                        "carried across remote execution."
                    ),
                    context="dynamic_block | block_state_validation",
                )
            _validate_value(item, path=f"{path}[{key!r}]")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_value(item, path=f"{path}[{index}]")
        return

    raise DynamicBlockError(
        public_message=(
            f"Custom Python block state `self.{path}` holds a "
            f"{type(value).__name__}, which cannot be carried across remote "
            "execution. Attributes set on `self` must be JSON-like values, "
            "numpy arrays, datetimes, sv.Detections, images or video "
            "metadata; keep other objects out of `self` or derive them per "
            "call."
        ),
        context="dynamic_block | block_state_validation",
    )


def _is_transportable_workflow_object(value: Any) -> bool:
    import supervision as sv
    from roboflow_workflows.execution_engine.entities.base import (
        VideoMetadata,
        WorkflowImageData,
    )

    is_transportable = isinstance(
        value, (sv.Detections, WorkflowImageData, VideoMetadata)
    )

    return is_transportable
