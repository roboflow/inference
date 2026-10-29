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
arrays, ``sv.Detections``, ``WorkflowImageData``, ``VideoMetadata``), plus
numpy bool/int/float scalars and tuples, which the transports do not carry as
such: ``normalize_block_state`` turns them into Python scalars and lists
before serialization on either side, so a block reads back ``False`` for the
``np.bool_`` it stored and a list for its tuple. Anything else — a tracker
object, an annotator, a model — is left out of the round trip, as is an
attribute whose serialized form is over ``BLOCK_STATE_MAX_BYTES``:
``partition_block_state`` drops it with a warning naming the attribute and
the reason, so the block rebuilds it on the next call exactly as it did
before state was carried at all. Blocks that cache such objects on ``self``
behind ``hasattr`` keep working; only the transportable attributes persist.
Dropping instead of refusing is deliberate: a hard error here would turn
blocks that run on hosted today into failures after the fact, and the
generic encoder stringifying the value would corrupt the block's next frame.

The echo is the output of user code, so the engine treats it as untrusted
input: ``apply_block_state`` refuses names the engine owns, dunders, and
names defined on the block class (usage metadata, ``run``) before it touches
the instance, and the sandbox leaves the first two out of the echo up front,
with the same warning as for a non-transportable value.
"""

from datetime import datetime
from typing import Any, Callable, Dict, FrozenSet, List, Tuple

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
# frame; an attribute over the cap is left out with a warning instead of
# slowing the whole stream down.
BLOCK_STATE_MAX_BYTES = 1024 * 1024

# One dropped attribute: its name and the reason it is not carried.
DroppedAttribute = Tuple[str, str]


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
    untouched. The whole echo is checked before anything is written, so a
    refused echo leaves the instance exactly as it was.

    Args:
        block: The engine-side dynamic block instance.
        state: The user attributes echoed back by the sandbox.

    Raises:
        DynamicBlockError: If ``state`` carries a key that is not a string,
            is an attribute the engine owns, is a dunder, or is defined on the
            block's class; user code could otherwise overwrite usage metadata
            or the block's methods from the sandbox.
    """
    for name in state:
        _ensure_state_name_allowed(name, block=block)

    instance_attributes = vars(block)
    engine_attributes = {
        name: value
        for name, value in instance_attributes.items()
        if name in ENGINE_ATTRIBUTE_NAMES
    }
    instance_attributes.clear()
    instance_attributes.update(engine_attributes)
    instance_attributes.update(state)


def _ensure_state_name_allowed(name: Any, *, block: Any) -> None:
    if not isinstance(name, str):
        raise DynamicBlockError(
            public_message=(
                f"Custom Python block state echoed a non-string attribute "
                f"name {name!r}; attributes on `self` must have string names."
            ),
            context="dynamic_block | block_state_apply",
        )
    if is_reserved_state_name(name) or hasattr(type(block), name):
        raise DynamicBlockError(
            public_message=(
                f"Custom Python block state `self.{name}` collides with an "
                "attribute the block or the engine owns; pick another name "
                "for state kept on `self`."
            ),
            context="dynamic_block | block_state_apply",
        )


def is_reserved_state_name(name: str) -> bool:
    """Tell whether ``name`` may never be set on ``self`` from the sandbox.

    Args:
        name: An attribute name echoed by the sandbox.

    Returns:
        ``True`` for engine-owned attributes and dunders, which the sandbox
        mirrors so the author sees the refusal without a round trip.
    """
    is_dunder = name.startswith("__") and name.endswith("__")
    is_reserved = is_dunder or name in ENGINE_ATTRIBUTE_NAMES

    return is_reserved


def normalize_block_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``state`` with values the transports cannot carry as-is replaced.

    numpy bool/int/float scalars become Python scalars and tuples become
    lists, recursively; everything else is passed through unchanged. Applied
    after validation on both ends of the round trip, so the block always
    reads back the same Python types whichever transport carried them.

    Args:
        state: Validated user attributes about to be serialized.

    Returns:
        A new dict with the same keys.
    """
    normalized_state = {name: _normalize_value(value) for name, value in state.items()}

    return normalized_state


def _normalize_value(value: Any) -> Any:
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {key: _normalize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]

    return value


def partition_block_state(
    state: Dict[str, Any], *, measure: Callable[[Dict[str, Any]], int]
) -> Tuple[Dict[str, Any], List[DroppedAttribute]]:
    """Split ``self`` state into what the transport carries and what it drops.

    An attribute is dropped when its name is reserved (engine-owned, dunder
    or not a string), when its value is not transportable, or when its own
    serialized size is over ``BLOCK_STATE_MAX_BYTES``. If the carried set is
    still over the cap, the largest attributes are dropped until it fits.
    Carried values are normalized (``normalize_block_state``) and ready to
    serialize.

    Args:
        state: User attributes about to be serialized.
        measure: Serialized size in bytes of a one-attribute dict, in the
            transport's own encoding.

    Returns:
        The carried state and the dropped attributes, each with the reason
        the caller should surface to the block's author.
    """
    carried: Dict[str, Any] = {}
    sizes: Dict[str, int] = {}
    dropped: List[DroppedAttribute] = []
    for name, value in state.items():
        if not isinstance(name, str) or is_reserved_state_name(name):
            dropped.append(
                (
                    str(name),
                    "the name collides with an attribute the engine owns; pick "
                    "another name for state kept on `self`",
                )
            )
            continue
        try:
            _validate_value(value, path=name)
        except DynamicBlockError as error:
            dropped.append((name, str(error)))
            continue
        normalized_value = _normalize_value(value)
        size = measure({name: normalized_value})
        if size > BLOCK_STATE_MAX_BYTES:
            dropped.append((name, _over_cap_reason(size)))
            continue
        carried[name] = normalized_value
        sizes[name] = size
    total = sum(sizes.values())
    for name in sorted(sizes, key=sizes.get, reverse=True):
        if total <= BLOCK_STATE_MAX_BYTES:
            break
        total -= sizes[name]
        del carried[name]
        dropped.append(
            (
                name,
                f"the state kept on `self` is {total + sizes[name]} bytes in "
                f"total, over the {BLOCK_STATE_MAX_BYTES} byte limit for remote "
                "execution",
            )
        )

    return carried, dropped


def _over_cap_reason(size: int) -> str:
    reason = (
        f"it serializes to {size} bytes, over the {BLOCK_STATE_MAX_BYTES} byte "
        "limit for remote execution; state is shipped to the sandbox and back "
        "on every call"
    )

    return reason


def describe_dropped_block_state(
    block_type_name: str, dropped: List[DroppedAttribute]
) -> List[str]:
    """One warning line per dropped attribute, for logs or the block's stderr.

    Args:
        block_type_name: The dynamic block's type, for the message.
        dropped: As returned by ``partition_block_state``.

    Returns:
        Messages telling the author the attribute is rebuilt on every call.
    """
    messages = [
        f"Custom Python block `{block_type_name}` state `self.{name}` is not "
        f"carried across remote execution and is rebuilt on every call: {reason}"
        for name, reason in dropped
    ]

    return messages


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
    if isinstance(value, (datetime, np.ndarray)):
        return
    if isinstance(value, (np.bool_, np.integer, np.floating)):
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
            "execution. Only JSON-like values, numpy arrays or bool/int/float "
            "scalars, datetimes, sv.Detections, images and video metadata "
            "persist between calls; other objects are rebuilt on every call."
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
