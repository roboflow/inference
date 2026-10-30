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
arrays of bool/int/float/str, ``sv.Detections``, ``WorkflowImageData``,
``VideoMetadata``), plus numpy bool/int/float scalars and tuples, which the
transports do not carry as such: values are normalized to Python scalars and
lists before serialization on either side, so a block reads back ``False``
for the ``np.bool_`` it stored and a list for its tuple. Images and
``sv.Detections`` ride the result serializers and are lossy: an image is
re-encoded as JPEG on every hop and only the ``data`` keys those serializers
know survive, which ``partition_block_state`` reports as a note.

Anything else (a tracker object, an annotator, a model, a value an encoder
refuses, a dict holding the ``_type`` wire marker) cannot be carried, nor can
an attribute whose serialized form is over ``BLOCK_STATE_MAX_BYTES``. When
any attribute is dropped, ``partition_block_state`` withholds the whole state
for that call and warns, naming every attribute: carrying the rest would
split state the block keeps consistent (a ``self.initialized`` flag guarding
the helper it initialized), so the block is left exactly where it was before
state was carried at all, rebuilding everything on the next call. Dropping
instead of refusing is deliberate: a hard error here would turn blocks that
run on hosted today into failures after the fact.

The echo is the output of user code, so the engine treats it as untrusted
input: ``apply_block_state`` refuses a non-mapping echo, names the engine
owns, dunders, and names defined on the block class (usage metadata, ``run``)
before it touches the instance, and the sandbox leaves the first two out of
the echo up front, with the same warning as for a non-transportable value.
"""

from datetime import datetime
from typing import Any, Callable, Dict, FrozenSet, List, NamedTuple, Tuple

import numpy as np
from roboflow_workflows.errors import DynamicBlockError
from roboflow_workflows.execution_engine.constants import (
    AREA_CONVERTED_KEY_IN_SV_DETECTIONS,
    AREA_KEY_IN_SV_DETECTIONS,
    BOUNDING_RECT_ANGLE_KEY_IN_SV_DETECTIONS,
    BOUNDING_RECT_HEIGHT_KEY_IN_SV_DETECTIONS,
    BOUNDING_RECT_RECT_KEY_IN_SV_DETECTIONS,
    BOUNDING_RECT_WIDTH_KEY_IN_SV_DETECTIONS,
    DETECTED_CODE_KEY,
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
    PARENT_COORDINATES_KEY,
    PARENT_DIMENSIONS_KEY,
    PARENT_ID_KEY,
    PATH_DEVIATION_KEY_IN_SV_DETECTIONS,
    POLYGON_KEY_IN_SV_DETECTIONS,
    RLE_MASK_KEY_IN_SV_DETECTIONS,
    ROOT_PARENT_COORDINATES_KEY,
    ROOT_PARENT_DIMENSIONS_KEY,
    ROOT_PARENT_ID_KEY,
    SMOOTHED_SPEED_KEY_IN_SV_DETECTIONS,
    SMOOTHED_VELOCITY_KEY_IN_SV_DETECTIONS,
    SPEED_KEY_IN_SV_DETECTIONS,
    TIME_IN_ZONE_KEY_IN_SV_DETECTIONS,
    VELOCITY_KEY_IN_SV_DETECTIONS,
)

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

# Both transports encode ints natively up to 64 bits (msgpack refuses wider
# ones); the signed range is what every decoder reads back as the same int.
_MIN_CARRIED_INT = -(2**63)
_MAX_CARRIED_INT = 2**63 - 1

# numpy dtype kinds the serializers carry losslessly through ``tolist()``:
# bool, signed/unsigned int, float and unicode. Complex, bytes, object,
# datetime64, timedelta64 and structured arrays break one encoder or another.
_CARRIED_NDARRAY_KINDS = frozenset("biufU")

# The ``_type`` key marks a typed value on the wire; a user dict carrying it
# would be decoded as that type on the other side.
WIRE_TYPE_MARKER_KEY = "_type"

# ``sv.Detections.data`` keys the transport serializer writes and the
# deserializer reads back; any other key is lost on every hop.
CARRIED_DETECTIONS_DATA_KEYS: FrozenSet[str] = frozenset(
    {
        "class_name",
        DETECTION_ID_KEY,
        IMAGE_DIMENSIONS_KEY,
        PARENT_ID_KEY,
        ROOT_PARENT_ID_KEY,
        PARENT_COORDINATES_KEY,
        PARENT_DIMENSIONS_KEY,
        ROOT_PARENT_COORDINATES_KEY,
        ROOT_PARENT_DIMENSIONS_KEY,
        PATH_DEVIATION_KEY_IN_SV_DETECTIONS,
        TIME_IN_ZONE_KEY_IN_SV_DETECTIONS,
        POLYGON_KEY_IN_SV_DETECTIONS,
        BOUNDING_RECT_ANGLE_KEY_IN_SV_DETECTIONS,
        BOUNDING_RECT_RECT_KEY_IN_SV_DETECTIONS,
        BOUNDING_RECT_HEIGHT_KEY_IN_SV_DETECTIONS,
        BOUNDING_RECT_WIDTH_KEY_IN_SV_DETECTIONS,
        KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS,
        KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS,
        KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS,
        KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
        DETECTED_CODE_KEY,
        VELOCITY_KEY_IN_SV_DETECTIONS,
        SPEED_KEY_IN_SV_DETECTIONS,
        SMOOTHED_VELOCITY_KEY_IN_SV_DETECTIONS,
        SMOOTHED_SPEED_KEY_IN_SV_DETECTIONS,
        AREA_KEY_IN_SV_DETECTIONS,
        AREA_CONVERTED_KEY_IN_SV_DETECTIONS,
        RLE_MASK_KEY_IN_SV_DETECTIONS,
        "width",
        "height",
    }
)


class BlockStatePartition(NamedTuple):
    """What ``partition_block_state`` decided for one call.

    Attributes:
        carried: Normalized state to serialize; empty whenever anything was
            dropped.
        dropped: Attributes that cannot be carried, with the reason.
        notes: Further messages for the block's author: state withheld
            because of a drop, or carried values the transport degrades.
    """

    carried: Dict[str, Any]
    dropped: List[DroppedAttribute]
    notes: List[str]


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
        DynamicBlockError: If ``state`` is not a dict, or carries a key that
            is not a string, is an attribute the engine owns, is a dunder, or
            is defined on the block's class; user code could otherwise
            overwrite usage metadata or the block's methods from the sandbox.
    """
    if not isinstance(state, dict):
        raise DynamicBlockError(
            public_message=(
                "Custom Python block state echoed by the sandbox is a "
                f"{type(state).__name__}, not a mapping of attribute names."
            ),
            context="dynamic_block | block_state_apply",
        )
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
    defined_on_class = any(name in vars(klass) for klass in type(block).__mro__)
    if is_reserved_state_name(name) or defined_on_class:
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
) -> BlockStatePartition:
    """Decide what of ``self`` state the transport carries on this call.

    An attribute is dropped when its name is reserved (engine-owned, dunder
    or not a string), when its value is not transportable, when the
    transport's encoder refuses it (``measure`` raising), or when its own
    serialized size is over ``BLOCK_STATE_MAX_BYTES``; if the rest is still
    over the cap, the largest attributes go until it fits. When anything is
    dropped, nothing is carried: see the module docstring.

    Args:
        state: User attributes about to be serialized.
        measure: Serialized size in bytes of a one-attribute dict, in the
            transport's own encoding. Doubles as the trial encode.

    Returns:
        The carried state, the dropped attributes with their reasons, and
        notes for the block's author.
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
            normalized_value = _normalize_value(value)
            size = measure({name: normalized_value})
        except DynamicBlockError as error:
            dropped.append((name, str(error)))
            continue
        except Exception as error:
            dropped.append((name, _encoder_refused_reason(error)))
            continue
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

    if dropped:
        notes = [_withheld_note(dropped=dropped, withheld=list(carried))]
        partition = BlockStatePartition(carried={}, dropped=dropped, notes=notes)

        return partition

    notes = _lossy_value_notes(carried)
    partition = BlockStatePartition(carried=carried, dropped=[], notes=notes)

    return partition


def _encoder_refused_reason(error: Exception) -> str:
    reason = "the transport cannot encode it " f"({type(error).__name__}: {error})"

    return reason


def _over_cap_reason(size: int) -> str:
    reason = (
        f"it serializes to {size} bytes, over the {BLOCK_STATE_MAX_BYTES} byte "
        "limit for remote execution; state is shipped to the sandbox and back "
        "on every call"
    )

    return reason


def _withheld_note(*, dropped: List[DroppedAttribute], withheld: List[str]) -> str:
    dropped_names = ", ".join(f"`self.{name}`" for name, _ in dropped)
    note = (
        f"Custom Python block state withheld for this call because {dropped_names} "
        "cannot be carried: none of the state kept on `self` persists to the "
        "next call, so the block rebuilds it as it did before state was carried"
    )
    if withheld:
        withheld_names = ", ".join(f"`self.{name}`" for name in withheld)
        note += f" (also not carried: {withheld_names})"

    return note


def _lossy_value_notes(state: Dict[str, Any]) -> List[str]:
    notes: List[str] = []
    for name, value in state.items():
        _collect_lossy_value_notes(value, path=name, notes=notes)

    return notes


def _collect_lossy_value_notes(value: Any, *, path: str, notes: List[str]) -> None:
    import supervision as sv
    from roboflow_workflows.execution_engine.entities.base import WorkflowImageData

    if isinstance(value, sv.Detections):
        lost_keys = sorted(set(value.data) - CARRIED_DETECTIONS_DATA_KEYS)
        if lost_keys:
            notes.append(
                f"Custom Python block state `self.{path}` is carried without "
                f"its sv.Detections data keys {lost_keys}; only the keys the "
                "workflow serializers know survive a call"
            )
        return
    if isinstance(value, WorkflowImageData):
        notes.append(
            f"Custom Python block state `self.{path}` is an image, which is "
            "re-encoded as JPEG on every call; keep `image.numpy_image` on "
            "`self` instead for exact pixels"
        )
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _collect_lossy_value_notes(item, path=f"{path}[{key!r}]", notes=notes)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _collect_lossy_value_notes(item, path=f"{path}[{index}]", notes=notes)


def describe_dropped_block_state(
    block_type_name: str, partition: BlockStatePartition
) -> List[str]:
    """Warning lines for a partition, for logs or the block's stderr.

    Args:
        block_type_name: The dynamic block's type, for the message.
        partition: As returned by ``partition_block_state``.

    Returns:
        One message per dropped attribute, then one per note.
    """
    messages = [
        f"Custom Python block `{block_type_name}` state `self.{name}` is not "
        f"carried across remote execution and is rebuilt on every call: {reason}"
        for name, reason in partition.dropped
    ]
    messages.extend(
        f"Custom Python block `{block_type_name}`: {note}" for note in partition.notes
    )

    return messages


def _validate_value(value: Any, *, path: str) -> None:
    if value is None or isinstance(value, (bool, float, str, bytes)):
        return
    if isinstance(value, int):
        _validate_int(int(value), path=path)
        return
    if isinstance(value, datetime):
        return
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in _CARRIED_NDARRAY_KINDS:
            _raise_not_transportable(f"numpy array of dtype {value.dtype}", path=path)
        if value.dtype.kind in "iu" and value.size:
            _validate_int(int(value.min()), path=path)
            _validate_int(int(value.max()), path=path)
        return
    if isinstance(value, (np.longdouble, np.clongdouble)) and not isinstance(
        value, (np.float64, np.complex128)
    ):
        _raise_not_transportable(type(value).__name__, path=path)
    if isinstance(value, np.bool_) or isinstance(value, np.floating):
        return
    if isinstance(value, np.integer):
        _validate_int(int(value), path=path)
        return
    if _is_transportable_workflow_object(value):
        _validate_workflow_object(value, path=path)
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
            if key == WIRE_TYPE_MARKER_KEY:
                raise DynamicBlockError(
                    public_message=(
                        f"Custom Python block state `self.{path}` has a "
                        f"`{WIRE_TYPE_MARKER_KEY}` key, which the transport "
                        "reserves to mark typed values; rename the key."
                    ),
                    context="dynamic_block | block_state_validation",
                )
            _validate_value(item, path=f"{path}[{key!r}]")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_value(item, path=f"{path}[{index}]")
        return

    _raise_not_transportable(type(value).__name__, path=path)


def _validate_int(value: int, *, path: str) -> None:
    if not _MIN_CARRIED_INT <= value <= _MAX_CARRIED_INT:
        raise DynamicBlockError(
            public_message=(
                f"Custom Python block state `self.{path}` holds an integer "
                "outside the signed 64-bit range, which cannot be carried "
                "across remote execution; store it as a string."
            ),
            context="dynamic_block | block_state_validation",
        )


def _validate_workflow_object(value: Any, *, path: str) -> None:
    import supervision as sv

    if not isinstance(value, sv.Detections) or len(value) == 0:
        return
    missing = []
    if value.confidence is None:
        missing.append("confidence")
    if value.class_id is None:
        missing.append("class_id")
    for key in ("class_name", DETECTION_ID_KEY):
        if key not in value.data:
            missing.append(f"data[{key!r}]")
    has_size = (
        IMAGE_DIMENSIONS_KEY in value.data
        or value.data.get(RLE_MASK_KEY_IN_SV_DETECTIONS) is not None
    )
    if not has_size:
        missing.append(f"data[{IMAGE_DIMENSIONS_KEY!r}]")
    if missing:
        raise DynamicBlockError(
            public_message=(
                f"Custom Python block state `self.{path}` is an sv.Detections "
                f"without {', '.join(missing)}, which the transport needs to "
                "carry it; detections from model blocks have them."
            ),
            context="dynamic_block | block_state_validation",
        )


def _raise_not_transportable(description: str, *, path: str) -> None:
    raise DynamicBlockError(
        public_message=(
            f"Custom Python block state `self.{path}` holds a "
            f"{description}, which cannot be carried across remote "
            "execution. Only JSON-like values, numpy arrays of bool/int/float/"
            "str or such scalars, datetimes, sv.Detections, images and video "
            "metadata persist between calls; other objects are rebuilt on "
            "every call."
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
