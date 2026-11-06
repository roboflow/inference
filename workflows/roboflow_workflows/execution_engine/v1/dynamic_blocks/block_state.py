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
``VideoMetadata``), as exact types, plus numpy bool/int/float/str/bytes
scalars, tuples and ``ZoneInfo`` datetimes, which are wrapped in wire markers
(``numpy_scalar``, ``tuple``, ``zoned_datetime``) because both transports
would otherwise bring them back as Python scalars, lists and fixed-offset
datetimes. Subclasses are not carried: they would come back as their base
type. Images and ``sv.Detections`` ride the result serializers and are
lossy: an image is re-encoded as JPEG on every hop and loses a root parent
id, and only the ``data`` keys those serializers know survive, which
``partition_block_state`` reports as notes.

Anything else (a tracker object, an annotator, a model, a value an encoder
refuses, a dict holding the ``_type`` wire marker or shaped like a
serialized image) cannot be carried, nor can
an attribute whose serialized form is over ``BLOCK_STATE_MAX_BYTES``, nor
state whose numpy arrays and scalars decode to more than eight times that. When
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

from datetime import datetime, timezone
from functools import lru_cache
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

# Attributes carried per call. A block keeping more on ``self`` is withheld
# rather than shipped; the engine refuses an echo with more, so a tampered
# sandbox cannot flood the engine's logs and warn-once set.
BLOCK_STATE_MAX_ATTRIBUTES = 1024

# Wire nesting depth of one attribute; senders drop deeper state.
_MAX_ECHO_DEPTH = 64

_IMAGE_WIRE_LEVELS = 3
_VIDEO_METADATA_WIRE_LEVELS = 2
_DETECTIONS_WIRE_LEVELS = 5

# Bytes the numpy arrays and scalars of one echo may make the engine allocate.
_MAX_ECHO_ALLOCATION_BYTES = 8 * BLOCK_STATE_MAX_BYTES

# Markers a state echo may contain; anything else is refused before decoding.
_ECHO_WIRE_TYPES = frozenset(
    {
        "datetime",
        "zoned_datetime",
        "bytes",
        "ndarray",
        "tuple",
        "numpy_scalar",
        "sv_detections",
        "workflow_image",
        "video_metadata",
    }
)

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

# Markers the state normalizer wraps values in that both transports would
# otherwise change: a tuple would come back as a list, a numpy scalar as the
# Python scalar (``.item()``, ``~``, fixed-width overflow and division by zero
# then behave differently), and a ``ZoneInfo`` datetime with a fixed offset.
# The decoders are shared with workflow inputs and block results, so a dict
# of exactly these shapes there is decoded too, like the older markers.
TUPLE_WIRE_TYPE = "tuple"
NUMPY_SCALAR_WIRE_TYPE = "numpy_scalar"
ZONED_DATETIME_WIRE_TYPE = "zoned_datetime"

# numpy dtype kinds a ``numpy_scalar`` marker may name when decoded.
NUMPY_SCALAR_WIRE_KINDS = frozenset("biufSU")

# numpy scalar types carried with the ``numpy_scalar`` marker, checked
# exactly: a subclass would come back as its base type. Excludes long
# doubles, complex, datetime64 and timedelta64, which no encoder carries.
CARRIED_NUMPY_SCALAR_TYPES: FrozenSet[type] = frozenset(
    scalar_type
    for scalar_type in set(np.sctypeDict.values())
    if issubclass(scalar_type, (np.bool_, np.integer, np.floating, np.str_, np.bytes_))
    and not issubclass(scalar_type, (np.timedelta64, np.longdouble))
)

# Ids the decoders substitute when a detection's parentage is not carried.
_PLACEHOLDER_PARENT_IDS = frozenset({"input", "result"})

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


# Detections data keys whose per-detection rows reach the wire unchanged.
_PASSED_THROUGH_DETECTIONS_DATA_KEYS: FrozenSet[str] = frozenset(
    {
        PATH_DEVIATION_KEY_IN_SV_DETECTIONS,
        TIME_IN_ZONE_KEY_IN_SV_DETECTIONS,
        DETECTED_CODE_KEY,
        BOUNDING_RECT_ANGLE_KEY_IN_SV_DETECTIONS,
        BOUNDING_RECT_RECT_KEY_IN_SV_DETECTIONS,
        BOUNDING_RECT_HEIGHT_KEY_IN_SV_DETECTIONS,
        BOUNDING_RECT_WIDTH_KEY_IN_SV_DETECTIONS,
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
    if type(value) in CARRIED_NUMPY_SCALAR_TYPES:
        marked = {
            WIRE_TYPE_MARKER_KEY: NUMPY_SCALAR_WIRE_TYPE,
            "dtype": numpy_scalar_wire_dtype(value),
            "value": value.item(),
        }
        return marked
    zone_key = _zone_key(value) if type(value) is datetime else None
    if zone_key is not None:
        # The wall time, fold and zone rebuild the value exactly, including
        # times in a DST gap or fold and the min/max sentinels, which an
        # instant converted back with ``astimezone`` would shift or overflow.
        marked = {
            WIRE_TYPE_MARKER_KEY: ZONED_DATETIME_WIRE_TYPE,
            "value": value.replace(tzinfo=None).isoformat(),
            "fold": value.fold,
            "zone": zone_key,
        }
        return marked
    if isinstance(value, dict):
        normalized = {
            _normalize_key(key): _normalize_value(item) for key, item in value.items()
        }
        return normalized
    if isinstance(value, tuple):
        marked = {
            WIRE_TYPE_MARKER_KEY: TUPLE_WIRE_TYPE,
            "value": [_normalize_value(item) for item in value],
        }
        return marked
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]

    return value


def numpy_scalar_wire_dtype(value: Any) -> str:
    """The dtype a ``numpy_scalar`` marker names for ``value``.

    The type code for numbers, so aliases such as ``np.longlong`` come back
    as themselves; the full dtype for strings, which needs the length.

    Args:
        value: A numpy scalar of one of ``CARRIED_NUMPY_SCALAR_TYPES``.

    Returns:
        A string ``np.dtype`` accepts.
    """
    if value.dtype.kind in "SU":
        return value.dtype.str

    return value.dtype.char


def decode_numpy_scalar(dtype_name: Any, value: Any) -> Any:
    """Build the numpy scalar a ``numpy_scalar`` marker describes, safely.

    The marker comes from sandboxed code, so its parts are checked before
    numpy allocates anything: an integer value under a bytes dtype would
    otherwise allocate that many bytes.

    Args:
        dtype_name: The marker's ``dtype``.
        value: The marker's decoded ``value``.

    Returns:
        The numpy scalar.

    Raises:
        ValueError: If the dtype is not carried, the value does not match it,
            or the scalar would be larger than any honest state value.
    """
    if not isinstance(dtype_name, str):
        raise ValueError("numpy scalar dtype must be a string")
    dtype = np.dtype(dtype_name)
    if dtype.kind not in NUMPY_SCALAR_WIRE_KINDS:
        raise ValueError(f"numpy scalar of dtype {dtype} is not carried")
    if dtype.itemsize > _MAX_ECHO_ALLOCATION_BYTES:
        raise ValueError(f"numpy scalar of dtype {dtype} is too large")
    expected_types = {
        "b": (bool,),
        "i": (int,),
        "u": (int,),
        "f": (float, int),
        "U": (str,),
        "S": (bytes,),
    }[dtype.kind]
    if type(value) not in expected_types:
        raise ValueError(
            f"numpy scalar of dtype {dtype} cannot hold a {type(value).__name__}"
        )
    if dtype.kind in "iu":
        limits = np.iinfo(dtype)
        if not limits.min <= value <= limits.max:
            raise ValueError(f"{value} is out of range for {dtype}")
    if (
        dtype.kind in "SU"
        and len(value) * (4 if dtype.kind == "U" else 1) > dtype.itemsize
    ):
        raise ValueError(f"value is longer than dtype {dtype}")
    scalar = dtype.type(value)

    return scalar


def check_echoed_wire_value(
    value: Any, *, path: str, depth: int = 0, allocated: int = 0
) -> int:
    """Refuse an echoed attribute, still in wire form, before it is decoded.

    The echo is written by sandboxed code, which can bypass the sandbox's own
    checks, so only the shapes the state normalizer and the serializers
    produce for carried values are let through to the shared decoders: known
    markers, arrays of carried dtypes, images as base64, and numpy arrays and
    scalars that together stay within one allocation budget per echo.

    Args:
        value: One attribute of the echo, as parsed from JSON or msgpack.
        path: The attribute path, for the message.
        depth: Nesting depth so far.
        allocated: Bytes the echo's arrays and numpy scalars checked so far
            make the engine allocate.

    Returns:
        ``allocated`` plus what this value's arrays and numpy scalars take.

    Raises:
        DynamicBlockError: Naming the attribute path and what was refused.
    """
    if depth > _MAX_ECHO_DEPTH:
        _raise_malformed_echo(path, "is nested too deeply")
    if isinstance(value, list):
        for item in value:
            allocated = check_echoed_wire_value(
                item, path=path, depth=depth + 1, allocated=allocated
            )
        return allocated
    if not isinstance(value, dict):
        return allocated
    wire_type = value.get(WIRE_TYPE_MARKER_KEY)
    if wire_type is None:
        if value.get("type") == "base64" and "value" in value:
            _raise_malformed_echo(path, "holds an untagged image")
    elif wire_type not in _ECHO_WIRE_TYPES:
        _raise_malformed_echo(path, f"holds a `{wire_type}` marker")
    elif wire_type == "ndarray":
        allocated += _check_echoed_ndarray(value, path=path)
    elif wire_type == "numpy_scalar" and isinstance(value.get("dtype"), str):
        allocated += _echoed_dtype(value, path=path).itemsize
    elif wire_type == "workflow_image" and value.get("type") != "base64":
        _raise_malformed_echo(path, "holds an image that is not base64")
    elif wire_type == "zoned_datetime" and value.get("fold") not in (0, 1):
        _raise_malformed_echo(path, "holds a zoned datetime with an invalid fold")
    if allocated > _MAX_ECHO_ALLOCATION_BYTES:
        _raise_malformed_echo(path, "holds arrays larger than any carried state")
    for key, item in value.items():
        if key != WIRE_TYPE_MARKER_KEY:
            allocated = check_echoed_wire_value(
                item, path=path, depth=depth + 1, allocated=allocated
            )

    return allocated


def _echoed_dtype(value: Dict[str, Any], *, path: str) -> np.dtype:
    try:
        dtype = np.dtype(value.get("dtype"))
    except (TypeError, ValueError) as error:
        raise DynamicBlockError(
            public_message=(
                f"Custom Python block state `self.{path}` returned from remote "
                f"execution holds an invalid dtype: {error}"
            ),
            context="dynamic_block | block_state_echo",
        ) from error

    return dtype


def _check_echoed_ndarray(value: Dict[str, Any], *, path: str) -> int:
    dtype = _echoed_dtype(value, path=path)
    if dtype.kind not in _CARRIED_NDARRAY_KINDS:
        _raise_malformed_echo(path, f"holds an array of dtype {dtype}")
    shape = value.get("shape")
    if not isinstance(shape, list) or not all(
        type(size) is int and size >= 0 for size in shape
    ):
        _raise_malformed_echo(path, "holds an array with an invalid shape")
    if dtype.kind in "SUV" and dtype.itemsize == 0 and 0 not in shape:
        _raise_malformed_echo(path, f"holds a non-empty array of dtype {dtype}")
    allocation = dtype.itemsize * float(np.prod(shape, dtype=np.float64))
    if allocation > _MAX_ECHO_ALLOCATION_BYTES:
        _raise_malformed_echo(path, "holds an array larger than any carried state")

    return int(allocation)


def _raise_malformed_echo(path: str, problem: str) -> None:
    raise DynamicBlockError(
        public_message=(
            f"Custom Python block state `self.{path}` returned from remote "
            f"execution {problem}, which the sandbox never sends for carried "
            "state; the echo is refused."
        ),
        context="dynamic_block | block_state_echo",
    )


def validate_echoed_block_state(state: Dict[str, Any]) -> None:
    """Check a decoded echo the way the state was checked before it was sent.

    Args:
        state: The echo after decoding, attribute by attribute.

    Raises:
        DynamicBlockError: If the echo has more attributes than are ever
            carried, or a value the engine would not have carried.
    """
    if len(state) > BLOCK_STATE_MAX_ATTRIBUTES:
        raise DynamicBlockError(
            public_message=(
                f"Custom Python block state returned from remote execution has "
                f"{len(state)} attributes, more than the "
                f"{BLOCK_STATE_MAX_ATTRIBUTES} ever carried; the echo is refused."
            ),
            context="dynamic_block | block_state_echo",
        )
    for name, value in state.items():
        _validate_value(value, path=name)


def _zone_key(value: datetime) -> Any:
    """The ``ZoneInfo`` key of an aware datetime, or ``None``."""
    from zoneinfo import ZoneInfo

    if type(value.tzinfo) is not ZoneInfo:
        return None

    return value.tzinfo.key


def _normalize_key(key: Any) -> Any:
    if type(key) is np.str_:
        return key.item()

    return key


def partition_block_state(
    state: Dict[str, Any], *, measure: Callable[[Dict[str, Any]], int]
) -> BlockStatePartition:
    """Decide what of ``self`` state the transport carries on this call.

    An attribute is dropped when its name is reserved (engine-owned, dunder
    or not a string), when its value is not transportable, when the
    transport's encoder refuses it (``measure`` raising), when its own
    serialized size is over ``BLOCK_STATE_MAX_BYTES``, or when its numpy
    arrays and scalars would make the receiver allocate more than the echo
    allows; if the rest is still over either cap, the largest attributes go
    until it fits. When anything is dropped, nothing is carried: see the
    module docstring.

    Args:
        state: User attributes about to be serialized.
        measure: Serialized size in bytes of a one-attribute dict, in the
            transport's own encoding. Doubles as the trial encode.

    Returns:
        The carried state, the dropped attributes with their reasons, and
        notes for the block's author.
    """
    if len(state) > BLOCK_STATE_MAX_ATTRIBUTES:
        reason = (
            f"the block keeps {len(state)} attributes on `self`, more than the "
            f"{BLOCK_STATE_MAX_ATTRIBUTES} carried across remote execution"
        )
        dropped = [(str(name), reason) for name in state]
        notes = [_withheld_note(dropped=dropped[:1], withheld=[])]
        partition = BlockStatePartition(carried={}, dropped=dropped[:1], notes=notes)

        return partition

    carried: Dict[str, Any] = {}
    raw_carried: Dict[str, Any] = {}
    sizes: Dict[str, int] = {}
    allocations: Dict[str, int] = {}
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
        allocation = _decoded_allocation(value)
        if allocation > _MAX_ECHO_ALLOCATION_BYTES:
            dropped.append((name, _over_allocation_reason(allocation)))
            continue
        carried[name] = normalized_value
        raw_carried[name] = value
        sizes[name] = size
        allocations[name] = allocation
    # The per-attribute sizes overcount the combined payload by one map
    # header each, so the total cap is decided on the combined encoding.
    total = sum(sizes.values())
    if total > BLOCK_STATE_MAX_BYTES:
        total = measure(carried)
    for name in sorted(sizes, key=sizes.get, reverse=True):
        if total <= BLOCK_STATE_MAX_BYTES:
            break
        del carried[name]
        dropped.append(
            (
                name,
                f"the state kept on `self` is {total} bytes in total, over the "
                f"{BLOCK_STATE_MAX_BYTES} byte limit for remote execution",
            )
        )
        total = measure(carried) if carried else 0
    total_allocation = sum(allocations[name] for name in carried)
    for name in sorted(carried, key=allocations.get, reverse=True):
        if total_allocation <= _MAX_ECHO_ALLOCATION_BYTES:
            break
        del carried[name]
        dropped.append(
            (
                name,
                f"the state kept on `self` takes {total_allocation} bytes to "
                f"decode in total, over the {_MAX_ECHO_ALLOCATION_BYTES} byte "
                "limit for remote execution",
            )
        )
        total_allocation -= allocations[name]

    if dropped:
        notes = [_withheld_note(dropped=dropped, withheld=list(carried))]
        partition = BlockStatePartition(carried={}, dropped=dropped, notes=notes)

        return partition

    # Notes read the values as the block holds them, so paths name the
    # block's own structure rather than the wire markers.
    notes = _lossy_value_notes(raw_carried)
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


def _over_allocation_reason(allocation: int) -> str:
    reason = (
        f"its numpy arrays and scalars take {allocation} bytes to decode, over "
        f"the {_MAX_ECHO_ALLOCATION_BYTES} byte limit for remote execution"
    )

    return reason


def _decoded_allocation(value: Any) -> int:
    """Bytes the receiver allocates for a validated value's arrays and scalars.

    Counts what ``check_echoed_wire_value`` charges on the wire form: every
    array and numpy scalar the state normalizer passes through, and inside
    ``sv.Detections`` the per-detection arrays the serializer leaves as arrays.
    """
    detections_type = _workflow_object_types()[0]
    value_type = type(value)
    if value_type in CARRIED_NUMPY_SCALAR_TYPES:
        return value.dtype.itemsize
    if value_type is np.ndarray:
        return value.dtype.itemsize * value.size
    if value_type is detections_type:
        allocation = sum(
            _detections_field_allocation(item)
            for key, item in value.data.items()
            if key in CARRIED_DETECTIONS_DATA_KEYS
        )
        return allocation
    if value_type is dict:
        return sum(_decoded_allocation(item) for item in value.values())
    if value_type in (list, tuple):
        return sum(_decoded_allocation(item) for item in value)

    return 0


def _detections_field_allocation(field: Any) -> int:
    # Rows of a one-dimensional field are scalars, never arrays, on the wire.
    if isinstance(field, np.ndarray) and field.dtype.kind != "O" and field.ndim < 2:
        return 0

    allocation = _payload_allocation(field)

    return allocation


def _payload_allocation(value: Any) -> int:
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "O":
            return sum(_payload_allocation(item) for item in value.ravel())
        return value.dtype.itemsize * value.size
    if isinstance(value, dict):
        return sum(_payload_allocation(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_payload_allocation(item) for item in value)

    return 0


def _withheld_note(*, dropped: List[DroppedAttribute], withheld: List[str]) -> str:
    dropped_names = ", ".join(f"`self.{name}`" for name, _ in dropped)
    note = (
        f"Custom Python block state withheld: {dropped_names} cannot be "
        "carried, so no state is sent to the sandbox and the block runs this "
        "call with an empty `self`, as it did before state was carried"
    )
    if withheld:
        withheld_names = ", ".join(f"`self.{name}`" for name in withheld)
        note += f" (also not sent: {withheld_names})"

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
        if len(value) == 0 and (value.data or value.metadata):
            notes.append(
                f"Custom Python block state `self.{path}` holds empty "
                "sv.Detections, which come back without their data and metadata"
            )
            return
        lost_keys = sorted(set(value.data) - CARRIED_DETECTIONS_DATA_KEYS, key=repr)
        if lost_keys:
            notes.append(
                f"Custom Python block state `self.{path}` is carried without "
                f"its sv.Detections data keys {lost_keys}; only the keys the "
                "workflow serializers know survive a call"
            )
        if _has_root_parent_id_equal_to_parent_id(value):
            notes.append(
                f"Custom Python block state `self.{path}` holds sv.Detections "
                f"whose `{ROOT_PARENT_ID_KEY}` equals their `{PARENT_ID_KEY}`; "
                f"it comes back replaced by a placeholder (`input` or "
                "`result`), because the transport only carries a root parent "
                "that differs from the parent"
            )
        if _has_integer_labels(value):
            notes.append(
                f"Custom Python block state `self.{path}` holds sv.Detections "
                "with integer class names or detection ids, which come back as "
                "strings"
            )
        if value.mask is not None and RLE_MASK_KEY_IN_SV_DETECTIONS not in value.data:
            notes.append(
                f"Custom Python block state `self.{path}` holds sv.Detections "
                "masks, which are carried as polygons: mask edges are "
                "approximated and a detection whose mask has no contour is lost"
            )
        return
    if isinstance(value, WorkflowImageData):
        notes.append(
            f"Custom Python block state `self.{path}` is an image, which is "
            "re-encoded as JPEG on every call, so its pixels drift"
        )
        parent_id = value.parent_metadata.parent_id
        is_root_image = parent_id == value.workflow_root_ancestor_metadata.parent_id
        if is_root_image and parent_id not in _PLACEHOLDER_PARENT_IDS:
            notes.append(
                f"Custom Python block state `self.{path}` is an image whose "
                "`parent_id` comes back replaced by a placeholder (`input` or "
                "`result`); the transport only carries the parentage of crops"
            )
        # The private attribute, so an image held only as base64 is not decoded.
        numpy_image = getattr(value, "_numpy_image", None)
        if numpy_image is not None and numpy_image.ndim == 2:
            notes.append(
                f"Custom Python block state `self.{path}` is a grayscale image, "
                "which comes back with three channels"
            )
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _collect_lossy_value_notes(item, path=f"{path}[{key!r}]", notes=notes)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _collect_lossy_value_notes(item, path=f"{path}[{index}]", notes=notes)


def _has_root_parent_id_equal_to_parent_id(detections: Any) -> bool:
    if PARENT_ID_KEY not in detections.data:
        return False
    if ROOT_PARENT_ID_KEY not in detections.data:
        return False
    parent_ids = [str(item) for item in detections.data[PARENT_ID_KEY]]
    root_parent_ids = [str(item) for item in detections.data[ROOT_PARENT_ID_KEY]]
    has_equal_ids = any(
        parent_id == root_parent_id and parent_id not in _PLACEHOLDER_PARENT_IDS
        for parent_id, root_parent_id in zip(parent_ids, root_parent_ids)
    )

    return has_equal_ids


def _has_integer_labels(detections: Any) -> bool:
    has_integer_labels = any(
        isinstance(detections.data.get(key), np.ndarray)
        and detections.data[key].dtype.kind in "iu"
        for key in ("class_name", DETECTION_ID_KEY)
    )

    return has_integer_labels


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


def _validate_value(value: Any, *, path: str, depth: int = 0) -> None:
    # Exact types only: a subclass (defaultdict, Counter, OrderedDict, a
    # namedtuple, an IntEnum member, a str or numpy scalar subclass) comes
    # back as its base type, which breaks the block from the next call on.
    detections_type, image_type, video_metadata_type = _workflow_object_types()
    value_type = type(value)
    if value is None or value_type in (bool, float, str):
        _validate_wire_depth(depth, path=path)
        return
    if value_type is bytes:
        _validate_wire_depth(depth + 1, path=path)
        return
    if value_type is datetime:
        _validate_datetime(value, path=path)
        _validate_wire_depth(depth + 1, path=path)
        return
    if value_type is int:
        _validate_int(value, path=path)
        _validate_wire_depth(depth, path=path)
        return
    if value_type in CARRIED_NUMPY_SCALAR_TYPES:
        if isinstance(value, np.integer):
            _validate_int(int(value), path=path)
        if value.dtype.kind in "SU" and _has_trailing_nul(value):
            _raise_not_transportable(
                f"numpy {value.dtype.name} with trailing NUL characters", path=path
            )
        # HTTP wraps a bytes scalar's value in a bytes marker.
        scalar_levels = 2 if value.dtype.kind == "S" else 1
        _validate_wire_depth(depth + scalar_levels, path=path)
        return
    if value_type is np.ndarray:
        _validate_ndarray(value, path=path, description="numpy array")
        _validate_wire_depth(depth + 1 + value.ndim, path=path)
        return
    if value_type is detections_type:
        _validate_detections(value, path=path, depth=depth)
        return
    if value_type is image_type:
        _validate_wire_depth(depth + _IMAGE_WIRE_LEVELS, path=path)
        return
    if value_type is video_metadata_type:
        _validate_wire_depth(depth + _VIDEO_METADATA_WIRE_LEVELS, path=path)
        return
    if value_type is dict:
        _validate_wire_depth(depth, path=path)
        _validate_dict(value, path=path, depth=depth)
        return
    if value_type in (list, tuple):
        item_depth = depth + (2 if value_type is tuple else 1)
        _validate_wire_depth(item_depth - 1, path=path)
        for index, item in enumerate(value):
            _validate_value(item, path=f"{path}[{index}]", depth=item_depth)
        return

    _raise_not_transportable(_describe_type(value_type), path=path)


def _validate_wire_depth(depth: int, *, path: str) -> None:
    if depth > _MAX_ECHO_DEPTH:
        raise DynamicBlockError(
            public_message=(
                f"Custom Python block state `self.{path}` is nested more than "
                f"{_MAX_ECHO_DEPTH} levels deep on the wire, which cannot be "
                "carried across remote execution; flatten it."
            ),
            context="dynamic_block | block_state_validation",
        )


def _has_trailing_nul(value: Any) -> bool:
    # numpy keeps the declared width but drops trailing NULs from the value,
    # so a wider dtype than the value means NULs that would not come back.
    width = 4 if value.dtype.kind == "U" else 1
    has_trailing_nul = value.dtype.itemsize > len(value.item()) * width

    return has_trailing_nul


@lru_cache(maxsize=1)
def _workflow_object_types() -> Tuple[type, type, type]:
    import supervision as sv
    from roboflow_workflows.execution_engine.entities.base import (
        VideoMetadata,
        WorkflowImageData,
    )

    return sv.Detections, WorkflowImageData, VideoMetadata


def _validate_datetime(value: datetime, *, path: str) -> None:
    from zoneinfo import ZoneInfo

    tzinfo = value.tzinfo
    if tzinfo is None or type(tzinfo) is timezone:
        return
    if type(tzinfo) is ZoneInfo and tzinfo.key is not None:
        return
    description = (
        "a ZoneInfo loaded from a file, without a key"
        if type(tzinfo) is ZoneInfo
        else f"a {type(tzinfo).__name__} time zone"
    )
    raise DynamicBlockError(
        public_message=(
            f"Custom Python block state `self.{path}` is a datetime with "
            f"{description}, which cannot be carried across remote execution; "
            "use `zoneinfo.ZoneInfo(key)` or `datetime.timezone`."
        ),
        context="dynamic_block | block_state_validation",
    )


def _describe_type(value_type: type) -> str:
    numpy_bases = [
        base for base in value_type.__mro__ if base in CARRIED_NUMPY_SCALAR_TYPES
    ]
    builtin_bases = [
        base
        for base in (bool, int, float, str, bytes, dict, list, tuple, datetime)
        if issubclass(value_type, base)
    ]
    base_names = [base.__name__ for base in numpy_bases + builtin_bases]
    if base_names:
        description = (
            f"{value_type.__name__} (a {base_names[0]} subclass, which would "
            f"come back as a plain {base_names[0]})"
        )
        return description

    return value_type.__name__


def _validate_dict(value: Dict[Any, Any], *, path: str, depth: int) -> None:
    _validate_dict_keys(value, path=path)
    for key, item in value.items():
        _validate_value(item, path=f"{path}[{key!r}]", depth=depth + 1)


def _validate_dict_keys(value: Dict[Any, Any], *, path: str) -> None:
    for key in value:
        if type(key) not in (str, np.str_):
            raise DynamicBlockError(
                public_message=(
                    f"Custom Python block state `self.{path}` has a "
                    f"non-string key of type {type(key).__name__}; only string "
                    "keys can be "
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
    if value.get("type") == "base64" and "value" in value:
        raise DynamicBlockError(
            public_message=(
                f"Custom Python block state `self.{path}` is a dict shaped "
                'like a serialized image (`type` "base64" with a `value`), '
                "which the transport would decode as an image; rename the keys "
                "or keep the image itself on `self`."
            ),
            context="dynamic_block | block_state_validation",
        )


def _validate_ndarray(value: np.ndarray, *, path: str, description: str) -> None:
    if value.dtype.kind not in _CARRIED_NDARRAY_KINDS:
        _raise_not_transportable(f"{description} of dtype {value.dtype}", path=path)
    if value.dtype.kind in "iu" and value.size:
        _validate_int(int(value.min()), path=path)
        _validate_int(int(value.max()), path=path)


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


def _validate_detections(value: Any, *, path: str, depth: int) -> None:
    _validate_wire_depth(depth + _DETECTIONS_WIRE_LEVELS, path=path)
    if len(value) == 0:
        return
    missing = []
    if value.confidence is None:
        missing.append("confidence")
    if value.class_id is None:
        missing.append("class_id")
    for key in ("class_name", DETECTION_ID_KEY):
        if key not in value.data:
            missing.append(f"data[{key!r}]")
    if IMAGE_DIMENSIONS_KEY not in value.data and not _rle_masks_carry_size(value):
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
    for field, array in (
        ("class_id", value.class_id),
        ("tracker_id", value.tracker_id),
    ):
        if array is not None:
            _validate_detections_field(array, path=f"{path}.{field}", depth=depth + 2)
    # Only the keys the transport carries: the others are dropped on the way
    # (and noted), so their contents never reach a decoder.
    for key, item in value.data.items():
        if key in CARRIED_DETECTIONS_DATA_KEYS:
            _validate_detections_field(
                item, path=f"{path}.data[{key!r}]", depth=depth + 2
            )
        if key in _PASSED_THROUGH_DETECTIONS_DATA_KEYS and isinstance(
            item, (np.ndarray, list)
        ):
            for row in item:
                _validate_row_arrays(row, path=f"{path}.data[{key!r}]")


def _validate_row_arrays(value: Any, *, path: str) -> None:
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in _CARRIED_NDARRAY_KINDS:
            raise DynamicBlockError(
                public_message=(
                    f"Custom Python block state `self.{path}` holds a numpy "
                    f"array of dtype {value.dtype} per detection, which the "
                    "transport cannot carry inside sv.Detections data; use "
                    "bool/int/float/str arrays there."
                ),
                context="dynamic_block | block_state_validation",
            )
        return
    if isinstance(value, dict):
        for item in value.values():
            _validate_row_arrays(item, path=path)
        return
    if isinstance(value, list):
        for item in value:
            _validate_row_arrays(item, path=path)


def _validate_detections_field(value: Any, *, path: str, depth: int) -> None:
    """Validate a carried Detections field by value, whatever its storage.

    The detections serializer passes these payloads through as they are, so
    the state markers do not apply inside them: a tuple would come back as a
    list and is refused, and so are the dicts the decoders would read as a
    typed value. ``depth`` is that of the per-detection list a field stands
    for, one level above each detection's value in the serialized form.
    """
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "iu" and value.size:
            _validate_int(int(np.max(value)), path=path)
            _validate_int(int(np.min(value)), path=path)
        elif value.dtype.kind == "O":
            for item in value.ravel():
                _validate_detections_field(
                    item, path=path, depth=depth + 1 + value.ndim
                )
        _validate_wire_depth(depth + 1 + value.ndim, path=path)
        return
    if isinstance(value, tuple):
        raise DynamicBlockError(
            public_message=(
                f"Custom Python block state `self.{path}` holds a tuple inside "
                "sv.Detections data, which the transport carries as a list; "
                "use a list there."
            ),
            context="dynamic_block | block_state_validation",
        )
    if isinstance(value, dict):
        _validate_dict_keys(value, path=path)
        _validate_wire_depth(depth, path=path)
        for key, item in value.items():
            _validate_detections_field(item, path=f"{path}[{key!r}]", depth=depth + 1)
        return
    if isinstance(value, list):
        _validate_wire_depth(depth, path=path)
        for index, item in enumerate(value):
            _validate_detections_field(item, path=f"{path}[{index}]", depth=depth + 1)
        return
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        _validate_int(int(value), path=path)
    _validate_wire_depth(depth + 1, path=path)


def _rle_masks_carry_size(detections: Any) -> bool:
    rle_masks = detections.data.get(RLE_MASK_KEY_IN_SV_DETECTIONS)
    if rle_masks is None:
        return False
    carry_size = all(
        isinstance(rle, dict)
        and isinstance(rle.get("size"), (list, tuple, np.ndarray))
        and len(rle["size"]) == 2
        for rle in rle_masks
    )

    return carry_size


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
