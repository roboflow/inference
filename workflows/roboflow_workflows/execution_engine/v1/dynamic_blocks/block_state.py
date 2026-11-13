"""``self`` state of a custom Python block across remote execution.

Under ``WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE=modal`` the sandbox cannot keep
the engine-side block instance, so every call sends the instance's user
attributes as JSON, the sandbox seeds ``self`` with them and echoes the state
the function left behind, and the engine writes it back onto the instance.

Only plain JSON values are carried: ``None``, ``bool``, ``int`` in the signed
64-bit range, finite ``float``, ``str``, ``list`` and ``dict`` with ``str``
keys, as exact types, nested at most 32 levels and at most 1 MiB encoded.
Anything else withholds the whole state for that call, so the next call
starts from an empty ``self``. The echo is treated as untrusted input:
problems with it clear the state with a warning and never fail the call.
"""

import hashlib
import json
import math
from typing import Any, Dict, FrozenSet, Optional, Tuple

from roboflow_workflows._compat_names import get_logger

logger = get_logger(__name__)

# Engine-set attributes; every other ``__dict__`` name is user state.
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

BLOCK_STATE_MAX_BYTES = 1024 * 1024
BLOCK_STATE_MAX_DEPTH = 32
_MAX_ECHO_LENGTH = 2 * BLOCK_STATE_MAX_BYTES
_MIN_INT = -(2**63)
_MAX_INT = 2**63 - 1
_MAX_INT_TOKEN_LENGTH = 20
_MAX_WARNED_NAME_LENGTH = 64
_MAX_WARNINGS = 1024
_WARNED: set = set()
_EMPTY_STATE_JSON = "{}"


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


def encode_block_state(block_type_name: str, state: Dict[str, Any]) -> str:
    """Encode user attributes for the request, or ``"{}"`` if they cannot be.

    Args:
        block_type_name: Block type, named in the warning.
        state: The block instance's user attributes.

    Returns:
        Compact JSON text of ``state``; ``"{}"`` with a warning when it does
        not encode as strict JSON or is over 1 MiB.
    """
    try:
        encoded = json.dumps(state, allow_nan=False, separators=(",", ":"))
    except Exception:
        _warn_state_reset(block_type_name, name="", reason="state is not plain JSON")
        return _EMPTY_STATE_JSON

    if len(encoded.encode("utf-8")) > BLOCK_STATE_MAX_BYTES:
        _warn_state_reset(block_type_name, name="", reason="state is over 1 MiB")
        return _EMPTY_STATE_JSON

    return encoded


def decode_echoed_block_state(block_type_name: str, echoed: Any) -> Dict[str, Any]:
    """Decode the state echoed by the sandbox, or ``{}`` if it is refused.

    Args:
        block_type_name: Block type, named in the warning.
        echoed: The response's ``block_state`` value; ``None`` when the server
            did not return one.

    Returns:
        The echoed user attributes; ``{}`` with a warning when the echo is
        missing, not a string, over 2 MiB, not a JSON object, or holds a value
        outside the carried types.
    """
    if echoed is None:
        _warn_state_reset(
            block_type_name, name="", reason="server did not return state"
        )
        return {}

    if not isinstance(echoed, str):
        _warn_state_reset(block_type_name, name="", reason="echo is not a string")
        return {}

    if len(echoed) > _MAX_ECHO_LENGTH:
        _warn_state_reset(block_type_name, name="", reason="echo is over 2 MiB")
        return {}

    try:
        state = json.loads(
            echoed, parse_int=_parse_bounded_int, parse_constant=_refuse_constant
        )
    except Exception:
        _warn_state_reset(block_type_name, name="", reason="echo is not valid JSON")
        return {}

    if not isinstance(state, dict):
        _warn_state_reset(block_type_name, name="", reason="echo is not an object")
        return {}

    problem = _find_out_of_contract_value(state)
    if problem is not None:
        name, reason = problem
        _warn_state_reset(block_type_name, name=name, reason=reason)
        return {}

    return state


def apply_block_state(block: Any, state: Dict[str, Any]) -> None:
    """Replace the user attributes of a dynamic block instance with ``state``.

    Attributes absent from ``state`` are removed, so a ``del self.x`` in the
    sandbox takes effect here too; engine attributes are kept. Every name is
    checked before the instance is touched: a name that is not a string, is
    an engine attribute, a dunder, or defined on any class of the block's MRO
    clears all user attributes instead, with a warning.

    Args:
        block: The engine-side dynamic block instance.
        state: The user attributes echoed by the sandbox.
    """
    refused_name = next(
        (name for name in state if not _is_assignable_name(name, block=block)),
        None,
    )

    instance_attributes = vars(block)
    engine_attributes = {
        name: value
        for name, value in instance_attributes.items()
        if name in ENGINE_ATTRIBUTE_NAMES
    }
    instance_attributes.clear()
    instance_attributes.update(engine_attributes)

    if refused_name is not None:
        block_type_name = getattr(type(block), "_usage_block_type", "")
        if not isinstance(refused_name, str):
            _warn_state_reset(
                block_type_name,
                name=type(refused_name).__name__,
                reason="attribute name is not a string",
            )
            return

        _warn_state_reset(
            block_type_name,
            name=refused_name,
            reason="attribute name is reserved by the block or the engine",
        )
        return

    instance_attributes.update(state)


def _is_assignable_name(name: Any, *, block: Any) -> bool:
    if not isinstance(name, str) or name in ENGINE_ATTRIBUTE_NAMES:
        return False

    if name.startswith("__") and name.endswith("__"):
        return False

    defined_on_class = any(name in vars(klass) for klass in type(block).__mro__)

    return not defined_on_class


def _parse_bounded_int(token: str) -> int:
    if len(token) > _MAX_INT_TOKEN_LENGTH:
        raise ValueError("integer token is too long")

    return int(token)


def _refuse_constant(token: str) -> None:
    raise ValueError(f"{token} is not carried")


def _find_out_of_contract_value(
    state: Dict[str, Any],
) -> Optional[Tuple[str, str]]:
    stack = [(name, value, 1) for name, value in state.items()]
    while stack:
        name, value, depth = stack.pop()
        if depth > BLOCK_STATE_MAX_DEPTH:
            return name, "value is nested deeper than 32 levels"

        if isinstance(value, bool) or value is None or isinstance(value, str):
            continue

        if isinstance(value, int):
            if not _MIN_INT <= value <= _MAX_INT:
                return name, "int is outside the signed 64-bit range"
            continue

        if isinstance(value, float):
            if not math.isfinite(value):
                return name, "float is not finite"
            continue

        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                return name, "dict key is not a string"
            stack.extend((name, item, depth + 1) for item in value.values())
            continue

        stack.extend((name, item, depth + 1) for item in value)

    return None


def _short_name(name: str) -> str:
    if len(name) <= _MAX_WARNED_NAME_LENGTH:
        return name

    digest = hashlib.sha1(name.encode("utf-8", "surrogatepass")).hexdigest()[:8]
    short_name = f"{name[:_MAX_WARNED_NAME_LENGTH]}…{digest}"

    return short_name


def _warn_state_reset(block_type_name: str, *, name: str, reason: str) -> None:
    short_name = _short_name(name)
    key = (block_type_name, short_name, reason)
    if key in _WARNED:
        return

    if len(_WARNED) >= _MAX_WARNINGS:
        _WARNED.clear()
    _WARNED.add(key)

    subject = f"self.{short_name}: " if short_name else ""
    logger.warning(
        "Custom Python block `%s`: %s%s; state reset.",
        block_type_name,
        subject,
        reason,
    )
