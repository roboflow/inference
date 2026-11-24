"""Per-source state and input checks shared by the sampling blocks.

State keys::

    ("global",)                         scope="global": V1's one state
    ("source", SampleContext.source_id)  scope="source" and the embedding has one
    ("position", index[:1])              otherwise (passive fallback)

The source is the ``embedding`` input's own sample context, so a selected
parameter without one (e.g. a step output computed from a literal) does not
hide it.

Every check here runs before a block touches its states, so a rejected call
changes nothing and does not take a ``max_sources`` slot.
"""

import threading
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    Generic,
    Hashable,
    Literal,
    Optional,
    TypeVar,
)

import torch
from pydantic import Field, StrictFloat, StrictInt
from roboflow_workflows.v2.engine.declaration import Block

__all__ = [
    "DEFAULT_MAX_SOURCES",
    "SamplingCapacityError",
    "Scope",
    "SourceStates",
    "UnitFloat",
    "WindowSize",
    "check_fraction",
    "check_like",
    "check_window_size",
    "state_key",
    "unit_vector",
]

DEFAULT_MAX_SOURCES = 256

Scope = Literal["source", "global"]
# Literal values are checked when the workflow compiles; selected ones in run().
UnitFloat = Annotated[StrictFloat, Field(ge=0, le=1)]
WindowSize = Annotated[StrictInt, Field(ge=1)]
State = TypeVar("State")


class SamplingCapacityError(RuntimeError):
    """A new source would exceed a sampling block's ``max_sources``."""


class SourceStates(Generic[State]):
    """Per-source states of one sampling block; at most ``max_sources`` of them.

    Callers hold ``lock`` around ``get``, ``add``, ``clear`` and the update of
    the returned state, so each update is whole.

    Args:
        block_type: Block type named in errors.
        create: Makes the empty state of a new source.
    """

    def __init__(self, block_type: str, *, create: Callable[[], State]) -> None:
        self.lock = threading.Lock()
        self._block_type = block_type
        self._create = create
        self._states: Dict[Hashable, State] = {}

    def get(self, key: Hashable) -> Optional[State]:
        """Return the state of a source.

        Args:
            key: Source key from ``state_key``.

        Returns:
            The state, or ``None`` before the source's first call.
        """
        state = self._states.get(key)

        return state

    def add(self, key: Hashable, *, max_sources: int) -> State:
        """Create and keep the empty state of a new source.

        Args:
            key: Source key from ``state_key``, not kept yet.
            max_sources: Most states this block may keep.

        Returns:
            The new state.

        Raises:
            SamplingCapacityError: When ``max_sources`` states exist already.
        """
        if len(self._states) >= max_sources:
            raise SamplingCapacityError(
                f"{self._block_type}: source {key!r} would be source "
                f"{len(self._states) + 1} of max_sources={max_sources}; raise "
                "max_sources, use scope='global', or reset processing to drop "
                "ended sources"
            )

        state = self._create()
        self._states[key] = state

        return state

    def clear(self) -> None:
        """Forget every source's state."""
        self._states.clear()


def state_key(block: Block, *, scope: str) -> Hashable:
    """Return the key of the state that the running call of ``block`` updates.

    Args:
        block: Sampling block whose ``run`` is executing; its Params field
            ``embedding`` holds the input that names the source.
        scope: ``"global"`` for one shared key, ``"source"`` for the
            embedding's source, or its top-level batch position when the
            embedding has no sample context.

    Returns:
        ``("global",)``, ``("source", source_id)`` or ``("position", index[:1])``.
    """
    if scope == "global":
        return ("global",)

    context = block.execution_context
    index = context.indices[0]
    sample = context.sample_at(index, field="embedding")
    if sample is not None:
        key = ("source", sample.source_id)
    else:
        key = ("position", index[:1])

    return key


def unit_vector(embedding: torch.Tensor) -> torch.Tensor:
    """Check an embedding and scale it to length 1; a zero vector stays zero.

    The finite check reuses the norm: it is NaN or Inf when any element is,
    and also when finite elements overflow the norm's dtype (e.g. 1e30 in
    float32), which is rejected too. Reading it is one device-to-host sync.

    Args:
        embedding: Non-empty 1-D floating tensor; not changed or kept.

    Returns:
        A new detached tensor of the same dtype and device.

    Raises:
        ValueError: When ``embedding`` is not a non-empty 1-D floating tensor
            or has a NaN, Inf or overflowing norm.
    """
    is_vector = (
        isinstance(embedding, torch.Tensor)
        and embedding.dim() == 1
        and embedding.numel() > 0
        and embedding.is_floating_point()
    )
    if not is_vector:
        found = (
            f"{tuple(embedding.shape)} {embedding.dtype}"
            if isinstance(embedding, torch.Tensor)
            else type(embedding).__name__
        )
        raise ValueError(
            f"embedding must be a non-empty 1-D floating tensor, got {found}"
        )

    embedding = embedding.detach()
    norm = torch.linalg.norm(embedding)
    if not bool(torch.isfinite(norm)):
        raise ValueError(
            f"embedding must be finite with a finite norm, got norm {float(norm)}"
        )

    # Dividing a zero vector by 1 keeps it zero without a host sync.
    unit = embedding / torch.where(norm == 0, torch.ones_like(norm), norm)

    return unit


def check_like(embedding: torch.Tensor, *, kept: torch.Tensor, block_type: str) -> None:
    """Fail when an embedding cannot join a source state that holds ``kept``.

    Args:
        embedding: Unit embedding of the current call.
        kept: Any tensor already in the source's state.
        block_type: Block type named in the error.

    Raises:
        ValueError: When the length, dtype or device differs.
    """
    if (
        embedding.shape != kept.shape
        or embedding.dtype != kept.dtype
        or embedding.device != kept.device
    ):
        raise ValueError(
            f"{block_type}: embedding {tuple(embedding.shape)} {embedding.dtype} "
            f"on {embedding.device} does not match this source's earlier "
            f"embeddings {tuple(kept.shape)} {kept.dtype} on {kept.device}; "
            "reset processing or use another source to change the embedding model"
        )


def check_fraction(value: Any, *, name: str) -> float:
    """Check a literal or selected parameter that must lie from 0 to 1.

    Args:
        value: Parameter value.
        name: Parameter name for the error.

    Returns:
        ``value`` as a float.

    Raises:
        ValueError: When ``value`` is not a number from 0 to 1.
    """
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not is_number or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a number from 0 to 1, got {value!r}")

    fraction = float(value)

    return fraction


def check_window_size(value: Any) -> int:
    """Check a literal or selected ``window_size``.

    Args:
        value: Parameter value.

    Returns:
        ``value`` unchanged.

    Raises:
        ValueError: When ``value`` is not an integer of at least 1.
    """
    is_integer = isinstance(value, int) and not isinstance(value, bool)
    if not is_integer or value < 1:
        raise ValueError(f"window_size must be an integer of at least 1, got {value!r}")

    return value
