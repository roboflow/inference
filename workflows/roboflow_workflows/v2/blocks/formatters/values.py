"""Value blocks: first available value and the current civil time.

Ports of V1 ``roboflow_core/first_non_empty_or_default@v1`` and
``roboflow_core/current_time@v1``::

    first_non_empty_or_default:  [a, b, c] ──▶ first value that is not None, else default
    current_time:                timezone  ──▶ timestamp, iso_string, date, time

``first_non_empty_or_default`` runs even when its inputs are missing, filtered
or gated, so it merges the outputs of conditional branches. As in V1, only
``None`` counts as empty: an empty list or detection set is a value.

``current_time`` reads civil time from the ``wall_clock`` constructor
resource, ``datetime.now`` by default. A session can inject another clock::

    session = plan.create_session({"wall_clock": lambda tz: fixed.astimezone(tz)})

The timezone must be one of V1's curated IANA names (``TIMEZONE_OPTIONS``).
"""

import re
from datetime import datetime, tzinfo
from typing import Any, Callable, Dict, List, Literal
from zoneinfo import ZoneInfo

from pydantic import Field, field_validator
from roboflow_workflows.core_steps.formatters.current_time.v1 import TIMEZONE_OPTIONS
from roboflow_workflows.v2.data.kinds import STRING_KIND, Kind
from roboflow_workflows.v2.engine.declaration import (
    DATA_SELECTOR_PATTERN,
    Block,
    BlockParams,
    Output,
    Ref,
)

WallClock = Callable[[tzinfo], datetime]
"""Return the current moment as an aware ``datetime`` in the given timezone."""

ALLOWED_TIMEZONES = tuple(name for name, _ in TIMEZONE_OPTIONS)
TimezoneName = Literal[ALLOWED_TIMEZONES]  # type: ignore[valid-type]


def _is_timestamp(payload: Any) -> bool:
    return isinstance(payload, datetime)


TIMESTAMP_KIND = Kind(
    name="timestamp",
    description="datetime.datetime; timezone-aware when a block says so.",
    validate=_is_timestamp,
)


class FirstNonEmptyOrDefaultBlock(Block):
    """Return the first input that is not ``None``, else ``default``.

    Example: ``data=["$steps.branch_a.output", "$steps.branch_b.output"]``
    outputs branch B's value when the gate skipped branch A.
    """

    type = "v2/first_non_empty_or_default@v1"
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


class CurrentTimeBlock(Block):
    """Report the current time in a curated IANA timezone.

    Example: ``timezone="Europe/Berlin"`` at 2026-07-01 10:00 UTC outputs
    ``iso_string="2026-07-01T12:00:00+02:00"``, ``date="2026-07-01"`` and
    ``time="12:00:00"``.
    """

    type = "v2/current_time@v1"
    metadata = {
        "name": "Current Time",
        "port_of": "roboflow_core/current_time@v1",
        "section": "advanced",
    }
    outputs = {
        "timestamp": Output(
            TIMESTAMP_KIND, description="The current moment, a timezone-aware datetime."
        ),
        "iso_string": Output(
            STRING_KIND, description="ISO 8601 text with the UTC offset."
        ),
        "date": Output(STRING_KIND, description="Local date, YYYY-MM-DD."),
        "time": Output(STRING_KIND, description="Local time, HH:MM:SS."),
    }

    class Params(BlockParams):
        timezone: TimezoneName | Ref(STRING_KIND) = Field(
            default="UTC",
            description="Curated IANA timezone name, e.g. America/New_York.",
            examples=["UTC", "Europe/Berlin", "$inputs.timezone"],
        )

        @field_validator("timezone", mode="before")
        @classmethod
        def _timezone_is_curated(cls, value: Any) -> Any:
            # Run before the Literal union for a short error; the engine also
            # runs this check after resolving a selected timezone.
            if not isinstance(value, str):
                return value
            if value not in ALLOWED_TIMEZONES and not re.fullmatch(
                DATA_SELECTOR_PATTERN, value
            ):
                raise ValueError(
                    f"unsupported timezone {value!r}; use one of the curated "
                    f"names, e.g. UTC, Europe/Berlin, America/New_York"
                )

            return value

    def __init__(self, *, wall_clock: WallClock = datetime.now) -> None:
        self._wall_clock = wall_clock

    def run(self, *, timezone: str) -> Dict[str, Any]:
        """Read the wall clock in ``timezone``.

        Args:
            timezone: Curated IANA timezone name, validated by ``Params``.

        Returns:
            ``timestamp``, ``iso_string``, ``date`` and ``time``.

        """
        now = self._wall_clock(ZoneInfo(timezone))

        return {
            "timestamp": now,
            "iso_string": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
        }
