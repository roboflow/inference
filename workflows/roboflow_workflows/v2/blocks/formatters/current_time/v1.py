"""Report the current time in a curated IANA timezone.

Canonical type: ``roboflow_core/current_time@v1``.
"""

import re
from datetime import datetime
from typing import Any, Dict, Literal
from zoneinfo import ZoneInfo
from pydantic import Field, field_validator
from roboflow_workflows.core_steps.formatters.current_time.v1 import TIMEZONE_OPTIONS
from roboflow_workflows.v2.blocks.formatters._shared.values import (
    TIMESTAMP_KIND,
    WallClock,
)
from roboflow_workflows.v2.data.kinds import STRING_KIND
from roboflow_workflows.v2.engine.declaration import (
    DATA_SELECTOR_PATTERN,
    Block,
    BlockParams,
    Output,
    Ref,
)


ALLOWED_TIMEZONES = tuple(name for name, _ in TIMEZONE_OPTIONS)

TimezoneName = Literal[ALLOWED_TIMEZONES]  # type: ignore[valid-type]


class CurrentTimeBlock(Block):
    """Report the current time in a curated IANA timezone.

    Example: ``timezone="Europe/Berlin"`` at 2026-07-01 10:00 UTC outputs
    ``iso_string="2026-07-01T12:00:00+02:00"``, ``date="2026-07-01"`` and
    ``time="12:00:00"``.
    """

    type = "roboflow_core/current_time@v1"
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
