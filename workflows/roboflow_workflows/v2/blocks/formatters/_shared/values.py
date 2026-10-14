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

from datetime import datetime, tzinfo
from typing import Any, Callable
from roboflow_workflows.v2.data.kinds import Kind


WallClock = Callable[[tzinfo], datetime]

def _is_timestamp(payload: Any) -> bool:
    return isinstance(payload, datetime)

TIMESTAMP_KIND = Kind(
    name="timestamp",
    description="datetime.datetime; timezone-aware when a block says so.",
    validate=_is_timestamp,
)
