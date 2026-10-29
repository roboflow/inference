"""Wall clock type and timestamp kind shared by time-aware blocks.

``WallClock`` types the ``wall_clock`` constructor resource of
``roboflow_core/current_time@v1``, the CSV formatter and the local file and S3
sinks; each defaults to ``datetime.now``. A session can inject another clock::

    session = plan.create_session({"wall_clock": lambda tz: fixed.astimezone(tz)})

``TIMESTAMP_KIND`` is the kind of ``current_time``'s ``timestamp`` output.
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
