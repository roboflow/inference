"""Date stamps used in labeling batch names."""

from datetime import datetime, timedelta

TIMESTAMP_FORMAT = "%Y_%m_%d"


def generate_today_timestamp() -> str:
    """Return the stamp of today.

    Returns:
        Local date formatted as ``YYYY_MM_DD``.
    """
    timestamp = datetime.today().strftime(TIMESTAMP_FORMAT)

    return timestamp


def generate_start_timestamp_for_this_week() -> str:
    """Return the stamp of the Monday of the current week.

    Returns:
        Local date formatted as ``YYYY_MM_DD``.
    """
    today = datetime.today()
    timestamp = (today - timedelta(days=today.weekday())).strftime(TIMESTAMP_FORMAT)

    return timestamp


def generate_start_timestamp_for_this_month() -> str:
    """Return the stamp of the first day of the current month.

    Returns:
        Local date formatted as ``YYYY_MM_DD``.
    """
    timestamp = datetime.today().replace(day=1).strftime(TIMESTAMP_FORMAT)

    return timestamp
