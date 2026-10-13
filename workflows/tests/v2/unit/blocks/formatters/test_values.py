"""``v2/first_non_empty_or_default@v1`` and ``v2/current_time@v1``."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError
from roboflow_workflows.v2.blocks.formatters.values import (
    TIMESTAMP_KIND,
    CurrentTimeBlock,
    FirstNonEmptyOrDefaultBlock,
)


def _first(values: list, **params) -> object:
    validated = FirstNonEmptyOrDefaultBlock.Params(
        data=[f"$steps.s{index}.output" for index in range(len(values))], **params
    )
    result = FirstNonEmptyOrDefaultBlock().run(data=values, default=validated.default)

    return result["output"]


def _time_at(moment: datetime, timezone_name: str) -> dict:
    params = CurrentTimeBlock.Params(timezone=timezone_name)
    block = CurrentTimeBlock(wall_clock=lambda tz: moment.astimezone(tz))

    return block.run(timezone=params.timezone)


def test_the_first_value_that_is_not_none_wins() -> None:
    assert _first([None, "b", "c"], default="d") == "b"


def test_empty_containers_and_falsy_values_are_values() -> None:
    assert _first([None, [], "c"]) == []
    assert _first([0, "b"]) == 0


def test_all_none_returns_the_default() -> None:
    assert _first([None, None], default="N/A") == "N/A"
    assert _first([None]) is None


def test_the_default_is_a_literal_even_when_it_looks_like_a_selector() -> None:
    assert _first([None], default="$inputs.x") == "$inputs.x"


def test_data_needs_at_least_one_selector() -> None:
    with pytest.raises(ValidationError, match="at least 1 item"):
        FirstNonEmptyOrDefaultBlock.Params(data=[])


def test_utc_is_the_default_timezone() -> None:
    moment = datetime(2026, 7, 1, 10, 0, 5, tzinfo=timezone.utc)

    result = _time_at(moment, CurrentTimeBlock.Params().timezone)

    assert result == {
        "timestamp": moment,
        "iso_string": "2026-07-01T10:00:05+00:00",
        "date": "2026-07-01",
        "time": "10:00:05",
    }
    assert result["timestamp"].tzinfo == ZoneInfo("UTC")


@pytest.mark.parametrize(
    ("moment", "zone", "iso"),
    [
        # Berlin: CEST (+02:00) in summer, CET (+01:00) in winter.
        (datetime(2026, 7, 1, 10, 0), "Europe/Berlin", "2026-07-01T12:00:00+02:00"),
        (datetime(2026, 1, 15, 10, 0), "Europe/Berlin", "2026-01-15T11:00:00+01:00"),
        # Half and quarter hour offsets.
        (datetime(2026, 1, 15, 10, 0), "Asia/Kolkata", "2026-01-15T15:30:00+05:30"),
        (datetime(2026, 1, 15, 10, 0), "Asia/Kathmandu", "2026-01-15T15:45:00+05:45"),
        # Local date differs from the UTC date.
        (datetime(2026, 3, 31, 23, 30), "Asia/Tokyo", "2026-04-01T08:30:00+09:00"),
        (
            datetime(2026, 4, 1, 2, 0),
            "America/Los_Angeles",
            "2026-03-31T19:00:00-07:00",
        ),
        # The first UTC instant after New York springs forward (2026-03-08).
        (datetime(2026, 3, 8, 7, 0), "America/New_York", "2026-03-08T03:00:00-04:00"),
    ],
)
def test_local_time_follows_the_timezone_offset_and_dst(
    moment: datetime, zone: str, iso: str
) -> None:
    result = _time_at(moment.replace(tzinfo=timezone.utc), zone)

    assert result["iso_string"] == iso
    assert result["date"] == iso[:10]
    assert result["time"] == iso[11:19]


def test_the_wall_clock_receives_the_zone_object() -> None:
    zones = []

    def clock(tz):
        zones.append(tz)
        return datetime(2026, 7, 1, tzinfo=timezone.utc).astimezone(tz)

    CurrentTimeBlock(wall_clock=clock).run(timezone="Asia/Tokyo")

    assert zones == [ZoneInfo("Asia/Tokyo")]


def test_the_default_wall_clock_reads_real_aware_time() -> None:
    before = datetime.now(timezone.utc)

    result = CurrentTimeBlock().run(timezone="UTC")

    assert before <= result["timestamp"] <= datetime.now(timezone.utc)
    assert TIMESTAMP_KIND.validate(result["timestamp"])


@pytest.mark.parametrize("zone", ["Europe/Paris", "Mars/Base", "utc"])
def test_timezones_outside_the_curated_list_are_rejected(zone: str) -> None:
    with pytest.raises(ValidationError, match="unsupported timezone.*curated"):
        CurrentTimeBlock.Params(timezone=zone)


def test_a_selected_timezone_is_accepted_for_later_validation() -> None:
    assert CurrentTimeBlock.Params(timezone="$inputs.tz").timezone == "$inputs.tz"
