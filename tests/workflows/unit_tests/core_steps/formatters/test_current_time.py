from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.formatters.current_time.v1 import (
    ALLOWED_TIMEZONES,
    TIMEZONE_METADATA,
    BlockManifest,
    CurrentTimeBlockV1,
)


def test_manifest_parsing_when_data_is_valid() -> None:
    # given
    data = {
        "type": "roboflow_core/current_time@v1",
        "name": "now",
        "timezone": "America/New_York",
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result.type == "roboflow_core/current_time@v1"
    assert result.name == "now"
    assert result.timezone == "America/New_York"


def test_manifest_parsing_defaults_to_utc() -> None:
    # given
    data = {"type": "roboflow_core/current_time@v1", "name": "now"}

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result.timezone == "UTC"


def test_manifest_accepts_selector_for_timezone() -> None:
    # given
    data = {
        "type": "roboflow_core/current_time@v1",
        "name": "now",
        "timezone": "$inputs.timezone",
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result.timezone == "$inputs.timezone"


def test_manifest_schema_exposes_timezone_dropdown_options() -> None:
    # when
    timezone_schema = BlockManifest.model_json_schema()["properties"]["timezone"]
    timezone_enum = next(
        item["enum"] for item in timezone_schema["anyOf"] if "enum" in item
    )

    # then
    assert timezone_enum == list(ALLOWED_TIMEZONES)
    assert len(timezone_enum) == 55
    assert "UTC" in timezone_enum
    assert "America/New_York" in timezone_enum
    assert "America/Bogota" in timezone_enum
    assert "Europe/Berlin" in timezone_enum
    assert "Europe/London" in timezone_enum
    assert "Africa/Lagos" in timezone_enum
    assert "Australia/Darwin" in timezone_enum
    assert "Australia/Adelaide" in timezone_enum
    assert "Europe/Paris" not in timezone_enum


def test_manifest_schema_exposes_timezone_dropdown_metadata() -> None:
    # when
    timezone_schema = BlockManifest.model_json_schema()["properties"]["timezone"]

    # then
    assert timezone_schema["values_metadata"] == TIMEZONE_METADATA
    assert (
        timezone_schema["values_metadata"]["America/New_York"]["name"]
        == "UTC-5/-4 Eastern Time (EST/EDT)"
    )
    assert "description" not in timezone_schema["values_metadata"]["America/New_York"]
    assert "-8" in timezone_schema["values_metadata"]["America/Los_Angeles"]["name"]
    assert "PST" in timezone_schema["values_metadata"]["America/Los_Angeles"]["name"]
    assert timezone_schema["values_metadata"]["Africa/Lagos"]["name"] == (
        "UTC+1 West Africa Time (WAT)"
    )
    assert timezone_schema["values_metadata"]["Australia/Darwin"]["name"] == (
        "UTC+9:30 Australian Central Standard Time (ACST)"
    )
    assert timezone_schema["values_metadata"]["Australia/Adelaide"]["name"] == (
        "UTC+9:30/+10:30 Australian Central Time (ACST/ACDT)"
    )


def test_manifest_parsing_when_type_is_invalid() -> None:
    # given
    data = {"type": "roboflow_core/not_current_time@v1", "name": "now"}

    # when / then
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(data)


def test_manifest_parsing_rejects_timezone_outside_curated_options() -> None:
    # given
    data = {
        "type": "roboflow_core/current_time@v1",
        "name": "now",
        "timezone": "Europe/Paris",
    }

    # when / then
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(data)


def test_run_returns_consistent_timestamp_for_valid_timezone() -> None:
    # given
    block = CurrentTimeBlockV1()

    # when
    result = block.run(timezone="America/New_York")

    # then
    assert set(result.keys()) == {"timestamp", "iso_string", "date", "time"}
    assert isinstance(result["timestamp"], datetime)
    assert result["timestamp"].tzinfo is not None
    # the derived strings agree with the datetime object
    assert result["iso_string"] == result["timestamp"].isoformat()
    assert result["date"] == result["timestamp"].strftime("%Y-%m-%d")
    assert result["time"] == result["timestamp"].strftime("%H:%M:%S")


def test_curated_options_distinguish_dst_and_fixed_timezone_rules() -> None:
    # given
    january = datetime(2026, 1, 15, 12)
    july = datetime(2026, 7, 15, 12)

    # when / then
    assert january.replace(tzinfo=ZoneInfo("America/New_York")).utcoffset() != (
        july.replace(tzinfo=ZoneInfo("America/New_York")).utcoffset()
    )
    assert january.replace(tzinfo=ZoneInfo("America/Bogota")).utcoffset() == (
        july.replace(tzinfo=ZoneInfo("America/Bogota")).utcoffset()
    )
    assert january.replace(tzinfo=ZoneInfo("Europe/Berlin")).utcoffset() != (
        july.replace(tzinfo=ZoneInfo("Europe/Berlin")).utcoffset()
    )
    assert january.replace(tzinfo=ZoneInfo("Africa/Lagos")).utcoffset() == (
        july.replace(tzinfo=ZoneInfo("Africa/Lagos")).utcoffset()
    )
    assert january.replace(tzinfo=ZoneInfo("Australia/Darwin")).utcoffset() == (
        july.replace(tzinfo=ZoneInfo("Australia/Darwin")).utcoffset()
    )
    assert january.replace(tzinfo=ZoneInfo("Australia/Adelaide")).utcoffset() != (
        july.replace(tzinfo=ZoneInfo("Australia/Adelaide")).utcoffset()
    )


def test_run_raises_for_unknown_timezone() -> None:
    # given
    block = CurrentTimeBlockV1()

    # when / then
    with pytest.raises(ValueError):
        block.run(timezone="Mars/Olympus_Mons")


def test_run_raises_for_timezone_outside_curated_options() -> None:
    # given
    block = CurrentTimeBlockV1()

    # when / then
    with pytest.raises(ValueError):
        block.run(timezone="Europe/Paris")
