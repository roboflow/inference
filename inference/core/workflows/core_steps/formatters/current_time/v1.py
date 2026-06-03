from datetime import datetime
from typing import Dict, List, Literal, Optional, Type, Union

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:
    from backports.zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    STRING_KIND,
    TIMESTAMP_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Output the current date and time for a given timezone.

Provide one of the curated timezone options (for example `America/New_York`,
`Europe/Berlin`, or `UTC`) and the block returns the current moment in that
timezone. The block produces a `timestamp` (a timezone-aware `datetime` object you
can pass to other blocks), along with ready-to-use `iso_string`, `date`, and
`time` strings.

The timezone may be a literal value typed into the block, or a reference to a workflow
input or another step's output.
"""

SHORT_DESCRIPTION = "Output the current date and time for a given timezone."
TIMEZONE_OPTIONS = (
    ("Etc/GMT+12", "UTC-12 International Date Line West"),
    ("Pacific/Pago_Pago", "UTC-11 Samoa Time (SST)"),
    ("Pacific/Honolulu", "UTC-10 Hawaii-Aleutian Time (HST/HAST)"),
    ("Pacific/Marquesas", "UTC-9:30 Marquesas Time (MART)"),
    ("America/Anchorage", "UTC-9/-8 Alaska Time (AKST/AKDT)"),
    ("Pacific/Gambier", "UTC-9 Gambier Time (GAMT)"),
    ("America/Los_Angeles", "UTC-8/-7 Pacific Time (PST/PDT)"),
    ("America/Denver", "UTC-7/-6 Mountain Time (MST/MDT)"),
    ("America/Phoenix", "UTC-7 Mountain Standard Time (MST)"),
    ("America/Chicago", "UTC-6/-5 Central Time (CST/CDT)"),
    ("America/Mexico_City", "UTC-6 Mexico / Central America Time (CST)"),
    ("America/New_York", "UTC-5/-4 Eastern Time (EST/EDT)"),
    ("America/Bogota", "UTC-5 Colombia / Peru Time (COT/PET)"),
    ("America/Halifax", "UTC-4/-3 Atlantic Time (AST/ADT)"),
    ("America/Puerto_Rico", "UTC-4 Atlantic Standard Time (AST)"),
    ("America/St_Johns", "UTC-3:30/-2:30 Newfoundland Time (NST/NDT)"),
    ("America/Sao_Paulo", "UTC-3 Brasilia Time (BRT)"),
    ("Atlantic/South_Georgia", "UTC-2 Mid-Atlantic Time (GST)"),
    ("Atlantic/Azores", "UTC-1/+0 Azores Time (AZOT/AZOST)"),
    ("Atlantic/Cape_Verde", "UTC-1 Cape Verde Time (CVT)"),
    ("UTC", "UTC+0 Greenwich Mean Time (GMT/WET)"),
    ("Europe/London", "UTC+0/+1 UK / Western European Time (GMT/BST/WET/WEST)"),
    ("Europe/Berlin", "UTC+1/+2 Central European Time (CET/CEST)"),
    ("Africa/Lagos", "UTC+1 West Africa Time (WAT)"),
    ("Europe/Kyiv", "UTC+2/+3 Eastern European Time (EET/EEST)"),
    ("Africa/Cairo", "UTC+2/+3 Egypt Time (EET/EEST)"),
    ("Africa/Johannesburg", "UTC+2 South Africa Time (SAST)"),
    ("Europe/Moscow", "UTC+3 Moscow Time (MSK)"),
    ("Europe/Istanbul", "UTC+3 Turkey Time (TRT)"),
    ("Africa/Nairobi", "UTC+3 East Africa Time (EAT)"),
    ("Asia/Tehran", "UTC+3:30 Iran Time (IRST)"),
    ("Asia/Dubai", "UTC+4 Gulf Time (GST)"),
    ("Asia/Kabul", "UTC+4:30 Afghanistan Time (AFT)"),
    ("Asia/Karachi", "UTC+5 Pakistan Time (PKT)"),
    ("Asia/Kolkata", "UTC+5:30 India Time (IST)"),
    ("Asia/Kathmandu", "UTC+5:45 Nepal Time (NPT)"),
    ("Asia/Dhaka", "UTC+6 Bangladesh Time (BST)"),
    ("Asia/Yangon", "UTC+6:30 Myanmar Time (MMT)"),
    ("Asia/Bangkok", "UTC+7 Indochina Time (ICT)"),
    ("Asia/Shanghai", "UTC+8 China / Western Australia Time (CST/AWST/PHT)"),
    ("Australia/Eucla", "UTC+8:45 Central Western Australia Time (ACWST)"),
    ("Asia/Tokyo", "UTC+9 Japan / Korea Time (JST/KST)"),
    ("Australia/Darwin", "UTC+9:30 Australian Central Standard Time (ACST)"),
    ("Australia/Adelaide", "UTC+9:30/+10:30 Australian Central Time (ACST/ACDT)"),
    ("Australia/Sydney", "UTC+10/+11 Australian Eastern Time (AEST/AEDT)"),
    ("Pacific/Port_Moresby", "UTC+10 Papua New Guinea Time (PGT)"),
    ("Australia/Lord_Howe", "UTC+10:30/+11 Lord Howe Time (LHST/LHDT)"),
    ("Pacific/Guadalcanal", "UTC+11 Solomon Islands Time (SBT)"),
    ("Pacific/Norfolk", "UTC+11/+12 Norfolk Island Time (NFT/NFDT)"),
    ("Pacific/Fiji", "UTC+12 Fiji Time (FJT)"),
    ("Pacific/Auckland", "UTC+12/+13 New Zealand Time (NZST/NZDT)"),
    ("Pacific/Chatham", "UTC+12:45/+13:45 Chatham Time (CHAST/CHADT)"),
    ("Pacific/Tongatapu", "UTC+13 Tonga Time (TOT)"),
    ("Pacific/Apia", "UTC+13 Samoa Time (WSST)"),
    ("Pacific/Kiritimati", "UTC+14 Line Islands Time (LINT)"),
)
ALLOWED_TIMEZONES = tuple(timezone for timezone, _ in TIMEZONE_OPTIONS)
ALLOWED_TIMEZONE_SET = frozenset(ALLOWED_TIMEZONES)
TIMEZONE_METADATA: Dict[str, Dict[str, str]] = {
    timezone: {"name": label} for timezone, label in TIMEZONE_OPTIONS
}


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Current Time",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "formatter",
            "ui_manifest": {
                "section": "advanced",
                "icon": "far fa-clock",
                "blockPriority": 10,
            },
        }
    )
    type: Literal["roboflow_core/current_time@v1"]
    timezone: Union[Literal[ALLOWED_TIMEZONES], Selector(kind=[STRING_KIND])] = Field(  # type: ignore
        default="UTC",
        description="Curated IANA timezone name to report the current time in.",
        examples=["UTC", "America/New_York", "Europe/Berlin", "$inputs.timezone"],
        json_schema_extra={"values_metadata": TIMEZONE_METADATA},
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="timestamp", kind=[TIMESTAMP_KIND]),
            OutputDefinition(name="iso_string", kind=[STRING_KIND]),
            OutputDefinition(name="date", kind=[STRING_KIND]),
            OutputDefinition(name="time", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class CurrentTimeBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(self, timezone: str = "UTC") -> BlockResult:
        if timezone not in ALLOWED_TIMEZONE_SET:
            raise ValueError(
                f"`roboflow_core/current_time@v1` received unsupported timezone '{timezone}'. "
                "Provide one of the curated timezone options shown in the block dropdown."
            )
        try:
            now = datetime.now(ZoneInfo(timezone))
        except ZoneInfoNotFoundError as error:
            raise ValueError(
                f"`roboflow_core/current_time@v1` received unknown timezone '{timezone}'. "
                "Provide one of the curated timezone options shown in the block dropdown."
            ) from error
        return {
            "timestamp": now,
            "iso_string": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
        }
