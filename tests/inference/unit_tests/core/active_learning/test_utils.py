from datetime import datetime
from unittest import mock
from unittest.mock import MagicMock

from inference.core.active_learning import utils
from inference.core.active_learning.utils import (
    generate_start_timestamp_for_this_month,
    generate_start_timestamp_for_this_week,
    generate_today_timestamp,
)


@mock.patch.object(utils, "datetime")
def test_generate_today_timestamp(datetime_mock: MagicMock) -> None:
    # given
    datetime_mock.today.return_value = datetime(year=1986, month=6, day=22)

    # when
    result = generate_today_timestamp()

    # then
    assert result == "1986_06_22"


@mock.patch.object(utils, "datetime")
def test_generate_start_timestamp_for_this_week(datetime_mock: MagicMock) -> None:
    # given
    datetime_mock.today.return_value = datetime(year=2023, month=10, day=26)

    # when
    result = generate_start_timestamp_for_this_week()

    # then
    assert result == "2023_10_23"


@mock.patch.object(utils, "datetime")
def test_generate_start_timestamp_for_this_month(datetime_mock: MagicMock) -> None:
    # given
    datetime_mock.today.return_value = datetime(year=2023, month=10, day=26)

    # when
    result = generate_start_timestamp_for_this_month()

    # then
    assert result == "2023_10_01"
