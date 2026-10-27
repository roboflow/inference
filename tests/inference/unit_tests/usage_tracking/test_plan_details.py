import sqlite3
from unittest import mock

import pytest

from inference.core.utils.sqlite_wrapper import SQLiteWrapper
from inference.usage_tracking.plan_details import PlanDetails


def test_load_from_cache():
    # given
    conn = sqlite3.connect(":memory:")
    q = SQLiteWrapper(
        db_file_path="",
        table_name="api_keys_plans",
        connection=conn,
        columns={
            "ts": "TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP",
            "api_key_hash": "TEXT NOT NULL",
            "is_enterprise": "INTEGER NOT NULL DEFAULT 1",
            "is_pro": "INTEGER NOT NULL DEFAULT 1",
            "is_trial": "INTEGER NOT NULL DEFAULT 0",
            "is_billed": "INTEGER NOT NULL DEFAULT 1",
            "offline_enabled": "INTEGER NOT NULL DEFAULT 1",
            "over_quota": "INTEGER NOT NULL DEFAULT 0",
        },
    )

    row1 = {
        "ts": "1970-01-01 00:00:00 UTC",
        "api_key_hash": "fake1",
        "is_enterprise": False,
        "is_pro": False,
        "is_trial": False,
        "is_billed": False,
        "offline_enabled": False,
        "over_quota": False,
    }
    row2 = {
        "ts": "1970-01-01 00:00:00 UTC",
        "api_key_hash": "fake2",
        "is_enterprise": True,
        "is_pro": True,
        "is_trial": True,
        "is_billed": True,
        "offline_enabled": True,
        "over_quota": True,
    }

    q.insert(connection=conn, row=row1)
    q.insert(connection=conn, row=row2)
    conn.commit()
    row1["id"] = 1
    row2["id"] = 2

    # when
    plan_details = PlanDetails(
        api_plan_endpoint_url="",
        webrtc_plans_endpoint_url="",
        sqlite_connection=conn,
    )
    conn.close()

    # then
    assert plan_details.api_keys_plans == {"fake1": row1, "fake2": row2}


def _plan_details_without_sqlite() -> PlanDetails:
    return PlanDetails(
        api_plan_endpoint_url="https://plans.example.invalid/details",
        webrtc_plans_endpoint_url="https://webrtc.example.invalid/plans",
        sqlite_cache_enabled=False,
    )


@mock.patch("inference.usage_tracking.plan_details.requests.get")
def test_cold_rate_limit_returns_the_unavailable_plan_and_retries(get_mock):
    # given
    get_mock.return_value = mock.Mock(status_code=429)
    plan_details = _plan_details_without_sqlite()

    # when
    plan = plan_details.get_api_key_plan(api_key="fixture-key")

    # then
    assert plan["is_enterprise"] is True
    assert plan["is_pro"] is True
    assert plan["is_billed"] is True
    assert plan["over_quota"] is False
    assert plan["ts"] == plan_details._ts_default

    # when the endpoint recovers
    get_mock.return_value = mock.Mock(status_code=200)
    get_mock.return_value.json.return_value = {"is_enterprise": True}
    recovered = plan_details.get_api_key_plan(api_key="fixture-key")
    plan_details.get_api_key_plan(api_key="fixture-key")

    # then
    assert get_mock.call_count == 2
    assert recovered["over_quota"] is False
    assert recovered["ts"] != plan_details._ts_default


@mock.patch("inference.usage_tracking.plan_details.requests.get")
def test_rate_limit_keeps_a_cached_plan(get_mock):
    # given
    get_mock.return_value = mock.Mock(status_code=200)
    get_mock.return_value.json.return_value = {"is_enterprise": True}
    plan_details = _plan_details_without_sqlite()
    plan = dict(plan_details.get_api_key_plan(api_key="fixture-key"))

    # when
    get_mock.return_value = mock.Mock(status_code=429)
    refreshed = plan_details.refresh_api_key_plan_cache(api_key="fixture-key")

    # then
    assert refreshed == plan
    assert plan_details.get_api_key_plan(api_key="fixture-key") == plan
    assert get_mock.call_count == 2


@pytest.mark.parametrize("status_code", [401, 403])
@mock.patch("inference.usage_tracking.plan_details.requests.get")
def test_rejected_new_credentials_are_denied_and_retried(get_mock, status_code):
    # given
    get_mock.return_value = mock.Mock(status_code=status_code)
    plan_details = _plan_details_without_sqlite()

    # when
    plan = plan_details.get_api_key_plan(api_key="fixture-key")
    plan_details.get_api_key_plan(api_key="fixture-key")

    # then
    assert plan["is_enterprise"] is False
    assert plan["over_quota"] is True
    assert plan["ts"] == plan_details._ts_default
    assert get_mock.call_count == 2
