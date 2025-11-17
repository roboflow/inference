import sqlite3

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
