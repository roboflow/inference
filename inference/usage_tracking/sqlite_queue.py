import json
import os
import sqlite3
from typing import Any, Dict, List, Optional

from inference.core.env import MODEL_CACHE_DIR
from inference.core.logger import logger
from inference.core.utils.sqlite_wrapper import SQLiteWrapper


class SQLiteQueue(SQLiteWrapper):
    def __init__(
        self,
        db_file_path: str = os.path.join(MODEL_CACHE_DIR, "usage.db"),
        table_name: str = "usage",
        sqlite_connection: Optional[sqlite3.Connection] = None,
    ):
        self._col_name = "payload"

        super().__init__(
            db_file_path=db_file_path,
            table_name=table_name,
            columns={self._col_name: "TEXT NOT NULL"},
            connection=sqlite_connection,
        )

    def put(self, payload: Any, sqlite_connection: Optional[sqlite3.Connection] = None):
        payload_str = json.dumps(payload)
        try:
            self.insert(
                row={self._col_name: payload_str},
                connection=sqlite_connection,
                with_exclusive=True,
            )
        except Exception:
            pass

    @staticmethod
    def full() -> bool:
        return False

    def empty(self, sqlite_connection: Optional[sqlite3.Connection] = None) -> bool:
        try:
            return self.count(connection=sqlite_connection) == 0
        except Exception:
            return True

    def get_nowait(
        self, sqlite_connection: Optional[sqlite3.Connection] = None
    ) -> List[Dict[str, Any]]:
        try:
            sqlite_payloads = self.flush(connection=sqlite_connection, limit=100)
        except Exception:
            return []

        usage_payloads = []
        for p in sqlite_payloads:
            try:
                usage_payloads.append(json.loads(p[self._col_name]))
            except Exception as exc:
                logger.debug("Failed to process sqlite payload %s - %s", p, exc)
        return usage_payloads

    def peek_payloads(self):
        with sqlite3.connect(self._db_file_path, timeout=1) as connection:
            rows = connection.execute(
                f"SELECT payload FROM {self._tbl_name} ORDER BY id LIMIT 100"
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def read_report_delivery(self):
        with sqlite3.connect(self._db_file_path, timeout=1) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS usage_report_delivery "
                "(id INTEGER PRIMARY KEY CHECK (id = 1), payload TEXT NOT NULL)"
            )
            row = connection.execute(
                "SELECT payload FROM usage_report_delivery WHERE id = 1"
            ).fetchone()
        return json.loads(row[0]) if row else {}

    def prepare_report_delivery(self, partition):
        """Replace raw rows and retain frozen reports in the same local commit."""
        with sqlite3.connect(self._db_file_path, timeout=1) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS usage_report_delivery "
                "(id INTEGER PRIMARY KEY CHECK (id = 1), payload TEXT NOT NULL)"
            )
            current = connection.execute(
                "SELECT payload FROM usage_report_delivery WHERE id = 1"
            ).fetchone()
            retained = json.loads(current[0]) if current else {}
            rows = connection.execute(
                f"SELECT id, payload FROM {self._tbl_name} ORDER BY id LIMIT 100"
            ).fetchall()
            prepared, legacy, deferred = partition(
                [json.loads(row[1]) for row in rows], retained
            )
            connection.executemany(
                f"DELETE FROM {self._tbl_name} WHERE id = ?",
                [(row[0],) for row in rows],
            )
            connection.executemany(
                f"INSERT INTO {self._tbl_name} (payload) VALUES (?)",
                [(json.dumps(payload),) for payload in deferred],
            )
            if prepared:
                for key, reports in prepared.items():
                    retained.setdefault(key, {}).update(reports)
                connection.execute(
                    "INSERT OR REPLACE INTO usage_report_delivery (id, payload) VALUES (1, ?)",
                    (json.dumps(retained),),
                )
        return prepared, legacy

    def acknowledge_reports(self, report_ids):
        with sqlite3.connect(self._db_file_path, timeout=1) as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT payload FROM usage_report_delivery WHERE id = 1"
            ).fetchone()
            if not current:
                return
            delivery = json.loads(current[0])
            remaining = {
                key: {
                    report_id: report
                    for report_id, report in reports.items()
                    if report_id not in report_ids
                }
                for key, reports in delivery.items()
            }
            remaining = {key: reports for key, reports in remaining.items() if reports}
            if remaining:
                connection.execute(
                    "UPDATE usage_report_delivery SET payload = ? WHERE id = 1",
                    (json.dumps(remaining),),
                )
            else:
                connection.execute("DELETE FROM usage_report_delivery WHERE id = 1")
