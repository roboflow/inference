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
