import json
import os
import sqlite3

from typing_extensions import Any, Dict, List, Optional

from inference.core.env import MODEL_CACHE_DIR
from inference.core.logger import logger
from inference.core.utils.sqlite_wrapper import SQLiteWrapper


class SQLiteQueue(SQLiteWrapper):
    def __init__(
        self,
        db_file_path: str = os.path.join(MODEL_CACHE_DIR, "usage.db"),
        table_name: str = "usage",
        connection: Optional[sqlite3.Connection] = None,
    ):
        self._col_name = "payload"

        super().__init__(
            db_file_path=db_file_path,
            table_name=table_name,
            columns={self._col_name: "TEXT NOT NULL"},
            connection=connection,
        )

    def put(self, payload: Any, connection: Optional[sqlite3.Connection] = None):
        payload_str = json.dumps(payload)
        try:
            self.insert(values={self._col_name: payload_str}, connection=connection)
        except Exception:
            pass

    @staticmethod
    def full() -> bool:
        return False

    def empty(self, connection: Optional[sqlite3.Connection] = None) -> bool:
        try:
            return self.count(connection=connection) == 0
        except Exception:
            return True

    def get_nowait(
        self, connection: Optional[sqlite3.Connection] = None
    ) -> List[Dict[str, Any]]:
        try:
            sqlite_payloads = self.flush(connection=connection, limit=100)
        except Exception:
            return []

        usage_payloads = []
        for p in sqlite_payloads:
            try:
                usage_payloads.append(json.loads(p[self._col_name]))
            except Exception as exc:
                logger.debug("Failed to process sqlite payload %s - %s", p, exc)
        return usage_payloads
