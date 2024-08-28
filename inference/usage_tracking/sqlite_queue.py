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
        self.insert(values={self._col_name: payload_str}, connection=connection)

    @staticmethod
    def full() -> bool:
        return False

    def empty(self, connection: Optional[sqlite3.Connection] = None) -> bool:
        return self.count(connection=connection) == 0

    def get_nowait(
        self, connection: Optional[sqlite3.Connection] = None
    ) -> List[Dict[str, Any]]:
        return self.flush(connection=connection, limit=100)
