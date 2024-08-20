import json
import os
import sqlite3

from typing_extensions import Any, Dict, List, Optional

from inference.core.env import MODEL_CACHE_DIR
from inference.core.logger import logger


class SQLiteQueue:
    def __init__(
        self,
        db_file_path: str = os.path.join(MODEL_CACHE_DIR, "usage.db"),
        connection: Optional[sqlite3.Connection] = None,
    ):
        self._tbl_name: str = "usage"
        self._col_name: str = "payload"
        self._db_file_path: str = db_file_path

        if not connection:
            if not os.path.exists(MODEL_CACHE_DIR):
                os.makedirs(MODEL_CACHE_DIR)
            connection: sqlite3.Connection = sqlite3.connect(db_file_path, timeout=1)
            self._create_table(connection=connection)
            connection.close()
        else:
            self._create_table(connection=connection)

    def _create_table(self, connection: sqlite3.Connection):
        sql_create_table = f"""CREATE TABLE IF NOT EXISTS {self._tbl_name} (
                                id INTEGER PRIMARY KEY,
                                {self._col_name} TEXT NOT NULL
                            );"""
        cursor = connection.cursor()
        cursor.execute("BEGIN EXCLUSIVE")
        try:
            cursor.execute(sql_create_table)
            connection.commit()
            cursor.close()
        except Exception as exc:
            connection.rollback()
            cursor.close()
            raise exc

    def _insert(self, payload: str, connection: sqlite3.Connection):
        cursor = connection.cursor()
        sql_insert = f"INSERT INTO {self._tbl_name} ({self._col_name}) VALUES (?)"

        try:
            cursor.execute("BEGIN EXCLUSIVE")
        except Exception as exc:
            logger.debug("Failed to store usage payload, %s", exc)
            return

        try:
            cursor.execute(sql_insert, (payload,))
            connection.commit()
        except Exception as exc:
            logger.debug("Failed to store usage payload '%s', %s", payload, exc)
            connection.rollback()

        cursor.close()

    def put(self, payload: Any, connection: Optional[sqlite3.Connection] = None):
        payload_str = json.dumps(payload)
        if not connection:
            try:
                connection: sqlite3.Connection = sqlite3.connect(
                    self._db_file_path, timeout=1
                )
                self._insert(payload=payload_str, connection=connection)
                connection.close()
            except Exception as exc:
                logger.debug("Failed to store usage records '%s', %s", payload, exc)
                return []
        else:
            self._insert(payload=payload_str, connection=connection)

    @staticmethod
    def full() -> bool:
        return False

    def _count_rows(self, connection: sqlite3.Connection) -> int:
        cursor = connection.cursor()
        sql_select = f"SELECT COUNT(*) FROM {self._tbl_name}"

        try:
            cursor.execute("BEGIN EXCLUSIVE")
        except Exception as exc:
            logger.debug("Failed to obtain records count, %s", exc)
            return 0

        count = 0
        try:
            cursor.execute(sql_select)
            count = int(cursor.fetchone()[0])
            connection.commit()
        except Exception as exc:
            logger.debug("Failed to obtain records count, %s", exc)
            connection.rollback()

        cursor.close()
        return count

    def empty(self, connection: Optional[sqlite3.Connection] = None) -> bool:
        rows_count = 0
        if not connection:
            try:
                connection: sqlite3.Connection = sqlite3.connect(
                    self._db_file_path, timeout=1
                )
                rows_count = self._count_rows(connection=connection)
                connection.close()
            except Exception as exc:
                logger.debug("Failed to obtain records count, %s", exc)
                return True
        else:
            rows_count = self._count_rows(connection=connection)

        return rows_count == 0

    def _flush_db(
        self, connection: sqlite3.Connection, limit: int = 100
    ) -> List[Dict[str, Any]]:
        cursor = connection.cursor()
        sql_select = f"SELECT id, {self._col_name} FROM {self._tbl_name} ORDER BY id ASC LIMIT {limit}"
        sql_delete = f"DELETE FROM {self._tbl_name} WHERE id >= ? and id <= ?"

        try:
            cursor.execute("BEGIN EXCLUSIVE")
        except Exception as exc:
            logger.debug("Failed to obtain records, %s", exc)
            return []

        payloads = []
        try:
            cursor.execute(sql_select)
            payloads = cursor.fetchall()
        except Exception as exc:
            logger.debug("Failed to obtain records, %s", exc)
            connection.rollback()
            return []

        parsed_payloads = []
        top_id = -1
        bottom_id = -1
        for _id, payload in payloads:
            top_id = max(top_id, _id)
            if bottom_id == -1:
                bottom_id = _id
            bottom_id = min(bottom_id, _id)
            try:
                parsed_payload = json.loads(payload)
                parsed_payloads.append(parsed_payload)
            except Exception as exc:
                logger.debug("Failed to parse usage payload %s, %s", payload, exc)

        try:
            cursor.execute(sql_delete, [bottom_id, top_id])
            connection.commit()
            cursor.close()
        except Exception as exc:
            logger.debug("Failed to obtain records, %s", exc)
            connection.rollback()

        return parsed_payloads

    def get_nowait(
        self, connection: Optional[sqlite3.Connection] = None
    ) -> List[Dict[str, Any]]:
        if not connection:
            try:
                connection: sqlite3.Connection = sqlite3.connect(
                    self._db_file_path, timeout=1
                )
                parsed_payloads = self._flush_db(connection=connection)
                connection.close()
                return parsed_payloads
            except Exception as exc:
                logger.debug("Failed to obtain records, %s", exc)
                return []
        else:
            return self._flush_db(connection=connection)
