import json
import os
import sqlite3

from typing_extensions import Any, Dict, List

from inference.core.env import MODEL_CACHE_DIR
from inference.core.logger import logger


class PersistentQueue:
    def __init__(self, db_file_path: str = os.path.join(MODEL_CACHE_DIR, "usage.db")):
        self._connection: sqlite3.Connection = sqlite3.connect(db_file_path, timeout=1)
        self._tbl_name: str = "usage"
        self._col_name: str = "payload"
        sql_create_table = f"""CREATE TABLE IF NOT EXISTS {self._tbl_name} (
                                id INTEGER PRIMARY KEY,
                                {self._col_name} TEXT NOT NULL
                            );"""
        cursor = self._connection.cursor()
        cursor.execute("BEGIN EXCLUSIVE")
        try:
            cursor.execute(sql_create_table)
            self._connection.commit()
            cursor.close()
        except Exception as exc:
            self._connection.rollback()
            cursor.close()
            raise exc

    def put(self, payload: Any):
        payload_str = json.dumps(payload)
        cursor = self._connection.cursor()
        sql_insert = f"INSERT INTO {self._tbl_name} ({self._col_name}) VALUES (?)"

        try:
            cursor.execute("BEGIN EXCLUSIVE")
        except Exception as exc:
            logger.debug("Failed to store usage payload, %s", exc)
            return

        try:
            cursor.execute(sql_insert, (payload_str,))
            self._connection.commit()
        except Exception as exc:
            logger.debug("Failed to store usage payload, %s", exc)
            self._connection.rollback()

        cursor.close()

    @staticmethod
    def full() -> bool:
        return False

    def empty(self) -> bool:
        cursor = self._connection.cursor()
        sql_select = f"SELECT COUNT(*) FROM {self._tbl_name}"
        count = 0

        try:
            cursor.execute("BEGIN EXCLUSIVE")
        except Exception as exc:
            logger.debug("Failed to obtain records count, %s", exc)
            return True

        try:
            cursor.execute(sql_select)
            count = int(cursor.fetchone()[0])
            self._connection.commit()
        except Exception as exc:
            logger.debug("Failed to store usage payload, %s", exc)
            self._connection.rollback()

        cursor.close()

        return count == 0

    def get_nowait(self) -> List[Dict[str, Any]]:
        cursor = self._connection.cursor()
        sql_select = f"SELECT {self._col_name} FROM {self._tbl_name}"
        sql_delete = f"DELETE FROM {self._tbl_name}"
        payloads = []

        try:
            cursor.execute("BEGIN EXCLUSIVE")
        except Exception as exc:
            logger.debug("Failed to obtain records, %s", exc)
            return []

        payloads = []
        try:
            cursor.execute(sql_select)
            payloads = cursor.fetchall()
            cursor.execute(sql_delete)
            self._connection.commit()
        except Exception as exc:
            logger.debug("Failed to store usage payload, %s", exc)
            self._connection.rollback()
            return []

        cursor.close()

        parsed_payloads = []
        for (payload,) in payloads:
            try:
                parsed_payloads.append(json.loads(payload))
            except Exception as exc:
                logger.debug("Failed to parse usage payload %s, %s", payload, exc)

        return parsed_payloads

    def __del__(self):
        cursor = self._connection.cursor()
        try:
            cursor.execute("BEGIN EXCLUSIVE")
            self._connection.commit()
            cursor.close()
            self._connection.close()
        except Exception as exc:
            logger.debug("Failed to safely close db connection, %s", exc)
