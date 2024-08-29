import os
import sqlite3
from typing import Any, Dict, List, Optional

from inference.core.logger import logger

ColName = str
ColType = str
ColValue = str


class SQLiteWrapper:
    def __init__(
        self,
        db_file_path: str,
        table_name: str,
        columns: Dict[ColName, ColType],
        connection: Optional[sqlite3.Connection] = None,
    ):
        self._db_file_path = db_file_path
        self._tbl_name = table_name
        self._columns = {**columns, **{"id": "INTEGER PRIMARY KEY"}}

        if not connection:
            os.makedirs(os.path.dirname(db_file_path), exist_ok=True)
            connection: sqlite3.Connection = sqlite3.connect(db_file_path, timeout=1)
            self.create_table(connection=connection)
            connection.close()
        else:
            self.create_table(connection=connection)

    def create_table(self, connection: Optional[sqlite3.Connection] = None):
        if not connection:
            connection: sqlite3.Connection = sqlite3.connect(
                self._db_file_path, timeout=1
            )
            self._create_table(connection=connection)
            connection.close()
        else:
            self._create_table(connection=connection)

    def _create_table(self, connection: sqlite3.Connection):
        sql_create_table = f"""CREATE TABLE IF NOT EXISTS {self._tbl_name}
                ({', '.join(f'{_col} {_type}' for _col, _type in self._columns.items())});
            """
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

    def insert(
        self,
        values: Dict[ColName, ColValue],
        connection: Optional[sqlite3.Connection] = None,
    ):
        if not connection:
            try:
                connection: sqlite3.Connection = sqlite3.connect(
                    self._db_file_path, timeout=1
                )
                self._insert(values=values, connection=connection)
                connection.close()
            except Exception as exc:
                logger.debug(
                    "Failed to store '%s' in %s - %s", values, self._tbl_name, exc
                )
                raise exc
        else:
            self._insert(values=values, connection=connection)

    def _insert(self, values: Dict[ColName, ColValue], connection: sqlite3.Connection):
        if not set(values.keys()).issubset(self._columns.keys()):
            logger.debug(
                "Cannot store '%s' in %s, requested column names do not match with table columns",
                values,
                self._tbl_name,
            )
            raise ValueError("Columns mismatch")
        cursor = connection.cursor()
        values = {k: v for k, v in values.items() if k != "id"}
        sql_insert = f"""INSERT INTO {self._tbl_name} ({', '.join(values.keys())})
                VALUES ({', '.join(['?'] * len(values))});
            """

        try:
            cursor.execute("BEGIN EXCLUSIVE")
        except Exception as exc:
            logger.debug("Failed to store '%s' in %s - %s", values, self._tbl_name, exc)
            raise exc

        try:
            cursor.execute(sql_insert, list(values.values()))
            connection.commit()
        except Exception as exc:
            logger.debug("Failed to store '%s' in %s - %s", values, self._tbl_name, exc)
            connection.rollback()
            raise exc
        cursor.close()

    def count(self, connection: Optional[sqlite3.Connection] = None) -> int:
        if not connection:
            try:
                connection: sqlite3.Connection = sqlite3.connect(
                    self._db_file_path, timeout=1
                )
                count = self._count(connection=connection)
                connection.close()
            except Exception as exc:
                logger.debug("Failed to obtain records count - %s", exc)
                raise exc
        else:
            count = self._count(connection=connection)
        return count

    def _count(self, connection: sqlite3.Connection) -> int:
        cursor = connection.cursor()
        sql_select = f"SELECT COUNT(*) FROM {self._tbl_name}"

        try:
            cursor.execute("BEGIN EXCLUSIVE")
        except Exception as exc:
            logger.debug("Failed to obtain records count - %s", exc)
            raise exc

        count = 0
        try:
            cursor.execute(sql_select)
            count = int(cursor.fetchone()[0])
            connection.commit()
        except Exception as exc:
            logger.debug("Failed to obtain records count - %s", exc)
            connection.rollback()
            raise exc
        cursor.close()

        return count

    def select(
        self,
        connection: Optional[sqlite3.Connection] = None,
        cursor: Optional[sqlite3.Cursor] = None,
        with_exclusive: bool = False,
        limit: int = 0,
    ) -> List[Dict[str, Any]]:
        if not connection and not cursor:
            try:
                connection: sqlite3.Connection = sqlite3.connect(
                    self._db_file_path, timeout=1
                )
                payloads = self._select(
                    connection=connection, with_exclusive=with_exclusive, limit=limit
                )
                connection.close()
            except Exception as exc:
                logger.debug("Failed to obtain records - %s", exc)
                raise exc
        elif connection and not cursor:
            payloads = self._select(
                connection=connection, with_exclusive=with_exclusive, limit=limit
            )
        elif connection and not with_exclusive:
            payloads = self._select(connection=connection, limit=limit)
        elif cursor and not with_exclusive:
            payloads = self._select(cursor=cursor, limit=limit)
        else:
            raise RuntimeError("Unsupported mode")
        return payloads

    def _select(
        self,
        connection: Optional[sqlite3.Connection] = None,
        cursor: Optional[sqlite3.Cursor] = None,
        with_exclusive: bool = False,
        limit: int = 0,
    ) -> List[Dict[str, Any]]:
        if not cursor:
            cursor = connection.cursor()
        if with_exclusive:
            try:
                cursor.execute("BEGIN EXCLUSIVE")
            except Exception as exc:
                logger.debug("Failed to obtain records - %s", exc)
                raise exc
        sql_select = f"""SELECT id, {', '.join(k for k in self._columns.keys() if k != 'id')}
                FROM {self._tbl_name}
                ORDER BY id ASC
            """
        if limit:
            sql_select = sql_select + f" LIMIT {limit}"
        payloads = []
        try:
            cursor.execute(sql_select)
            payloads = cursor.fetchall()
            if with_exclusive:
                connection.commit()
        except Exception as exc:
            logger.debug("Failed to obtain records - %s", exc)
            connection.rollback()
            raise exc

        rows = []
        for _id, *row in payloads:
            row = {
                k: v
                for k, v in zip([k for k in self._columns.keys() if k != "id"], row)
            }
            row["id"] = _id
            rows.append(row)

        return rows

    def flush(
        self, connection: Optional[sqlite3.Connection] = None, limit: int = 0
    ) -> List[Dict[str, Any]]:
        if not connection:
            try:
                connection: sqlite3.Connection = sqlite3.connect(
                    self._db_file_path, timeout=1
                )
                payloads = self._flush(connection=connection, limit=limit)
                connection.close()
            except Exception as exc:
                logger.debug("Failed to flush db - %s", exc)
                raise exc
        else:
            payloads = self._flush(connection=connection, limit=limit)
        return payloads

    def _flush(
        self, connection: sqlite3.Connection, limit: int = 0
    ) -> List[Dict[str, Any]]:
        cursor = connection.cursor()
        try:
            cursor.execute("BEGIN EXCLUSIVE")
        except Exception as exc:
            logger.debug("Failed to obtain records - %s", exc)
            raise exc

        rows = self.select(connection=connection, cursor=cursor, limit=limit)

        top_id = -1
        bottom_id = -1
        for row in rows:
            _id = row["id"]
            top_id = max(top_id, _id)
            if bottom_id == -1:
                bottom_id = _id
            bottom_id = min(bottom_id, _id)

        sql_delete = f"DELETE FROM {self._tbl_name} WHERE id >= ? and id <= ?"
        try:
            cursor.execute(sql_delete, [bottom_id, top_id])
            connection.commit()
            cursor.close()
        except Exception as exc:
            logger.debug("Failed to delete records - %s", exc)
            connection.rollback()
            raise exc

        return rows
