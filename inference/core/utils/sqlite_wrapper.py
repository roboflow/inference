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

        self._columns = columns

        self._id_col_name = "id"
        self._columns[self._id_col_name] = "INTEGER PRIMARY KEY"

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
        row: Dict[ColName, ColValue],
        connection: Optional[sqlite3.Connection] = None,
        cursor: Optional[sqlite3.Cursor] = None,
        with_exclusive: bool = False,
    ):
        if not connection and not cursor:
            try:
                connection: sqlite3.Connection = sqlite3.connect(
                    self._db_file_path, timeout=1
                )
                self._insert(
                    row=row, connection=connection, with_exclusive=with_exclusive
                )
                connection.close()
            except Exception as exc:
                logger.debug(
                    "Failed to store '%s' in %s - %s", row, self._tbl_name, exc
                )
                raise exc
        elif connection and not cursor:
            self._insert(row=row, connection=connection, with_exclusive=with_exclusive)
        elif connection and not with_exclusive:
            self._insert(row=row, connection=connection)
        elif cursor and not with_exclusive:
            self._insert(row=row, cursor=cursor)
        else:
            raise RuntimeError("Unsupported mode")

    def _insert(
        self,
        row: Dict[ColName, ColValue],
        connection: Optional[sqlite3.Connection] = None,
        cursor: Optional[sqlite3.Cursor] = None,
        with_exclusive: bool = False,
    ):
        if not set(row.keys()).issubset(self._columns.keys()):
            logger.debug(
                "Cannot store '%s' in %s, requested column names do not match with table columns",
                row,
                self._tbl_name,
            )
            raise ValueError("Columns mismatch")

        cursor_needs_closing = False
        if not cursor:
            cursor = connection.cursor()
            cursor_needs_closing = True

        if with_exclusive:
            try:
                cursor.execute("BEGIN EXCLUSIVE")
            except Exception as exc:
                logger.debug(
                    "Failed to store '%s' in %s - %s", row, self._tbl_name, exc
                )
                raise exc

        values = {k: v for k, v in row.items() if k != "id"}
        sql_insert = f"""INSERT INTO {self._tbl_name} ({', '.join(values.keys())})
                VALUES ({', '.join(['?'] * len(values))});
            """

        try:
            cursor.execute(sql_insert, list(values.values()))
            if with_exclusive:
                connection.commit()
        except Exception as exc:
            logger.debug("Failed to store '%s' in %s - %s", values, self._tbl_name, exc)
            connection.rollback()
            raise exc

        if cursor_needs_closing:
            cursor.close()

    def count(
        self,
        connection: Optional[sqlite3.Connection] = None,
        cursor: Optional[sqlite3.Cursor] = None,
        with_exclusive: bool = False,
    ) -> int:
        if not connection and not cursor:
            try:
                connection: sqlite3.Connection = sqlite3.connect(
                    self._db_file_path, timeout=1
                )
                count = self._count(
                    connection=connection, with_exclusive=with_exclusive
                )
                connection.close()
            except Exception as exc:
                logger.debug("Failed to obtain records count - %s", exc)
                raise exc
        elif connection and not cursor:
            count = self._count(connection=connection, with_exclusive=with_exclusive)
        elif connection and not with_exclusive:
            count = self._count(connection=connection)
        elif cursor and not with_exclusive:
            count = self._count(cursor=cursor)
        else:
            raise RuntimeError("Unsupported mode")
        return count

    def _count(
        self,
        connection: Optional[sqlite3.Connection] = None,
        cursor: Optional[sqlite3.Cursor] = None,
        with_exclusive: bool = False,
    ) -> int:
        cursor_needs_closing = False
        if not cursor:
            cursor = connection.cursor()
            cursor_needs_closing = True

        if with_exclusive:
            try:
                cursor.execute("BEGIN EXCLUSIVE")
            except Exception as exc:
                logger.debug("Failed to obtain records count - %s", exc)
                raise exc

        sql_select = f"SELECT COUNT(*) FROM {self._tbl_name}"

        count = 0
        try:
            cursor.execute(sql_select)
            count = int(cursor.fetchone()[0])
            if with_exclusive:
                connection.commit()
        except Exception as exc:
            logger.debug("Failed to obtain records count - %s", exc)
            connection.rollback()
            raise exc

        if cursor_needs_closing:
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
                rows = self._select(
                    connection=connection, with_exclusive=with_exclusive, limit=limit
                )
                connection.close()
            except Exception as exc:
                logger.debug("Failed to obtain records - %s", exc)
                raise exc
        elif connection and not cursor:
            rows = self._select(
                connection=connection, with_exclusive=with_exclusive, limit=limit
            )
        elif connection and not with_exclusive:
            rows = self._select(connection=connection, limit=limit)
        elif cursor and not with_exclusive:
            rows = self._select(cursor=cursor, limit=limit)
        else:
            raise RuntimeError("Unsupported mode")
        return rows

    def _select(
        self,
        connection: Optional[sqlite3.Connection] = None,
        cursor: Optional[sqlite3.Cursor] = None,
        with_exclusive: bool = False,
        limit: int = 0,
    ) -> List[Dict[str, Any]]:
        cursor_needs_closing = False
        if not cursor:
            cursor = connection.cursor()
            cursor_needs_closing = True

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

        try:
            cursor.execute(sql_select)
            sqlite_rows = cursor.fetchall()
            if with_exclusive:
                connection.commit()
        except Exception as exc:
            logger.debug("Failed to obtain records - %s", exc)
            connection.rollback()
            raise exc

        rows = []
        for _id, *row in sqlite_rows:
            row = {
                k: v
                for k, v in zip([k for k in self._columns.keys() if k != "id"], row)
            }
            row["id"] = _id
            rows.append(row)

        if cursor_needs_closing:
            cursor.close()

        return rows

    def flush(
        self, connection: Optional[sqlite3.Connection] = None, limit: int = 0
    ) -> List[Dict[str, Any]]:
        if not connection:
            try:
                connection: sqlite3.Connection = sqlite3.connect(
                    self._db_file_path, timeout=1
                )
                rows = self._flush(connection=connection, limit=limit)
                connection.close()
            except Exception as exc:
                logger.debug("Failed to flush db - %s", exc)
                raise exc
        else:
            rows = self._flush(connection=connection, limit=limit)
        return rows

    def _flush(
        self, connection: sqlite3.Connection, limit: int = 0
    ) -> List[Dict[str, Any]]:
        cursor = connection.cursor()
        try:
            cursor.execute("BEGIN EXCLUSIVE")
        except Exception as exc:
            logger.debug("Failed to obtain records - %s", exc)
            raise exc

        try:
            rows = self.select(cursor=cursor, limit=limit)
            self.delete(rows=rows, cursor=cursor)
            connection.commit()
            cursor.close()
        except Exception as exc:
            logger.debug("Failed to delete records - %s", exc)
            connection.rollback()
            raise exc

        return rows

    def delete(
        self,
        rows: List[Dict[ColName, ColValue]],
        connection: Optional[sqlite3.Connection] = None,
        cursor: Optional[sqlite3.Cursor] = None,
        with_exclusive: bool = False,
    ) -> List[Dict[str, Any]]:
        if not connection and not cursor:
            try:
                connection: sqlite3.Connection = sqlite3.connect(
                    self._db_file_path, timeout=1
                )
                deleted = self._delete(
                    rows=rows, connection=connection, with_exclusive=with_exclusive
                )
                connection.close()
            except Exception as exc:
                logger.debug("Failed to obtain records - %s", exc)
                raise exc
        elif connection and not cursor:
            deleted = self._delete(
                rows=rows, connection=connection, with_exclusive=with_exclusive
            )
        elif connection and not with_exclusive:
            deleted = self._delete(rows=rows, connection=connection)
        elif cursor and not with_exclusive:
            deleted = self._delete(rows=rows, cursor=cursor)
        else:
            raise RuntimeError("Unsupported mode")
        return deleted

    def _delete(
        self,
        rows: List[Dict[ColName, ColValue]],
        connection: Optional[sqlite3.Connection] = None,
        cursor: Optional[sqlite3.Cursor] = None,
        with_exclusive: bool = False,
    ) -> List[Dict[str, Any]]:
        keys = [r["id"] for r in rows if "id" in r]
        if not keys:
            logger.debug("No row with 'id' key found in %s", rows)
            return []

        cursor_needs_closing = False
        if not cursor:
            cursor = connection.cursor()
            cursor_needs_closing = True

        if with_exclusive:
            try:
                cursor.execute("BEGIN EXCLUSIVE")
            except Exception as exc:
                logger.debug("Failed to delete records - %s", exc)
                raise exc

        sql_delete = f"""DELETE
                FROM {self._tbl_name}
                WHERE "id" in ({', '.join(['?'] * len(keys))})
            """
        sql_select = f"""SELECT *
                FROM {self._tbl_name}
                WHERE "id" in ({', '.join(['?'] * len(keys))})
            """

        try:
            cursor.execute(sql_delete, keys)
        except Exception as exc:
            logger.debug("Failed to delete records - %s", exc)
            connection.rollback()
            raise exc

        try:
            cursor.execute(sql_select, keys)
            payloads = cursor.fetchall()
            if with_exclusive:
                connection.commit()
        except Exception as exc:
            logger.debug("Failed to delete records - %s", exc)
            connection.rollback()
            raise exc

        if cursor_needs_closing:
            cursor.close()

        _ids = set()
        for _id, *_ in payloads:
            _ids.add(_id)

        return [r for r in rows if "id" in r and r["id"] not in _ids]

    def refresh(
        self,
        rows: List[Dict[ColName, ColValue]],
        connection: Optional[sqlite3.Connection] = None,
    ) -> List[Dict[str, Any]]:
        if not connection:
            try:
                connection: sqlite3.Connection = sqlite3.connect(
                    self._db_file_path, timeout=1
                )
                payloads = self._refresh(rows=rows, connection=connection)
                connection.close()
            except Exception as exc:
                logger.debug("Failed to flush db - %s", exc)
                raise exc
        else:
            payloads = self._refresh(rows=rows, connection=connection)
        return payloads

    def _refresh(
        self, rows: List[Dict[ColName, ColValue]], connection: sqlite3.Connection
    ) -> List[Dict[str, Any]]:
        cursor = connection.cursor()
        try:
            cursor.execute("BEGIN EXCLUSIVE")
        except Exception as exc:
            logger.debug("Failed to obtain records - %s", exc)
            raise exc

        try:
            self.delete(rows=rows, cursor=cursor)
        except Exception as exc:
            logger.debug("Failed to delete records - %s", exc)
            connection.rollback()
            raise exc

        try:
            for r in rows:
                self.insert(row=r, cursor=cursor)
            connection.commit()
        except Exception as exc:
            logger.debug("Failed to insert records - %s", exc)
            connection.rollback()
            raise exc

        try:
            rows = self.select(cursor=cursor)
            cursor.close()
        except Exception as exc:
            logger.debug("Failed to delete records - %s", exc)
            connection.rollback()
            raise exc

        return rows
