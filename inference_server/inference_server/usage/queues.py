"""Queues holding usage payloads between aggregation and sending."""

import json
import logging
import sqlite3
import time
from enum import Enum
from pathlib import Path
from queue import Empty, Queue
from threading import Lock
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)
from uuid import uuid4

from inference_server import configuration

logger = logging.getLogger(__name__)

T = TypeVar("T")

SQLITE_FILE_NAME = "usage.db"
SQLITE_TIMEOUT_S = 1
SQLITE_FLUSH_LIMIT = 100
SQLITE_MAX_PAYLOAD_AGE_S = 30 * 24 * 3600
STALE_PAYLOADS_LOG_LINE = (
    "Deleted %s usage payloads older than %s days from the persistent queue"
)
UNDECODABLE_PAYLOAD_LOG_LINE = "Failed to process a stored usage payload: %s"
REDIS_SORTED_SET_NAME = "UsageCollector"


class PutUnconfirmed(Enum):
    """Outcome of a write that may have reached the store."""

    UNCONFIRMED = "unconfirmed"


UNCONFIRMED = PutUnconfirmed.UNCONFIRMED


def split_payload_by_api_key_hash(
    payload: Any, known_hashes: Set[str]
) -> Tuple[Optional[Any], Optional[Any]]:
    """Split a stored payload into the rows of known API key hashes and the rest.

    Args:
        payload: Rows by API key hash and usage key, or a list of such payloads.
        known_hashes: Hashes the caller can resolve to an API key.

    Returns:
        The known part and the unknown part, in the shape of the payload; each
        is None when it holds nothing. A payload that is not a mapping is
        unknown as a whole.
    """
    if isinstance(payload, list):
        known_parts = []
        unknown_parts = []
        for element in payload:
            known, unknown = split_payload_by_api_key_hash(element, known_hashes)
            if known is not None:
                known_parts.append(known)
            if unknown is not None:
                unknown_parts.append(unknown)

        return known_parts or None, unknown_parts or None
    if not isinstance(payload, dict):
        return None, payload

    known = {
        api_key_hash: rows
        for api_key_hash, rows in payload.items()
        if api_key_hash in known_hashes
    }
    unknown = {
        api_key_hash: rows
        for api_key_hash, rows in payload.items()
        if api_key_hash not in known_hashes
    }

    return known or None, unknown or None


def _newest_timestamp_ns(payload: Any) -> Optional[int]:
    elements = payload if isinstance(payload, list) else [payload]
    newest = None
    for element in elements:
        if not isinstance(element, dict):
            return None
        for rows in element.values():
            if not isinstance(rows, dict):
                return None
            for row in rows.values():
                if not isinstance(row, dict):
                    return None
                stamp = row.get("timestamp_stop") or row.get("timestamp_start")
                if isinstance(stamp, bool) or not isinstance(stamp, (int, float)):
                    return None
                newest = stamp if newest is None else max(newest, stamp)

    return newest


def _is_stale(payload: Any, now_ns: int) -> bool:
    newest = _newest_timestamp_ns(payload)
    if newest is None:
        return False
    stale = now_ns - newest > SQLITE_MAX_PAYLOAD_AGE_S * 1_000_000_000

    return stale


class MemoryQueue(Queue):
    """In-memory usage queue of one process, read by its own sender only."""

    def take(self, known_hashes: Set[str]) -> List[Any]:
        """Remove and return the rows of known API key hashes.

        The rows of other hashes are put back behind them, in their order. The
        caller serialises every access to the queue.

        Args:
            known_hashes: Hashes the caller can resolve to an API key.

        Returns:
            The known part of every queued payload, oldest first.
        """
        taken: List[Any] = []
        kept: List[Any] = []
        for _ in range(self.qsize()):
            try:
                payload = self.get_nowait()
            except Empty:
                break
            known, unknown = split_payload_by_api_key_hash(payload, known_hashes)
            if known is not None:
                taken.append(known)
            if unknown is not None:
                kept.append(unknown)
        for payload in kept:
            self.put_nowait(payload)

        return taken


class SQLiteQueue:
    """Usage payloads persisted as JSON text in a SQLite table."""

    read_batch = SQLITE_FLUSH_LIMIT

    def __init__(
        self,
        db_file_path: Optional[Union[str, Path]] = None,
        table_name: str = "usage",
        sqlite_connection: Optional[sqlite3.Connection] = None,
    ) -> None:
        """Create the table when it does not exist.

        Args:
            db_file_path: Database file; ``usage.db`` in ``MODEL_CACHE_DIR``
                when omitted.
            table_name: Table holding the payloads.
            sqlite_connection: Connection used instead of opening the file.
        """
        if db_file_path is None:
            db_file_path = Path(configuration.MODEL_CACHE_DIR) / SQLITE_FILE_NAME
        self._db_file_path = Path(db_file_path)
        self._tbl_name = table_name

        if sqlite_connection is None:
            self._db_file_path.parent.mkdir(parents=True, exist_ok=True)
        self._run(self._create_table, sqlite_connection)

    def _run(
        self,
        operation: Callable[[sqlite3.Connection], T],
        connection: Optional[sqlite3.Connection],
    ) -> T:
        if connection is not None:
            return operation(connection)

        connection = sqlite3.connect(str(self._db_file_path), timeout=SQLITE_TIMEOUT_S)
        try:
            result = operation(connection)
        finally:
            try:
                connection.close()
            except Exception as error:
                logger.error(
                    "Failed to close the usage database: %s", type(error).__name__
                )

        return result

    def _in_exclusive_transaction(
        self,
        connection: sqlite3.Connection,
        operation: Callable[[sqlite3.Cursor], T],
    ) -> T:
        cursor = connection.cursor()
        try:
            cursor.execute("BEGIN EXCLUSIVE")
            try:
                result = operation(cursor)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        finally:
            cursor.close()

        return result

    def _create_table(self, connection: sqlite3.Connection) -> None:
        sql_create_table = (
            f"CREATE TABLE IF NOT EXISTS {self._tbl_name} "
            "(payload TEXT NOT NULL, id INTEGER PRIMARY KEY);"
        )
        self._in_exclusive_transaction(
            connection, lambda cursor: cursor.execute(sql_create_table)
        )

    def _insert(self, connection: sqlite3.Connection, *, payload_str: str) -> None:
        sql_insert = f"INSERT INTO {self._tbl_name} (payload) VALUES (?);"
        self._in_exclusive_transaction(
            connection, lambda cursor: cursor.execute(sql_insert, [payload_str])
        )

    def _count(self, connection: sqlite3.Connection) -> int:
        cursor = connection.cursor()
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {self._tbl_name}")
            count = int(cursor.fetchone()[0])
        finally:
            cursor.close()

        return count

    def _take_batch(
        self,
        connection: sqlite3.Connection,
        *,
        known_hashes: Optional[Set[str]],
        limit: int,
    ) -> Tuple[List[Any], int]:
        sql_select = (
            f"SELECT id, payload FROM {self._tbl_name} "
            "WHERE id > ? ORDER BY id ASC LIMIT ?"
        )
        sql_delete = f"DELETE FROM {self._tbl_name} WHERE id = ?"
        sql_update = f"UPDATE {self._tbl_name} SET payload = ? WHERE id = ?"
        now_ns = time.time_ns()

        def scan(cursor: sqlite3.Cursor) -> Tuple[List[Any], int]:
            taken: List[Any] = []
            stale = 0
            last_id = 0
            while len(taken) < limit:
                cursor.execute(sql_select, [last_id, limit - len(taken)])
                rows = cursor.fetchall()
                if not rows:
                    break
                for row_id, payload_str in rows:
                    last_id = row_id
                    try:
                        payload = json.loads(payload_str)
                    except Exception as error:
                        logger.debug(UNDECODABLE_PAYLOAD_LOG_LINE, type(error).__name__)
                        cursor.execute(sql_delete, [row_id])
                        continue
                    if known_hashes is None:
                        known, unknown = payload, None
                    else:
                        known, unknown = split_payload_by_api_key_hash(
                            payload, known_hashes
                        )
                    if unknown is None:
                        cursor.execute(sql_delete, [row_id])
                    elif known is not None:
                        cursor.execute(sql_update, [json.dumps(unknown), row_id])
                    elif _is_stale(unknown, now_ns):
                        cursor.execute(sql_delete, [row_id])
                        stale += 1
                    if isinstance(known, list):
                        taken.extend(known)
                    elif known is not None:
                        taken.append(known)

            return taken, stale

        outcome = self._in_exclusive_transaction(connection, scan)

        return outcome

    def put(
        self, payload: Any, sqlite_connection: Optional[sqlite3.Connection] = None
    ) -> bool:
        """Store one payload; a failure is logged and reported.

        Args:
            payload: JSON-serialisable payload.
            sqlite_connection: Connection used instead of opening the file.

        Returns:
            True when the payload was committed, False when it was not stored.
        """
        try:
            payload_str = json.dumps(payload)
            self._run(
                lambda connection: self._insert(connection, payload_str=payload_str),
                sqlite_connection,
            )
        except Exception as error:
            logger.error("Failed to store usage records: %s", type(error).__name__)
            return False

        return True

    @staticmethod
    def full() -> bool:
        """Tell whether the queue refuses payloads.

        Returns:
            Always False: the table has no size limit.
        """
        return False

    def qsize(self, sqlite_connection: Optional[sqlite3.Connection] = None) -> int:
        """Count the stored payloads.

        Args:
            sqlite_connection: Connection used instead of opening the file.

        Returns:
            The number of stored payloads; zero when the table cannot be read.
        """
        try:
            count = self._run(self._count, sqlite_connection)
        except Exception:
            return 0

        return count

    def empty(self, sqlite_connection: Optional[sqlite3.Connection] = None) -> bool:
        """Tell whether no payload is stored.

        Args:
            sqlite_connection: Connection used instead of opening the file.

        Returns:
            True when the table is empty or cannot be read.
        """
        empty = self.qsize(sqlite_connection=sqlite_connection) == 0

        return empty

    def take(
        self,
        known_hashes: Optional[Set[str]] = None,
        *,
        limit: int = SQLITE_FLUSH_LIMIT,
        sqlite_connection: Optional[sqlite3.Connection] = None,
    ) -> List[Any]:
        """Remove and return the oldest payloads of known API key hashes.

        The table is scanned oldest first inside one exclusive transaction
        until ``limit`` payloads are collected. A payload holding rows of other
        hashes keeps those rows under its own id; a payload holding only such
        rows is left untouched, unless every row of it stopped more than
        ``SQLITE_MAX_PAYLOAD_AGE_S`` ago, in which case it is deleted and
        counted in one warning line. A payload that cannot be decoded is
        deleted.

        Args:
            known_hashes: Hashes the caller can resolve to an API key; None
                takes every payload.
            limit: Most payloads returned by one call.
            sqlite_connection: Connection used instead of opening the file.

        Returns:
            The decoded payloads, oldest first; empty when the table cannot be
            read.
        """
        try:
            taken, stale = self._run(
                lambda connection: self._take_batch(
                    connection, known_hashes=known_hashes, limit=limit
                ),
                sqlite_connection,
            )
        except Exception:
            return []
        if stale:
            logger.warning(
                STALE_PAYLOADS_LOG_LINE, stale, SQLITE_MAX_PAYLOAD_AGE_S // 86400
            )

        return taken

    def get_nowait(
        self, sqlite_connection: Optional[sqlite3.Connection] = None
    ) -> List[Dict[str, Any]]:
        """Remove and return up to 100 of the oldest payloads, whatever their hash.

        Args:
            sqlite_connection: Connection used instead of opening the file.

        Returns:
            The decoded payloads, oldest first; empty when the table cannot be
            read.
        """
        usage_payloads = self.take(sqlite_connection=sqlite_connection)

        return usage_payloads


def _build_redis_client() -> Any:
    import redis

    client = redis.Redis(
        host=configuration.REDIS_HOST,
        port=configuration.REDIS_PORT,
        db=0,
        decode_responses=False,
        ssl=configuration.REDIS_SSL,
        socket_timeout=configuration.REDIS_TIMEOUT,
        socket_connect_timeout=configuration.REDIS_TIMEOUT,
    )

    return client


class RedisQueue:
    """Write-only queue: stored keys are read by an external service."""

    def __init__(
        self,
        hash_tag: str = "UsageCollector",
        redis_client: Optional[Any] = None,
    ) -> None:
        """Bind the queue to a Redis client.

        Args:
            hash_tag: Hash tag every stored key starts with.
            redis_client: Client to write with; built from the ``REDIS_*``
                settings when omitted.

        Raises:
            ImportError: If no client is given and ``redis`` is not installed.
        """
        if redis_client is None:
            redis_client = _build_redis_client()
        self._prefix: str = f"{{{hash_tag}}}:{time.time()}:{uuid4().hex[:5]}"
        self._redis_client = redis_client
        self._increment: int = 0
        self._lock: Lock = Lock()

    def put(self, payload: Any) -> Union[bool, PutUnconfirmed]:
        """Store one payload with a single attempt; it is never retried.

        The queue is write-only, so a write that was not acknowledged may
        still have landed and a retry could be consumed twice.

        Args:
            payload: JSON text, or a JSON-serialisable payload.

        Returns:
            True when every command was acknowledged, ``UNCONFIRMED`` otherwise.
        """
        if not isinstance(payload, str):
            try:
                payload = json.dumps(payload)
            except Exception as error:
                logger.error(
                    "Failed to serialise usage records: %s", type(error).__name__
                )
                return UNCONFIRMED

        with self._lock:
            try:
                self._increment += 1
                redis_key = f"{self._prefix}:{self._increment}"
                redis_pipeline = self._redis_client.pipeline()
                redis_pipeline.set(
                    name=redis_key,
                    value=payload,
                )
                redis_pipeline.zadd(
                    name=REDIS_SORTED_SET_NAME,
                    mapping={redis_key: time.time()},
                )
                results = redis_pipeline.execute()
                if not all(results):
                    logger.error(
                        "Failed to store usage records (partial insert): "
                        "%s of %s commands succeeded",
                        sum(1 for result in results if result),
                        len(results),
                    )
                    return UNCONFIRMED
            except Exception as error:
                logger.error("Failed to store usage records: %s", type(error).__name__)
                return UNCONFIRMED

        return True

    @staticmethod
    def full() -> bool:
        """Tell whether the queue refuses payloads.

        Returns:
            Always False.
        """
        return False

    def empty(self) -> bool:
        """Tell whether there is anything to read back.

        Returns:
            Always True: the queue is never read by the server.
        """
        return True

    def get_nowait(self) -> List[Dict[str, Any]]:
        """Read nothing back.

        Returns:
            Always an empty list.
        """
        return []
