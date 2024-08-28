import sqlite3

from inference.usage_tracking.sqlite_queue import SQLiteQueue


def test_empty():
    # given
    conn = sqlite3.connect(":memory:")
    q = SQLiteQueue(connection=conn)

    # then
    assert q.empty(connection=conn) is True
    conn.close()


def test_not_empty():
    # given
    conn = sqlite3.connect(":memory:")
    q = SQLiteQueue(connection=conn)

    # when
    q.put("test", connection=conn)

    # then
    assert q.empty(connection=conn) is False
    conn.close()


def test_get_nowait():
    # given
    conn = sqlite3.connect(":memory:")
    q = SQLiteQueue(connection=conn)

    # when
    q.put("test", connection=conn)
    q.put("test", connection=conn)
    q.put("test", connection=conn)

    # then
    assert len(q.get_nowait(connection=conn)) == 3
    assert q.empty(connection=conn) is True
    conn.close()
