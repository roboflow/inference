import sqlite3

from inference.core.utils.sqlite_wrapper import SQLiteWrapper


def test_count_empty_table():
    # given
    conn = sqlite3.connect(":memory:")
    q = SQLiteWrapper(db_file_path="", table_name="test", columns={"col1": "TEXT"}, connection=conn)

    # then
    assert q.count(connection=conn) == 0
    conn.close()


def test_insert():
    # given
    conn = sqlite3.connect(":memory:")
    q = SQLiteWrapper(db_file_path="", table_name="test", columns={"col1": "TEXT"}, connection=conn)

    # when
    q.insert(values={"col1": "lorem"}, connection=conn)

    # then
    assert q.count(connection=conn) == 1
    conn.close()


def test_select_no_limit():
    # given
    conn = sqlite3.connect(":memory:")
    q = SQLiteWrapper(db_file_path="", table_name="test", columns={"col1": "TEXT"}, connection=conn)

    # when
    q.insert(values={"col1": "lorem"}, connection=conn)
    q.insert(values={"col1": "ipsum"}, connection=conn)
    values = q.select(connection=conn)

    # then
    assert values == [
        {"id": 1, "col1": "lorem"},
        {"id": 2, "col1": "ipsum"}
    ]
    conn.close()


def test_select_limit():
    # given
    conn = sqlite3.connect(":memory:")
    q = SQLiteWrapper(db_file_path="", table_name="test", columns={"col1": "TEXT"}, connection=conn)

    # when
    q.insert(values={"col1": "lorem"}, connection=conn)
    q.insert(values={"col1": "ipsum"}, connection=conn)
    values = q.select(connection=conn, limit=1)

    # then
    assert values == [
        {"id": 1, "col1": "lorem"},
    ]
    conn.close()


def test_flush_no_limit():
    # given
    conn = sqlite3.connect(":memory:")
    q = SQLiteWrapper(db_file_path="", table_name="test", columns={"col1": "TEXT"}, connection=conn)

    # when
    q.insert(values={"col1": "lorem"}, connection=conn)
    q.insert(values={"col1": "ipsum"}, connection=conn)
    values = q.flush(connection=conn)

    # then
    assert values == [
        {"id": 1, "col1": "lorem"},
        {"id": 2, "col1": "ipsum"}
    ]
    assert q.count(connection=conn) == 0
    conn.close()


def test_flush_limit():
    # given
    conn = sqlite3.connect(":memory:")
    q = SQLiteWrapper(db_file_path="", table_name="test", columns={"col1": "TEXT"}, connection=conn)

    # when
    q.insert(values={"col1": "lorem"}, connection=conn)
    q.insert(values={"col1": "ipsum"}, connection=conn)
    values = q.flush(connection=conn, limit=1)

    # then
    assert values == [
        {"id": 1, "col1": "lorem"},
    ]
    assert q.count(connection=conn) == 1
    conn.close()
