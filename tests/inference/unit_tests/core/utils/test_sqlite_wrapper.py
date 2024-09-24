import sqlite3

import pytest

from inference.core.utils.sqlite_wrapper import SQLiteWrapper


def test_count_empty_table():
    # given
    conn = sqlite3.connect(":memory:")
    q = SQLiteWrapper(
        db_file_path="", table_name="test", columns={"col1": "TEXT"}, connection=conn
    )

    # then
    assert q.count(connection=conn) == 0
    conn.close()


def test_insert():
    # given
    conn = sqlite3.connect(":memory:")
    q = SQLiteWrapper(
        db_file_path="", table_name="test", columns={"col1": "TEXT"}, connection=conn
    )

    # when
    q.insert(row={"col1": "lorem"}, connection=conn)

    # then
    assert q.count(connection=conn) == 1
    conn.close()


def test_insert_incorrect_columns():
    # given
    conn = sqlite3.connect(":memory:")
    q = SQLiteWrapper(
        db_file_path="", table_name="test", columns={"col1": "TEXT"}, connection=conn
    )

    with pytest.raises(ValueError):
        q.insert(row={"col2": "lorem"}, connection=conn)

    conn.close()


def test_select_no_limit():
    # given
    conn = sqlite3.connect(":memory:")
    q = SQLiteWrapper(
        db_file_path="", table_name="test", columns={"col1": "TEXT"}, connection=conn
    )

    # when
    q.insert(row={"col1": "lorem"}, connection=conn)
    q.insert(row={"col1": "ipsum"}, connection=conn)
    rows = q.select(connection=conn)

    # then
    assert rows == [{"id": 1, "col1": "lorem"}, {"id": 2, "col1": "ipsum"}]
    conn.close()


def test_select_with_exclusive():
    # given
    conn = sqlite3.connect(":memory:")
    q = SQLiteWrapper(
        db_file_path="", table_name="test", columns={"col1": "TEXT"}, connection=conn
    )

    # when
    q.insert(row={"col1": "lorem"}, connection=conn)
    q.insert(row={"col1": "ipsum"}, connection=conn)
    conn.commit()
    rows = q.select(connection=conn, with_exclusive=True)

    # then
    assert rows == [{"id": 1, "col1": "lorem"}, {"id": 2, "col1": "ipsum"}]
    conn.close()


def test_select_from_cursor():
    # given
    conn = sqlite3.connect(":memory:")
    curr = conn.cursor()
    q = SQLiteWrapper(
        db_file_path="", table_name="test", columns={"col1": "TEXT"}, connection=conn
    )

    # when
    q.insert(row={"col1": "lorem"}, connection=conn)
    q.insert(row={"col1": "ipsum"}, connection=conn)
    rows = q.select(cursor=curr)

    # then
    assert rows == [{"id": 1, "col1": "lorem"}, {"id": 2, "col1": "ipsum"}]
    conn.close()


def test_select_limit():
    # given
    conn = sqlite3.connect(":memory:")
    q = SQLiteWrapper(
        db_file_path="", table_name="test", columns={"col1": "TEXT"}, connection=conn
    )

    # when
    q.insert(row={"col1": "lorem"}, connection=conn)
    q.insert(row={"col1": "ipsum"}, connection=conn)
    rows = q.select(connection=conn, limit=1)

    # then
    assert rows == [
        {"id": 1, "col1": "lorem"},
    ]
    conn.close()


def test_flush_no_limit():
    # given
    conn = sqlite3.connect(":memory:")
    q = SQLiteWrapper(
        db_file_path="", table_name="test", columns={"col1": "TEXT"}, connection=conn
    )

    # when
    q.insert(row={"col1": "lorem"}, connection=conn)
    q.insert(row={"col1": "ipsum"}, connection=conn)
    conn.commit()
    rows = q.flush(connection=conn)

    # then
    assert rows == [{"id": 1, "col1": "lorem"}, {"id": 2, "col1": "ipsum"}]
    assert q.count(connection=conn) == 0
    conn.close()


def test_flush_limit():
    # given
    conn = sqlite3.connect(":memory:")
    q = SQLiteWrapper(
        db_file_path="", table_name="test", columns={"col1": "TEXT"}, connection=conn
    )

    # when
    q.insert(row={"col1": "lorem"}, connection=conn)
    q.insert(row={"col1": "ipsum"}, connection=conn)
    conn.commit()
    rows = q.flush(connection=conn, limit=1)

    # then
    assert rows == [
        {"id": 1, "col1": "lorem"},
    ]
    assert q.count(connection=conn) == 1
    conn.close()


def test_delete():
    # given
    conn = sqlite3.connect(":memory:")
    q = SQLiteWrapper(
        db_file_path="", table_name="test", columns={"col1": "TEXT"}, connection=conn
    )

    # when
    q.insert(row={"col1": "lorem"}, connection=conn)
    q.insert(row={"col1": "ipsum"}, connection=conn)
    q.insert(row={"col1": "dolor"}, connection=conn)
    rows = q.select(connection=conn)
    rows_to_be_deleted = rows[:-1]
    rows_to_be_kept = rows[-1:]
    deleted_rows = q.delete(connection=conn, rows=rows_to_be_deleted)

    # then
    assert deleted_rows == rows_to_be_deleted
    assert q.select(connection=conn) == rows_to_be_kept
    conn.close()


def test_delete_non_existent():
    # given
    conn = sqlite3.connect(":memory:")
    q = SQLiteWrapper(
        db_file_path="", table_name="test", columns={"col1": "TEXT"}, connection=conn
    )

    # when
    q.insert(row={"col1": "lorem"}, connection=conn)
    q.insert(row={"col1": "ipsum"}, connection=conn)
    rows = q.select(connection=conn)
    rows_to_be_deleted = [rows[0], {"col1": "dolor"}]
    rows_to_be_kept = rows[1]
    deleted_rows = q.delete(connection=conn, rows=rows_to_be_deleted)

    # then
    assert deleted_rows == [rows[0]]
    assert q.select(connection=conn) == [rows_to_be_kept]
    conn.close()


def test_delete_with_exclusive():
    # given
    conn = sqlite3.connect(":memory:")
    q = SQLiteWrapper(
        db_file_path="", table_name="test", columns={"col1": "TEXT"}, connection=conn
    )

    # when
    q.insert(row={"col1": "lorem"}, connection=conn)
    q.insert(row={"col1": "ipsum"}, connection=conn)
    q.insert(row={"col1": "dolor"}, connection=conn)
    conn.commit()
    rows = q.select(connection=conn)
    rows_to_be_deleted = rows[:-1]
    rows_to_be_kept = rows[-1:]
    deleted_rows = q.delete(
        connection=conn, rows=rows_to_be_deleted, with_exclusive=True
    )

    # then
    assert deleted_rows == rows_to_be_deleted
    assert q.select(connection=conn) == rows_to_be_kept
    conn.close()


def test_delete_from_cursor():
    # given
    conn = sqlite3.connect(":memory:")
    curr = conn.cursor()
    q = SQLiteWrapper(
        db_file_path="", table_name="test", columns={"col1": "TEXT"}, connection=conn
    )

    # when
    q.insert(row={"col1": "lorem"}, connection=conn)
    q.insert(row={"col1": "ipsum"}, connection=conn)
    q.insert(row={"col1": "dolor"}, connection=conn)
    rows = q.select(connection=conn)
    rows_to_be_deleted = rows[:-1]
    rows_to_be_kept = rows[-1:]
    deleted_rows = q.delete(cursor=curr, rows=rows_to_be_deleted)

    # then
    assert deleted_rows == rows_to_be_deleted
    assert q.select(connection=conn) == rows_to_be_kept
    conn.close()


def test_refresh_db_empty():
    # given
    conn = sqlite3.connect(":memory:")
    q = SQLiteWrapper(
        db_file_path="", table_name="test", columns={"col1": "TEXT"}, connection=conn
    )

    # when
    rows = [{"col1": "lorem"}, {"col1": "ipsum"}]
    refreshed_rows = q.refresh(connection=conn, rows=rows)

    # then
    assert refreshed_rows == [{"id": 1, "col1": "lorem"}, {"id": 2, "col1": "ipsum"}]
    assert q.count(connection=conn) == 2
    conn.close()


def test_refresh_rows_exist():
    # given
    conn = sqlite3.connect(":memory:")
    q = SQLiteWrapper(
        db_file_path="", table_name="test", columns={"col1": "TEXT"}, connection=conn
    )

    # when
    q.insert(row={"col1": "lorem"}, connection=conn)
    q.insert(row={"col1": "ipsum"}, connection=conn)
    conn.commit()
    rows = q.select(connection=conn)
    rows[0]["col1"] = "foo"
    rows[1]["col1"] = "bar"
    rows.append({"col1": "baz"})
    refreshed_rows = q.refresh(connection=conn, rows=rows)

    # then
    assert refreshed_rows == [
        {"id": 1, "col1": "foo"},
        {"id": 2, "col1": "bar"},
        {"id": 3, "col1": "baz"},
    ]
    assert q.count(connection=conn) == 3
    conn.close()
