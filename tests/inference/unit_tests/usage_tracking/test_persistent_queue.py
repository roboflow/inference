from typing_extensions import Any, List

from inference.core.env import LAMBDA
from inference.usage_tracking.persistent_queue import PersistentQueue


def test_empty():
    # given
    q = PersistentQueue(db_file_path=":memory:")

    # then
    assert q.empty() is True


def test_not_empty():
    # given
    q = PersistentQueue(db_file_path=":memory:")

    # when
    q.put("test")

    # then
    assert q.empty() is False


def test_get_nowait():
    # given
    q = PersistentQueue(db_file_path=":memory:")

    # when
    q.put("test")
    q.put("test")
    q.put("test")

    # then
    assert len(q.get_nowait()) == 3
    assert q.empty() is True
