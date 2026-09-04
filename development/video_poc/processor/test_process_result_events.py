import json
import queue

from job_process import PROTOCOL_VERSION, bounded_child_event
from processor import EventBus, ProcessJobRun, _ChildWorker


def result_event(frame_id):
    return {
        "frameId": frame_id,
        "timestamp": "2026-08-14T12:00:00Z",
        "latencyMs": 12.5,
        "outputs": {
            "predictions": [{"class": "car", "confidence": 0.9}],
            "visualization": {
                "type": "image_ref",
                "output": "visualization",
            },
        },
    }


def test_child_result_queue_is_latest_value_and_never_grows():
    child = _ChildWorker.__new__(_ChildWorker)
    child._result_events = queue.Queue(maxsize=1)

    assert child.publish_result_event(result_event(1)) is True
    assert child.publish_result_event(result_event(2)) is True

    queued = child._result_events.get_nowait()
    assert queued["type"] == "result"
    assert queued["result"]["frameId"] == 2
    assert child._result_events.empty()


def test_parent_republishes_child_json_without_image_bytes():
    parent = ProcessJobRun.__new__(ProcessJobRun)
    parent.events = EventBus()
    event = bounded_child_event(
        {
            "version": PROTOCOL_VERSION,
            "type": "result",
            "result": result_event(7),
        }
    )

    parent._apply_child_event(event)

    published, cursor = parent.events.since(None)
    assert cursor == 1
    assert [json.loads(item) for item in published] == [result_event(7)]
    assert "pixels" not in published[0]
    assert "tensor" not in published[0]
