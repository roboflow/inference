"""CPU-only tests execute the production accounting/queue functions in isolation."""

import ast
import time
import uuid
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Lock
from types import SimpleNamespace
from typing import *

import pytest

# Locate repository root without importing optional camera/GPU dependencies.
source = next(
    parent / "inference/core/interfaces/camera/video_source.py"
    for parent in Path(__file__).parents
    if (parent / "inference/core/interfaces/camera/video_source.py").exists()
)
tree = ast.parse(source.read_text())
names = {"SourceFrameAccounting", "get_from_queue", "decode_video_frame_to_buffer"}
selected = [node for node in tree.body if getattr(node, "name", None) in names]
namespace = dict(globals())
namespace.update(
    VideoFrame=lambda **kw: SimpleNamespace(**kw),
    UpdateSeverity=SimpleNamespace(DEBUG="DEBUG"),
)
events = []


def send_update(**kw):
    update = SimpleNamespace(event_type=kw["event_type"], payload=kw["payload"])
    for handler in kw["status_update_handlers"]:
        handler(update)


namespace["send_video_source_status_update"] = send_update
exec(
    compile(
        ast.fix_missing_locations(
            ast.Module(
                body=[
                    ast.ImportFrom(
                        module="__future__",
                        names=[ast.alias(name="annotations")],
                        level=0,
                    ),
                    *selected,
                ],
                type_ignores=[],
            )
        ),
        str(source),
        "exec",
    ),
    namespace,
)
Accounting = namespace["SourceFrameAccounting"]
get_queue = namespace["get_from_queue"]
decode = namespace["decode_video_frame_to_buffer"]


def test_eager_purge_records_each_discard_and_returns_same_latest():
    queue = Queue()
    for value in (11, 12, 13):
        queue.put(value)
    discarded, reads = [], []
    result = get_queue(
        queue,
        purge=True,
        on_discard=discarded.append,
        on_successful_read=lambda: reads.append(True),
    )
    assert result == 13
    assert discarded == [11, 12]
    assert len(reads) == 3 and queue.unfinished_tasks == 0


def test_lazy_and_default_callers_keep_original_queue_behavior():
    queue = Queue()
    queue.put(11)
    queue.put(12)
    discarded = []
    assert get_queue(queue, on_discard=discarded.append) == 11
    assert discarded == []
    assert get_queue(queue, purge=True) == 12
    assert get_queue(queue, timeout=0, purge=True, on_discard=discarded.append) is None


def test_accounting_bounded_causes_and_source_local_counts():
    first, second = Accounting(), Accounting()
    for event, payload in [
        ("FRAME_CAPTURED", {"frame_id": 3}),
        ("FRAME_ENQUEUED", {"frame_id": 3}),
        ("FRAME_CONSUMED", {"frame_id": 3}),
        ("FRAME_DROPPED", {"cause": "EAGER queue purge"}),
    ]:
        first(SimpleNamespace(event_type=event, payload=payload))
    for value in range(100):
        first(
            SimpleNamespace(event_type="FRAME_DROPPED", payload={"cause": str(value)})
        )
    result = first.snapshot()
    assert result["counts"] == {
        "captured": 1,
        "enqueued": 1,
        "returned": 1,
        "retrieve_failed": 0,
    }
    assert result["dropped_by_cause"] == {"EAGER queue purge": 1, "other": 100}
    assert result["last_frame_ids"]["returned"] == 3
    assert second.snapshot()["counts"]["captured"] == 0


def test_enqueue_and_retrieve_failure_events_do_not_change_selection():
    accounting = Accounting()
    queue = Queue()
    queue._frame_accounting_handlers = [accounting]
    producer = SimpleNamespace(retrieve=lambda: (True, "pixels"))
    monitor = SimpleNamespace(tick=lambda: None)
    assert decode(datetime.now(), 7, producer, queue, monitor, 0)
    assert queue.get_nowait().frame_id == 7
    producer.retrieve = lambda: (False, None)
    assert not decode(datetime.now(), 8, producer, queue, monitor, 0)
    assert accounting.snapshot()["counts"]["enqueued"] == 1
    assert accounting.snapshot()["counts"]["retrieve_failed"] == 1


def test_subsampling_emits_explicit_drop_without_retrieving():
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "VideoConsumer"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "consume_frame"
    )
    env = dict(
        namespace,
        FRAME_CAPTURED_EVENT="FRAME_CAPTURED",
        timedelta=__import__("datetime").timedelta,
    )
    drops = []
    env["send_frame_drop_update"] = lambda **kw: drops.append(kw)
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=[
                        ast.ImportFrom(
                            module="__future__",
                            names=[ast.alias(name="annotations")],
                            level=0,
                        ),
                        method,
                    ],
                    type_ignores=[],
                )
            ),
            str(source),
            "exec",
        ),
        env,
    )
    consumer = SimpleNamespace(
        _is_source_video_file=False,
        _timestamp_created=None,
        _frame_counter=0,
        _stream_consumption_pace_monitor=SimpleNamespace(tick=lambda: None, fps=30),
        _status_update_handlers=[],
        _video_fps_should_be_sub_sampled=lambda: True,
    )
    assert env["consume_frame"](
        consumer, SimpleNamespace(grab=lambda: True), 30, False, Queue(), True, 0
    )
    assert consumer._frame_counter == 1
    assert drops[0]["cause"] == "desired source FPS subsampling"
    assert drops[0]["frame_id"] == 1


def test_source_accounting_is_opt_in_and_does_not_mix_shared_handlers(monkeypatch):
    import os
    from threading import Event

    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VideoSource"
    )
    methods = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name in ("__init__", "frame_accounting")
    ]
    stub = ast.ClassDef(
        name="SourceSubset", bases=[], keywords=[], body=methods, decorator_list=[]
    )
    frame_type = type("Frame", (SimpleNamespace,), {})
    env = dict(
        namespace,
        os=os,
        Event=Event,
        VideoFrame=frame_type,
        sanitize_source_reference=lambda x: x,
        StreamState=SimpleNamespace(NOT_STARTED="idle"),
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=[
                        ast.ImportFrom(
                            module="__future__",
                            names=[ast.alias(name="annotations")],
                            level=0,
                        ),
                        stub,
                    ],
                    type_ignores=[],
                )
            ),
            str(source),
            "exec",
        ),
        env,
    )
    monkeypatch.setenv("ENABLE_RUNTIME_DIAGNOSTICS", "true")
    shared = []

    def create():
        consumer = SimpleNamespace(_frame_counter=3, _status_update_handlers=shared)
        return (
            env["SourceSubset"]("safe", Queue(), shared, None, consumer, None, 0),
            consumer,
        )

    first, consumer = create()
    second, _ = create()
    for handler in consumer._status_update_handlers:
        handler(SimpleNamespace(event_type="FRAME_CAPTURED", payload={"frame_id": 3}))
    first._frames_buffer.put(frame_type(frame_id=3))
    assert first.frame_accounting["counts"]["captured"] == 1
    assert first.frame_accounting["queue_pending"] == 1
    assert first.frame_accounting["queue_first_frame_id"] == 3
    assert second.frame_accounting["counts"]["captured"] == 0
    assert shared == []
    monkeypatch.setenv("ENABLE_RUNTIME_DIAGNOSTICS", "false")
    monkeypatch.setenv("INFERENCE_MODELS_RUNTIME_DIAGNOSTICS", "false")
    disabled, _ = create()
    assert disabled.frame_accounting is None


def test_generic_discard_callback_reports_none_items_too():
    queue = Queue()
    queue.put(None)
    queue.put(1)
    discarded = []
    assert get_queue(queue, purge=True, on_discard=discarded.append) == 1
    assert discarded == [None]


def test_generation_changes_on_new_producer_without_resetting_lifetime_counts():
    accounting = Accounting()
    accounting(SimpleNamespace(event_type="FRAME_CAPTURED", payload={"frame_id": 1}))
    first = accounting.snapshot()
    assert first["generation"] != Accounting().snapshot()["generation"]
    accounting.new_generation()
    second = accounting.snapshot()
    assert first["generation"] != second["generation"]
    assert second["counts"] == first["counts"]
