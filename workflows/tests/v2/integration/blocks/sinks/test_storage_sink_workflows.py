"""Compiled workflows with the S3 and event writer sinks.

These run the real compiler, session and executor. S3 uploads go to a
recording fake client (the unit tests use a stubbed boto3 client);
events go to a loopback HTTP service on ``127.0.0.1``. Credentials are
invented::

    name, count ─▶ v2/csv_formatter@v1 ─▶ v2/s3_sink@v1 ◀─ secrets store (host values)
    image, boxes ─▶ v2/event_writer_sink@v1 ─▶ 127.0.0.1 (+ sink_effects)
"""

import json
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterator, List

import pytest
import torch
from roboflow_workflows.v2.blocks.formatters.csv import CsvFormatterBlock
from roboflow_workflows.v2.blocks.secrets import (
    EnvironmentSecretsStoreBlock,
    StaticSecrets,
)
from roboflow_workflows.v2.blocks.sinks.event_writer import EventWriterSinkBlock
from roboflow_workflows.v2.blocks.sinks.runtime import BackgroundEffects
from roboflow_workflows.v2.blocks.sinks.s3 import S3SinkBlock
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.secrets import SECRET_KIND
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import StepExecutionError

from inference_models.models.base.object_detection import Detections

KEY_ID = "AKIATESTKEYID000"
SECRET_KEY = "test-SECRET-access-key-9c1e"
CATALOGUE = Catalogue(
    [
        CsvFormatterBlock,
        EnvironmentSecretsStoreBlock,
        S3SinkBlock,
        EventWriterSinkBlock,
    ],
    kinds=[SECRET_KIND],
)


def _output(name: str, selector: str) -> Dict[str, str]:
    return {"type": "JsonField", "name": name, "selector": selector}


def _clock(tz: Any) -> datetime:
    return datetime(2026, 10, 13, 8, 0, tzinfo=timezone.utc).astimezone(tz)


def _s3_plan(**sink: Any) -> Any:
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowBatchInput", "name": "name", "kind": ["string"]},
            {"type": "WorkflowBatchInput", "name": "count", "kind": ["integer"]},
        ],
        "steps": [
            {
                "type": "v2/environment_secrets_store@v1",
                "name": "secrets",
                "variables_storing_secrets": ["AWS_KEY_ID", "AWS_SECRET"],
            },
            {
                "type": "v2/csv_formatter@v1",
                "name": "csv",
                "columns_data": {"name": "$inputs.name", "count": "$inputs.count"},
            },
            {
                "type": "v2/s3_sink@v1",
                "name": "s3",
                "content": "$steps.csv.csv_content",
                "output_mode": "append_log",
                "bucket_name": "results",
                "s3_prefix": "logs",
                "aws_access_key_id": "$steps.secrets.aws_key_id",
                "aws_secret_access_key": "$steps.secrets.aws_secret",
                **sink,
            },
        ],
        "outputs": [
            _output("error_status", "$steps.s3.error_status"),
            _output("message", "$steps.s3.message"),
        ],
    }
    plan = compile_workflow(definition, catalogue=CATALOGUE)

    return plan


class RecordingS3:
    """Client factory whose client records ``put_object`` calls.

    The unit tests cover a real boto3 client under ``Stubber``; here only
    what the compiled step uploads matters.
    """

    def __init__(self) -> None:
        self.credentials: List[Dict[str, Any]] = []
        self.uploads: List[Dict[str, Any]] = []

    def __call__(self, **credentials: Any) -> "RecordingS3":
        self.credentials.append(credentials)
        return self

    def put_object(self, **request: Any) -> None:
        self.uploads.append(request)


def _host_secrets() -> StaticSecrets:
    return StaticSecrets({"AWS_KEY_ID": KEY_ID, "AWS_SECRET": SECRET_KEY})


def test_append_log_reuploads_one_growing_object_with_store_credentials() -> None:
    s3 = RecordingS3()
    session = _s3_plan().create_session(
        resources={"secrets": _host_secrets(), "s3_client_factory": s3}
    )

    # when: two runs; the CSV formatter makes one CSV text per run
    rows = [
        session.run({"name": [name], "count": [count]}).rows()[0]
        for name, count in (("a", 2), ("b", 5))
    ]

    assert [row["error_status"] for row in rows] == [False, False]
    first, second = (upload["Body"].decode() for upload in s3.uploads)
    assert first.splitlines()[0] == "name,count,timestamp"
    assert second.startswith(first)
    assert [line.split(",")[:2] for line in second.splitlines()] == [
        ["name", "count"],
        ["a", "2"],
        ["b", "5"],
    ]
    keys = {upload["Key"] for upload in s3.uploads}
    assert len(keys) == 1
    assert rows[1]["message"] == f"Data saved successfully to s3://results/{keys.pop()}"
    assert s3.credentials[0] == {
        "aws_access_key_id": KEY_ID,
        "aws_secret_access_key": SECRET_KEY,
        "aws_region": None,
    }


def test_upload_error_reaches_outputs_scrubbed() -> None:
    class LeakyClient:
        def put_object(self, **request: Any) -> None:
            raise RuntimeError(f"rejected {SECRET_KEY}")

    session = _s3_plan().create_session(
        resources={
            "secrets": _host_secrets(),
            "s3_client_factory": lambda **credentials: LeakyClient(),
        }
    )

    rows = session.run({"name": ["a"], "count": [1]}).rows()

    assert rows == [
        {
            "error_status": True,
            "message": "rejected <secret AWS_SECRET>",
        }
    ]


def test_missing_store_secret_fails_the_step_without_upload() -> None:
    s3 = RecordingS3()
    session = _s3_plan().create_session(
        resources={
            "secrets": StaticSecrets({"AWS_KEY_ID": KEY_ID}),
            "s3_client_factory": s3,
        }
    )

    with pytest.raises(StepExecutionError) as raised:
        session.run({"name": ["a"], "count": [1]})

    assert "ambient AWS credentials are not used" in str(raised.value.__cause__)
    assert s3.credentials == []


def test_max_entries_below_one_is_rejected_at_compile_time() -> None:
    with pytest.raises(Exception, match="max_entries_per_file"):
        _s3_plan(max_entries_per_file=0)


class EventService:
    """Loopback Event Ingestion Service answering 201 with an id."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        service = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                service.events.append(json.loads(self.rfile.read(length)))
                body = json.dumps({"id": f"evt-{len(service.events)}"}).encode()
                self.send_response(201)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: Any) -> None:
                return None

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self.url = "http://{}:{}".format(*self._server.server_address[:2])

    def __enter__(self) -> "EventService":
        self._thread.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def service() -> Iterator[EventService]:
    with EventService() as running:
        yield running


def _event_plan(url: str, **sink: Any) -> Any:
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]},
            {
                "type": "WorkflowBatchInput",
                "name": "boxes",
                "kind": ["object_detection_prediction"],
            },
            {"type": "WorkflowParameter", "name": "line"},
        ],
        "steps": [
            {
                "type": "v2/event_writer_sink@v1",
                "name": "events",
                "event_ingestion_url": url,
                "event_schema": "inventory_count",
                "location": "dock-2",
                "item_type": "pallet",
                "output_image": "$inputs.image",
                "object_detections": "$inputs.boxes",
                "custom_metadata": {"line": "$inputs.line"},
                **sink,
            }
        ],
        "outputs": [
            _output("error_status", "$steps.events.error_status"),
            _output("event_id", "$steps.events.event_id"),
            _output("message", "$steps.events.message"),
        ],
    }
    plan = compile_workflow(definition, catalogue=CATALOGUE)

    return plan


def _frame_inputs(frames: int) -> Dict[str, Any]:
    boxes = Detections(
        xyxy=torch.tensor([[2.0, 2.0, 10.0, 6.0]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([0.75]),
        image_metadata={"class_names": {0: "pallet"}},
    )
    images = [
        ImageData.from_tensor(torch.zeros((3, 16, 16), dtype=torch.uint8))
        for _ in range(frames)
    ]
    inputs = {"image": images, "boxes": [boxes] * frames, "line": "A1"}

    return inputs


def test_sync_events_return_ids_and_carry_boxes(service: EventService) -> None:
    session = _event_plan(service.url, fire_and_forget=False).create_session(
        resources={"wall_clock": _clock}
    )

    rows = session.run(_frame_inputs(2)).rows()

    assert [row["event_id"] for row in rows] == ["evt-1", "evt-2"]
    event = service.events[0]
    assert event["event_schema"] == "inventory_count"
    assert event["event_data"] == {"location": "dock-2", "itemType": "pallet"}
    assert event["custom_metadata"] == {"line": "A1"}
    assert event["images"][0]["objectDetections"] == [
        {
            "class": "pallet",
            "x": 6.0,
            "y": 4.0,
            "width": 8.0,
            "height": 4.0,
            "confidence": 0.75,
        }
    ]


def test_fire_and_forget_events_go_through_host_background_effects(
    service: EventService,
) -> None:
    with BackgroundEffects(max_workers=2) as effects:
        session = _event_plan(service.url).create_session(
            resources={"sink_effects": effects}
        )
        rows = session.run(_frame_inputs(3)).rows()
        assert effects.drain(timeout=10)

    assert {row["message"] for row in rows} == {"Event sent in background task"}
    assert {row["event_id"] for row in rows} == {""}
    assert len(service.events) == 3
    assert effects.stats().completed == 3


def test_unknown_event_schema_is_rejected_at_compile_time() -> None:
    with pytest.raises(Exception, match="event_schema"):
        _event_plan("http://127.0.0.1:9", event_schema="maintenance")
