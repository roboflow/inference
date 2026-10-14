"""Compiled workflows with the S3 and event writer sinks.

These run the real compiler, session and executor. S3 uploads go to a
recording fake client (the unit tests use a stubbed boto3 client);
events go to a loopback HTTP service on ``127.0.0.1``. Credentials are
invented::

    name, count ─▶ roboflow_core/csv_formatter@v1 ─▶ roboflow_core/s3_sink@v1 ◀─ secrets store (host values)
"""

import json
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterator, List

import pytest
import torch
from roboflow_workflows.v2.blocks.formatters.csv_formatter.v1 import CsvFormatterBlock
from roboflow_workflows.v2.blocks.secrets import (
    EnvironmentSecretsStoreBlock,
    StaticSecrets,
)

from roboflow_workflows.v2.blocks.sinks.runtime import BackgroundEffects
from roboflow_workflows.v2.blocks.sinks.s3_sink.v1 import S3SinkBlock
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
                "type": "roboflow_core/environment_secrets_store@v1",
                "name": "secrets",
                "variables_storing_secrets": ["AWS_KEY_ID", "AWS_SECRET"],
            },
            {
                "type": "roboflow_core/csv_formatter@v1",
                "name": "csv",
                "columns_data": {"name": "$inputs.name", "count": "$inputs.count"},
            },
            {
                "type": "roboflow_core/s3_sink@v1",
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














