"""Unit tests for the Event Writer sink block (v1).

Covers:
- Manifest validation (all 4 schemas, custom_metadata selector values)
- _build_event_data (per-schema field mapping, None stripping)
- run() disable_sink, synchronous, fire-and-forget paths
- run() API key env var handling
- run() custom_metadata payload inclusion
- _detections_to_v2_object_detections
- _detections_to_v2_instance_segmentations
- _classifications_to_v2
- _keypoints_to_v2
- _build_image_entry
- _execute_event_request (success, HTTP error, timeout, connection error)
"""

import os
from concurrent.futures import ThreadPoolExecutor
from unittest import mock
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import supervision as sv
from fastapi import BackgroundTasks

from inference.enterprise.workflows.enterprise_blocks.sinks.event_writer.v1 import (
    BlockManifest,
    EventWriterSinkBlockV1,
    _build_event_data,
    _build_image_entry,
    _classifications_to_v2,
    _detections_to_v2_instance_segmentations,
    _detections_to_v2_object_detections,
    _execute_event_request,
    _keypoints_to_v2,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image(base64: str = "abc123") -> MagicMock:
    img = MagicMock()
    img.base64_image = base64
    return img


def _make_block(background_tasks=None, thread_pool=None) -> EventWriterSinkBlockV1:
    return EventWriterSinkBlockV1(
        background_tasks=background_tasks,
        thread_pool_executor=thread_pool,
    )


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------


class TestBlockManifest:

    def test_quality_check_schema_valid(self):
        raw = {
            "type": "roboflow_enterprise/event_writer_sink@v1",
            "name": "qc_writer",
            "event_ingestion_url": "http://localhost:8001",
            "event_schema": "quality_check",
            "output_image": "$inputs.image",
            "qc_result": "pass",
            "fire_and_forget": False,
            "disable_sink": False,
        }
        result = BlockManifest.model_validate(raw)
        assert result.event_schema == "quality_check"
        assert result.qc_result == "pass"

    def test_inventory_count_schema_valid(self):
        raw = {
            "type": "roboflow_enterprise/event_writer_sink@v1",
            "name": "inv_writer",
            "event_ingestion_url": "http://localhost:8001",
            "event_schema": "inventory_count",
            "output_image": "$inputs.image",
            "location": "warehouse-A",
            "item_count": 42,
            "item_type": "widget",
            "fire_and_forget": False,
            "disable_sink": False,
        }
        result = BlockManifest.model_validate(raw)
        assert result.event_schema == "inventory_count"
        assert result.location == "warehouse-A"
        assert result.item_count == 42

    def test_safety_alert_schema_valid(self):
        raw = {
            "type": "roboflow_enterprise/event_writer_sink@v1",
            "name": "safety_writer",
            "event_ingestion_url": "http://localhost:8001",
            "event_schema": "safety_alert",
            "output_image": "$inputs.image",
            "alert_type": "no_hardhat",
            "severity": "high",
            "alert_description": "Worker without PPE",
            "fire_and_forget": False,
            "disable_sink": False,
        }
        result = BlockManifest.model_validate(raw)
        assert result.event_schema == "safety_alert"
        assert result.severity == "high"

    def test_custom_schema_valid(self):
        raw = {
            "type": "roboflow_enterprise/event_writer_sink@v1",
            "name": "custom_writer",
            "event_ingestion_url": "http://localhost:8001",
            "event_schema": "custom",
            "output_image": "$inputs.image",
            "custom_value": "my custom value",
            "fire_and_forget": False,
            "disable_sink": False,
        }
        result = BlockManifest.model_validate(raw)
        assert result.event_schema == "custom"
        assert result.custom_value == "my custom value"

    def test_custom_metadata_with_literal_values(self):
        raw = {
            "type": "roboflow_enterprise/event_writer_sink@v1",
            "name": "writer",
            "event_ingestion_url": "http://localhost:8001",
            "event_schema": "custom",
            "output_image": "$inputs.image",
            "custom_metadata": {
                "line": "A1",
                "shift": "morning",
                "count": 5,
                "rate": 3.14,
                "active": True,
            },
            "fire_and_forget": False,
            "disable_sink": False,
        }
        result = BlockManifest.model_validate(raw)
        assert result.custom_metadata["line"] == "A1"
        assert result.custom_metadata["count"] == 5
        assert result.custom_metadata["rate"] == 3.14
        assert result.custom_metadata["active"] is True

    def test_custom_metadata_with_selector_values(self):
        """Selector strings must be accepted as dict values so the workflow engine
        can resolve them at runtime."""
        raw = {
            "type": "roboflow_enterprise/event_writer_sink@v1",
            "name": "writer",
            "event_ingestion_url": "http://localhost:8001",
            "event_schema": "custom",
            "output_image": "$inputs.image",
            "custom_metadata": {
                "dyn_str": "$steps.formatter.label",
                "dyn_int": "$steps.formatter.count",
                "lit_str": "hello",
            },
            "fire_and_forget": False,
            "disable_sink": False,
        }
        result = BlockManifest.model_validate(raw)
        assert result.custom_metadata["dyn_str"] == "$steps.formatter.label"
        assert result.custom_metadata["dyn_int"] == "$steps.formatter.count"
        assert result.custom_metadata["lit_str"] == "hello"

    def test_custom_metadata_defaults_to_empty_dict(self):
        raw = {
            "type": "roboflow_enterprise/event_writer_sink@v1",
            "name": "writer",
            "event_ingestion_url": "http://localhost:8001",
            "event_schema": "custom",
            "output_image": "$inputs.image",
            "fire_and_forget": False,
            "disable_sink": False,
        }
        result = BlockManifest.model_validate(raw)
        assert result.custom_metadata == {}

    def test_fire_and_forget_accepts_selector(self):
        raw = {
            "type": "roboflow_enterprise/event_writer_sink@v1",
            "name": "writer",
            "event_ingestion_url": "http://localhost:8001",
            "event_schema": "custom",
            "output_image": "$inputs.image",
            "fire_and_forget": "$inputs.fire_and_forget",
            "disable_sink": False,
        }
        result = BlockManifest.model_validate(raw)
        assert result.fire_and_forget == "$inputs.fire_and_forget"

    def test_external_id_accepted(self):
        raw = {
            "type": "roboflow_enterprise/event_writer_sink@v1",
            "name": "writer",
            "event_ingestion_url": "http://localhost:8001",
            "event_schema": "quality_check",
            "output_image": "$inputs.image",
            "external_id": "batch-001",
            "fire_and_forget": False,
            "disable_sink": False,
        }
        result = BlockManifest.model_validate(raw)
        assert result.external_id == "batch-001"


# ---------------------------------------------------------------------------
# _build_event_data
# ---------------------------------------------------------------------------


class TestBuildEventData:

    def test_quality_check_all_fields(self):
        data = _build_event_data("quality_check", qc_result="pass", external_id="ext-1")
        assert data == {"result": "pass", "externalId": "ext-1"}

    def test_quality_check_strips_none(self):
        data = _build_event_data("quality_check", qc_result="fail")
        assert data == {"result": "fail"}
        assert "externalId" not in data

    def test_inventory_count_all_fields(self):
        data = _build_event_data(
            "inventory_count",
            location="zone-A",
            item_count=10,
            item_type="widget",
            external_id="ext-2",
        )
        assert data == {
            "location": "zone-A",
            "itemCount": 10,
            "itemType": "widget",
            "externalId": "ext-2",
        }

    def test_inventory_count_partial_fields(self):
        data = _build_event_data("inventory_count", location="zone-B", item_count=5)
        assert data["location"] == "zone-B"
        assert data["itemCount"] == 5
        assert "itemType" not in data

    def test_safety_alert_all_fields(self):
        data = _build_event_data(
            "safety_alert",
            alert_type="no_hardhat",
            severity="high",
            alert_description="PPE violation",
            external_id="ext-3",
        )
        assert data == {
            "alertType": "no_hardhat",
            "severity": "high",
            "description": "PPE violation",
            "externalId": "ext-3",
        }

    def test_custom_all_fields(self):
        data = _build_event_data(
            "custom", custom_value="hello world", external_id="ext-4"
        )
        assert data == {"value": "hello world", "externalId": "ext-4"}

    def test_unknown_schema_returns_empty(self):
        data = _build_event_data("operator_feedback")
        assert data == {}


# ---------------------------------------------------------------------------
# _detections_to_v2_object_detections
# ---------------------------------------------------------------------------


class TestDetectionsToV2ObjectDetections:

    def test_none_returns_empty_list(self):
        assert _detections_to_v2_object_detections(None) == []

    def test_converts_correctly(self):
        detections = sv.Detections(
            xyxy=np.array([[10.0, 20.0, 110.0, 120.0], [50.0, 60.0, 150.0, 160.0]]),
            confidence=np.array([0.9, 0.8]),
            data={"class_name": np.array(["person", "car"])},
        )
        result = _detections_to_v2_object_detections(detections)
        assert len(result) == 2
        assert result[0]["class"] == "person"
        assert result[0]["x"] == pytest.approx(60.0)
        assert result[0]["y"] == pytest.approx(70.0)
        assert result[0]["width"] == pytest.approx(100.0)
        assert result[0]["height"] == pytest.approx(100.0)
        assert result[0]["confidence"] == pytest.approx(0.9)

    def test_missing_confidence_uses_zero(self):
        detections = sv.Detections(
            xyxy=np.array([[0.0, 0.0, 10.0, 10.0]]),
            confidence=None,
            data={"class_name": np.array(["thing"])},
        )
        result = _detections_to_v2_object_detections(detections)
        assert result[0]["confidence"] == 0.0

    def test_missing_class_name_uses_unknown(self):
        detections = sv.Detections(
            xyxy=np.array([[0.0, 0.0, 10.0, 10.0]]),
            confidence=np.array([0.5]),
            data={},
        )
        result = _detections_to_v2_object_detections(detections)
        assert result[0]["class"] == "unknown"


# ---------------------------------------------------------------------------
# _detections_to_v2_instance_segmentations
# ---------------------------------------------------------------------------


class TestDetectionsToV2InstanceSegmentations:

    def test_none_returns_empty_list(self):
        assert _detections_to_v2_instance_segmentations(None) == []

    def test_with_valid_mask_produces_entry(self):
        # Create a filled square mask — findContours will produce a polygon
        mask = np.zeros((200, 200), dtype=bool)
        mask[50:100, 50:100] = True
        masks = np.array([mask])
        detections = sv.Detections(
            xyxy=np.array([[10.0, 20.0, 110.0, 120.0]]),
            confidence=np.array([0.85]),
            data={"class_name": np.array(["person"])},
            mask=masks,
        )
        result = _detections_to_v2_instance_segmentations(detections)
        assert len(result) == 1
        assert result[0]["class"] == "person"
        assert len(result[0]["points"]) >= 3

    def test_no_mask_entry_excluded(self):
        """Without a mask, the detection has no polygon points (< 3) and is excluded."""
        detections = sv.Detections(
            xyxy=np.array([[10.0, 20.0, 110.0, 120.0]]),
            confidence=np.array([0.85]),
            data={"class_name": np.array(["person"])},
        )
        result = _detections_to_v2_instance_segmentations(detections)
        assert result == []


# ---------------------------------------------------------------------------
# _classifications_to_v2
# ---------------------------------------------------------------------------


class TestClassificationsToV2:

    def test_none_returns_empty_list(self):
        assert _classifications_to_v2(None) == []

    def test_dict_format(self):
        predictions = {
            "class_name": ["cat", "dog"],
            "confidence": [0.7, 0.3],
        }
        result = _classifications_to_v2(predictions)
        assert len(result) == 2
        assert result[0] == {"class": "cat", "confidence": pytest.approx(0.7)}
        assert result[1] == {"class": "dog", "confidence": pytest.approx(0.3)}

    def test_sv_detections_format(self):
        preds = sv.Detections(
            xyxy=np.array([[0.0, 0.0, 1.0, 1.0]]),
            confidence=np.array([0.95]),
            data={"class_name": np.array(["hardhat"])},
        )
        result = _classifications_to_v2(preds)
        assert len(result) == 1
        assert result[0]["class"] == "hardhat"
        assert result[0]["confidence"] == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# _keypoints_to_v2
# ---------------------------------------------------------------------------


class TestKeypointsToV2:

    def test_none_returns_empty_list(self):
        assert _keypoints_to_v2(None) == []

    def test_converts_keypoints_correctly(self):
        detections = sv.Detections(
            xyxy=np.array([[10.0, 20.0, 110.0, 120.0]]),
            confidence=np.array([0.9]),
            data={
                "class_name": np.array(["person"]),
                "keypoints_xy": np.array([[[30.0, 40.0], [50.0, 60.0], [70.0, 80.0]]]),
            },
        )
        result = _keypoints_to_v2(detections)
        assert len(result) == 1
        kps = result[0]["keypoints"]
        assert len(kps) == 3
        assert kps[0] == {"id": 0, "x": pytest.approx(30.0), "y": pytest.approx(40.0)}

    def test_detection_with_no_keypoints_excluded(self):
        """Entry with empty keypoints list is excluded from output."""
        detections = sv.Detections(
            xyxy=np.array([[10.0, 20.0, 110.0, 120.0]]),
            confidence=np.array([0.9]),
            data={"class_name": np.array(["person"])},
        )
        result = _keypoints_to_v2(detections)
        assert result == []


# ---------------------------------------------------------------------------
# _build_image_entry
# ---------------------------------------------------------------------------


class TestBuildImageEntry:

    def test_output_image_only(self):
        img = _make_image("base64output")
        entry = _build_image_entry(output_image=img)
        assert entry == {"base64Image": "base64output"}

    def test_with_input_image(self):
        out_img = _make_image("out")
        in_img = _make_image("inp")
        entry = _build_image_entry(output_image=out_img, input_image=in_img)
        assert entry["inputBase64Image"] == "inp"

    def test_with_image_label(self):
        img = _make_image("out")
        entry = _build_image_entry(output_image=img, image_label="defect-scan")
        assert entry["label"] == "defect-scan"

    def test_with_object_detections(self):
        img = _make_image("out")
        detections = sv.Detections(
            xyxy=np.array([[0.0, 0.0, 50.0, 50.0]]),
            confidence=np.array([0.9]),
            data={"class_name": np.array(["person"])},
        )
        entry = _build_image_entry(output_image=img, object_detections=detections)
        assert "objectDetections" in entry
        assert len(entry["objectDetections"]) == 1

    def test_empty_detections_not_included(self):
        img = _make_image("out")
        empty = sv.Detections.empty()
        entry = _build_image_entry(output_image=img, object_detections=empty)
        assert "objectDetections" not in entry


# ---------------------------------------------------------------------------
# _execute_event_request
# ---------------------------------------------------------------------------


class TestExecuteEventRequest:

    def test_201_returns_event_id(self):
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "event-abc-123"}

        with patch("requests.post", return_value=mock_response):
            error, msg, event_id = _execute_event_request(
                url="http://localhost:8001/v2/events",
                payload={"event_schema": "custom"},
                api_key=None,
                timeout=5,
            )

        assert error is False
        assert event_id == "event-abc-123"
        assert "successfully" in msg

    def test_api_key_added_to_header(self):
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "x"}

        with patch("requests.post", return_value=mock_response) as mock_post:
            _execute_event_request(
                url="http://localhost:8001/v2/events",
                payload={},
                api_key="my-secret-key",
                timeout=5,
            )

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["X-API-Key"] == "my-secret-key"

    def test_no_api_key_no_header(self):
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "x"}

        with patch("requests.post", return_value=mock_response) as mock_post:
            _execute_event_request(
                url="http://localhost:8001/v2/events",
                payload={},
                api_key=None,
                timeout=5,
            )

        headers = mock_post.call_args.kwargs["headers"]
        assert "X-API-Key" not in headers

    def test_400_returns_error(self):
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"detail": "bad request"}

        with patch("requests.post", return_value=mock_response):
            error, msg, event_id = _execute_event_request(
                url="http://localhost:8001/v2/events",
                payload={},
                api_key=None,
                timeout=5,
            )

        assert error is True
        assert event_id == ""
        assert "400" in msg

    def test_529_returns_backpressure_message(self):
        mock_response = MagicMock()
        mock_response.status_code = 529
        mock_response.json.return_value = {"detail": "at capacity"}

        with patch("requests.post", return_value=mock_response):
            error, msg, event_id = _execute_event_request(
                url="http://localhost:8001/v2/events",
                payload={},
                api_key=None,
                timeout=5,
            )

        assert error is True
        assert "529" in msg
        assert "backpressure" in msg.lower() or "capacity" in msg.lower()

    def test_timeout_returns_error(self):
        import requests

        with patch("requests.post", side_effect=requests.exceptions.Timeout()):
            error, msg, event_id = _execute_event_request(
                url="http://localhost:8001/v2/events",
                payload={},
                api_key=None,
                timeout=5,
            )

        assert error is True
        assert "timed out" in msg.lower()
        assert event_id == ""

    def test_connection_error_returns_error(self):
        import requests

        with patch(
            "requests.post", side_effect=requests.exceptions.ConnectionError("refused")
        ):
            error, msg, event_id = _execute_event_request(
                url="http://localhost:8001/v2/events",
                payload={},
                api_key=None,
                timeout=5,
            )

        assert error is True
        assert event_id == ""


# ---------------------------------------------------------------------------
# EventWriterSinkBlockV1.run()
# ---------------------------------------------------------------------------


class TestEventWriterRun:

    def test_disable_sink_skips_request(self):
        block = _make_block()
        result = block.run(
            event_ingestion_url="http://localhost:8001",
            event_schema="custom",
            output_image=_make_image(),
            fire_and_forget=False,
            disable_sink=True,
            request_timeout=5,
        )
        assert result["error_status"] is False
        assert "disabled" in result["message"]

    def test_synchronous_success_returns_event_id(self):
        block = _make_block()
        with patch(
            "inference.enterprise.workflows.enterprise_blocks.sinks.event_writer.v1._execute_event_request",
            return_value=(False, "Event created successfully", "evt-999"),
        ):
            result = block.run(
                event_ingestion_url="http://localhost:8001",
                event_schema="custom",
                output_image=_make_image(),
                fire_and_forget=False,
                disable_sink=False,
                request_timeout=5,
                custom_value="hello",
            )

        assert result["error_status"] is False
        assert result["event_id"] == "evt-999"

    def test_synchronous_failure_returns_error(self):
        block = _make_block()
        with patch(
            "inference.enterprise.workflows.enterprise_blocks.sinks.event_writer.v1._execute_event_request",
            return_value=(True, "HTTP 500: server error", ""),
        ):
            result = block.run(
                event_ingestion_url="http://localhost:8001",
                event_schema="custom",
                output_image=_make_image(),
                fire_and_forget=False,
                disable_sink=False,
                request_timeout=5,
            )

        assert result["error_status"] is True
        assert result["event_id"] == ""

    def test_fire_and_forget_with_background_tasks(self):
        bg = MagicMock(spec=BackgroundTasks)
        block = _make_block(background_tasks=bg)

        with patch(
            "inference.enterprise.workflows.enterprise_blocks.sinks.event_writer.v1._execute_event_request",
        ):
            result = block.run(
                event_ingestion_url="http://localhost:8001",
                event_schema="custom",
                output_image=_make_image(),
                fire_and_forget=True,
                disable_sink=False,
                request_timeout=5,
            )

        bg.add_task.assert_called_once()
        assert result["error_status"] is False
        assert result["event_id"] == ""
        assert "background" in result["message"].lower()

    def test_fire_and_forget_with_thread_pool(self):
        pool = MagicMock(spec=ThreadPoolExecutor)
        block = _make_block(thread_pool=pool)

        result = block.run(
            event_ingestion_url="http://localhost:8001",
            event_schema="custom",
            output_image=_make_image(),
            fire_and_forget=True,
            disable_sink=False,
            request_timeout=5,
        )

        pool.submit.assert_called_once()
        assert result["error_status"] is False

    def test_url_trailing_slash_stripped(self):
        block = _make_block()
        captured = {}

        def capture(**kwargs):
            captured.update(kwargs)
            return (False, "ok", "id-1")

        with patch(
            "inference.enterprise.workflows.enterprise_blocks.sinks.event_writer.v1._execute_event_request",
            side_effect=capture,
        ):
            block.run(
                event_ingestion_url="http://localhost:8001///",
                event_schema="custom",
                output_image=_make_image(),
                fire_and_forget=False,
                disable_sink=False,
                request_timeout=5,
            )

        assert captured["url"] == "http://localhost:8001/v2/events"

    def test_api_key_read_from_env_var(self):
        block = _make_block()
        captured = {}

        def capture(**kwargs):
            captured.update(kwargs)
            return (False, "ok", "id-1")

        with patch.dict(os.environ, {"EVENT_INGESTION_API_KEY": "test-key-xyz"}):
            with patch(
                "inference.enterprise.workflows.enterprise_blocks.sinks.event_writer.v1._execute_event_request",
                side_effect=capture,
            ):
                block.run(
                    event_ingestion_url="http://localhost:8001",
                    event_schema="custom",
                    output_image=_make_image(),
                    fire_and_forget=False,
                    disable_sink=False,
                    request_timeout=5,
                )

        assert captured["api_key"] == "test-key-xyz"

    def test_no_api_key_env_var_passes_none(self):
        block = _make_block()
        captured = {}

        def capture(**kwargs):
            captured.update(kwargs)
            return (False, "ok", "id-1")

        env = {k: v for k, v in os.environ.items() if k != "EVENT_INGESTION_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with patch(
                "inference.enterprise.workflows.enterprise_blocks.sinks.event_writer.v1._execute_event_request",
                side_effect=capture,
            ):
                block.run(
                    event_ingestion_url="http://localhost:8001",
                    event_schema="custom",
                    output_image=_make_image(),
                    fire_and_forget=False,
                    disable_sink=False,
                    request_timeout=5,
                )

        assert captured["api_key"] is None

    def test_custom_metadata_included_in_payload_when_non_empty(self):
        block = _make_block()
        captured = {}

        def capture(**kwargs):
            captured.update(kwargs)
            return (False, "ok", "id-1")

        with patch(
            "inference.enterprise.workflows.enterprise_blocks.sinks.event_writer.v1._execute_event_request",
            side_effect=capture,
        ):
            block.run(
                event_ingestion_url="http://localhost:8001",
                event_schema="custom",
                output_image=_make_image(),
                fire_and_forget=False,
                disable_sink=False,
                request_timeout=5,
                custom_metadata={"line": "A1", "shift": "morning"},
            )

        assert captured["payload"]["custom_metadata"] == {
            "line": "A1",
            "shift": "morning",
        }

    def test_custom_metadata_omitted_from_payload_when_none(self):
        """custom_metadata is only omitted when None (the default), not when empty dict."""
        block = _make_block()
        captured = {}

        def capture(**kwargs):
            captured.update(kwargs)
            return (False, "ok", "id-1")

        with patch(
            "inference.enterprise.workflows.enterprise_blocks.sinks.event_writer.v1._execute_event_request",
            side_effect=capture,
        ):
            block.run(
                event_ingestion_url="http://localhost:8001",
                event_schema="custom",
                output_image=_make_image(),
                fire_and_forget=False,
                disable_sink=False,
                request_timeout=5,
                custom_metadata=None,
            )

        assert "custom_metadata" not in captured["payload"]

    def test_quality_check_payload_shape(self):
        block = _make_block()
        captured = {}

        def capture(**kwargs):
            captured.update(kwargs)
            return (False, "ok", "id-1")

        with patch(
            "inference.enterprise.workflows.enterprise_blocks.sinks.event_writer.v1._execute_event_request",
            side_effect=capture,
        ):
            block.run(
                event_ingestion_url="http://localhost:8001",
                event_schema="quality_check",
                output_image=_make_image("out64"),
                fire_and_forget=False,
                disable_sink=False,
                request_timeout=5,
                qc_result="pass",
                external_id="batch-001",
            )

        payload = captured["payload"]
        assert payload["event_schema"] == "quality_check"
        assert payload["event_data"] == {"result": "pass", "externalId": "batch-001"}
        assert payload["images"][0]["base64Image"] == "out64"
        assert "inference_timestamp" in payload
