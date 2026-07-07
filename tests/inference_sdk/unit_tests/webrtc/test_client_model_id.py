"""Unit tests for model_id mode in WebRTCClient.stream() and predictions pairing."""

import base64
import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import supervision as sv

from inference_sdk.http.errors import InvalidParameterError
from inference_sdk.webrtc.client import WebRTCClient
from inference_sdk.webrtc.model_workflows import (
    TASK_TYPE_TO_BLOCK,
    apply_model_id_defaults,
    build_model_workflow,
)
from inference_sdk.webrtc.config import StreamConfig
from inference_sdk.webrtc.session import SessionState, VideoMetadata, WebRTCSession


def _mock_ort_response(task_type: str):
    """Build a MagicMock mimicking the Roboflow /ort endpoint response."""
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"ort": {"type": task_type}}
    return response


@pytest.fixture
def client():
    """Create a WebRTCClient without triggering the experimental warning noise."""
    return WebRTCClient(api_url="http://localhost:9001", api_key="test_key")


def _make_session(model_mode=True):
    """Build a WebRTCSession in model mode without initializing WebRTC."""
    with patch("inference_sdk.webrtc.session._check_webrtc_dependencies"):
        session = WebRTCSession(
            api_url="http://localhost:9001",
            api_key="test_key",
            source=MagicMock(),
            image_input_name="image",
            workflow_config={},
            stream_config=StreamConfig(
                stream_output=["image"], data_output=["predictions"]
            ),
            model_mode=model_mode,
            predictions_output="predictions",
        )
    return session


def _encode_base64_image(img: np.ndarray) -> str:
    """Encode a BGR image as base64 JPEG (matches server-side base64 output)."""
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _predictions_dict():
    """A serialized object-detection response with one 'car' box."""
    return {
        "predictions": [
            {
                "x": 50.0,
                "y": 50.0,
                "width": 20.0,
                "height": 20.0,
                "confidence": 0.9,
                "class": "car",
                "class_id": 0,
            }
        ],
        "image": {"width": 100, "height": 100},
    }


class TestStreamValidation:
    """Tests for the workflow/model_id mutual-exclusion validation."""

    def test_raises_when_both_workflow_and_model_id(self, client):
        source = MagicMock()
        with pytest.raises(InvalidParameterError, match="Exactly one"):
            client.stream(source=source, workflow="wf", model_id="rfdetr-nano")

    def test_raises_when_neither_workflow_nor_model_id(self, client):
        source = MagicMock()
        with pytest.raises(InvalidParameterError, match="Exactly one"):
            client.stream(source=source)


class TestBuildModelWorkflow:
    """Tests for the generated workflow specification."""

    def test_workflow_spec_built_correctly(self):
        spec = build_model_workflow("rfdetr-nano", "object-detection")
        assert spec == {
            "version": "1.0",
            "inputs": [{"type": "InferenceImage", "name": "image"}],
            "steps": [
                {
                    "type": "roboflow_core/roboflow_object_detection_model@v2",
                    "name": "model",
                    "images": "$inputs.image",
                    "model_id": "rfdetr-nano",
                }
            ],
            "outputs": [
                {
                    "type": "JsonField",
                    "name": "predictions",
                    "coordinates_system": "own",
                    "selector": "$steps.model.predictions",
                },
                {
                    "type": "JsonField",
                    "name": "image",
                    "selector": "$inputs.image",
                },
            ],
        }

    @pytest.mark.parametrize(
        "task_type,expected_block",
        [
            (
                "object-detection",
                "roboflow_core/roboflow_object_detection_model@v2",
            ),
            (
                "instance-segmentation",
                "roboflow_core/roboflow_instance_segmentation_model@v2",
            ),
            (
                "classification",
                "roboflow_core/roboflow_classification_model@v2",
            ),
            (
                "multi-label-classification",
                "roboflow_core/roboflow_multi_label_classification_model@v2",
            ),
            (
                "keypoint-detection",
                "roboflow_core/roboflow_keypoint_detection_model@v2",
            ),
            (
                "semantic-segmentation",
                "roboflow_core/roboflow_semantic_segmentation_model@v2",
            ),
        ],
    )
    def test_each_task_type_maps_to_correct_block(self, task_type, expected_block):
        spec = build_model_workflow("some/1", task_type)
        assert spec["steps"][0]["type"] == expected_block
        # Every task type exposes its predictions under the "predictions" output.
        assert spec["outputs"][0]["name"] == "predictions"
        assert spec["outputs"][0]["selector"] == "$steps.model.predictions"
        assert spec["outputs"][0]["coordinates_system"] == "own"

    def test_build_workflow_rejects_unknown_task_type(self):
        with pytest.raises(InvalidParameterError, match="Unsupported task_type"):
            build_model_workflow("some/1", "not-a-task")

    def test_stream_passes_spec_to_session(self, client):
        source = MagicMock()
        with patch(
            "inference_sdk.webrtc.client.WebRTCSession"
        ) as mock_session_cls:
            client.stream(
                source=source,
                model_id="rfdetr-nano",
                task_type="object-detection",
            )
        _, kwargs = mock_session_cls.call_args
        assert kwargs["model_mode"] is True
        assert kwargs["predictions_output"] == "predictions"
        assert kwargs["workflow_config"] == {
            "workflow_specification": build_model_workflow(
                "rfdetr-nano", "object-detection"
            )
        }


class TestTaskTypeResolution:
    """Tests for resolving the model task type (explicit and via the API)."""

    def test_explicit_task_type_skips_http(self, client):
        source = MagicMock()
        with patch("inference_sdk.webrtc.model_workflows.requests.get") as mock_get, patch(
            "inference_sdk.webrtc.client.WebRTCSession"
        ) as mock_session_cls:
            client.stream(
                source=source,
                model_id="rfdetr-nano",
                task_type="instance-segmentation",
            )
        mock_get.assert_not_called()
        _, kwargs = mock_session_cls.call_args
        spec = kwargs["workflow_config"]["workflow_specification"]
        assert (
            spec["steps"][0]["type"]
            == "roboflow_core/roboflow_instance_segmentation_model@v2"
        )

    def test_auto_resolution_calls_api_and_selects_block(self, client):
        source = MagicMock()
        with patch(
            "inference_sdk.webrtc.model_workflows.requests.get",
            return_value=_mock_ort_response("keypoint-detection"),
        ) as mock_get, patch(
            "inference_sdk.webrtc.client.WebRTCSession"
        ) as mock_session_cls:
            client.stream(source=source, model_id="rfdetr-nano")
        # Lookup happened once.
        assert mock_get.call_count == 1
        call_url = mock_get.call_args[0][0]
        # Alias resolved before lookup: rfdetr-nano -> coco/38.
        assert call_url.endswith("/ort/coco/38")
        params = mock_get.call_args.kwargs["params"]
        assert params["api_key"] == "test_key"
        assert params["device"] == "sdk"
        _, kwargs = mock_session_cls.call_args
        spec = kwargs["workflow_config"]["workflow_specification"]
        assert (
            spec["steps"][0]["type"]
            == "roboflow_core/roboflow_keypoint_detection_model@v2"
        )

    def test_invalid_explicit_task_type_raises(self, client):
        source = MagicMock()
        with patch("inference_sdk.webrtc.model_workflows.requests.get") as mock_get:
            with pytest.raises(InvalidParameterError, match="Unsupported task_type"):
                client.stream(
                    source=source, model_id="rfdetr-nano", task_type="bogus"
                )
        mock_get.assert_not_called()

    def test_task_type_with_workflow_raises(self, client):
        source = MagicMock()
        with pytest.raises(
            InvalidParameterError, match="'task_type' is only valid together with"
        ):
            client.stream(
                source=source,
                workflow="wf",
                workspace="ws",
                task_type="object-detection",
            )

    def test_versionless_model_id_raises(self, client):
        source = MagicMock()
        with patch("inference_sdk.webrtc.model_workflows.requests.get") as mock_get:
            with pytest.raises(InvalidParameterError, match="dataset/version"):
                client.stream(source=source, model_id="just-a-name-no-version")
        mock_get.assert_not_called()

    def test_lookup_failure_raises_helpful_error(self, client):
        source = MagicMock()
        with patch(
            "inference_sdk.webrtc.model_workflows.requests.get",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(RuntimeError, match="task_type=") as exc_info:
                client.stream(source=source, model_id="rfdetr-nano")
        # Error should point the user at the escape hatch.
        assert "Failed to resolve task type" in str(exc_info.value)

    def test_unsupported_api_task_type_raises(self, client):
        source = MagicMock()
        with patch(
            "inference_sdk.webrtc.model_workflows.requests.get",
            return_value=_mock_ort_response("some-exotic-task"),
        ):
            with pytest.raises(InvalidParameterError, match="not supported"):
                client.stream(source=source, model_id="rfdetr-nano")

    def test_all_supported_task_types_resolvable(self, client):
        # Sanity: every key in the map resolves end-to-end via the API path.
        for task_type in TASK_TYPE_TO_BLOCK:
            source = MagicMock()
            with patch(
                "inference_sdk.webrtc.model_workflows.requests.get",
                return_value=_mock_ort_response(task_type),
            ), patch("inference_sdk.webrtc.client.WebRTCSession") as mock_session_cls:
                client.stream(source=source, model_id="rfdetr-nano")
            _, kwargs = mock_session_cls.call_args
            spec = kwargs["workflow_config"]["workflow_specification"]
            assert (
                spec["steps"][0]["type"]
                == TASK_TYPE_TO_BLOCK[task_type]["block_type"]
            )


class TestModelIdDefaults:
    """Tests for StreamConfig defaults in model_id mode."""

    def test_defaults_when_config_none(self):
        config = apply_model_id_defaults(None)
        assert config.stream_output == ["image"]
        assert config.data_output == ["predictions"]

    def test_defaults_fill_empty_lists(self):
        config = apply_model_id_defaults(StreamConfig())
        assert config.stream_output == ["image"]
        assert config.data_output == ["predictions"]

    def test_user_stream_output_preserved(self):
        user = StreamConfig(stream_output=["annotated"], realtime_processing=False)
        config = apply_model_id_defaults(user)
        # user-set stream_output kept, empty data_output filled
        assert config.stream_output == ["annotated"]
        assert config.data_output == ["predictions"]
        # other settings untouched
        assert config.realtime_processing is False

    def test_user_data_output_preserved(self):
        user = StreamConfig(data_output=["custom"])
        config = apply_model_id_defaults(user)
        assert config.data_output == ["custom"]
        assert config.stream_output == ["image"]


class TestDatachannelVideoPairing:
    """Datachannel-video path (VideoFileSource): image + predictions same message."""

    def test_on_frame_receives_raw_predictions_dict(self):
        session = _make_session()
        session._video_through_datachannel = True

        received = []

        @session.on_frame
        def handler(frame, data):
            received.append((frame, data))

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        predictions = _predictions_dict()
        serialized = {
            "image": {"type": "base64", "value": _encode_base64_image(img)},
            "predictions": predictions,
        }
        metadata = VideoMetadata(frame_id=1, received_at=datetime.now(), pts=1)
        session._handle_datachannel_video_frame(serialized, metadata)
        session._video_queue.put_nowait(None)  # end stream

        session._state = SessionState.STARTED
        with patch.object(session, "_ensure_started"):
            session._run_model_mode()

        assert len(received) == 1
        frame, data = received[0]
        assert frame.shape == (100, 100, 3)
        # Raw dict passed through unchanged (same object, same keys/values).
        assert data is predictions
        assert data == _predictions_dict()
        # And it round-trips into sv.Detections in user code.
        detections = sv.Detections.from_inference(data)
        assert len(detections) == 1
        assert detections.xyxy.tolist() == [[40.0, 40.0, 60.0, 60.0]]
        assert detections.class_id.tolist() == [0]
        assert detections.data["class_name"].tolist() == ["car"]

    def test_missing_predictions_yields_synthesized_empty_dict(self):
        session = _make_session()
        session._video_through_datachannel = True

        received = []

        @session.on_frame
        def handler(frame, data):
            received.append(data)

        img = np.zeros((120, 80, 3), dtype=np.uint8)  # H=120, W=80
        serialized = {"image": {"type": "base64", "value": _encode_base64_image(img)}}
        metadata = VideoMetadata(frame_id=1, received_at=datetime.now(), pts=1)
        session._handle_datachannel_video_frame(serialized, metadata)
        session._video_queue.put_nowait(None)

        session._state = SessionState.STARTED
        with patch.object(session, "_ensure_started"):
            session._run_model_mode()

        assert len(received) == 1
        data = received[0]
        # Synthesized inference-shaped dict from the frame's shape, never None.
        assert data == {"image": {"width": 80, "height": 120}, "predictions": []}
        assert len(sv.Detections.from_inference(data)) == 0


class TestTrackPairing:
    """Video-track path (webcam/RTSP): pair track frame and predictions by pts."""

    def test_frame_then_predictions(self):
        session = _make_session()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        metadata = VideoMetadata(frame_id=7, received_at=datetime.now(), pts=7)

        # frame arrives first, stashed
        session._pair_track_frame(7, img, metadata)
        assert session._video_queue.empty()
        assert 7 in session._pending_frames

        # predictions arrive, pairing completes
        predictions = _predictions_dict()
        session._pair_track_predictions(7, predictions)

        item = session._video_queue.get_nowait()
        frame, data, md = item
        assert np.array_equal(frame, img)
        # Raw dict passed through unchanged.
        assert data is predictions
        assert data == _predictions_dict()
        assert md.frame_id == 7
        assert 7 not in session._pending_frames

    def test_predictions_then_frame(self):
        session = _make_session()
        img = np.ones((100, 100, 3), dtype=np.uint8)
        metadata = VideoMetadata(frame_id=9, received_at=datetime.now(), pts=9)

        predictions = _predictions_dict()
        session._pair_track_predictions(9, predictions)
        assert session._video_queue.empty()
        assert 9 in session._pending_predictions

        session._pair_track_frame(9, img, metadata)
        item = session._video_queue.get_nowait()
        frame, data, md = item
        assert np.array_equal(frame, img)
        assert data is predictions
        assert 9 not in session._pending_predictions

    def test_overflow_evicts_frame_with_synthesized_empty_dict(self):
        session = _make_session()
        session._pairing_max_size = 3

        # Insert more frames than the pairing buffer allows, none matched.
        # Distinct shapes so we can assert width/height come from the frame.
        for pts in range(5):
            img = np.zeros((10 + pts, 20 + pts, 3), dtype=np.uint8)
            md = VideoMetadata(frame_id=pts, received_at=datetime.now(), pts=pts)
            session._pair_track_frame(pts, img, md)

        # 5 inserted, buffer capped at 3 -> 2 evicted onto the queue as empty.
        assert len(session._pending_frames) == 3
        evicted = []
        while not session._video_queue.empty():
            evicted.append(session._video_queue.get_nowait())
        assert len(evicted) == 2
        # pts 0 and 1 are the oldest, so evicted first.
        for expected_pts, (frame, data, md) in zip((0, 1), evicted):
            assert md.pts == expected_pts
            height, width = frame.shape[:2]
            assert data == {
                "image": {"width": width, "height": height},
                "predictions": [],
            }
            assert len(sv.Detections.from_inference(data)) == 0

    def test_pts_none_delivers_synthesized_empty_dict_immediately(self):
        session = _make_session()
        img = np.zeros((30, 40, 3), dtype=np.uint8)  # H=30, W=40
        md = VideoMetadata(frame_id=0, received_at=datetime.now(), pts=None)
        session._pair_track_frame(None, img, md)
        frame, data, _ = session._video_queue.get_nowait()
        assert np.array_equal(frame, img)
        assert data == {"image": {"width": 40, "height": 30}, "predictions": []}
        assert len(sv.Detections.from_inference(data)) == 0


class TestOnDataMessagePairing:
    """End-to-end pairing through the datachannel message handler (track path)."""

    def test_on_data_message_pairs_predictions(self):
        session = _make_session()
        # Not the datachannel-video path (webcam/RTSP)
        session._video_through_datachannel = False

        # Simulate a frame already waiting on the track for pts=42.
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        md = VideoMetadata(frame_id=42, received_at=datetime.now(), pts=42)
        session._pending_frames[42] = (img, md)

        # Drive the datachannel message handler directly by building the
        # message dict and invoking the pairing path it uses.
        predictions = _predictions_dict()
        serialized = {"predictions": predictions}
        message = {
            "video_metadata": {
                "frame_id": 42,
                "received_at": datetime.now().isoformat(),
                "pts": 42,
            },
            "serialized_output_data": serialized,
        }
        # Emulate the branch inside _on_data_message for model mode.
        parsed = json.loads(json.dumps(message))
        metadata = session._parse_video_metadata(parsed["video_metadata"], errors=[])
        parsed_predictions = parsed["serialized_output_data"].get("predictions")
        session._pair_track_predictions(metadata.pts, parsed_predictions)

        frame, data, out_md = session._video_queue.get_nowait()
        assert np.array_equal(frame, img)
        # Raw dict passed through unchanged (equal to what was sent).
        assert data == predictions
        assert out_md.frame_id == 42


class TestHandlerArity:
    """2-arg and 3-arg on_frame handler support in model mode."""

    def test_two_arg_handler(self):
        session = _make_session()
        calls = []

        @session.on_frame
        def handler(frame, data):
            calls.append((frame, data))

        img = np.zeros((10, 10, 3), dtype=np.uint8)
        data = _predictions_dict()
        md = VideoMetadata(frame_id=1, received_at=datetime.now(), pts=1)
        session._video_queue.put_nowait((img, data, md))
        session._video_queue.put_nowait(None)

        session._state = SessionState.STARTED
        with patch.object(session, "_ensure_started"):
            session._run_model_mode()

        assert len(calls) == 1
        assert calls[0][1] is data

    def test_three_arg_handler(self):
        session = _make_session()
        calls = []

        @session.on_frame
        def handler(frame, data, metadata):
            calls.append((frame, data, metadata))

        img = np.zeros((10, 10, 3), dtype=np.uint8)
        data = _predictions_dict()
        md = VideoMetadata(frame_id=5, received_at=datetime.now(), pts=5)
        session._video_queue.put_nowait((img, data, md))
        session._video_queue.put_nowait(None)

        session._state = SessionState.STARTED
        with patch.object(session, "_ensure_started"):
            session._run_model_mode()

        assert len(calls) == 1
        assert calls[0][1] is data
        assert calls[0][2].frame_id == 5


class TestVideoIteratorModelMode:
    """video() iterator yields (frame, data) in model mode."""

    def test_video_yields_frame_data(self):
        session = _make_session()
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        data = _predictions_dict()
        md = VideoMetadata(frame_id=1, received_at=datetime.now(), pts=1)
        session._video_queue.put_nowait((img, data, md))
        session._video_queue.put_nowait(None)

        with patch.object(session, "_ensure_started"):
            items = list(session.video())

        assert len(items) == 1
        frame, out_data = items[0]
        assert np.array_equal(frame, img)
        # Raw dict, not sv.Detections.
        assert out_data is data
        assert len(sv.Detections.from_inference(out_data)) == 1
