import json
from datetime import datetime
from functools import partial
from unittest.mock import MagicMock

import numpy as np

from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
)
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.sinks import (
    UDPSink,
    active_learning_sink,
    multi_sink,
    render_boxes,
)


def test_render_boxes_completes_successfully() -> None:
    # given
    video_frame = VideoFrame(
        image=np.ones((1920, 1920, 3), dtype=np.uint8) * 255,
        frame_id=1,
        frame_timestamp=datetime.now(),
    )
    predictions = ObjectDetectionInferenceResponse(
        predictions=[
            ObjectDetectionPrediction(
                **{
                    "x": 10,
                    "y": 20,
                    "width": 30,
                    "height": 40,
                    "confidence": 0.9,
                    "class": "car",
                    "class_id": 3,
                }
            )
        ],
        image=InferenceResponseImage(width=1920, height=1080),
    ).dict(
        by_alias=True,
        exclude_none=True,
    )
    captured_images = []

    def capture_image(image: np.ndarray) -> None:
        captured_images.append(image)

    # when
    render_boxes(
        video_frame=video_frame,
        predictions=predictions,
        on_frame_rendered=capture_image,
    )

    # then
    assert (
        len(captured_images) == 1
    ), "One capture_image() side effect expected after rendering"
    assert captured_images[0].shape == (
        720,
        1280,
        3,
    ), "capture_image() should be called against resized image dictated by default parameter"


def test_udp_sends_data_through_socket() -> None:
    # given
    socket = MagicMock()
    frame_timestamp = datetime.now()
    video_frame = VideoFrame(
        image=np.ones((1920, 1920, 3), dtype=np.uint8) * 255,
        frame_id=1,
        frame_timestamp=frame_timestamp,
    )
    udp_sink = UDPSink(
        ip_address="127.0.0.1",
        port=9090,
        udp_socket=socket,
    )

    # when
    udp_sink.send_predictions(
        video_frame=video_frame, predictions={"some": "prediction"}
    )

    # then
    socket.sendto.assert_called_once()
    assert socket.sendto.call_args[0][1] == (
        "127.0.0.1",
        9090,
    ), "Data must be sent to 127.0.0.1:9090"
    decoded_message = json.loads(socket.sendto.call_args[0][0])
    assert (
        decoded_message["some"] == "prediction"
    ), "Payload of message that was sent must match the prediction emitted"
    assert decoded_message["inference_metadata"]["frame_id"] == 1
    assert (
        decoded_message["inference_metadata"]["frame_decoding_time"]
        == frame_timestamp.isoformat()
    )
    assert "emission_time" in decoded_message["inference_metadata"]


def test_multi_sink_when_error_occurs() -> None:
    # given
    video_frame = VideoFrame(
        image=np.ones((128, 128, 3), dtype=np.uint8) * 255,
        frame_id=1,
        frame_timestamp=datetime.now(),
    )
    predictions = {"some": "prediction"}

    calls = []

    def faulty_sink(predict: dict, frame: VideoFrame) -> None:
        calls.append((frame, predict))
        raise Exception()

    # when
    multi_sink(
        video_frame=video_frame,
        predictions=predictions,
        sinks=[faulty_sink, faulty_sink],
    )

    # then
    assert len(calls) == 2, "Despite error in sink, all handlers mus be called"
    assert calls[0] == (
        video_frame,
        predictions,
    ), "Call must happen according to contract (VideoFrame, dict)"
    assert calls[1] == (
        video_frame,
        predictions,
    ), "Call must happen according to contract (VideoFrame, dict)"


def test_multi_sink_when_no_error_occurs() -> None:
    # given
    video_frame = VideoFrame(
        image=np.ones((128, 128, 3), dtype=np.uint8) * 255,
        frame_id=1,
        frame_timestamp=datetime.now(),
    )
    predictions = {"some": "prediction"}

    calls = []

    def correct_sink(predict: dict, frame: VideoFrame) -> None:
        calls.append((frame, predict))

    # when
    multi_sink(
        video_frame=video_frame,
        predictions=predictions,
        sinks=[correct_sink, correct_sink],
    )

    # then
    assert len(calls) == 2, "All handlers must be called"
    assert calls[0] == (
        video_frame,
        predictions,
    ), "Call must happen according to contract (VideoFrame, dict)"
    assert calls[1] == (
        video_frame,
        predictions,
    ), "Call must happen according to contract (VideoFrame, dict)"


def test_active_learning_sink() -> None:
    # given
    active_learning_middleware = MagicMock()
    sink = partial(
        active_learning_sink,
        active_learning_middleware=active_learning_middleware,
        model_type="object-detection",
    )
    video_frame = VideoFrame(
        image=np.ones((128, 128, 3), dtype=np.uint8) * 255,
        frame_id=1,
        frame_timestamp=datetime.now(),
    )
    predictions = {"some": "prediction"}

    # when
    sink(predictions, video_frame)

    # then
    active_learning_middleware.register.assert_called_once_with(
        inference_input=video_frame.image,
        prediction=predictions,
        prediction_type="object-detection",
        disable_preproc_auto_orient=False,
    )
