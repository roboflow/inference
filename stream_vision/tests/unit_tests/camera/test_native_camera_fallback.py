import cv2
import pytest
from streamvision.camera import video_source
from streamvision.camera.fourcc import parse_fourcc


@pytest.mark.parametrize("failure_stage", ["construction", "negotiation"])
@pytest.mark.parametrize(
    "property_name", ["fourcc", "frame_width", "frame_height", "fps"]
)
@pytest.mark.parametrize("accepts_set", [False, True])
def test_native_camera_fallback_rejects_wrong_mode(
    monkeypatch, failure_stage, property_name, accepts_set
):
    capture, source, updates = _fallback_source(
        monkeypatch,
        failure_stage=failure_stage,
        wrong_property=property_name,
        accepts_set=accepts_set,
    )

    with pytest.raises(ValueError, match=f"cv2 camera mode mismatch: {property_name}"):
        source.start()

    assert source.describe_source().state is video_source.StreamState.ERROR
    assert capture.released
    assert any(
        update.payload.get("new_state") is video_source.StreamState.ERROR
        for update in updates
    )


@pytest.mark.parametrize("failure_stage", ["construction", "negotiation"])
def test_native_camera_fallback_accepts_requested_mode(monkeypatch, failure_stage):
    capture, source, _ = _fallback_source(monkeypatch, failure_stage=failure_stage)

    source.start()
    source._stream_consumption_thread.join(timeout=2)

    properties = source.describe_source().source_properties
    assert (properties.width, properties.height, properties.fps) == (1920, 1080, 29.97)
    source.terminate(wait_on_frames_consumption=False)
    source._release_video()


def _fallback_source(
    monkeypatch, *, failure_stage, wrong_property=None, accepts_set=True
):
    requested = {
        "fourcc": "MJPG",
        "frame_width": 1920,
        "frame_height": 1080,
        "fps": 30000 / 1001,
    }
    effective = {**requested, "fourcc": parse_fourcc("MJPG"), "fps": 29.97}
    if wrong_property is not None:
        effective[wrong_property] = {
            "fourcc": parse_fourcc("YUYV"),
            "frame_width": 2304,
            "frame_height": 1536,
            "fps": 2,
        }[wrong_property]

    class Capture:
        released = False

        def isOpened(self):
            return True

        def set(self, key, value):
            # Drivers can report success while silently retaining another mode.
            return accepts_set

        def get(self, key):
            for name, value in effective.items():
                if key == getattr(cv2, "CAP_PROP_" + name.upper()):
                    return value
            return 0

        def release(self):
            self.released = True

    class Native:
        def isOpened(self):
            return True

        def initialize_source_properties(self, properties):
            raise RuntimeError("native negotiation failed")

        def release(self):
            pass

    capture = Capture()
    monkeypatch.setattr(video_source.cv2, "VideoCapture", lambda *args: capture)
    monkeypatch.setattr(video_source, "RUNS_ON_JETSON", True)
    monkeypatch.setattr(video_source, "ENABLE_TENSOR_DATA_REPRESENTATION", True)
    monkeypatch.setattr(
        video_source,
        "_build_default_producer",
        lambda *args, **kwargs: (
            Native()
            if failure_stage == "negotiation"
            else video_source.CV2VideoFrameProducer(0)
        ),
    )
    monkeypatch.setattr(
        video_source.VideoSource,
        "_consume_video",
        lambda self: self._change_state(video_source.StreamState.RUNNING),
    )
    updates = []
    source = video_source.VideoSource.init(
        0, video_source_properties=requested, status_update_handlers=[updates.append]
    )
    return capture, source, updates
