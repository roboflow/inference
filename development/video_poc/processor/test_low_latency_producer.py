import importlib
import sys
from dataclasses import dataclass
from types import ModuleType

import numpy as np


# Keep this POC worker unit test independent of the full inference dependency
# graph.  The producer only needs these two camera interface types at import
# time; the staging image supplies the real module.
entities = ModuleType("inference.core.interfaces.camera.entities")


class VideoFrameProducer:
    pass


@dataclass
class SourceProperties:
    width: int
    height: int
    total_frames: int
    is_file: bool
    fps: float
    is_reconnectable: bool


entities.VideoFrameProducer = VideoFrameProducer
entities.SourceProperties = SourceProperties
_module_names = (
    "inference",
    "inference.core",
    "inference.core.interfaces",
    "inference.core.interfaces.camera",
    "inference.core.interfaces.camera.entities",
)
_previous_modules = {name: sys.modules.get(name) for name in _module_names}
try:
    for package in _module_names[:-1]:
        sys.modules.setdefault(package, ModuleType(package))
    sys.modules.setdefault(_module_names[-1], entities)
    LowLatencyRtspProducer = importlib.import_module(
        "low_latency_producer"
    ).LowLatencyRtspProducer
finally:
    for name, previous in _previous_modules.items():
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


class _Frame:
    def __init__(self) -> None:
        self.conversion_formats = []
        self.image = np.zeros((2, 3, 3), dtype=np.uint8)

    def to_ndarray(self, format):
        self.conversion_formats.append(format)
        return self.image


class _Packet:
    def __init__(self, frame) -> None:
        self.frame = frame

    def decode(self):
        return [self.frame]


def _producer_with_frames(*frames):
    producer = LowLatencyRtspProducer.__new__(LowLatencyRtspProducer)
    producer._demuxer = iter(_Packet(frame) for frame in frames)
    producer._pending = None
    producer._open = True
    return producer


def test_grab_defers_host_materialisation_until_retrieve() -> None:
    frame = _Frame()
    producer = _producer_with_frames(frame)

    assert producer.grab() is True
    assert frame.conversion_formats == []

    success, image = producer.retrieve()

    assert success is True
    assert image is frame.image
    assert frame.conversion_formats == ["bgr24"]


def test_unretrieved_frame_is_replaced_without_host_materialisation() -> None:
    skipped_frame = _Frame()
    selected_frame = _Frame()
    producer = _producer_with_frames(skipped_frame, selected_frame)

    assert producer.grab() is True
    assert producer.grab() is True
    success, image = producer.retrieve()

    assert success is True
    assert image is selected_frame.image
    assert skipped_frame.conversion_formats == []
    assert selected_frame.conversion_formats == ["bgr24"]
