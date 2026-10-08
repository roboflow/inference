"""``video/stream@v1`` and ``video/stream_set@v1`` Workflows 2.0 sources.

Both sources read frames through ``VideoSource`` with the bounded reader core
in ``reader.py``. Constructors acquire nothing; ``open`` starts one member
thread per stream, ``read`` returns the next emission, ``close`` stops all
threads within ``stop_timeout``.

What each frame carries::

    ImageData            CHW uint8 RGB tensor: the producer's tensor as is
                         (CUDA or CPU), or one CPU copy of an OpenCV BGR array
      .video_metadata    video_identifier = member id, frame_number = VideoSource
                         frame id, frame_timestamp = VideoSource host estimate
    SampleContext        source_id = member id, source_metadata {"stream": index}
    TemporalContext      observed = engine clock when the member thread took
                         the frame from VideoSource; media = None; capture = None

Member id: ``"<source name>"`` for ``video/stream@v1`` and
``"<source name>/<index>"`` for ``video/stream_set@v1``. Source names are
compiler-checked selector segments without ``/``, so the ids are unambiguous.

Media PTS and capture time stay ``None``: no producer exposes either, and
``VideoFrame.frame_timestamp`` is wall-clock time taken before the grab (or a
file-creation-time estimate), not a capture or media time.

A stream set emits a ``Batch`` over the stationary sample axis ``camera``; the
index of each item is ``(stream index,)``. Missing cameras are absent, never
compacted: read results with ``outputs.data[name].iter_with_indices()``;
``GroupResult.rows()`` pads absent positions with ``None`` rows. The physical
batch is not a time grouping. Engine overload policies and controls apply to
the whole set, not to one camera; use separate ``video/stream@v1`` sources
when cameras need their own.
"""

import threading
from typing import Any, List, Optional, Tuple, Union

import torch
from pydantic import Field, ValidationInfo, field_validator
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.execution_engine.v2.blocks.image_data import ImageData
from roboflow_workflows.execution_engine.v2.blocks.kinds import IMAGE_KIND
from roboflow_workflows.execution_engine.v2.data import (
    Axis,
    Batch,
    EntryLayout,
    EntryMetadata,
    InputValue,
    SampleContext,
    TemporalContext,
    Timestamp,
)
from roboflow_workflows.execution_engine.v2.sources import (
    Emission,
    Source,
    SourceOutput,
    SourceParams,
)
from streamvision.camera.entities import VideoFrame
from streamvision.workflows_v2.reader import (
    DEFAULT_BUFFER_SIZE,
    DEFAULT_COLLECTION_TIMEOUT,
    DEFAULT_DECODER,
    DEFAULT_END,
    DEFAULT_FAILURE_POLICY,
    DEFAULT_HANDOFF_SIZE,
    DEFAULT_STOP_TIMEOUT,
    Decoder,
    EndPolicy,
    FailurePolicy,
    FrameCollector,
    MemberReader,
    check_non_negative_seconds,
    check_positive_int,
)
from streamvision.workflows_v2.statistics import VideoStatistics

__all__ = ["VideoStream", "VideoStreamSet", "frame_to_image"]

CAMERAS = EntryLayout((Axis(id="camera", kind="sample", stationary=True),))

Frame = Tuple[ImageData, Timestamp, int]
"""A handoff item: image, acquisition time, stream index."""


def frame_to_image(frame: VideoFrame, *, member_id: str) -> ImageData:
    """Wrap a VideoSource frame as an ``ImageData`` with its video metadata.

    Args:
        frame: Frame from ``VideoSource.read_frame``. A tensor must be a
            ``(3, height, width)`` uint8 RGB tensor (what the hardware
            producers deliver); an array is an OpenCV BGR or grayscale image.
        member_id: Member id, used as ``video_identifier``.

    Returns:
        The image; a tensor is wrapped without a copy or a device move.
    """
    metadata = VideoMetadata(
        video_identifier=member_id,
        frame_number=frame.frame_id,
        frame_timestamp=frame.frame_timestamp,
        fps=frame.fps,
        measured_fps=frame.measured_fps,
        comes_from_video_file=frame.comes_from_video_file,
    )
    pixels = frame.image
    if isinstance(pixels, torch.Tensor):
        image = ImageData.from_tensor(pixels, video_metadata=metadata)
    else:
        rgb = pixels[..., ::-1] if pixels.ndim == 3 else pixels
        image = ImageData.from_numpy_rgb(rgb, video_metadata=metadata)

    return image


class _StreamParams(SourceParams):
    """Settings of every stream, shared by both sources' ``Params``."""

    decoder: Decoder = Field(
        default=DEFAULT_DECODER,
        description=(
            "auto (VideoSource choice, OpenCV fallback), opencv, "
            "gstreamer_cuda or jetson (forced, no fallback)."
        ),
    )
    buffer_size: int = Field(
        default=DEFAULT_BUFFER_SIZE,
        description="VideoSource frame buffer capacity per stream (B).",
    )
    handoff_size: int = Field(
        default=DEFAULT_HANDOFF_SIZE,
        description="Frames per stream waiting for the engine (S).",
    )
    stall_timeout: Optional[float] = Field(
        default=None,
        description="Seconds without a frame that count as a stream failure.",
    )
    stop_timeout: float = Field(
        default=DEFAULT_STOP_TIMEOUT,
        description="Seconds close waits for all stream threads.",
    )

    @field_validator("buffer_size", "handoff_size", mode="before")
    @classmethod
    def _check_capacity(cls, value: Any, info: ValidationInfo) -> Any:
        return check_positive_int(info.field_name, value)

    @field_validator("stall_timeout", mode="before")
    @classmethod
    def _check_stall_timeout(cls, value: Any) -> Any:
        return check_non_negative_seconds("stall_timeout", value, optional=True)

    @field_validator("stop_timeout", mode="before")
    @classmethod
    def _check_stop_timeout(cls, value: Any) -> Any:
        return check_non_negative_seconds("stop_timeout", value, optional=False)


class _VideoSourceBase(Source):
    """Shared lifecycle of the two video sources."""

    def __init__(
        self,
        *,
        video_statistics: Optional[VideoStatistics] = None,
        video_producer_factories: Optional[dict] = None,
    ) -> None:
        self._statistics = video_statistics
        self._factories = video_producer_factories or {}
        self._collector: Optional[FrameCollector] = None

    def _open_members(
        self,
        references: List[Union[str, int]],
        *,
        member_ids: List[str],
        decoder: str,
        buffer_size: int,
        handoff_size: int,
        **collector_settings: Any,
    ) -> None:
        condition = threading.Condition()
        readers = [
            MemberReader(
                member_id,
                reference=reference,
                condition=condition,
                convert=_converter(member_id=member_id, index=index),
                decoder=decoder,
                buffer_size=buffer_size,
                handoff_size=handoff_size,
                producer_factory=self._factories.get(reference),
            )
            for index, (member_id, reference) in enumerate(zip(member_ids, references))
        ]
        self._collector = FrameCollector(
            readers,
            condition=condition,
            stop_event=self.stop_event,
            **collector_settings,
        )
        if self._statistics is not None:
            self._statistics._register(self.source_name, readers)
        self._collector.start()

    def close(self) -> None:
        if self._collector is not None:
            self._collector.close()


def _converter(*, member_id: str, index: int):
    def convert(frame: VideoFrame, acquired: Timestamp) -> Frame:
        image = frame_to_image(frame, member_id=member_id)

        return image, acquired, index

    return convert


def _member_metadata(
    source: Source, member_id: str, acquired: Timestamp, index: int
) -> Tuple[SampleContext, TemporalContext]:
    sample = SampleContext(
        source_id=member_id,
        source_type=source.type,
        source_metadata={"stream": index},
    )
    temporal = TemporalContext(
        observed_coverage=acquired, media_coverage=None, capture_coverage=None
    )

    return sample, temporal


class VideoStream(_VideoSourceBase):
    """One video file, stream or camera; one ``ImageData`` per pulse.

    Resources (both optional):
        video_statistics: ``VideoStatistics`` receiving the final counters.
        video_producer_factories: ``{reference: callable}``, a supported host
            and test hook. A key matches a declared reference by plain
            equality, without sanitising or type conversion (``0`` and
            ``"0"`` differ). VideoSource then calls the callable instead of
            choosing a decoder; ``decoder`` is ignored, and the statistics
            report decoder ``"factory"`` with the sanitized reference.
    """

    type = "video/stream@v1"
    outputs = {"image": SourceOutput(IMAGE_KIND)}

    class Params(_StreamParams):
        reference: Union[str, int] = Field(
            description="Video file path, stream URL or camera device index.",
            examples=["rtsp://camera-1/stream", "/data/video.mp4", 0],
        )
        on_failure: FailurePolicy = Field(
            default=DEFAULT_FAILURE_POLICY,
            description="fail: the run fails; end_member: the source ends.",
        )

    def open(
        self,
        *,
        reference: Union[str, int],
        decoder: str,
        buffer_size: int,
        handoff_size: int,
        on_failure: str,
        stall_timeout: Optional[float],
        stop_timeout: float,
    ) -> None:
        self._open_members(
            [reference],
            member_ids=[self.source_name],
            decoder=decoder,
            buffer_size=buffer_size,
            handoff_size=handoff_size,
            batch_size=1,
            on_member_failure=on_failure,
            stall_timeout=stall_timeout,
            stop_timeout=stop_timeout,
        )

    def read(self) -> Optional[Emission]:
        batch = self._collector.collect()
        if batch is None:
            return None

        _, (image, acquired, index) = batch[0]
        sample, temporal = _member_metadata(self, self.source_name, acquired, index)
        value = InputValue(
            image, EntryMetadata(sample={(): sample}, temporal={(): temporal})
        )
        emission = Emission({"image": value})

        return emission


class VideoStreamSet(_VideoSourceBase):
    """Several streams; each pulse holds up to ``batch_size`` ready cameras.

    A pulse never waits for every camera: it starts with the first ready
    camera, waits at most ``collection_timeout`` for more, and takes at most
    one frame per camera, starting after the camera served last.

    Resources: as ``VideoStream``.
    """

    type = "video/stream_set@v1"
    outputs = {"image": SourceOutput(IMAGE_KIND, layout=CAMERAS)}

    class Params(_StreamParams):
        references: List[Union[str, int]] = Field(
            description="One reference per camera; the list position is its index.",
            min_length=1,
        )
        batch_size: Optional[int] = Field(
            default=None,
            description="Most cameras per pulse; None means every camera.",
        )
        collection_timeout: float = Field(
            default=DEFAULT_COLLECTION_TIMEOUT,
            description="Seconds to wait for more ready cameras after the first.",
        )
        end: EndPolicy = Field(
            default=DEFAULT_END,
            description=(
                "all: end when every camera ended and its frames were taken; "
                "any: end at the first camera end, dropping frames still "
                "waiting from the other cameras."
            ),
        )
        on_member_failure: FailurePolicy = Field(
            default=DEFAULT_FAILURE_POLICY,
            description="fail: the run fails; end_member: only that camera ends.",
        )

        @field_validator("batch_size", mode="before")
        @classmethod
        def _check_batch_size(cls, value: Any) -> Any:
            if value is None:
                return value

            return check_positive_int("batch_size", value)

        @field_validator("collection_timeout", mode="before")
        @classmethod
        def _check_collection_timeout(cls, value: Any) -> Any:
            return check_non_negative_seconds(
                "collection_timeout", value, optional=False
            )

    def open(
        self,
        *,
        references: List[Union[str, int]],
        batch_size: Optional[int],
        collection_timeout: float,
        end: str,
        on_member_failure: str,
        decoder: str,
        buffer_size: int,
        handoff_size: int,
        stall_timeout: Optional[float],
        stop_timeout: float,
    ) -> None:
        member_ids = [f"{self.source_name}/{index}" for index in range(len(references))]
        self._open_members(
            list(references),
            member_ids=member_ids,
            decoder=decoder,
            buffer_size=buffer_size,
            handoff_size=handoff_size,
            batch_size=len(references) if batch_size is None else batch_size,
            collection_timeout=collection_timeout,
            end=end,
            on_member_failure=on_member_failure,
            stall_timeout=stall_timeout,
            stop_timeout=stop_timeout,
        )

    def read(self) -> Optional[Emission]:
        batch = self._collector.collect()
        if batch is None:
            return None

        images, indices, sample, temporal = [], [], {}, {}
        for _, (image, acquired, index) in batch:
            member_id = f"{self.source_name}/{index}"
            position = (index,)
            images.append(image)
            indices.append(position)
            sample[position], temporal[position] = _member_metadata(
                self, member_id, acquired, index
            )
        value = InputValue(
            Batch.of(images, indices=indices),
            EntryMetadata(sample=sample, temporal=temporal),
        )
        emission = Emission({"image": value})

        return emission
