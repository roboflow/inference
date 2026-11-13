"""Decode self-contained offline clips without opening secondary resources."""

import math
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator, Optional, Tuple, Union

import av
import cv2
import numpy as np

# FFmpeg checks this before calling the selected demuxer's read_header. Do not
# replace it with a check of container.format after av.open: that is too late.
_VIDEO_DEMUXERS = "mov,matroska,webm,avi,mpegts,mpeg,ogg,flv,asf,h264,hevc"
_DECODE_ERROR = "Video could not be decoded."


class VideoDecodeError(ValueError):
    """An offline clip is invalid or requires access to another resource."""


class VideoSizeError(VideoDecodeError):
    """An offline frame exceeds the configured decoded-input limits."""


class _SecondaryIO:
    def __init__(self):
        self.attempted = False

    def __call__(self, url, flags, options):
        self.attempted = True
        self.check()

    def check(self):
        if self.attempted:
            raise VideoDecodeError(
                "Videos referencing other files or URLs are not supported."
            )


class VideoReader:
    """Read one video stream from an already restricted container.

    Instances belong to the context returned by ``open_video``. Frames are BGR
    arrays with display rotation applied, matching the offline OpenCV reader.

    Attributes:
        fps (float): Declared frame rate, or zero when unavailable.
        frame_count (int): Declared count or duration-based estimate; zero if unknown.
    """

    def __init__(self, container, *, secondary_io: _SecondaryIO, max_pixels=None):
        if not container.streams.video:
            raise VideoDecodeError(_DECODE_ERROR)

        stream = container.streams.video[0]
        if stream.codec_context is None:
            raise VideoDecodeError(_DECODE_ERROR)
        self._max_pixels = max_pixels
        if max_pixels is not None:
            self._check_size(stream.codec_context.width, stream.codec_context.height)
            # av.open options cover probing, but PyAV creates a fresh codec
            # context for subsequent decoding. Configure that context as well.
            stream.codec_context.options = {
                **stream.codec_context.options,
                "max_pixels": str(max_pixels),
            }
            stream.codec_context.thread_count = 1
        rate = stream.average_rate
        if container.format.name == "h264":
            # Raw H.264 can report the demuxer's default 25 fps as its average
            # even when the bitstream declares another rate. Container formats
            # still use their average rate to preserve variable-rate sampling.
            rate = stream.guessed_rate or rate

        self.fps = float(rate or stream.guessed_rate or 0)
        self.frame_count = stream.frames
        if not self.frame_count and self.fps > 0 and math.isfinite(self.fps):
            # Match OpenCV's duration-based estimate for containers such as
            # Matroska/TS that do not declare nb_frames. Otherwise every probe
            # would trigger a full recount before the caller's duration limit.
            duration = 0.0
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration = container.duration / av.time_base
            self.frame_count = max(0, math.floor(duration * self.fps + 0.5))
        self._frames = container.decode(stream)
        self._secondary_io = secondary_io

    def _check_size(self, width, height):
        # FFmpeg can clear dimensions when max_pixels rejects the probe.
        # Under an active limit, unusable dimensions also fail the size policy.
        if self._max_pixels is not None and (
            width <= 0 or height <= 0 or width * height > self._max_pixels
        ):
            raise VideoSizeError(
                "Video frame exceeds the source pixel limit or has unusable dimensions."
            )

    def read(
        self,
        *,
        rgb: bool = False,
        max_bytes: Optional[int] = None,
        expected_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[np.ndarray]:
        """Decode the next frame without seeking.

        Args:
            rgb: Return RGB instead of BGR pixels.
            max_bytes: Maximum bytes for the returned RGB/BGR allocation.
            expected_size: Required display height and width, if already known.

        Returns:
            Contiguous BGR (or RGB) uint8 pixels, or None at the end of the stream.

        Raises:
            VideoDecodeError: If decoding fails or attempts secondary I/O.
            VideoSizeError: If a frame exceeds the pixel or output byte limit.
        """
        self._secondary_io.check()
        try:
            frame = next(self._frames, None)
            self._secondary_io.check()
            if frame is None:
                return None

            self._check_size(frame.width, frame.height)
            if max_bytes is not None and frame.width * frame.height * 3 > max_bytes:
                raise VideoSizeError(
                    "Video frame exceeds the original-frame byte budget."
                )
            size = (frame.height, frame.width)
            if frame.rotation in (-270, -90, 90, 270):
                size = size[::-1]
            if expected_size is not None and size != expected_size:
                raise VideoDecodeError("All frames of a video must have one size.")
            pixels = frame.to_ndarray(format="rgb24" if rgb else "bgr24")
            # OpenCV's default auto-orientation supports quarter turns too.
            rotation = frame.rotation
            if rotation in (90, -270):
                pixels = cv2.rotate(pixels, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif rotation in (180, -180):
                pixels = cv2.rotate(pixels, cv2.ROTATE_180)
            elif rotation in (270, -90):
                pixels = cv2.rotate(pixels, cv2.ROTATE_90_CLOCKWISE)
            pixels = np.ascontiguousarray(pixels)
        except av.FFmpegError as error:
            raise VideoDecodeError(_DECODE_ERROR) from error

        return pixels


@contextmanager
def open_video(
    path: Union[str, Path, BinaryIO], *, max_pixels: Optional[int] = None
) -> Iterator[VideoReader]:
    """Open only the bytes of a server-owned offline video file.

    The primary file uses Python I/O. FFmpeg's built-in protocols are disabled,
    and its secondary-open callback rejects all additional files and URLs.
    The format allowlist excludes manifests and image sequences; external MOV
    tracks stay disabled even though standalone MOV/MP4 containers are allowed.

    Args:
        path (str or Path): Temporary file produced by the caller's guarded
            download or base64 ingestion, or its already-open binary handle.
            Caller-provided handles remain owned by the caller.
        max_pixels: Optional source-pixel limit for probing and frame decoding.

    Yields:
        VideoReader: A sequential reader, valid only inside this context.

    Raises:
        VideoDecodeError: If opening fails or the input is not a supported,
            self-contained video. Error messages omit paths and URLs.
    """
    if max_pixels is not None and max_pixels <= 0:
        raise ValueError("max_pixels must be positive")

    secondary_io = _SecondaryIO()
    with ExitStack() as stack:
        try:
            source = (
                stack.enter_context(Path(path).open("rb"))
                if isinstance(path, (str, Path))
                else path
            )
            codec_options = {}
            if max_pixels is not None:
                codec_options = {"max_pixels": str(max_pixels), "threads": "1"}
            container = stack.enter_context(
                av.open(
                    source,
                    mode="r",
                    io_open=secondary_io,
                    options={
                        **codec_options,
                        "protocol_whitelist": "",
                        "format_whitelist": _VIDEO_DEMUXERS,
                        "enable_drefs": "0",
                        "use_absolute_path": "0",
                    },
                )
            )
            secondary_io.check()
            reader = VideoReader(
                container, secondary_io=secondary_io, max_pixels=max_pixels
            )
        except VideoDecodeError:
            raise
        except (OSError, ValueError) as error:
            raise VideoDecodeError(_DECODE_ERROR) from error

        yield reader
