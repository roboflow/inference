"""Decode self-contained offline clips without opening secondary resources."""

import math
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Iterator, Optional, Union

import av
import numpy as np

# FFmpeg checks this before calling the selected demuxer's read_header. Do not
# replace it with a check of container.format after av.open: that is too late.
_VIDEO_DEMUXERS = "mov,matroska,webm,avi,mpegts,mpeg,ogg,flv,asf,h264,hevc"
_DECODE_ERROR = "Video could not be decoded."


class VideoDecodeError(ValueError):
    """An offline clip is invalid or requires access to another resource."""


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

    def __init__(self, container, *, secondary_io: _SecondaryIO):
        if not container.streams.video:
            raise VideoDecodeError(_DECODE_ERROR)

        stream = container.streams.video[0]
        self.fps = float(stream.average_rate or stream.guessed_rate or 0)
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

    def read(self) -> Optional[np.ndarray]:
        """Decode the next frame without seeking.

        Returns:
            Contiguous BGR uint8 pixels, or None at the end of the stream.

        Raises:
            VideoDecodeError: If decoding fails or attempts secondary I/O.
        """
        self._secondary_io.check()
        try:
            frame = next(self._frames, None)
            self._secondary_io.check()
            if frame is None:
                return None

            pixels = frame.to_ndarray(format="bgr24")
            # OpenCV's default auto-orientation supports quarter turns too.
            rotation = frame.rotation
            if rotation in (-270, -180, -90, 90, 180, 270):
                pixels = np.rot90(pixels, k=round(rotation / 90))
            pixels = np.ascontiguousarray(pixels)
        except av.FFmpegError as error:
            raise VideoDecodeError(_DECODE_ERROR) from error

        return pixels


@contextmanager
def open_video(path: Union[str, Path]) -> Iterator[VideoReader]:
    """Open only the bytes of a server-owned offline video file.

    The primary file uses Python I/O. FFmpeg's built-in protocols are disabled,
    and its secondary-open callback rejects all additional files and URLs.
    The format allowlist excludes manifests and image sequences; external MOV
    tracks stay disabled even though standalone MOV/MP4 containers are allowed.

    Args:
        path (str or Path): Temporary file produced by the caller's guarded
            download or base64 ingestion. Never a user-supplied filesystem path.

    Yields:
        VideoReader: A sequential reader, valid only inside this context.

    Raises:
        VideoDecodeError: If opening fails or the input is not a supported,
            self-contained video. Error messages omit paths and URLs.
    """
    secondary_io = _SecondaryIO()
    with ExitStack() as stack:
        try:
            source = stack.enter_context(Path(path).open("rb"))
            container = stack.enter_context(
                av.open(
                    source,
                    mode="r",
                    io_open=secondary_io,
                    options={
                        "protocol_whitelist": "",
                        "format_whitelist": _VIDEO_DEMUXERS,
                        "enable_drefs": "0",
                        "use_absolute_path": "0",
                    },
                )
            )
            secondary_io.check()
            reader = VideoReader(container, secondary_io=secondary_io)
        except VideoDecodeError:
            raise
        except (OSError, ValueError) as error:
            raise VideoDecodeError(_DECODE_ERROR) from error

        yield reader
