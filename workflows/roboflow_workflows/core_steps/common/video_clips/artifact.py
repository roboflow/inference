"""`ClipArtifact`: a muxed MP4 clip owned by whoever uploads it.

Clips are always muxed to a temporary file first (`movflags=+faststart` rewrites
the file in place, which needs a real path). Clips up to
`WORKFLOWS_VIDEO_CLIP_INMEMORY_MAX_BYTES` (default 64 MiB) are then read into
memory and the file is deleted; larger clips stay on disk under
`WORKFLOWS_VIDEO_CLIP_TEMP_DIR` (default: the system temp directory) until
`release()`.
"""

import io
import logging
import tempfile
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Callable, Iterator, Optional, Union

from roboflow_workflows.core_steps.common.video_clips import (
    WORKFLOWS_VIDEO_CLIP_INMEMORY_MAX_BYTES,
    WORKFLOWS_VIDEO_CLIP_TEMP_DIR,
)
from roboflow_workflows.core_steps.common.video_clips.report import ClipReport

logger = logging.getLogger(__name__)

CLIP_CONTENT_TYPE = "video/mp4"
TEMP_FILE_PREFIX = "roboflow-video-clip-"


def clip_temp_dir() -> Path:
    """Return the directory clips are muxed into, creating it if needed.

    Returns:
        `WORKFLOWS_VIDEO_CLIP_TEMP_DIR`, or the system temp directory.
    """
    directory = WORKFLOWS_VIDEO_CLIP_TEMP_DIR or Path(tempfile.gettempdir())
    directory.mkdir(parents=True, exist_ok=True)

    return directory


def allocate_clip_path(directory: Optional[Path] = None) -> Path:
    """Return a fresh, unused path for a clip file.

    Args:
        directory: Target directory; `clip_temp_dir()` when ``None``.

    Returns:
        Path of a not-yet-existing ``.mp4`` file.
    """
    target_directory = directory or clip_temp_dir()
    path = target_directory / f"{TEMP_FILE_PREFIX}{uuid.uuid4().hex}.mp4"

    return path


class ClipArtifact:
    """An encoded clip, in memory or in a temporary file.

    Args:
        bytes_or_path: The MP4 bytes, or the path of the temporary MP4 file.
        size_bytes: Size of the MP4 in bytes.
        duration_seconds: Output (playback) duration in seconds.
        report: The clip report.
        content_type: MIME type of the clip.
        on_release: Optional hook called once by the first `release()`. Rings
            drop their packet references when the MP4 is muxed, not here.
    """

    def __init__(
        self,
        bytes_or_path: Union[bytes, Path],
        *,
        size_bytes: int,
        duration_seconds: float,
        report: ClipReport,
        content_type: str = CLIP_CONTENT_TYPE,
        on_release: Optional[Callable[[], None]] = None,
    ):
        self.bytes_or_path = bytes_or_path
        self.size_bytes = size_bytes
        self.duration_seconds = duration_seconds
        self.report = report
        self.content_type = content_type
        self._on_release = on_release
        self._lock = threading.Lock()
        self._released = False

    @classmethod
    def from_muxed_file(
        cls,
        path: Path,
        *,
        duration_seconds: float,
        report: ClipReport,
        inmemory_max_bytes: Optional[int] = None,
        on_release: Optional[Callable[[], None]] = None,
    ) -> "ClipArtifact":
        """Wrap a freshly muxed MP4 file, loading it into memory when small.

        Args:
            path: The muxed MP4 file; deleted here when loaded into memory.
            duration_seconds: Output duration in seconds.
            report: The clip report.
            inmemory_max_bytes: In-memory threshold; defaults to
                `WORKFLOWS_VIDEO_CLIP_INMEMORY_MAX_BYTES`.
            on_release: Called once by the first `release()`.

        Returns:
            The artifact.
        """
        threshold = (
            WORKFLOWS_VIDEO_CLIP_INMEMORY_MAX_BYTES
            if inmemory_max_bytes is None
            else inmemory_max_bytes
        )
        size_bytes = path.stat().st_size
        content: Union[bytes, Path] = path
        if size_bytes <= threshold:
            content = path.read_bytes()
            path.unlink(missing_ok=True)

        artifact = cls(
            content,
            size_bytes=size_bytes,
            duration_seconds=duration_seconds,
            report=report,
            on_release=on_release,
        )

        return artifact

    @property
    def in_memory(self) -> bool:
        """True when the clip bytes are held in memory."""
        return isinstance(self.bytes_or_path, bytes)

    @property
    def released(self) -> bool:
        """True once `release()` was called."""
        return self._released

    def read_bytes(self) -> bytes:
        """Return the whole clip as bytes.

        Returns:
            The MP4 bytes.

        Raises:
            RuntimeError: If the artifact was already released.
        """
        self._ensure_not_released()
        if isinstance(self.bytes_or_path, bytes):
            return self.bytes_or_path

        content = self.bytes_or_path.read_bytes()

        return content

    @contextmanager
    def open_stream(self) -> Iterator[BinaryIO]:
        """Open the clip for streaming reads (for example as a PUT body).

        Yields:
            A binary file object positioned at the start of the clip.

        Raises:
            RuntimeError: If the artifact was already released.
        """
        self._ensure_not_released()
        if isinstance(self.bytes_or_path, bytes):
            yield io.BytesIO(self.bytes_or_path)
            return

        with self.bytes_or_path.open("rb") as stream:
            yield stream

    def release(self) -> None:
        """Free the clip: delete its temporary file and run `on_release`.

        Idempotent and thread-safe.
        """
        with self._lock:
            if self._released:
                return None
            self._released = True
            on_release, self._on_release = self._on_release, None
            content = self.bytes_or_path
            self.bytes_or_path = b""

        if isinstance(content, Path):
            try:
                content.unlink(missing_ok=True)
            except OSError as error:
                logger.warning(
                    "Could not delete video clip file %s: %s", content, error
                )
        if on_release is not None:
            on_release()

    def _ensure_not_released(self) -> None:
        if self._released:
            raise RuntimeError("The video clip artifact was already released")
