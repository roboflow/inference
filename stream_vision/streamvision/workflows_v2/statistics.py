"""Final per-member counters and end reasons of the video sources.

Pass one ``VideoStatistics`` as the ``video_statistics`` resource; every video
source of the session registers its members in it when it opens::

    statistics = VideoStatistics()
    session = plan.create_session({"video_statistics": statistics})
    run = session.start(handlers=handlers)
    run.wait()
    statistics.members()["cams/3"].end_reason      # e.g. "end_of_stream"

The record of a member survives a run without any emission, a failed run and
``close``. The next run of a source with the same name replaces its records.

Frame accounting of one member once its source closed::

    grabbed              VideoSource grabs (decoder handed over a frame)
      = video_source_dropped + read      (VideoSource buffer of B frames)
    read                 frames the member thread took from VideoSource
      = discarded        read after the member stopped or failed
      + evicted          oldest frame of a full handoff (S frames) replaced
      + consumed         emitted to the engine
      + cleared          waiting in the handoff when the member failed or closed

``grabbed`` exceeds ``video_source_dropped + read`` only by a grab whose
retrieve failed (the end of that member).

End reasons: ``end_of_stream`` (a video file ended), ``stopped`` (the engine
or a policy stopped the member), and the failures ``open_failed``,
``source_error`` (VideoSource capture thread error), ``live_stream_ended``
(VideoSource cannot tell a live disconnect from a regular end), ``stalled``
(no frame within ``stall_timeout``) and ``frame_error`` (a frame could not be
converted). Nothing before VideoSource's grab
is counted: frames lost in the network, decoder or appsink are not visible
here. ``native`` holds the hardware bridge's own counters, frozen when the
producer was released, for forced CUDA decoders only.
"""

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Tuple

if TYPE_CHECKING:
    from streamvision.workflows_v2.reader import MemberReader

__all__ = ["MemberStatistics", "VideoStatistics"]


@dataclass(frozen=True)
class MemberStatistics:
    """Snapshot of one member's counters and end.

    Attributes:
        member_id: ``"<source>"`` for a single stream, ``"<source>/<index>"``
            for a member of a set; the same string as the frame's
            ``SampleContext.source_id`` and ``VideoMetadata.video_identifier``.
        reference: Video reference with credentials removed.
        decoder: Requested decoder (``auto``, ``opencv``, ``gstreamer_cuda``,
            ``jetson``), or ``factory`` when an injected producer factory
            built the producer.
        is_file: Whether VideoSource reported a file; ``None`` before open.
        grabbed: VideoSource grabs.
        video_source_dropped: Frames VideoSource dropped (its full buffer).
        read: Frames the member thread read from VideoSource.
        discarded: Frames read after the member stopped or failed.
        evicted: Frames replaced as the oldest of a full handoff.
        consumed: Frames emitted to the engine.
        cleared: Frames waiting in the handoff when it was cleared.
        native: Counters of the hardware bridge at producer release; ``None``
            when the decoder has none or they could not be read.
        end_reason: Why the member ended; ``None`` while it runs.
        end_detail: Human-readable detail of ``end_reason``.
        closed: ``True`` once the member's threads ended, VideoSource
            terminated and the producer was released within the stop timeout.
        close_errors: Stop problems: threads still alive, terminate or
            producer release failures.
    """

    member_id: str
    reference: str
    decoder: str
    is_file: Optional[bool]
    grabbed: int
    video_source_dropped: int
    read: int
    discarded: int
    evicted: int
    consumed: int
    cleared: int
    native: Optional[Mapping[str, int]]
    end_reason: Optional[str]
    end_detail: Optional[str]
    closed: bool
    close_errors: Tuple[str, ...]


class VideoStatistics:
    """Statistics resource shared by the video sources of a session.

    Thread-safe. Snapshots taken during a run are consistent per member for
    the handoff counters and may lag for the VideoSource counters; snapshots
    taken after the run finished are final.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sources: Dict[str, List["MemberReader"]] = {}

    def members(self) -> Dict[str, MemberStatistics]:
        """Return a snapshot of every registered member, keyed by member id.

        Returns:
            One ``MemberStatistics`` per member of every registered source.
        """
        with self._lock:
            readers = [reader for group in self._sources.values() for reader in group]
        snapshot = {reader.member_id: reader.statistics() for reader in readers}

        return snapshot

    def source(self, source_name: str) -> Dict[str, MemberStatistics]:
        """Return a snapshot of one source's members, keyed by member id.

        Args:
            source_name: Declared name of the source.

        Returns:
            The members in index order; empty when the source never opened.
        """
        with self._lock:
            readers = list(self._sources.get(source_name, ()))
        snapshot = {reader.member_id: reader.statistics() for reader in readers}

        return snapshot

    def _register(self, source_name: str, readers: List["MemberReader"]) -> None:
        # Called by a source's open(); replaces the previous run's records.
        with self._lock:
            self._sources[source_name] = list(readers)
