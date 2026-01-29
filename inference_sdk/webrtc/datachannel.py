"""WebRTC data channel binary chunking utilities."""

import asyncio
import os
import struct
import time
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple

from inference_sdk.config import (
    WEBRTC_VIDEO_UPLOAD_BUFFER_LIMIT,
    WEBRTC_VIDEO_UPLOAD_CHUNK_SIZE,
)

if TYPE_CHECKING:
    from aiortc import RTCDataChannel

# Interval for sending keepalive pings to server watchdog during uploads
# Note: This does NOT prevent ICE consent expiry - that's handled by yielding
# to the event loop so aioice can send STUN packets. This ping is for the
# server-side watchdog (Modal container timeout).
WATCHDOG_PING_INTERVAL = 3.0  # seconds

# Pre-compiled struct for parsing 12-byte header (3 x uint32 little-endian)
_HEADER_STRUCT = struct.Struct("<III")


def _parse_chunked_binary_message(message: bytes) -> Tuple[int, int, int, bytes]:
    """Parse a binary message with standard 12-byte header.

    Format: [frame_id: 4][chunk_index: 4][total_chunks: 4][payload: N]
    All integers are uint32 little-endian.

    Returns: (frame_id, chunk_index, total_chunks, payload)
    """
    if len(message) < 12:
        raise ValueError(f"Message too short: {len(message)} bytes (expected >= 12)")

    frame_id, chunk_index, total_chunks = _HEADER_STRUCT.unpack(message[0:12])
    payload = message[12:]
    return frame_id, chunk_index, total_chunks, payload


class ChunkReassembler:
    """Helper to reassemble chunked binary messages."""

    def __init__(self):
        """Initialize the chunk reassembler."""
        self._chunks: Dict[int, Dict[int, bytes]] = (
            {}
        )  # {frame_id: {chunk_index: data}}
        self._total: Dict[int, int] = {}  # {frame_id: total_chunks}

    def add_chunk(self, message: bytes) -> Tuple[Optional[bytes], Optional[int]]:
        """Parse and add a chunk, returning complete payload and frame_id if all chunks received.

        Args:
            message: Raw binary message with 12-byte header

        Returns:
            Tuple of (payload, frame_id) if complete, (None, None) otherwise
        """
        # Parse the binary message
        frame_id, chunk_index, total_chunks, chunk_data = _parse_chunked_binary_message(
            message
        )

        # Initialize buffers for new frame
        if frame_id not in self._chunks:
            self._chunks[frame_id] = {}
            self._total[frame_id] = total_chunks

        # Store chunk
        self._chunks[frame_id][chunk_index] = chunk_data

        # Check if all chunks received
        if len(self._chunks[frame_id]) >= total_chunks:
            # Reassemble in order
            complete_payload = b"".join(
                self._chunks[frame_id][i] for i in range(total_chunks)
            )

            # Clean up buffers for completed frame - this is the key part!
            del self._chunks[frame_id]
            del self._total[frame_id]

            return complete_payload, frame_id

        return None, None


def create_video_upload_chunk(
    chunk_index: int, total_chunks: int, data: bytes
) -> bytes:
    """Create a video upload chunk message.

    Format: [chunk_index:u32][total_chunks:u32][payload]
    All integers are uint32 little-endian.

    Args:
        chunk_index: Zero-based index of this chunk
        total_chunks: Total number of chunks in the file
        data: Chunk payload bytes

    Returns:
        Binary message with 8-byte header + payload
    """
    return struct.pack("<II", chunk_index, total_chunks) + data


class VideoFileUploader:
    """Uploads a video file through a WebRTC datachannel in chunks.

    Protocol: [chunk_index:u32][total_chunks:u32][payload]
    Server auto-completes when all chunks received.

    Features:
    - Backpressure handling via bufferedAmount monitoring with low watermark
    - Event loop yielding every N chunks to prevent starvation (critical for
      ICE consent - aioice needs event loop time to send STUN refresh packets)
    - Periodic pings to keep server watchdog alive
    - Progress callback support
    """

    def __init__(
        self,
        path: str,
        channel: "RTCDataChannel",
        chunk_size: int = WEBRTC_VIDEO_UPLOAD_CHUNK_SIZE,
        buffer_limit: int = WEBRTC_VIDEO_UPLOAD_BUFFER_LIMIT,
        watchdog_ping_interval: float = WATCHDOG_PING_INTERVAL,
    ):
        """Initialize video file uploader.

        Args:
            path: Path to the video file to upload
            channel: RTCDataChannel to send chunks through
            chunk_size: Size of each chunk in bytes (default 48KB)
            buffer_limit: Max buffered bytes before applying backpressure
            watchdog_ping_interval: Seconds between watchdog pings (default 3s)
        """
        self._path = path
        self._channel = channel
        self._chunk_size = chunk_size
        self._buffer_limit = buffer_limit
        self._buffer_low = buffer_limit // 4  # Low watermark for backpressure
        self._watchdog_ping_interval = watchdog_ping_interval
        self._file_size = os.path.getsize(path)
        self._total_chunks = (self._file_size + chunk_size - 1) // chunk_size
        self._uploaded_chunks = 0
        self._last_watchdog_ping = time.time()

    @property
    def total_chunks(self) -> int:
        """Total number of chunks to upload."""
        return self._total_chunks

    @property
    def uploaded_chunks(self) -> int:
        """Number of chunks uploaded so far."""
        return self._uploaded_chunks

    @property
    def file_size(self) -> int:
        """Size of the file in bytes."""
        return self._file_size

    def _send_watchdog_ping(self) -> None:
        """Send ping to keep server watchdog alive if interval has elapsed.
        
        Note: This does NOT prevent ICE consent expiry - ICE consent is maintained
        by aioice sending STUN Binding Indications at the transport layer.
        This ping keeps the server-side watchdog (e.g., Modal container timeout)
        alive during long uploads. Server echoes these back.
        """
        now = time.time()
        if now - self._last_watchdog_ping >= self._watchdog_ping_interval:
            if self._channel.readyState == "open":
                # Send 1-byte ping that server echoes back
                self._channel.send(b"\x00")
                self._last_watchdog_ping = now

    async def upload(
        self, on_progress: Optional[Callable[[int, int], None]] = None
    ) -> None:
        """Upload the file in chunks with backpressure handling.

        Args:
            on_progress: Optional callback called after each chunk with
                (uploaded_chunks, total_chunks)

        Raises:
            RuntimeError: If channel closes during upload
        """
        with open(self._path, "rb") as f:
            for chunk_idx in range(self._total_chunks):
                if self._channel.readyState != "open":
                    raise RuntimeError("Upload channel closed during upload")

                chunk_data = f.read(self._chunk_size)
                message = create_video_upload_chunk(
                    chunk_idx, self._total_chunks, chunk_data
                )

                # Backpressure: wait for buffer to drain to low watermark
                # The asyncio.sleep() yields to event loop, allowing aioice to
                # send STUN consent refresh packets (critical for ICE health)
                while self._channel.bufferedAmount > self._buffer_limit:
                    self._send_watchdog_ping()
                    await asyncio.sleep(0.01)
                    # Check channel still open after waiting
                    if self._channel.readyState != "open":
                        raise RuntimeError("Upload channel closed during backpressure wait")

                self._channel.send(message)
                self._uploaded_chunks = chunk_idx + 1

                if on_progress:
                    on_progress(self._uploaded_chunks, self._total_chunks)
                
                # Ping server watchdog periodically
                self._send_watchdog_ping()
                
                # CRITICAL: Yield to event loop every 10 chunks
                # Without this, aioice cannot send STUN Binding Indications to
                # refresh ICE consent (expires after ~30s without refresh).
                # Event loop starvation is the root cause of "Consent to send expired".
                if chunk_idx % 10 == 0:
                    await asyncio.sleep(0)