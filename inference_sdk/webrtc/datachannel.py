"""WebRTC data channel binary chunking utilities."""

import struct
from typing import Dict, Optional, Tuple


def _parse_chunked_binary_message(message: bytes) -> Tuple[int, int, int, bytes]:
    """Parse a binary message with standard 12-byte header.

    Format: [frame_id: 4][chunk_index: 4][total_chunks: 4][payload: N]
    All integers are uint32 little-endian.

    Returns: (frame_id, chunk_index, total_chunks, payload)
    """
    if len(message) < 12:
        raise ValueError(f"Message too short: {len(message)} bytes (expected >= 12)")

    frame_id, chunk_index, total_chunks = struct.unpack("<III", message[0:12])
    payload = message[12:]
    return frame_id, chunk_index, total_chunks, payload


class ChunkReassembler:
    """Helper to reassemble chunked binary messages."""

    def __init__(self):
        """Initialize the chunk reassembler."""
        self._chunks: Dict[int, Dict[int, bytes]] = {}  # {frame_id: {chunk_index: data}}
        self._total: Dict[int, int] = {}  # {frame_id: total_chunks}

    def add_chunk(self, message: bytes) -> Tuple[Optional[bytes], Optional[int]]:
        """Parse and add a chunk, returning complete payload and frame_id if all chunks received.

        Args:
            message: Raw binary message with 12-byte header

        Returns:
            Tuple of (payload, frame_id) if complete, (None, None) otherwise
        """
        # Parse the binary message
        frame_id, chunk_index, total_chunks, chunk_data = _parse_chunked_binary_message(message)

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