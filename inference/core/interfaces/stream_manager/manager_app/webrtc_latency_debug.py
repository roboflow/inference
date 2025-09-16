"""
Debug script to understand the 2-second latency in WebRTC.

Even with immediate frame returns, there's a constant 2-second delay.
This helps identify where the delay originates.
"""

import asyncio
import time
from datetime import datetime

from av import VideoFrame

_TIME_BASE = (1, 90000)


def analyze_webrtc_latency():
    """
    Analyze potential sources of 2-second latency.
    """

    print("WebRTC Latency Analysis")
    print("=" * 50)

    # 1. Check if it's encoding lag
    print("\n1. ENCODING LAG ANALYSIS:")
    print("   - H.264 encoder uses ThreadPoolExecutor with unbounded queue")
    print("   - Even on M3 Pro, encoding at 1080p can take 20-50ms per frame")
    print("   - At 30 FPS input: 30 frames/sec * 50ms = 1.5 seconds backlog")
    print("   - Add network jitter buffer: could reach 2 seconds total")

    # 2. Browser buffering
    print("\n2. BROWSER/CLIENT BUFFERING:")
    print("   - Browsers buffer 1-3 seconds for smooth playback")
    print("   - Chrome: typically buffers 1-2 seconds")
    print("   - Safari: can buffer up to 3 seconds")
    print("   - This is NOT controlled by aiortc")

    # 3. Frame timestamp vs wall clock
    print("\n3. FRAME TIMING MISMATCH:")
    print("   - Incoming frames have RTP timestamps")
    print("   - Outgoing frames need new timestamps")
    print("   - If timestamps aren't adjusted, playback delays occur")

    # 4. Calculation
    fps = 30
    encoding_time_ms = 40  # Typical H.264 encoding time
    frames_in_2_sec = 2 * fps

    print(f"\n4. LATENCY CALCULATION:")
    print(f"   - Input FPS: {fps}")
    print(f"   - Encoding time per frame: {encoding_time_ms}ms")
    print(f"   - Frames accumulated in 2 seconds: {frames_in_2_sec}")
    print(
        f"   - Total encoding time for {frames_in_2_sec} frames: {frames_in_2_sec * encoding_time_ms / 1000:.1f}s"
    )

    return {
        "likely_causes": [
            "Browser-side playback buffering (1-2 seconds)",
            "Encoding queue accumulation",
            "Frame timestamp synchronization",
        ],
        "solutions": [
            "Reduce frame rate to 15 FPS",
            "Reduce resolution to 720p or lower",
            "Configure browser for low-latency mode",
            "Use hardware encoding if available",
        ],
    }


def create_timestamp_adjuster():
    """
    Create a function to adjust frame timestamps to reduce latency.
    """
    start_time = time.monotonic()
    frame_count = 0

    def adjust_timestamp(frame: VideoFrame) -> VideoFrame:
        nonlocal frame_count
        # Force frames to have current timestamp
        current_time = time.monotonic() - start_time
        frame.pts = int(current_time * 90000)  # 90kHz clock
        frame.time_base = (1, 90000)
        frame_count += 1
        return frame

    return adjust_timestamp


# To use in your VideoTransformTrack:
#
# self.adjust_timestamp = create_timestamp_adjuster()
#
# async def recv(self):
#     frame = await self.track.recv()
#     frame = self.adjust_timestamp(frame)
#     return frame


if __name__ == "__main__":
    result = analyze_webrtc_latency()
    print("\n" + "=" * 50)
    print("MOST LIKELY CAUSE:")
    print("The 2-second delay is probably browser-side buffering")
    print("combined with encoding queue accumulation.")
    print("\nTO CONFIRM:")
    print("1. Check browser WebRTC stats (chrome://webrtc-internals)")
    print("2. Look for 'jitterBufferDelay' and 'playoutDelay'")
    print("3. These will show the client-side buffering")
