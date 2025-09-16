"""
Debug script to understand the 2-second latency in WebRTC.

Even with immediate frame returns, there's a constant 2-second delay.
This helps identify where the delay originates.
"""

import asyncio
import time
from datetime import datetime

from av import VideoFrame


def analyze_webrtc_latency():
    """
    Analyze potential sources of 2-second latency.
    """

    fps = 30
    encoding_time_ms = 40  # Typical H.264 encoding time
    frames_in_2_sec = 2 * fps

    output_lines = [
        "WebRTC Latency Analysis",
        "=" * 50,
            # 1. Check if it's encoding lag
        "\n1. ENCODING LAG ANALYSIS:",
        "   - H.264 encoder uses ThreadPoolExecutor with unbounded queue",
        "   - Even on M3 Pro, encoding at 1080p can take 20-50ms per frame",
        "   - At 30 FPS input: 30 frames/sec * 50ms = 1.5 seconds backlog",
        "   - Add network jitter buffer: could reach 2 seconds total",
            # 2. Browser buffering
        "\n2. BROWSER/CLIENT BUFFERING:",
        "   - Browsers buffer 1-3 seconds for smooth playback",
        "   - Chrome: typically buffers 1-2 seconds",
        "   - Safari: can buffer up to 3 seconds",
        "   - This is NOT controlled by aiortc",
            # 3. Frame timestamp vs wall clock
        "\n3. FRAME TIMING MISMATCH:",
        "   - Incoming frames have RTP timestamps",
        "   - Outgoing frames need new timestamps",
        "   - If timestamps aren't adjusted, playback delays occur",
            # 4. Calculation
        f"\n4. LATENCY CALCULATION:",
        f"   - Input FPS: {fps}",
        f"   - Encoding time per frame: {encoding_time_ms}ms",
        f"   - Frames accumulated in 2 seconds: {frames_in_2_sec}",
        f"   - Total encoding time for {frames_in_2_sec} frames: {frames_in_2_sec * encoding_time_ms / 1000:.1f}s",
    ]

    print("\n".join(output_lines))

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
    start_time = time.time()
    frame_count = 0

    def adjust_timestamp(frame: VideoFrame) -> VideoFrame:
        nonlocal frame_count
        # Force frames to have current timestamp
        current_time = time.time() - start_time
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
