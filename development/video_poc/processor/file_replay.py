"""Build the live RTSP replay used for uploaded-file stream jobs.

Uploaded files are not guaranteed to already have a low-latency encoding.  In
particular, stream-copying H.264 with B-frames into RTSP is incompatible with
the processor's fail-fast/low-delay decoder: disabling the decoder reorder
buffer can expose frames in decode order rather than presentation order.

The connector solves the same problem for its file sources by normalising them
to H.264 with zerolatency tuning and no B-frames.  Keep the worker-side replay
equivalent so an uploaded recording behaves like the connector stream made
from that recording.
"""


DEFAULT_BITRATE_KBPS = 1500


def build_file_replay_command(
    ffmpeg_bin: str,
    source_path: str,
    publish_url: str,
    bitrate_kbps: int = DEFAULT_BITRATE_KBPS,
):
    """Return a real-time, looping, low-latency H.264 RTSP publisher command."""
    bitrate_kbps = int(bitrate_kbps)
    if bitrate_kbps <= 0:
        raise ValueError("file replay bitrate must be positive")
    # A half-second VBV window bounds RTSP/TCP bursts without introducing a
    # standing frame buffer.  This is the same envelope used by rf-connector.
    buffer_kbps = max(1, bitrate_kbps // 2)
    return [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-re",
        "-stream_loop",
        "-1",
        "-i",
        source_path,
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-tune",
        "zerolatency",
        "-profile:v",
        "baseline",
        "-pix_fmt",
        "yuv420p",
        "-bf",
        "0",
        "-b:v",
        f"{bitrate_kbps}k",
        "-maxrate",
        f"{bitrate_kbps}k",
        "-bufsize",
        f"{buffer_kbps}k",
        # A viewer can only join after an IDR.  Use stream time rather than a
        # fixed GOP frame count so native 24/30/60 FPS inputs behave alike.
        "-force_key_frames",
        "expr:gte(t,n_forced)",
        "-f",
        "rtsp",
        "-rtsp_transport",
        "tcp",
        publish_url,
    ]
