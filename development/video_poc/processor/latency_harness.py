"""Glass-to-glass latency harness for the video POC ingest/publish legs.

Publishes a stream whose pixels ENCODE wallclock time (32 vertical bars = a
32-bit millisecond counter), then reads it back through any leg and computes
latency per frame with no OCR, no shared state, and no browser:

    latency_ms = now_ms - decoded_ms

Modes:
  publish            generate + push the pixel clock to the relay (rtsp)
  probe              read a stream via cv2.VideoCapture and report latency
                     (set OPENCV_FFMPEG_CAPTURE_OPTIONS in the env to test
                     capture configurations; pass --n-threads to test
                     CAP_PROP_N_THREADS as an open parameter)

Typical experiment (three terminals or backgrounded):
  python latency_harness.py publish
  OPENCV_FFMPEG_CAPTURE_OPTIONS="max_delay;0" python latency_harness.py probe

The bars are 20px wide at 640x120 so they survive x264 quantization cleanly.
"""

import argparse
import os
import subprocess
import sys
import time

WIDTH, HEIGHT, FPS, BARS = 640, 120, 30, 32
BAR_W = WIDTH // BARS
FFMPEG = os.getenv(
    "FFMPEG_BIN",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bin", "ffmpeg"),
)
if not os.path.exists(FFMPEG):
    FFMPEG = "ffmpeg"


def now_ms() -> int:
    return int(time.time() * 1000) & 0xFFFFFFFF


def publish(url: str):
    import numpy as np

    proc = subprocess.Popen(
        [
            FFMPEG, "-hide_banner", "-loglevel", "warning",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{WIDTH}x{HEIGHT}",
            "-r", str(FPS), "-i", "-",
            # mirror the connector's encode settings
            "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
            "-pix_fmt", "yuv420p", "-g", "30", "-bf", "0",
            "-f", "rtsp", "-rtsp_transport", "tcp", url,
        ],
        stdin=subprocess.PIPE,
    )
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    interval = 1.0 / FPS
    print(f"[harness] publishing pixel clock to {url}")
    try:
        while True:
            tick = time.time()
            value = now_ms()
            for i in range(BARS):
                bit = (value >> (BARS - 1 - i)) & 1
                frame[:, i * BAR_W : (i + 1) * BAR_W, :] = 255 if bit else 0
            proc.stdin.write(frame.tobytes())
            delay = interval - (time.time() - tick)
            if delay > 0:
                time.sleep(delay)
    except (KeyboardInterrupt, BrokenPipeError):
        pass
    finally:
        try:
            proc.terminate()
        except Exception:
            pass


def decode_frame(image) -> int:
    value = 0
    y0, y1 = HEIGHT // 4, 3 * HEIGHT // 4
    for i in range(BARS):
        x0 = i * BAR_W + 4
        x1 = (i + 1) * BAR_W - 4
        mean = image[y0:y1, x0:x1].mean()
        value = (value << 1) | (1 if mean > 127 else 0)
    return value


def probe_ffmpeg(url: str, samples: int, extra_flags):
    """Read via the ffmpeg CLI (bypasses OpenCV entirely) to isolate whether
    the RTSP read latency lives in ffmpeg's client stack or cv2's wrapper."""
    import numpy as np

    cmd = [FFMPEG, "-hide_banner", "-loglevel", "warning", "-rtsp_transport", "tcp"]
    if extra_flags:
        cmd += extra_flags.split()
    cmd += ["-i", url, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    print(f"[harness] ffmpeg cmd: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=WIDTH * HEIGHT * 3)
    frame_bytes = WIDTH * HEIGHT * 3
    lat = []
    warmup = 30
    while len(lat) < samples + warmup:
        buf = proc.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            print("[harness] ffmpeg pipe closed")
            break
        image = np.frombuffer(buf, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
        value = decode_frame(image)
        delta = (now_ms() - value) & 0xFFFFFFFF
        if delta < 60_000:
            lat.append(delta)
    proc.terminate()
    lat = lat[warmup:]
    if not lat:
        print("[harness] no valid samples")
        sys.exit(2)
    lat.sort()
    n = len(lat)
    print(
        f"[harness] samples={n} p50={lat[n // 2]}ms p90={lat[int(n * 0.9)]}ms "
        f"min={lat[0]}ms max={lat[-1]}ms"
    )


def probe(url: str, samples: int, n_threads):
    import cv2

    print(f"[harness] OPENCV_FFMPEG_CAPTURE_OPTIONS={os.getenv('OPENCV_FFMPEG_CAPTURE_OPTIONS', '<unset>')}")
    if n_threads is not None:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, int(n_threads)])
        print(f"[harness] CAP_PROP_N_THREADS={n_threads} (open param)")
    else:
        cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("[harness] failed to open", url)
        sys.exit(2)
    lat = []
    warmup = 30
    while len(lat) < samples + warmup:
        ok, image = cap.read()
        if not ok:
            print("[harness] read failed")
            break
        value = decode_frame(image)
        delta = (now_ms() - value) & 0xFFFFFFFF
        if delta < 60_000:  # sanity: ignore garbled decodes
            lat.append(delta)
    cap.release()
    lat = lat[warmup:]
    if not lat:
        print("[harness] no valid samples")
        sys.exit(2)
    lat.sort()
    n = len(lat)
    print(
        f"[harness] samples={n} p50={lat[n // 2]}ms p90={lat[int(n * 0.9)]}ms "
        f"min={lat[0]}ms max={lat[-1]}ms"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["publish", "probe", "probe-ffmpeg"])
    parser.add_argument("--url", default="rtsp://127.0.0.1:8554/lat-test")
    parser.add_argument("--samples", type=int, default=150)
    parser.add_argument("--n-threads", type=int, default=None)
    parser.add_argument("--ffmpeg-flags", default="")
    args = parser.parse_args()
    if args.mode == "publish":
        publish(args.url)
    elif args.mode == "probe-ffmpeg":
        probe_ffmpeg(args.url, args.samples, args.ffmpeg_flags)
    else:
        probe(args.url, args.samples, args.n_threads)
