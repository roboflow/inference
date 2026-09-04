#!/usr/bin/env python3
"""One bounded publisher or reader for the multi-cell staging campaign.

The probe emits a pixel-clock stream or reads it with PyAV.  Reader reports
contain per-frame monotonic arrival times, per-frame glass-to-glass latency, and
encoded packet payload bytes.  Credentials and media paths are never reported.
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import time
import urllib.parse

BARS = 32
URL_PATTERN = re.compile(r"(?:https?|rtsp|rtsps)://[^\s\"']+", re.I)


def redact_url(value):
    try:
        parsed = urllib.parse.urlsplit(str(value))
    except ValueError:
        return "[url-redacted]"
    if not parsed.scheme or not parsed.hostname:
        return "[value-redacted]"
    port = ":%d" % parsed.port if parsed.port else ""
    return "%s://%s%s/[path-redacted]" % (parsed.scheme, parsed.hostname, port)


def redact_text(value):
    return URL_PATTERN.sub(lambda match: redact_url(match.group(0)), str(value))


def expand_url(template, stream):
    if not template or "{stream}" not in template:
        raise ValueError("BENCH_URL_TEMPLATE must contain {stream}")
    return template.replace("{stream}", stream)


def now_clock_ms():
    return int(time.time() * 1000) & 0xFFFFFFFF


def paint_clock(frame, value):
    width = frame.shape[1]
    bar_width = width // BARS
    if bar_width < 4:
        raise ValueError("frame is too narrow for pixel-clock-v1")
    for index in range(BARS):
        bit = (value >> (BARS - 1 - index)) & 1
        x0 = index * bar_width
        x1 = width if index == BARS - 1 else (index + 1) * bar_width
        frame[:, x0:x1, :] = 255 if bit else 0
    return frame


def decode_clock(frame):
    width = frame.shape[1]
    bar_width = width // BARS
    if bar_width < 4:
        raise ValueError("decoded frame is too narrow for pixel-clock-v1")
    value = 0
    y0 = frame.shape[0] // 4
    y1 = frame.shape[0] * 3 // 4
    for index in range(BARS):
        x0 = index * bar_width + max(1, bar_width // 5)
        x1 = (index + 1) * bar_width - max(1, bar_width // 5)
        value = (value << 1) | (1 if frame[y0:y1, x0:x1].mean() > 127 else 0)
    return value


def build_publish_command(ffmpeg, width, height, fps, bitrate_bps, url):
    return [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        "%dx%d" % (width, height),
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-tune",
        "zerolatency",
        "-bf",
        "0",
        "-g",
        str(max(1, int(round(fps * 2)))),
        "-b:v",
        str(int(bitrate_bps)),
        "-minrate",
        str(int(bitrate_bps)),
        "-maxrate",
        str(int(bitrate_bps)),
        "-bufsize",
        str(int(bitrate_bps)),
        "-x264-params",
        "nal-hrd=cbr:force-cfr=1",
        "-f",
        "rtsp",
        "-rtsp_transport",
        "tcp",
        url,
    ]


def identity_from_environment():
    expected_node = os.environ.get("BENCH_EXPECTED_NODE")
    observed_node = os.environ.get("BENCH_OBSERVED_NODE")
    if not expected_node or not observed_node or expected_node != observed_node:
        raise ValueError("observed node does not match immutable expected node")
    required = (
        "BENCH_RUN_ID",
        "BENCH_EXPECTED_CELL",
        "BENCH_EXPECTED_NODE_UID",
        "BENCH_EXPECTED_NETWORK_SHA256",
        "BENCH_SOURCE_PLACEMENT_GENERATION",
        "BENCH_FIXTURE_SHA256",
        "BENCH_POD_UID",
    )
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        raise ValueError(
            "required identity environment is missing: %s" % ",".join(missing)
        )
    return {
        "runId": os.environ["BENCH_RUN_ID"],
        "cell": os.environ["BENCH_EXPECTED_CELL"],
        "nodeName": observed_node,
        "expectedNodeUid": os.environ["BENCH_EXPECTED_NODE_UID"],
        "expectedNetworkSha256": os.environ["BENCH_EXPECTED_NETWORK_SHA256"],
        "sourcePlacementGeneration": int(
            os.environ["BENCH_SOURCE_PLACEMENT_GENERATION"]
        ),
        "fixtureSha256": os.environ["BENCH_FIXTURE_SHA256"],
        "podUid": os.environ["BENCH_POD_UID"],
    }


def _dimensions():
    width = int(os.environ.get("BENCH_WIDTH", "0"))
    height = int(os.environ.get("BENCH_HEIGHT", "0"))
    fps = float(os.environ.get("BENCH_EXPECTED_FPS", "0"))
    bitrate = int(os.environ.get("BENCH_BITRATE_BPS", "0"))
    warmup = float(os.environ.get("BENCH_WARMUP_SECONDS", "0"))
    measure = float(os.environ.get("BENCH_MEASURE_SECONDS", "0"))
    startup_grace = float(os.environ.get("BENCH_STARTUP_GRACE_SECONDS", "0"))
    shutdown_margin = float(os.environ.get("BENCH_SHUTDOWN_MARGIN_SECONDS", "0"))
    if (
        min(
            width,
            height,
            fps,
            bitrate,
            warmup,
            measure,
            startup_grace,
            shutdown_margin,
        )
        <= 0
    ):
        raise ValueError("fixture dimensions, rates, and timing must be positive")
    return (
        width,
        height,
        fps,
        bitrate,
        warmup,
        measure,
        startup_grace,
        shutdown_margin,
    )


def publish_clock(url, stopped):
    import numpy as np

    (
        width,
        height,
        fps,
        bitrate,
        warmup,
        measure,
        startup_grace,
        shutdown_margin,
    ) = _dimensions()
    ffmpeg = shutil.which(os.environ.get("FFMPEG_BIN", "ffmpeg"))
    if not ffmpeg:
        raise ValueError("ffmpeg is not installed")
    command = build_publish_command(ffmpeg, width, height, fps, bitrate, url)
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    interval = 1.0 / fps
    # Publisher starts before relay readiness and processor claim barriers. Keep
    # it alive for the whole bounded control-plane grace plus probe window.
    # Retain a bounded media tail after the reader's measurement deadline. It
    # prevents an exactly-on-budget readiness/claim sequence from racing the
    # publisher's final frame, while leaving at least half the Kubernetes
    # shutdown margin for ffmpeg exit and termination-report flush.
    media_tail = min(5.0, shutdown_margin / 2.0)
    deadline = time.monotonic() + startup_grace + warmup + measure + media_tail
    next_frame = time.monotonic()
    frames = 0
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    error = None
    try:
        while time.monotonic() < deadline and not stopped["requested"]:
            delay = next_frame - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            paint_clock(frame, now_clock_ms())
            process.stdin.write(frame.tobytes())
            frames += 1
            next_frame += interval
    except (BrokenPipeError, OSError) as exc:
        error = redact_text(exc)
    finally:
        if process.stdin:
            process.stdin.close()
        try:
            return_code = process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            return_code = process.wait(timeout=5)
        stderr = redact_text(process.stderr.read().decode("utf-8", "replace"))[-2000:]
    return {
        "success": not stopped["requested"] and error is None and return_code == 0,
        "generatedFrames": frames,
        "returnCode": return_code,
        "error": error,
        "stderr": stderr,
        "command": [redact_text(value) for value in command],
    }


def probe_clock(url, stopped, decode):
    import av

    (
        _width,
        _height,
        _fps,
        _bitrate,
        warmup,
        measure,
        _startup_grace,
        _shutdown_margin,
    ) = _dimensions()
    connect_start_ns = time.monotonic_ns()
    container = av.open(
        url,
        mode="r",
        options={
            "rtsp_transport": "tcp",
            "fflags": "nobuffer",
            "flags": "low_delay",
            "max_delay": "0",
            "rw_timeout": "5000000",
        },
    )
    video = next(
        (stream for stream in container.streams if stream.type == "video"), None
    )
    if video is None:
        raise ValueError("media stream contains no video track")
    measurement_start = time.monotonic() + warmup
    deadline = measurement_start + measure
    first_frame_ns = None
    arrivals = []
    latencies = []
    encoded_payload_bytes = 0
    packets = 0
    try:
        for packet in container.demux(video):
            now = time.monotonic()
            if stopped["requested"] or now >= deadline:
                break
            if packet.size:
                packets += 1
                if now >= measurement_start:
                    encoded_payload_bytes += int(packet.size)
            if not decode:
                continue
            for frame in packet.decode():
                arrival_ns = time.monotonic_ns()
                first_frame_ns = first_frame_ns or arrival_ns
                if time.monotonic() < measurement_start:
                    continue
                image = frame.to_ndarray(format="bgr24")
                encoded_ms = decode_clock(image)
                latency_ms = (now_clock_ms() - encoded_ms) & 0xFFFFFFFF
                if latency_ms > 60_000:
                    raise ValueError("pixel-clock frame failed integrity bound")
                arrivals.append(arrival_ns)
                latencies.append(latency_ms)
    finally:
        container.close()
    return {
        "success": not stopped["requested"]
        and packets > 0
        and (bool(arrivals) if decode else True),
        "connectStartMonotonicNs": connect_start_ns,
        "firstDecodedFrameMonotonicNs": first_frame_ns,
        "decodedFrameArrivalMonotonicNs": arrivals,
        "pixelClockLatencyMs": latencies,
        "latencySource": "pixel-clock-v1" if decode else None,
        "encodedPayloadBytes": encoded_payload_bytes,
        "encodedPayloadBytesSource": "pyav-packet-size",
        "packets": packets,
        "measurementStartMonotonicNs": int(measurement_start * 1e9),
        "measurementEndMonotonicNs": time.monotonic_ns(),
    }


def write_report(path, report):
    payload = json.dumps(report, sort_keys=True, separators=(",", ":"))
    Path = __import__("pathlib").Path
    Path(path).write_text(payload + "\n")
    print("BENCHMARK_FINAL_JSON=" + payload, flush=True)


def run(args):
    identity = identity_from_environment()
    stream = os.environ.get("BENCH_STREAM")
    url = expand_url(os.environ.get("BENCH_URL_TEMPLATE"), stream)
    stopped = {"requested": False}

    def request_stop(_signum, _frame):
        stopped["requested"] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started = time.time()
    try:
        if args.role == "publish-clock":
            result = publish_clock(url, stopped)
        else:
            result = probe_clock(url, stopped, decode=args.role == "probe-clock")
        report = {
            "schemaVersion": 1,
            "role": args.role,
            "identity": identity,
            "endpoint": redact_url(url),
            "startedUnixSeconds": started,
            "finishedUnixSeconds": time.time(),
            **result,
        }
    except (
        Exception
    ) as error:  # bounded terminal evidence is more useful than a traceback
        report = {
            "schemaVersion": 1,
            "role": args.role,
            "identity": identity,
            "endpoint": redact_url(url),
            "startedUnixSeconds": started,
            "finishedUnixSeconds": time.time(),
            "success": False,
            "error": redact_text(error),
        }
    write_report(args.report_path, report)
    return 0 if report["success"] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role", required=True, choices=("publish-clock", "probe-clock", "probe-copy")
    )
    parser.add_argument("--report-path", default="/dev/termination-log")
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
