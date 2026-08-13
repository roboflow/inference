#!/usr/bin/env python3
"""Run one distributed relay benchmark client and emit a structured report.

Media URLs are read from environment variables so credentials never appear in
the Kubernetes Job arguments. The final report is written to the configured
path and emitted as one ``BENCHMARK_FINAL_JSON=...`` log line for collection by
the controller.
"""

import argparse
import http.server
import json
import os
import platform
import re
import resource
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse


ROLES = ("publish-copy", "read-copy", "read-decode")
URL_PATTERN = re.compile(r"(?:https?|rtsp|rtmp|srt)://[^\s\"']+", re.IGNORECASE)


def redact_url(value):
    """Retain only a URL's scheme and host; redact credentials and media path."""
    try:
        parsed = urllib.parse.urlsplit(str(value))
    except ValueError:
        return "[url-redacted]"
    if not parsed.scheme or not parsed.netloc:
        return str(value)
    hostname = parsed.hostname or "host-redacted"
    try:
        port = parsed.port
    except ValueError:
        port = None
    if port:
        hostname = "%s:%d" % (hostname, port)
    credentials = "[credentials-redacted]@" if parsed.username else ""
    path = "/[path-redacted]" if parsed.path not in ("", "/") else parsed.path
    query = "?[query-redacted]" if parsed.query else ""
    return "%s://%s%s%s%s" % (
        parsed.scheme,
        credentials,
        hostname,
        path,
        query,
    )


def redact_text(value):
    return URL_PATTERN.sub(lambda match: redact_url(match.group(0)), str(value))


def resolve_url_from_environment(name, stream, require_stream_placeholder=False):
    value = os.environ.get(name)
    if not value:
        raise ValueError("required URL environment variable is unset: %s" % name)
    if require_stream_placeholder and "{stream}" not in value:
        raise ValueError(
            "URL template environment variable %s must contain {stream}" % name
        )
    return value.replace("{stream}", stream)


def build_ffmpeg_command(role, ffmpeg, input_url, output_url=None):
    common = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-stats_period",
        "1",
        "-progress",
        "pipe:1",
    ]
    if role == "publish-copy":
        if not output_url:
            raise ValueError("publish-copy requires an output URL")
        return common + [
            "-re",
            "-stream_loop",
            "-1",
            "-i",
            input_url,
            "-map",
            "0:v:0",
            "-an",
            "-c",
            "copy",
            "-f",
            "rtsp",
            "-rtsp_transport",
            "tcp",
            output_url,
        ]
    if role == "read-copy":
        return common + [
            "-rtsp_transport",
            "tcp",
            "-i",
            input_url,
            "-map",
            "0:v:0",
            "-an",
            "-c",
            "copy",
            "-f",
            "null",
            "-",
        ]
    if role == "read-decode":
        return common + [
            "-rtsp_transport",
            "tcp",
            "-i",
            input_url,
            "-map",
            "0:v:0",
            "-an",
            "-f",
            "null",
            "-",
        ]
    raise ValueError("unsupported role: %s" % role)


def ffmpeg_version(ffmpeg):
    result = subprocess.run(
        [ffmpeg, "-version"],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = result.stdout.splitlines()
    return lines[0] if lines else "unknown"


def redacted_command(command):
    return [redact_text(part) for part in command]


def _number(value, integer=False):
    try:
        return int(value) if integer else float(value)
    except (TypeError, ValueError):
        return None


def _rate(value):
    match = re.match(r"^([-+0-9.eE]+)([kmg]?bits/s)?$", str(value).strip())
    if not match:
        return None
    number = _number(match.group(1))
    if number is None:
        return None
    multipliers = {"bits/s": 1.0, "kbits/s": 1000.0, "mbits/s": 1e6, "gbits/s": 1e9}
    return number * multipliers.get((match.group(2) or "bits/s").lower(), 1.0)


class ProgressAccumulator:
    """Parse records produced by ``ffmpeg -progress pipe:1``."""

    def __init__(self):
        self.current = {}
        self.records = 0
        self.first_progress_at = None
        self.first_media_at = None
        self.last_progress_at = None
        self.max_frame = 0
        self.max_total_size = 0
        self.max_out_time_us = 0
        self.drop_frames = 0
        self.dup_frames = 0
        self.reported_fps = None
        self.reported_bitrate_bps = None
        self.speed = None

    def feed_line(self, line, now=None):
        if "=" not in line:
            return False
        key, value = line.rstrip("\r\n").split("=", 1)
        self.current[key] = value
        if key != "progress":
            return False
        now = time.time() if now is None else now
        self.records += 1
        self.first_progress_at = self.first_progress_at or now
        self.last_progress_at = now
        self.max_frame = max(
            self.max_frame, _number(self.current.get("frame"), True) or 0
        )
        self.max_total_size = max(
            self.max_total_size, _number(self.current.get("total_size"), True) or 0
        )
        out_time_us = _number(self.current.get("out_time_us"), True)
        if out_time_us is None:
            out_time_ms = _number(self.current.get("out_time_ms"), True)
            out_time_us = out_time_ms if out_time_ms is not None else 0
        self.max_out_time_us = max(self.max_out_time_us, out_time_us or 0)
        if self.first_media_at is None and (
            self.max_frame > 0 or self.max_out_time_us > 0
        ):
            self.first_media_at = now
        self.drop_frames = max(
            self.drop_frames, _number(self.current.get("drop_frames"), True) or 0
        )
        self.dup_frames = max(
            self.dup_frames, _number(self.current.get("dup_frames"), True) or 0
        )
        self.reported_fps = _number(self.current.get("fps"))
        self.reported_bitrate_bps = _rate(self.current.get("bitrate"))
        speed = str(self.current.get("speed", "")).rstrip("x")
        self.speed = _number(speed)
        self.current = {}
        return True

    def summary(self):
        return {
            "records": self.records,
            "firstProgressAt": self.first_progress_at,
            "firstMediaAt": self.first_media_at,
            "lastProgressAt": self.last_progress_at,
            "frames": self.max_frame,
            "bytes": self.max_total_size,
            "mediaDurationSeconds": self.max_out_time_us / 1e6,
            "dropFrames": self.drop_frames,
            "duplicateFrames": self.dup_frames,
            "reportedFps": self.reported_fps,
            "reportedBitrateBps": self.reported_bitrate_bps,
            "speed": self.speed,
        }


def aggregate_progress(attempts, measured_seconds):
    progress = [attempt["progress"] for attempt in attempts]
    frames = sum(item["frames"] for item in progress)
    media_seconds = sum(item["mediaDurationSeconds"] for item in progress)
    last = progress[-1] if progress else {}
    return {
        "attempts": len(attempts),
        "frames": frames,
        "bytes": sum(item["bytes"] for item in progress),
        "mediaDurationSeconds": media_seconds,
        "dropFrames": sum(item["dropFrames"] for item in progress),
        "duplicateFrames": sum(item["duplicateFrames"] for item in progress),
        "deliveredFps": frames / measured_seconds if measured_seconds > 0 else 0.0,
        "mediaToWallRatio": (
            media_seconds / measured_seconds if measured_seconds > 0 else 0.0
        ),
        "lastReportedFps": last.get("reportedFps"),
        "lastReportedBitrateBps": last.get("reportedBitrateBps"),
        "lastReportedSpeed": last.get("speed"),
    }


def _prometheus_escape(value):
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def render_prometheus(state):
    labels = ",".join(
        '%s="%s"' % (key, _prometheus_escape(state[key]))
        for key in ("role", "location")
    )
    progress = state["progress"]
    return "\n".join(
        [
            "# TYPE video_relay_benchmark_agent_running gauge",
            "video_relay_benchmark_agent_running{%s} %d" % (labels, state["running"]),
            "# TYPE video_relay_benchmark_agent_progress_records_total counter",
            "video_relay_benchmark_agent_progress_records_total{%s} %d"
            % (labels, progress.records),
            "# TYPE video_relay_benchmark_agent_frames gauge",
            "video_relay_benchmark_agent_frames{%s} %d" % (labels, progress.max_frame),
            "# TYPE video_relay_benchmark_agent_reconnects_total counter",
            "video_relay_benchmark_agent_reconnects_total{%s} %d"
            % (labels, state["reconnects"]),
            "",
        ]
    )


class _MetricsHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/metrics":
            self.send_error(404)
            return
        payload = self.server.render_metrics().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format, *_args):
        return


class MetricsServer:
    def __init__(self, port, render_metrics):
        self.port = port
        self.render_metrics = render_metrics
        self.server = None
        self.thread = None

    def start(self):
        if self.port <= 0:
            return
        self.server = http.server.ThreadingHTTPServer(
            ("0.0.0.0", self.port), _MetricsHandler
        )
        self.server.render_metrics = self.render_metrics
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.thread is not None:
            self.thread.join(timeout=2)


def _stop_process(process):
    if process.poll() is not None:
        return process.returncode
    try:
        os.killpg(process.pid, signal.SIGTERM)
        return process.wait(timeout=8)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.wait(timeout=3)


def _placement():
    return {
        "hostname": platform.node(),
        "podName": os.environ.get("POD_NAME"),
        "podUid": os.environ.get("POD_UID"),
        "namespace": os.environ.get("POD_NAMESPACE"),
        "nodeName": os.environ.get("NODE_NAME"),
        "requestedNodeInstanceType": os.environ.get(
            "REQUESTED_NODE_INSTANCE_TYPE"
        ),
        "requestedImageReference": os.environ.get("AGENT_IMAGE_REFERENCE"),
        "expectedCell": os.environ.get("EXPECTED_CELL_ID"),
        "expectedClusterContext": os.environ.get("EXPECTED_CLUSTER_CONTEXT"),
        "mediaPath": os.environ.get("MEDIA_PATH"),
        "mediaPathKind": os.environ.get("MEDIA_PATH_KIND"),
    }


def run_agent(args):
    ffmpeg = shutil.which(args.ffmpeg)
    if ffmpeg is None:
        raise ValueError("ffmpeg not found: %s" % args.ffmpeg)
    input_url = resolve_url_from_environment(
        args.input_url_env,
        args.stream,
        require_stream_placeholder=args.role != "publish-copy",
    )
    output_url = None
    if args.role == "publish-copy":
        output_url = resolve_url_from_environment(
            args.output_url_env, args.stream, require_stream_placeholder=True
        )
    command = build_ffmpeg_command(args.role, ffmpeg, input_url, output_url)
    version = ffmpeg_version(ffmpeg)
    if args.dry_run:
        return {
            "schemaVersion": 1,
            "dryRun": True,
            "runId": args.run_id,
            "role": args.role,
            "location": args.location,
            "stream": args.stream,
            "ffmpeg": version,
            "command": redacted_command(command),
        }

    scheduled_at = time.time()
    stopped = {"requested": False}

    def request_stop(_signum, _frame):
        stopped["requested"] = True

    old_sigterm = signal.signal(signal.SIGTERM, request_stop)
    old_sigint = signal.signal(signal.SIGINT, request_stop)
    delay_deadline = time.monotonic() + args.start_delay_seconds
    while time.monotonic() < delay_deadline and not stopped["requested"]:
        time.sleep(min(0.25, delay_deadline - time.monotonic()))
    started_at = time.time()
    deadline = time.monotonic() + args.duration_seconds
    attempts = []
    reconnects = 0
    stop_reason = None
    live_progress = ProgressAccumulator()
    metrics_state = {
        "role": args.role,
        "location": args.location,
        "running": 1,
        "reconnects": 0,
        "progress": live_progress,
    }
    metrics_server = MetricsServer(
        args.metrics_port, lambda: render_prometheus(metrics_state)
    )
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    metrics_server.start()
    try:
        while time.monotonic() < deadline and not stopped["requested"]:
            attempt_started_at = time.time()
            attempt_started_monotonic = time.monotonic()
            progress = ProgressAccumulator()
            metrics_state["progress"] = progress
            last_progress_monotonic = None
            stderr = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=stderr,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            threshold_reason = None
            while process.poll() is None and time.monotonic() < deadline:
                if stopped["requested"]:
                    threshold_reason = "signal"
                    break
                now = time.monotonic()
                if progress.first_media_at is None:
                    if now - attempt_started_monotonic > args.max_startup_seconds:
                        threshold_reason = "startup-timeout"
                        break
                elif now - last_progress_monotonic > args.max_progress_stall_seconds:
                    threshold_reason = "progress-stall"
                    break
                for key, _mask in selector.select(timeout=0.25):
                    line = key.fileobj.readline()
                    if line:
                        if progress.feed_line(line):
                            last_progress_monotonic = time.monotonic()
            reached_deadline = time.monotonic() >= deadline
            return_code = _stop_process(process)
            for line in process.stdout:
                progress.feed_line(line)
            process.stdout.close()
            selector.close()
            stderr.seek(0)
            stderr_tail = redact_text(stderr.read()[-4000:])
            stderr.close()
            attempt_ended_at = time.time()
            attempts.append(
                {
                    "attempt": len(attempts) + 1,
                    "startedAt": attempt_started_at,
                    "endedAt": attempt_ended_at,
                    "returnCode": return_code,
                    "thresholdReason": threshold_reason,
                    "stderrTail": stderr_tail,
                    "progress": progress.summary(),
                }
            )
            if threshold_reason:
                stop_reason = threshold_reason
                break
            if reached_deadline:
                stop_reason = "duration-complete"
                break
            if process.returncode == 0:
                stop_reason = "ffmpeg-exited"
            else:
                stop_reason = "ffmpeg-failed"
            if reconnects >= args.max_reconnects:
                break
            reconnects += 1
            metrics_state["reconnects"] = reconnects
            time.sleep(
                min(
                    args.reconnect_delay_seconds,
                    max(0.0, deadline - time.monotonic()),
                )
            )
        if stopped["requested"] and stop_reason is None:
            stop_reason = "signal"
    finally:
        metrics_state["running"] = 0
        metrics_server.stop()
        signal.signal(signal.SIGTERM, old_sigterm)
        signal.signal(signal.SIGINT, old_sigint)

    ended_at = time.time()
    measured_seconds = max(0.0, ended_at - started_at)
    progress_summary = aggregate_progress(attempts, measured_seconds)
    media_times = [
        attempt["progress"]["firstMediaAt"]
        for attempt in attempts
        if attempt["progress"]["firstMediaAt"] is not None
    ]
    progress_summary["startupSeconds"] = (
        min(media_times) - started_at if media_times else None
    )
    threshold_failures = []
    if args.role == "read-decode" and args.expected_fps:
        ratio = progress_summary["deliveredFps"] / args.expected_fps
        progress_summary["deliveredFpsRatio"] = ratio
        if ratio < args.min_delivered_fps_ratio:
            threshold_failures.append("delivered-fps-below-threshold")
    if stop_reason not in ("duration-complete",):
        threshold_failures.append(stop_reason or "unknown-stop")
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "schemaVersion": 1,
        "dryRun": False,
        "runId": args.run_id,
        "role": args.role,
        "location": args.location,
        "stream": args.stream,
        "startedAt": started_at,
        "scheduledAt": scheduled_at,
        "startDelaySeconds": started_at - scheduled_at,
        "endedAt": ended_at,
        "durationSeconds": measured_seconds,
        "stopReason": stop_reason,
        "status": "passed" if not threshold_failures else "failed",
        "thresholdFailures": threshold_failures,
        "reconnectCount": reconnects,
        "command": redacted_command(command),
        "ffmpeg": version,
        "progress": progress_summary,
        "attempts": attempts,
        "placement": _placement(),
        "resources": {
            "childUserCpuSeconds": usage_after.ru_utime - usage_before.ru_utime,
            "childSystemCpuSeconds": usage_after.ru_stime - usage_before.ru_stime,
            "childMaxRss": usage_after.ru_maxrss,
            "childMaxRssUnit": "KiB",
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }


def write_report(report, path):
    payload = json.dumps(report, sort_keys=True)
    if path and path != "-":
        with open(path, "w") as output:
            json.dump(report, output, indent=2, sort_keys=True)
            output.write("\n")
    print("BENCHMARK_FINAL_JSON=" + payload, flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=ROLES)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--location", required=True)
    parser.add_argument("--stream", required=True)
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument("--start-delay-seconds", type=float, default=0)
    parser.add_argument("--input-url-env", default="BENCH_INPUT_URL")
    parser.add_argument("--output-url-env", default="BENCH_OUTPUT_URL")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--report-path", default="/dev/termination-log")
    parser.add_argument("--metrics-port", type=int, default=9091)
    parser.add_argument("--max-startup-seconds", type=float, default=30)
    parser.add_argument("--max-progress-stall-seconds", type=float, default=15)
    parser.add_argument("--max-reconnects", type=int, default=0)
    parser.add_argument("--reconnect-delay-seconds", type=float, default=1)
    parser.add_argument("--expected-fps", type=float)
    parser.add_argument("--min-delivered-fps-ratio", type=float, default=0.95)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.duration_seconds <= 0:
        parser.error("--duration-seconds must be positive")
    if args.start_delay_seconds < 0:
        parser.error("--start-delay-seconds cannot be negative")
    if args.max_startup_seconds <= 0 or args.max_progress_stall_seconds <= 0:
        parser.error("startup and progress-stall thresholds must be positive")
    if args.max_reconnects < 0:
        parser.error("--max-reconnects cannot be negative")
    if not 0 < args.min_delivered_fps_ratio <= 1:
        parser.error("--min-delivered-fps-ratio must be in (0, 1]")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        report = run_agent(args)
    except Exception as error:
        report = {
            "schemaVersion": 1,
            "runId": getattr(args, "run_id", None),
            "role": getattr(args, "role", None),
            "status": "failed",
            "thresholdFailures": ["agent-error"],
            "error": redact_text(error),
        }
    write_report(report, args.report_path)
    return 0 if report.get("status", "passed") == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
