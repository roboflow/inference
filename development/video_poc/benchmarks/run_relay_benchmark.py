#!/usr/bin/env python3
"""Run reproducible MediaMTX publisher/reader capacity scenarios.

Publishers replay an existing encoded fixture in real time with ``-c copy`` so
the load generator does not spend CPU encoding synthetic video. Readers consume
and discard decoded frames outside MediaMTX. Optional Prometheus endpoints are
sampled and aggregated without retaining their labels.

Run active load only in staging or a dedicated performance environment.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path

METRIC_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)
SAFE_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
URL_CREDENTIALS = re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*://)[^/@\s]+@")
URL_QUERY = re.compile(r"(?P<base>[a-zA-Z][a-zA-Z0-9+.-]*://[^?\s]+)\?[^\s]+")


def sanitize(text):
    text = URL_CREDENTIALS.sub(r"\g<scheme>[credentials-redacted]@", str(text))
    return URL_QUERY.sub(r"\g<base>?[query-redacted]", text)


def parse_prometheus(text, prefixes=()):
    """Sum samples by metric name and intentionally discard all labels."""
    totals = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = METRIC_LINE.match(line)
        if not match:
            continue
        name = match.group("name")
        if prefixes and not any(name.startswith(prefix) for prefix in prefixes):
            continue
        totals[name] = totals.get(name, 0.0) + float(match.group("value"))
    return totals


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ffmpeg_version(ffmpeg):
    result = subprocess.run(
        [ffmpeg, "-version"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()[0]


def publisher_command(ffmpeg, fixture, url):
    return [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-re",
        "-stream_loop",
        "-1",
        "-i",
        str(fixture),
        "-map",
        "0:v:0",
        "-c",
        "copy",
        "-f",
        "rtsp",
        "-rtsp_transport",
        "tcp",
        url,
    ]


def reader_command(ffmpeg, url):
    return [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-rtsp_transport",
        "tcp",
        "-i",
        url,
        "-map",
        "0:v:0",
        "-an",
        "-f",
        "null",
        "-",
    ]


def redacted_command(command):
    return [sanitize(part) if "://" in str(part) else str(part) for part in command]


def _expand(value):
    return os.path.expandvars(str(value))


def load_config(path):
    config_path = Path(path).resolve()
    with config_path.open() as source:
        config = json.load(source)

    fixture = Path(_expand(config.get("fixture", ""))).expanduser()
    if not fixture.is_absolute():
        fixture = (config_path.parent / fixture).resolve()
    if not fixture.is_file():
        raise ValueError(f"fixture does not exist: {fixture}")

    publish_template = _expand(
        config.get(
            "publishUrlTemplate",
            "rtsp://127.0.0.1:8554/{stream}",
        )
    )
    read_template = _expand(config.get("readUrlTemplate", publish_template))
    for key, template in (
        ("publishUrlTemplate", publish_template),
        ("readUrlTemplate", read_template),
    ):
        if "{stream}" not in template:
            raise ValueError(f"{key} must include {{stream}}")

    scenarios = config.get("scenarios") or []
    if not scenarios:
        raise ValueError("config must contain at least one scenario")
    names = set()
    normalized = []
    for raw in scenarios:
        name = str(raw.get("name") or "")
        if not SAFE_NAME.fullmatch(name) or name in names:
            raise ValueError(
                f"scenario name must be unique and filesystem-safe: {name!r}"
            )
        names.add(name)
        scenario = {
            "name": name,
            "sources": int(raw.get("sources", 1)),
            "readersPerSource": int(raw.get("readersPerSource", 1)),
            "publisherWarmupSeconds": float(raw.get("publisherWarmupSeconds", 3)),
            "warmupSeconds": float(raw.get("warmupSeconds", 10)),
            "durationSeconds": float(raw.get("durationSeconds", 60)),
            "sampleIntervalSeconds": float(raw.get("sampleIntervalSeconds", 5)),
        }
        if scenario["sources"] < 1 or scenario["readersPerSource"] < 0:
            raise ValueError(f"scenario {name} has invalid source/reader counts")
        if (
            min(
                scenario["publisherWarmupSeconds"],
                scenario["warmupSeconds"],
                scenario["durationSeconds"],
                scenario["sampleIntervalSeconds"],
            )
            < 0
        ):
            raise ValueError(f"scenario {name} has a negative duration")
        if scenario["sampleIntervalSeconds"] == 0:
            raise ValueError(f"scenario {name} sample interval must be positive")
        normalized.append(scenario)

    metrics = {}
    for name, raw in (config.get("metrics") or {}).items():
        if not SAFE_NAME.fullmatch(name):
            raise ValueError(f"metrics endpoint name is not safe: {name!r}")
        if isinstance(raw, str):
            raw = {"url": raw}
        metrics[name] = {
            "url": _expand(raw["url"]),
            "prefixes": tuple(raw.get("prefixes") or ()),
        }

    return {
        "configPath": str(config_path),
        "fixture": fixture,
        "publishUrlTemplate": publish_template,
        "readUrlTemplate": read_template,
        "ffmpeg": _expand(config.get("ffmpeg", "ffmpeg")),
        "metrics": metrics,
        "scenarios": normalized,
    }


class ManagedProcess:
    def __init__(self, role, command):
        self.role = role
        self.command = command
        self.stderr = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
        self.process = None

    def start(self):
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=self.stderr,
            start_new_session=True,
        )

    def poll(self):
        return self.process.poll() if self.process is not None else None

    def stop(self):
        if self.process is None or self.process.poll() is not None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
            self.process.wait(timeout=8)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process.wait(timeout=3)

    def summary(self):
        return_code = self.poll()
        self.stderr.seek(0)
        tail = self.stderr.read()[-4000:]
        return {
            "role": self.role,
            "returnCode": return_code,
            "stderrTail": sanitize(tail),
        }

    def close(self):
        self.stderr.close()


def _sample_metrics(endpoints):
    sample = {"timestamp": time.time(), "endpoints": {}}
    for name, endpoint in endpoints.items():
        try:
            request = urllib.request.Request(endpoint["url"], method="GET")
            with urllib.request.urlopen(request, timeout=5) as response:
                payload = response.read().decode("utf-8", errors="replace")
            sample["endpoints"][name] = {
                "metrics": parse_prometheus(payload, endpoint["prefixes"])
            }
        except Exception as error:
            sample["endpoints"][name] = {"error": sanitize(error)}
    return sample


def _sleep_while_healthy(processes, seconds, sample_interval, metrics, samples):
    deadline = time.monotonic() + seconds
    next_sample = time.monotonic()
    early_exits = []
    while time.monotonic() < deadline:
        for process in processes:
            return_code = process.poll()
            if return_code is not None:
                early_exits.append({"role": process.role, "returnCode": return_code})
        if early_exits:
            return early_exits
        now = time.monotonic()
        if metrics and now >= next_sample:
            samples.append(_sample_metrics(metrics))
            next_sample = now + sample_interval
        time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
    return early_exits


def run_scenario(config, scenario, run_id, dry_run=False):
    ffmpeg = shutil.which(config["ffmpeg"])
    if ffmpeg is None:
        raise ValueError(f"ffmpeg not found: {config['ffmpeg']}")

    streams = [
        f"bench-{run_id}-{index + 1:04d}" for index in range(scenario["sources"])
    ]
    publishers = []
    readers = []
    for stream in streams:
        publish_url = config["publishUrlTemplate"].format(stream=stream)
        publishers.append(
            ManagedProcess(
                f"publisher:{stream}",
                publisher_command(ffmpeg, config["fixture"], publish_url),
            )
        )
        read_url = config["readUrlTemplate"].format(stream=stream)
        for reader_index in range(scenario["readersPerSource"]):
            readers.append(
                ManagedProcess(
                    f"reader:{stream}:{reader_index + 1}",
                    reader_command(ffmpeg, read_url),
                )
            )

    result = {
        "name": scenario["name"],
        "parameters": scenario,
        "streams": len(streams),
        "readers": len(readers),
        "commands": [
            redacted_command(process.command) for process in publishers + readers
        ],
        "startedAt": time.time(),
        "metricSamples": [],
        "earlyExits": [],
    }
    if dry_run:
        result["dryRun"] = True
        return result

    processes = publishers + readers
    try:
        for process in publishers:
            process.start()
        result["earlyExits"].extend(
            _sleep_while_healthy(
                publishers,
                scenario["publisherWarmupSeconds"],
                scenario["sampleIntervalSeconds"],
                config["metrics"],
                result["metricSamples"],
            )
        )
        if not result["earlyExits"]:
            for process in readers:
                process.start()
            result["earlyExits"].extend(
                _sleep_while_healthy(
                    processes,
                    scenario["warmupSeconds"],
                    scenario["sampleIntervalSeconds"],
                    config["metrics"],
                    result["metricSamples"],
                )
            )
        if not result["earlyExits"]:
            result["measurementStartedAt"] = time.time()
            result["earlyExits"].extend(
                _sleep_while_healthy(
                    processes,
                    scenario["durationSeconds"],
                    scenario["sampleIntervalSeconds"],
                    config["metrics"],
                    result["metricSamples"],
                )
            )
            result["measurementEndedAt"] = time.time()
    finally:
        for process in reversed(processes):
            process.stop()
        result["processes"] = [process.summary() for process in processes]
        for process in processes:
            process.close()
        result["endedAt"] = time.time()

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="benchmark matrix JSON")
    parser.add_argument("--scenario", action="append", help="run only this scenario")
    parser.add_argument("--output-dir", default="benchmark-results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    requested = set(args.scenario or [])
    scenarios = [
        scenario
        for scenario in config["scenarios"]
        if not requested or scenario["name"] in requested
    ]
    missing = requested - {scenario["name"] for scenario in scenarios}
    if missing:
        parser.error(f"unknown scenarios: {', '.join(sorted(missing))}")

    ffmpeg = shutil.which(config["ffmpeg"])
    if ffmpeg is None:
        parser.error(f"ffmpeg not found: {config['ffmpeg']}")

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schemaVersion": 1,
        "runId": run_id,
        "dryRun": args.dry_run,
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "fixture": {
            "path": str(config["fixture"]),
            "bytes": config["fixture"].stat().st_size,
            "sha256": _sha256(config["fixture"]),
        },
        "ffmpeg": _ffmpeg_version(ffmpeg),
        "configPath": config["configPath"],
        "scenarios": [],
    }
    for scenario in scenarios:
        print(f"[benchmark] {scenario['name']}", flush=True)
        report["scenarios"].append(
            run_scenario(config, scenario, run_id, dry_run=args.dry_run)
        )

    output_path = output_dir / f"relay-{run_id}.json"
    with output_path.open("w") as output:
        json.dump(report, output, indent=2, sort_keys=True)
        output.write("\n")
    print(output_path)
    return 1 if any(item["earlyExits"] for item in report["scenarios"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
