#!/usr/bin/env python3
"""Run tenant-aware concurrent HTTP clients against a staging MMP server."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import ipaddress
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlencode, urlparse


@dataclass(frozen=True)
class ClientSpec:
    tenant_id: str
    api_key_env: str
    model_id: str
    client_id: str = ""
    concurrency: int = 1
    target_fps: float = 0.0
    instance: str = ""
    device: str = ""
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunSpec:
    server_url: str
    duration_s: float
    warmup_s: float
    sample_interval_s: float
    clients: tuple[ClientSpec, ...]
    request_timeout_s: float = 60.0
    max_latency_samples_per_client: int = 250_000


@dataclass
class TimeBucket:
    requests: int = 0
    successes: int = 0
    errors: int = 0
    latency_sum_ms: float = 0.0
    latency_max_ms: float = 0.0

    def record(self, latency_ms: float, success: bool) -> None:
        self.requests += 1
        self.successes += int(success)
        self.errors += int(not success)
        self.latency_sum_ms += latency_ms
        self.latency_max_ms = max(self.latency_max_ms, latency_ms)

    def report(self, second: int) -> dict[str, Any]:
        return {
            "second": second,
            "requests": self.requests,
            "successes": self.successes,
            "errors": self.errors,
            "avg_latency_ms": (
                self.latency_sum_ms / self.requests if self.requests else None
            ),
            "max_latency_ms": self.latency_max_ms if self.requests else None,
        }


class ClientRecorder:
    def __init__(self, spec: ClientSpec, max_samples: int, seed: int) -> None:
        self.spec = spec
        self.max_samples = max_samples
        self.random = random.Random(seed)
        self.requests = 0
        self.successes = 0
        self.errors = 0
        self.status_counts: Counter[str] = Counter()
        self.error_counts: Counter[str] = Counter()
        self.latencies_ms: list[float] = []
        self.sampled_requests = 0
        self.buckets: defaultdict[int, TimeBucket] = defaultdict(TimeBucket)
        self.first_request_offset_s: float | None = None
        self.first_success_offset_s: float | None = None
        self.first_success_latency_ms: float | None = None

    def record(
        self,
        *,
        offset_s: float,
        measured_offset_s: float,
        latency_ms: float,
        status: int | None,
        error: str | None,
        measured: bool,
    ) -> None:
        if self.first_request_offset_s is None:
            self.first_request_offset_s = offset_s
        success = status is not None and 200 <= status < 300 and error is None
        if success and self.first_success_offset_s is None:
            self.first_success_offset_s = offset_s + latency_ms / 1000.0
            self.first_success_latency_ms = latency_ms
        if not measured:
            return

        self.requests += 1
        self.successes += int(success)
        self.errors += int(not success)
        self.status_counts[str(status) if status is not None else "transport"] += 1
        if error:
            self.error_counts[error] += 1
        self.buckets[max(0, int(measured_offset_s))].record(latency_ms, success)
        self.sampled_requests += 1
        if len(self.latencies_ms) < self.max_samples:
            self.latencies_ms.append(latency_ms)
        else:
            candidate = self.random.randrange(self.sampled_requests)
            if candidate < self.max_samples:
                self.latencies_ms[candidate] = latency_ms

    def report(self, measured_duration_s: float) -> dict[str, Any]:
        latencies = sorted(self.latencies_ms)
        return {
            "client_id": client_name(self.spec),
            "tenant_id": self.spec.tenant_id,
            "model_id": self.spec.model_id,
            "routing_key": routing_key(self.spec),
            "instance": self.spec.instance,
            "device": self.spec.device,
            "concurrency": self.spec.concurrency,
            "target_fps": self.spec.target_fps,
            "api_key_env": self.spec.api_key_env,
            "requests": self.requests,
            "successes": self.successes,
            "errors": self.errors,
            "success_rate": self.successes / self.requests if self.requests else 0.0,
            "delivered_fps": self.successes / max(measured_duration_s, 1e-9),
            "status_counts": dict(sorted(self.status_counts.items())),
            "error_counts": dict(self.error_counts.most_common()),
            "latency_sample_count": len(latencies),
            "latency_population_count": self.sampled_requests,
            "latency_ms": latency_summary(latencies),
            "first_request_offset_s": self.first_request_offset_s,
            "first_success_offset_s": self.first_success_offset_s,
            "first_success_latency_ms": self.first_success_latency_ms,
            "time_buckets": [
                self.buckets[second].report(second) for second in sorted(self.buckets)
            ],
        }


def percentile(sorted_values: Sequence[float], quantile: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def latency_summary(sorted_values: Sequence[float]) -> dict[str, float | None]:
    if not sorted_values:
        return {"p50": None, "p95": None, "p99": None, "max": None, "mean": None}
    return {
        "p50": percentile(sorted_values, 0.50),
        "p95": percentile(sorted_values, 0.95),
        "p99": percentile(sorted_values, 0.99),
        "max": sorted_values[-1],
        "mean": statistics.fmean(sorted_values),
    }


def jain_fairness(values: Iterable[float]) -> float | None:
    usable = [max(0.0, value) for value in values]
    if not usable or not any(usable):
        return None
    return sum(usable) ** 2 / (len(usable) * sum(v * v for v in usable))


def client_name(client: ClientSpec) -> str:
    return client.client_id or client.tenant_id


def routing_key(client: ClientSpec) -> str:
    return (
        f"{client.model_id}:{client.instance}" if client.instance else client.model_id
    )


def _is_safe_staging_host(host: str | None) -> bool:
    if not host:
        return False
    normalized = host.lower().rstrip(".")
    if normalized in {"localhost", "host.docker.internal"}:
        return True
    try:
        address = ipaddress.ip_address(normalized)
        return address.is_private or address.is_loopback or address.is_link_local
    except ValueError:
        pass
    return (
        normalized.endswith(".roboflow.one")
        or normalized.endswith(".svc")
        or normalized.endswith(".svc.cluster.local")
        or "staging" in normalized
    )


def validate_staging_url(server_url: str) -> str:
    parsed = urlparse(server_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("server_url must use http or https")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("server_url must not contain credentials, query, or fragment")
    if not _is_safe_staging_host(parsed.hostname):
        raise ValueError(
            f"refusing non-staging benchmark target {parsed.hostname!r}; "
            "use localhost/private IP, Kubernetes service DNS, or a staging host"
        )
    return server_url.rstrip("/")


def load_spec(path: Path) -> RunSpec:
    raw = json.loads(path.read_text())
    clients = tuple(ClientSpec(**client) for client in raw.get("clients", []))
    spec = RunSpec(
        server_url=validate_staging_url(raw["server_url"]),
        duration_s=float(raw.get("duration_s", 120)),
        warmup_s=float(raw.get("warmup_s", 15)),
        sample_interval_s=float(raw.get("sample_interval_s", 1)),
        clients=clients,
        request_timeout_s=float(raw.get("request_timeout_s", 60)),
        max_latency_samples_per_client=int(
            raw.get("max_latency_samples_per_client", 250_000)
        ),
    )
    validate_spec(spec)
    return spec


def validate_spec(spec: RunSpec) -> None:
    if not spec.clients:
        raise ValueError("at least one client is required")
    if spec.duration_s <= 0 or spec.warmup_s < 0 or spec.sample_interval_s <= 0:
        raise ValueError(
            "duration/sample interval must be positive and warmup nonnegative"
        )
    if spec.request_timeout_s <= 0 or spec.max_latency_samples_per_client <= 0:
        raise ValueError("request timeout and sample cap must be positive")
    seen: set[str] = set()
    for client in spec.clients:
        name = client_name(client)
        if not client.tenant_id or not name or name in seen:
            raise ValueError(
                "tenant_id must be non-empty and client_id (or tenant_id) must be unique"
            )
        seen.add(name)
        if not client.api_key_env or not client.model_id:
            raise ValueError(f"{client.tenant_id}: api_key_env and model_id required")
        if client.concurrency <= 0 or client.target_fps < 0:
            raise ValueError(f"{client.tenant_id}: invalid concurrency/target_fps")


def _client_url(server_url: str, client: ClientSpec) -> str:
    query: dict[str, Any] = {
        "model_id": client.model_id,
        "format": "json",
    }
    if client.instance:
        query["instance"] = client.instance
    if client.device:
        query["device"] = client.device
    query.update(client.params)
    return f"{server_url}/infer?{urlencode(query, doseq=True)}"


def _safe_error(status: int | None, body: str | None, exc: Exception | None) -> str:
    if exc is not None:
        return f"{type(exc).__name__}: {str(exc)[:160]}"
    compact = " ".join((body or "").split())[:160]
    return f"HTTP {status}: {compact}" if compact else f"HTTP {status}"


async def _run_worker(
    *,
    session: Any,
    server_url: str,
    client: ClientSpec,
    key: str,
    image: bytes,
    recorder: ClientRecorder,
    started: float,
    measure_started: float,
    stop_at: float,
    worker_index: int,
) -> None:
    url = _client_url(server_url, client)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "image/jpeg"}
    interval_s = (
        client.concurrency / client.target_fps if client.target_fps > 0 else 0.0
    )
    next_start = started
    if interval_s:
        next_start += worker_index * interval_s / client.concurrency

    while True:
        now = time.monotonic()
        if now >= stop_at:
            return
        if interval_s and now < next_start:
            await asyncio.sleep(min(next_start - now, max(0.0, stop_at - now)))
            now = time.monotonic()
            if now >= stop_at:
                return
        request_started = time.monotonic()
        status: int | None = None
        error: str | None = None
        body: str | None = None
        caught: Exception | None = None
        try:
            async with session.post(url, data=image, headers=headers) as response:
                status = response.status
                payload = await response.read()
                if not 200 <= status < 300:
                    body = payload.decode("utf-8", errors="replace")
        except Exception as exc:  # preserve all transport errors in evidence
            caught = exc
        finished = time.monotonic()
        if caught is not None or status is None or not 200 <= status < 300:
            error = _safe_error(status, body, caught)
        recorder.record(
            offset_s=request_started - started,
            measured_offset_s=request_started - measure_started,
            latency_ms=(finished - request_started) * 1000,
            status=status,
            error=error,
            measured=request_started >= measure_started,
        )
        if interval_s:
            next_start += interval_s
            if next_start < finished - interval_s:
                next_start = finished


async def _metrics_sampler(
    *,
    session: Any,
    server_url: str,
    key: str,
    started: float,
    stop_at: float,
    interval_s: float,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    url = f"{server_url}/v2/server/metrics"
    headers = {"Authorization": f"Bearer {key}"}
    while time.monotonic() < stop_at:
        sample_started = time.monotonic()
        try:
            async with session.get(url, headers=headers) as response:
                payload: Any
                if response.status == 200:
                    payload = await response.json()
                else:
                    payload = {"error": f"HTTP {response.status}"}
                samples.append(
                    {
                        "offset_s": sample_started - started,
                        "status": response.status,
                        "metrics": payload,
                    }
                )
        except Exception as exc:
            samples.append(
                {
                    "offset_s": sample_started - started,
                    "status": None,
                    "metrics": {"error": f"{type(exc).__name__}: {str(exc)[:160]}"},
                }
            )
        await asyncio.sleep(max(0.0, interval_s - (time.monotonic() - sample_started)))
    return samples


def _model_load_evidence(client_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    models: dict[str, dict[str, Any]] = {}
    for report in client_reports:
        model_id = str(report["routing_key"])
        entry = models.setdefault(
            model_id,
            {"first_success_offset_s": None, "first_success_latency_ms": None},
        )
        offset = report.get("first_success_offset_s")
        if offset is not None and (
            entry["first_success_offset_s"] is None
            or offset < entry["first_success_offset_s"]
        ):
            entry["first_success_offset_s"] = offset
            entry["first_success_latency_ms"] = report.get("first_success_latency_ms")
    return models


def _metrics_evidence(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [sample for sample in samples if sample.get("status") == 200]
    if not valid:
        return {"sample_count": 0}
    gpu_utils: list[float] = []
    gpu_memory: list[float] = []
    gpu_power: list[float] = []
    free_slots: list[int] = []
    pending: list[int] = []
    rejects: list[int] = []
    per_model_peak_vram: dict[str, float] = {}
    for sample in valid:
        metrics = sample["metrics"]
        for gpu in metrics.get("gpus", []):
            if gpu.get("utilization_pct") is not None:
                gpu_utils.append(float(gpu["utilization_pct"]))
            if gpu.get("memory_used_mb") is not None:
                gpu_memory.append(float(gpu["memory_used_mb"]))
            if gpu.get("power_w") is not None:
                gpu_power.append(float(gpu["power_w"]))
        for model_id, value in metrics.get("per_model_gpu_mb", {}).items():
            per_model_peak_vram[model_id] = max(
                per_model_peak_vram.get(model_id, 0.0), float(value)
            )
        if metrics.get("mmp_free_slots") is not None:
            free_slots.append(int(metrics["mmp_free_slots"]))
        if metrics.get("mmp_pending") is not None:
            pending.append(int(metrics["mmp_pending"]))
        if metrics.get("mmp_rejects_pool_full") is not None:
            rejects.append(int(metrics["mmp_rejects_pool_full"]))

    first_metrics = valid[0]["metrics"]
    last_metrics = valid[-1]["metrics"]
    model_deltas: dict[str, dict[str, Any]] = {}
    first_models = first_metrics.get("mmp_models", {})
    last_models = last_metrics.get("mmp_models", {})
    for model_id in sorted(set(first_models) | set(last_models)):
        before = first_models.get(model_id, {})
        after = last_models.get(model_id, {})
        model_deltas[model_id] = {
            "worker_pid": after.get("worker_pid"),
            "backend_type": after.get("backend_type"),
            "device": after.get("device"),
            "max_batch_size": after.get("max_batch_size"),
            "inference_count_delta": int(after.get("inference_count", 0))
            - int(before.get("inference_count", 0)),
            "batch_count_delta": int(after.get("batch_count", 0))
            - int(before.get("batch_count", 0)),
            "avg_batch_size_final": after.get("avg_batch_size"),
            "throughput_fps_final": after.get("throughput_fps"),
            "latency_p50_ms_final": after.get("latency_p50_ms"),
            "latency_p95_ms_final": after.get("latency_p95_ms"),
            "latency_p99_ms_final": after.get("latency_p99_ms"),
            "avg_decode_ms_final": after.get("avg_decode_ms"),
            "avg_infer_ms_final": after.get("avg_infer_ms"),
            "avg_write_ms_final": after.get("avg_write_ms"),
            "error_count_delta": int(after.get("error_count", 0))
            - int(before.get("error_count", 0)),
        }

    return {
        "sample_count": len(valid),
        "first_offset_s": valid[0]["offset_s"],
        "last_offset_s": valid[-1]["offset_s"],
        "gpu_utilization_pct": _numeric_summary(gpu_utils),
        "gpu_memory_used_mb": _numeric_summary(gpu_memory),
        "gpu_power_w": _numeric_summary(gpu_power),
        "mmp_free_slots": _numeric_summary(free_slots),
        "mmp_pending": _numeric_summary(pending),
        "mmp_pool_full_rejects_delta": (
            rejects[-1] - rejects[0] if len(rejects) >= 2 else 0
        ),
        "per_model_peak_vram_mb": per_model_peak_vram,
        "model_deltas": model_deltas,
        "initial": first_metrics,
        "final": last_metrics,
    }


def _numeric_summary(values: Sequence[float | int]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def _client_seed(client: ClientSpec) -> int:
    digest = hashlib.sha256(
        f"{client_name(client)}\0{client.tenant_id}\0{client.model_id}\0{client.instance}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big")


async def execute(spec: RunSpec, image: bytes) -> dict[str, Any]:
    try:
        import aiohttp
    except ImportError as exc:
        raise RuntimeError("aiohttp is required to run the benchmark") from exc

    keys: dict[str, str] = {}
    for client in spec.clients:
        key = os.environ.get(client.api_key_env, "")
        if not key:
            raise ValueError(
                f"{client.tenant_id}: environment variable {client.api_key_env} is empty"
            )
        keys[client_name(client)] = key

    timeout = aiohttp.ClientTimeout(total=spec.request_timeout_s)
    connector = aiohttp.TCPConnector(limit=0, enable_cleanup_closed=True)
    started = time.monotonic()
    measure_started = started + spec.warmup_s
    stop_at = measure_started + spec.duration_s
    recorders = {
        client_name(client): ClientRecorder(
            client, spec.max_latency_samples_per_client, _client_seed(client)
        )
        for client in spec.clients
    }

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        first_key = keys[client_name(spec.clients[0])]
        sampler = asyncio.create_task(
            _metrics_sampler(
                session=session,
                server_url=spec.server_url,
                key=first_key,
                started=started,
                stop_at=stop_at,
                interval_s=spec.sample_interval_s,
            )
        )
        workers = [
            asyncio.create_task(
                _run_worker(
                    session=session,
                    server_url=spec.server_url,
                    client=client,
                    key=keys[client_name(client)],
                    image=image,
                    recorder=recorders[client_name(client)],
                    started=started,
                    measure_started=measure_started,
                    stop_at=stop_at,
                    worker_index=worker_index,
                )
            )
            for client in spec.clients
            for worker_index in range(client.concurrency)
        ]
        await asyncio.gather(*workers)
        metrics_samples = await sampler
    finished = time.monotonic()

    client_reports = [
        recorders[client_name(client)].report(spec.duration_s)
        for client in spec.clients
    ]
    delivered = [float(report["delivered_fps"]) for report in client_reports]
    normalized = [
        (
            float(report["delivered_fps"]) / float(report["target_fps"])
            if float(report["target_fps"]) > 0
            else float(report["delivered_fps"])
        )
        for report in client_reports
    ]
    return {
        "schema_version": 1,
        "run": {
            "server_url": spec.server_url,
            "duration_s": spec.duration_s,
            "warmup_s": spec.warmup_s,
            "actual_elapsed_s": finished - started,
            "started_unix_s": time.time() - (finished - started),
            "finished_unix_s": time.time(),
            "image_bytes": len(image),
            "image_sha256": hashlib.sha256(image).hexdigest(),
            "source_revision": os.environ.get(
                "MMP_BENCHMARK_SOURCE_REVISION", "unknown"
            ),
            "image_ref": os.environ.get("MMP_BENCHMARK_IMAGE_REF", "unknown"),
            "node_name": os.environ.get("NODE_NAME", "unknown"),
            "pod_name": os.environ.get("POD_NAME", "unknown"),
            "python": platform.python_version(),
            "clients": [asdict(client) for client in spec.clients],
        },
        "aggregate": {
            "requests": sum(int(r["requests"]) for r in client_reports),
            "successes": sum(int(r["successes"]) for r in client_reports),
            "errors": sum(int(r["errors"]) for r in client_reports),
            "delivered_fps": sum(delivered),
            "jain_fairness_delivered_fps": jain_fairness(delivered),
            "jain_fairness_normalized_to_target": jain_fairness(normalized),
        },
        "clients": client_reports,
        "model_load_evidence": _model_load_evidence(client_reports),
        "metrics_evidence": _metrics_evidence(metrics_samples),
        "metrics_samples": metrics_samples,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate spec/image without sending requests or requiring API keys",
    )
    parser.add_argument(
        "--fail-on-errors",
        action="store_true",
        help="return exit 2 when the completed run records any request errors",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        spec = load_spec(args.spec)
        image = args.image.read_bytes()
        if not image:
            raise ValueError("image file is empty")
        if args.validate_only:
            print(
                json.dumps(
                    {
                        "valid": True,
                        "clients": len(spec.clients),
                        "total_concurrency": sum(c.concurrency for c in spec.clients),
                        "image_bytes": len(image),
                    },
                    indent=2,
                )
            )
            return 0
        report = asyncio.run(execute(spec, image))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report["aggregate"], indent=2, sort_keys=True))
        if args.fail_on_errors and report["aggregate"]["errors"]:
            return 2
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
