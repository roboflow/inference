#!/usr/bin/env python3
"""Attach staging Prometheus resource evidence to an API benchmark report.

The collector is deliberately tied to ``ck8s-stg`` and queries Prometheus from
inside its existing pod. It performs no Kubernetes writes and never reads an
API key or source URL. Processor pods are taken from the report's sanitized
runtime identity, which keeps DCGM and cAdvisor samples joined to the exact
single-L40S workers that ran the jobs.
"""

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import urllib.parse
from datetime import datetime
from pathlib import Path


STAGING_CONTEXT = "ck8s-stg"
PROMETHEUS_NAMESPACE = "monitoring"
PROMETHEUS_SELECTOR = "prometheus=kube-prometheus-stack-prometheus"
PROMETHEUS_CONTAINER = "prometheus"
PROMETHEUS_URL = "http://127.0.0.1:9090"


def parse_timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def report_processor_pods(report):
    pods = set()
    for job in report.get("jobs") or []:
        hostname = ((job.get("stats") or {}).get("runtime") or {}).get("hostname")
        if hostname:
            pods.add(str(hostname))
    if not pods:
        raise ValueError("report has no sanitized runtime hostname")
    return sorted(pods)


def metric_queries(pods):
    pod_pattern = "|".join(urllib.parse.quote(pod, safe="-") for pod in pods)
    processor = (
        'namespace="video-proc",container="processor",'
        f'pod=~"{pod_pattern}"'
    )
    dcgm = (
        'exported_namespace="video-proc",exported_container="processor",'
        f'exported_pod=~"{pod_pattern}"'
    )
    relay = 'namespace="video-proc",pod=~"mediamtx-.*"'
    return {
        "processorCpuCores": (
            f"sum(rate(container_cpu_usage_seconds_total{{{processor}}}[1m]))"
        ),
        "processorMemoryWorkingSetBytes": (
            f"sum(container_memory_working_set_bytes{{{processor}}})"
        ),
        "gpuUtilPercent": f"max(DCGM_FI_DEV_GPU_UTIL{{{dcgm}}})",
        "gpuFramebufferUsedMiB": f"sum(DCGM_FI_DEV_FB_USED{{{dcgm}}})",
        "gpuDecoderUtilPercent": f"max(DCGM_FI_DEV_DEC_UTIL{{{dcgm}}})",
        "gpuEncoderUtilPercent": f"max(DCGM_FI_DEV_ENC_UTIL{{{dcgm}}})",
        "gpuMemoryCopyUtilPercent": f"max(DCGM_FI_DEV_MEM_COPY_UTIL{{{dcgm}}})",
        "relayCpuCores": (
            "sum(rate(container_cpu_usage_seconds_total{"
            + relay
            + ',container="mediamtx"}[1m]))'
        ),
        "relayMemoryWorkingSetBytes": (
            f"sum(container_memory_working_set_bytes{{{relay},container=\"mediamtx\"}})"
        ),
        "relayReaders": f"sum(paths_readers{{{relay}}})",
        "relayIngressBytesPerSecond": (
            f"sum(rate(paths_bytes_received{{{relay}}}[1m]))"
        ),
        "relayEgressBytesPerSecond": f"sum(rate(paths_bytes_sent{{{relay}}}[1m]))",
        "relayRtspPacketsLostPerSecond": (
            f"sum(rate(rtsp_sessions_rtp_packets_lost{{{relay}}}[1m]))"
        ),
        "relayRtspPacketsInErrorPerSecond": (
            f"sum(rate(rtsp_sessions_rtp_packets_in_error{{{relay}}}[1m]))"
        ),
    }


def discover_prometheus_pod(context):
    command = [
        "kubectl",
        "--context",
        context,
        "-n",
        PROMETHEUS_NAMESPACE,
        "get",
        "pods",
        "-l",
        PROMETHEUS_SELECTOR,
        "-o",
        "json",
    ]
    payload = json.loads(subprocess.run(command, check=True, capture_output=True).stdout)
    ready = []
    for item in payload.get("items") or []:
        conditions = item.get("status", {}).get("conditions") or []
        if any(
            condition.get("type") == "Ready" and condition.get("status") == "True"
            for condition in conditions
        ):
            ready.append(item["metadata"]["name"])
    if len(ready) != 1:
        raise RuntimeError(f"expected one ready staging Prometheus pod, found {ready}")
    return ready[0]


def query_range(context, prometheus_pod, query, start, end, step):
    parameters = urllib.parse.urlencode(
        {"query": query, "start": start, "end": end, "step": step}
    )
    url = f"{PROMETHEUS_URL}/api/v1/query_range?{parameters}"
    command = [
        "kubectl",
        "--context",
        context,
        "-n",
        PROMETHEUS_NAMESPACE,
        "exec",
        prometheus_pod,
        "-c",
        PROMETHEUS_CONTAINER,
        "--",
        "wget",
        "-qO-",
        url,
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    payload = json.loads(result.stdout)
    if payload.get("status") != "success":
        raise RuntimeError(f"Prometheus query failed: {payload}")
    return payload["data"]["result"]


def sample_values(series):
    samples = []
    for item in series:
        for timestamp, raw_value in item.get("values") or []:
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                samples.append([float(timestamp), value])
    samples.sort(key=lambda item: item[0])
    return samples


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(samples):
    values = [item[1] for item in samples]
    if not values:
        return {"count": 0, "min": None, "mean": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "min": round(min(values), 6),
        "mean": round(statistics.fmean(values), 6),
        "p95": round(percentile(values, 0.95), 6),
        "max": round(max(values), 6),
    }


def collect(report_path, context, step_seconds):
    if context != STAGING_CONTEXT:
        raise ValueError(f"context must be exactly {STAGING_CONTEXT}")
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    start = parse_timestamp(report["measurementStartedAt"])
    end = parse_timestamp(report["measurementEndedAt"])
    pods = report_processor_pods(report)
    prometheus_pod = discover_prometheus_pod(context)
    metrics = {}
    for name, query in metric_queries(pods).items():
        series = query_range(context, prometheus_pod, query, start, end, step_seconds)
        samples = sample_values(series)
        metrics[name] = {
            "query": query,
            "summary": summarize(samples),
            "series": series,
        }
    return {
        "schemaVersion": 1,
        "environment": "staging",
        "clusterContext": context,
        "sourceReport": str(report_path),
        "sourceReportSha256": hashlib.sha256(report_bytes).hexdigest(),
        "runId": report.get("runId"),
        "measurementStartedAt": report["measurementStartedAt"],
        "measurementEndedAt": report["measurementEndedAt"],
        "sampleStepSeconds": step_seconds,
        "processorPods": pods,
        "prometheusPod": prometheus_pod,
        "metrics": metrics,
    }


def write_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--context", default=STAGING_CONTEXT)
    parser.add_argument("--step-seconds", type=int, default=15)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.step_seconds < 5 or args.step_seconds > 60:
        parser.error("--step-seconds must be between 5 and 60")
    return args


def main():
    args = parse_args()
    output = args.output or args.report.with_name(args.report.stem + "-resources.json")
    write_atomic(output, collect(args.report.resolve(), args.context, args.step_seconds))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
