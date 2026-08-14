#!/usr/bin/env python3
"""Collect CPU-specific resource evidence for one staging API report.

This extends the common read-only capacity collector with CFS throttling,
thread-count, and restart evidence. It performs no API or Kubernetes writes.
"""

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_DIR))

import collect_staging_capacity_telemetry as common  # noqa: E402
from verify_rollout_patch import validate_patch_document  # noqa: E402


def cpu_queries(pods):
    # Rebuild the selector explicitly rather than accepting arbitrary report
    # text in PromQL. report_processor_pods already returns runtime hostnames;
    # quote() constrains every alternation to hostname-safe characters.
    pod_pattern = "|".join(
        common.urllib.parse.quote(pod, safe="-") for pod in sorted(pods)
    )
    selector = 'namespace="video-proc",container="processor",' f'pod=~"{pod_pattern}"'
    return {
        "processorCpuThrottledPeriodsRatio": (
            "sum(rate(container_cpu_cfs_throttled_periods_total{"
            + selector
            + "}[1m])) / clamp_min(sum(rate(container_cpu_cfs_periods_total{"
            + selector
            + "}[1m])), 0.000001)"
        ),
        "processorThreads": f"sum(container_threads{{{selector}}})",
        "processorRestarts": (
            "sum by (pod) (kube_pod_container_status_restarts_total{" + selector + "})"
        ),
        "processorCpuRequests": (
            "sum by (pod,node) (kube_pod_container_resource_requests{"
            + selector
            + ',resource="cpu",unit="core"})'
        ),
        "processorMemoryRequestsBytes": (
            "sum by (pod,node) (kube_pod_container_resource_requests{"
            + selector
            + ',resource="memory",unit="byte"})'
        ),
        "processorCpuLimits": (
            "sum by (pod,node) (kube_pod_container_resource_limits{"
            + selector
            + ',resource="cpu",unit="core"})'
        ),
        "processorMemoryLimitsBytes": (
            "sum by (pod,node) (kube_pod_container_resource_limits{"
            + selector
            + ',resource="memory",unit="byte"})'
        ),
        "processorGuaranteedQos": (
            'max by (pod,qos_class) (kube_pod_status_qos_class{namespace="video-proc",'
            f'pod=~"{pod_pattern}",qos_class="Guaranteed"}})'
        ),
        "processorOomKilled": (
            'sum(kube_pod_container_status_last_terminated_reason{namespace="video-proc",'
            f'container="processor",pod=~"{pod_pattern}",reason="OOMKilled"}})'
        ),
        "processorEvicted": (
            'sum(kube_pod_status_reason{namespace="video-proc",'
            f'pod=~"{pod_pattern}",reason="Evicted"}})'
        ),
        "processorActiveJobs": (
            f'sum(video_processor_active_jobs{{namespace="video-proc",pod=~"{pod_pattern}"}})'
        ),
        "processorImageInfo": (
            'max by (pod,image_id,node) (kube_pod_container_info{namespace="video-proc",'
            f'container="processor",pod=~"{pod_pattern}"}})'
        ),
    }


def _metric_values(metric):
    return [
        value for _timestamp, value in common.sample_values(metric.get("series") or [])
    ]


def _series_delta(metric):
    total = 0.0
    for series in metric.get("series") or []:
        values = common.sample_values([series])
        if values:
            total += values[-1][1] - values[0][1]
    return round(total, 6)


def _series_initial(metric):
    total = 0.0
    found = False
    for series in metric.get("series") or []:
        values = common.sample_values([series])
        if values:
            found = True
            total += values[0][1]
    return round(total, 6) if found else None


def _series_pods(metric):
    return {
        item.get("metric", {}).get("pod")
        for item in metric.get("series") or []
        if item.get("metric", {}).get("pod")
    }


def _temporal_coverage(metric, start, end, step_seconds, expected_pods=None):
    series = metric.get("series") or []
    samples = common.sample_values(series)
    duration = max(0.0, end - start)
    expected_count = max(1, int(duration / step_seconds))
    per_series = []
    for item in series:
        item_samples = common.sample_values([item])
        span = (
            item_samples[-1][0] - item_samples[0][0] if len(item_samples) >= 2 else 0.0
        )
        per_series.append(
            {
                "metric": item.get("metric") or {},
                "sampleCount": len(item_samples),
                "expectedSampleCount": expected_count,
                "sampleRatio": len(item_samples) / expected_count,
                "spanSeconds": span,
                "expectedSpanSeconds": duration,
                "covered": len(item_samples) >= 0.9 * expected_count
                and span >= 0.9 * duration,
            }
        )
    expected_series_count = (
        max(1, len(expected_pods)) if expected_pods is not None else 1
    )
    pods_match = expected_pods is None or _series_pods(metric) == set(expected_pods)
    cardinality_matches = len(series) == expected_series_count
    return {
        "sampleCount": len(samples),
        "expectedSampleCount": expected_count * expected_series_count,
        "sampleRatio": len(samples) / (expected_count * expected_series_count),
        "perSeries": per_series,
        "podsMatch": pods_match,
        "seriesCount": len(series),
        "expectedSeriesCount": expected_series_count,
        "cardinalityMatches": cardinality_matches,
        "covered": bool(per_series)
        and all(item["covered"] for item in per_series)
        and pods_match
        and cardinality_matches,
    }


def node_cpu_query(nodes):
    pattern = "|".join(
        common.urllib.parse.quote(node, safe=".-") for node in sorted(nodes)
    )
    return (
        'sum by (node) (rate(container_cpu_usage_seconds_total{node=~"'
        + pattern
        + '",container!=""}[1m])) / max by (node) '
        '(kube_node_status_allocatable{node=~"'
        + pattern
        + '",resource="cpu",unit="core"})'
    )


def _runtime_identities(report):
    identities = []
    for job in report.get("jobs") or []:
        runtime = (job.get("stats") or {}).get("runtime") or {}
        if runtime:
            identities.append(runtime)
    return identities


def _expected_variant(report, catalog):
    identities = _runtime_identities(report)
    variants = {item.get("variant") for item in identities if item.get("variant")}
    if len(variants) != 1:
        return None, None, None
    variant = variants.pop()
    match = re.fullmatch(r"(cpu-[248])-(thread|process)", variant)
    if match is None:
        return variant, None, None
    return variant, match.group(1), match.group(2)


def expected_runtime_from_patch(patch, catalog):
    validate_patch_document(catalog, patch)
    containers = patch["spec"]["template"]["spec"]["containers"]
    if len(containers) != 1 or containers[0].get("name") != "processor":
        raise ValueError("expected patch must contain one processor container")
    container = containers[0]
    environment = {item["name"]: item.get("value") for item in container["env"]}
    variant = environment.get("VIDEO_PROC_RUNTIME_VARIANT")
    match = re.fullmatch(r"(cpu-[248])-(thread|process)", str(variant or ""))
    if match is None:
        raise ValueError("expected patch has an invalid CPU runtime variant")
    if environment.get("VIDEO_PROC_IMAGE") != container.get("image"):
        raise ValueError("expected patch image identity is inconsistent")
    return {
        "variant": variant,
        "size": match.group(1),
        "topology": match.group(2),
        "image": container["image"],
        "revision": environment.get("VIDEO_PROC_GIT_SHA"),
    }


def certification_evidence(metrics, report, catalog, expected, step_seconds=15):
    required = {
        "processorCpuCores",
        "processorMemoryWorkingSetBytes",
        "processorCpuThrottledPeriodsRatio",
        "processorThreads",
        "processorRestarts",
        "processorCpuRequests",
        "processorMemoryRequestsBytes",
        "processorCpuLimits",
        "processorMemoryLimitsBytes",
        "processorGuaranteedQos",
        "processorActiveJobs",
        "processorImageInfo",
        "processorNodeCpuUtilizationRatio",
    }
    start = common.parse_timestamp(report["measurementStartedAt"])
    measurement_end = common.parse_timestamp(report["measurementEndedAt"])
    lifecycle_end = common.parse_timestamp(
        report.get("endedAt") or report["measurementEndedAt"]
    )
    expected_pods = {
        item.get("hostname")
        for item in _runtime_identities(report)
        if item.get("hostname")
    }
    pod_metrics = {
        "processorRestarts",
        "processorCpuRequests",
        "processorMemoryRequestsBytes",
        "processorCpuLimits",
        "processorMemoryLimitsBytes",
        "processorGuaranteedQos",
        "processorImageInfo",
    }
    lifecycle_metrics = {
        "processorRestarts",
        "processorActiveJobs",
        "processorImageInfo",
    }
    coverage = {}
    for name in sorted(required):
        metric_end = lifecycle_end if name in lifecycle_metrics else measurement_end
        coverage[name] = _temporal_coverage(
            metrics.get(name) or {},
            start,
            metric_end,
            step_seconds,
            expected_pods if name in pod_metrics else None,
        )
    restart_series = (metrics.get("processorRestarts") or {}).get("series") or []
    restart_values_by_series = [common.sample_values([item]) for item in restart_series]
    active = _metric_values(metrics.get("processorActiveJobs") or {})
    oom = _metric_values(metrics.get("processorOomKilled") or {})
    evicted = _metric_values(metrics.get("processorEvicted") or {})
    qos = _metric_values(metrics.get("processorGuaranteedQos") or {})
    node_cpu = _metric_values(metrics.get("processorNodeCpuUtilizationRatio") or {})
    identities = _runtime_identities(report)
    variant, size_name, topology = _expected_variant(report, catalog)
    size = (catalog.get("sizeClasses") or {}).get(expected["size"])
    process_ids = [item.get("processId") for item in identities]
    images = {item.get("image") for item in identities if item.get("image")}
    revisions = {item.get("revision") for item in identities if item.get("revision")}
    cpu_requests = _metric_values(metrics.get("processorCpuRequests") or {})
    cpu_limits = _metric_values(metrics.get("processorCpuLimits") or {})
    memory_requests = _metric_values(metrics.get("processorMemoryRequestsBytes") or {})
    memory_limits = _metric_values(metrics.get("processorMemoryLimitsBytes") or {})
    cpu_usage_summary = common.summarize(
        common.sample_values(
            (metrics.get("processorCpuCores") or {}).get("series") or []
        )
    )
    memory_usage_summary = common.summarize(
        common.sample_values(
            (metrics.get("processorMemoryWorkingSetBytes") or {}).get("series") or []
        )
    )
    throttling = _metric_values(metrics.get("processorCpuThrottledPeriodsRatio") or {})
    image_ids = {
        item.get("metric", {}).get("image_id")
        for item in (metrics.get("processorImageInfo") or {}).get("series") or []
        if item.get("metric", {}).get("image_id")
    }
    expected_memory = int(size["memory"].removesuffix("Gi")) * 1024**3 if size else None
    checks = {
        "allRequiredMetricsCovered": all(item["covered"] for item in coverage.values()),
        "allRestartSamplesZero": bool(restart_values_by_series)
        and all(
            values and all(value == 0 for _timestamp, value in values)
            for values in restart_values_by_series
        ),
        "restartCountersMonotonic": bool(restart_values_by_series)
        and all(
            all(later[1] >= earlier[1] for earlier, later in zip(values, values[1:]))
            for values in restart_values_by_series
        ),
        "activeJobsReturnedToZero": bool(active) and active[-1] == 0,
        # These state metrics exist only when the reason occurred. Absence is
        # meaningful only after the exact pod has coverage in image/restart/QoS
        # series, which are required above; synthetic vector(0) is forbidden.
        "noOomKilled": not oom or max(oom) == 0,
        "notEvicted": not evicted or max(evicted) == 0,
        "guaranteedQosObserved": bool(qos) and min(qos) == 1,
        "runtimeIdentityComplete": bool(identities)
        and len(identities) == report.get("plannedConcurrency")
        and all(process_ids)
        and bool(expected_pods)
        and len(images) == 1
        and len(revisions) == 1,
        "knownCpuVariant": size is not None and topology in {"thread", "process"},
        "runtimeVariantMatchesExpected": variant == expected["variant"],
        "runtimeImageMatchesExpected": images == {expected["image"]},
        "runtimeRevisionMatchesExpected": revisions == {expected["revision"]},
        "runtimeTopologyMatchesExpected": topology == expected["topology"],
        "cpuRequestMatchesVariant": bool(size)
        and bool(cpu_requests)
        and set(cpu_requests) == {float(size["cpu"])},
        "cpuLimitMatchesVariant": bool(size)
        and bool(cpu_limits)
        and set(cpu_limits) == {float(size["cpu"])},
        "memoryRequestMatchesVariant": expected_memory is not None
        and bool(memory_requests)
        and set(memory_requests) == {float(expected_memory)},
        "memoryLimitMatchesVariant": expected_memory is not None
        and bool(memory_limits)
        and set(memory_limits) == {float(expected_memory)},
        "pidTopologyMatchesVariant": bool(process_ids)
        and (
            len(set(process_ids)) == 1
            if expected["topology"] == "thread"
            else (
                len(set(process_ids)) == len(process_ids)
                if expected["topology"] == "process"
                else False
            )
        ),
        "nodeCpuBelow80Percent": bool(node_cpu) and max(node_cpu) <= 0.80,
        "processorCpuHeadroom": bool(size)
        and cpu_usage_summary.get("p95") is not None
        and cpu_usage_summary["p95"] <= 0.90 * float(size["cpu"]),
        "processorMemoryHeadroom": expected_memory is not None
        and memory_usage_summary.get("max") is not None
        and memory_usage_summary["max"] <= 0.80 * expected_memory,
        "cfsThrottlingBelowOnePercent": bool(throttling) and max(throttling) <= 0.01,
        "prometheusImageMatchesRuntime": len(images) == 1
        and bool(image_ids)
        and all(expected["image"].split("@", 1)[-1] in item for item in image_ids),
    }
    return {
        "coverage": coverage,
        "restartSeries": restart_values_by_series,
        "finalActiveJobs": active[-1] if active else None,
        "runtimeVariant": variant,
        "runtimeImages": sorted(images),
        "runtimeRevisions": sorted(revisions),
        "runtimeProcessIds": process_ids,
        "prometheusImageIds": sorted(image_ids),
        "checks": checks,
        "evidenceComplete": all(checks.values()),
    }


def collect(report_path, context, step_seconds, expected_patch_path):
    payload = common.collect(report_path, context, step_seconds)
    report = json.loads(report_path.read_text())
    start = common.parse_timestamp(report["measurementStartedAt"])
    # Lifecycle evidence must include the bounded cleanup/retirement window,
    # not stop at the last measurement sample while jobs are intentionally live.
    measurement_end = common.parse_timestamp(report["measurementEndedAt"])
    lifecycle_end = max(
        common.parse_timestamp(report.get("endedAt") or report["measurementEndedAt"]),
        time.time(),
    )
    prometheus_pod = payload["prometheusPod"]
    for name, query in cpu_queries(payload["processorPods"]).items():
        query_end = (
            lifecycle_end
            if name
            in {
                "processorRestarts",
                "processorOomKilled",
                "processorEvicted",
                "processorActiveJobs",
                "processorImageInfo",
            }
            else measurement_end
        )
        series = common.query_range(
            context, prometheus_pod, query, start, query_end, step_seconds
        )
        payload["metrics"][name] = {
            "query": query,
            "summary": common.summarize(common.sample_values(series)),
            "series": series,
        }
    nodes = {
        item.get("metric", {}).get("node")
        for item in payload["metrics"]["processorCpuRequests"]["series"]
        if item.get("metric", {}).get("node")
    }
    if nodes:
        query = node_cpu_query(nodes)
        series = common.query_range(
            context, prometheus_pod, query, start, measurement_end, step_seconds
        )
        payload["metrics"]["processorNodeCpuUtilizationRatio"] = {
            "query": query,
            "summary": common.summarize(common.sample_values(series)),
            "series": series,
        }
    payload["collectorVariant"] = "cpu-sizing-v1"
    catalog = json.loads(Path(__file__).with_name("size_classes.json").read_text())
    expected_patch_bytes = expected_patch_path.read_bytes()
    expected = expected_runtime_from_patch(json.loads(expected_patch_bytes), catalog)
    payload["expectedRuntime"] = expected
    payload["expectedPatchSha256"] = hashlib.sha256(expected_patch_bytes).hexdigest()
    payload["certificationEvidence"] = certification_evidence(
        payload["metrics"], report, catalog, expected, step_seconds
    )
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--context", default=common.STAGING_CONTEXT)
    parser.add_argument("--step-seconds", type=int, default=15)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-patch", type=Path, required=True)
    args = parser.parse_args()
    if args.step_seconds < 5 or args.step_seconds > 60:
        parser.error("--step-seconds must be between 5 and 60")
    return args


def main():
    args = parse_args()
    report = args.report.resolve()
    output = args.output or report.with_name(report.stem + "-cpu-resources.json")
    common.write_atomic(
        output,
        collect(
            report,
            args.context,
            args.step_seconds,
            args.expected_patch.resolve(),
        ),
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
