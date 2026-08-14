#!/usr/bin/env python3
"""Certify one staging video soak and enforce the 15m/1h/4h/12h ladder.

The API report proves job behavior. The independently collected Prometheus
artifact proves processor, GPU, and relay resource behavior. This analyzer
hash-binds both inputs and fails closed on missing samples, counter resets,
requeues, output stalls, relay loss, incomplete cleanup, or memory growth.
"""

import argparse
import copy
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

from report import _histogram_latency_delta

MIB = 1024 * 1024
MAX_CLUSTER_IDENTITY_VALIDITY_SECONDS = 48 * 60 * 60
TERMINAL_STATES = {"cancelled", "completed", "error"}
COMMON_METRICS = {
    "processorCpuCores",
    "processorMemoryWorkingSetBytes",
    "processorContainerRestarts",
    "relayCpuCores",
    "relayMemoryWorkingSetBytes",
    "relayContainerRestarts",
    "relayPodIdentity",
    "relayReaders",
    "relayIngressBytesPerSecond",
    "relayEgressBytesPerSecond",
    "relayOutputPathCount",
    "relayOutputIngressBytesPerSecond",
    "relayRtspPacketsLostPerSecond",
    "relayRtspPacketsInErrorPerSecond",
}
GPU_METRICS = {
    "gpuUtilPercent",
    "gpuFramebufferUsedMiB",
    "gpuDecoderUtilPercent",
    "gpuMemoryCopyUtilPercent",
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(path):
    return json.loads(Path(path).read_text())


def _canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_corpus_bundle(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = _load(manifest_path)
    profiles = {}
    specifications = {}
    for raw in manifest.get("profiles") or []:
        profile_id = raw.get("id")
        relative = Path(str(raw.get("spec") or ""))
        specification_path = (manifest_path.parent / relative).resolve()
        if (
            not profile_id
            or profile_id in profiles
            or not relative.parts
            or relative.is_absolute()
            or manifest_path.parent not in specification_path.parents
        ):
            raise ValueError("corpus profile identity or specification path is invalid")
        specification = _load(specification_path)
        specifications[str(relative)] = specification
        profiles[profile_id] = {**raw, "specification": specification}
    if not profiles:
        raise ValueError("corpus manifest contains no profiles")
    return {
        "path": manifest_path,
        "profiles": profiles,
        "sha256": _canonical_sha256(
            {"manifest": manifest, "specifications": specifications}
        ),
    }


def _timestamp(value):
    if not value:
        raise ValueError("required timestamp is missing")
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()


def _finite_number(value, field):
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a finite number") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be a finite number")
    return parsed


def _metric_samples(metric):
    """Aggregate Prometheus series at each timestamp.

    The collector queries sums/maxes today, but aggregation here prevents an
    accidental label split from making coverage look larger or choosing one
    arbitrary series for memory leak analysis.
    """

    values = {}
    for series in metric.get("series") or []:
        for raw_time, raw_value in series.get("values") or []:
            timestamp = _finite_number(raw_time, "metric timestamp")
            value = _finite_number(raw_value, "metric value")
            values[timestamp] = values.get(timestamp, 0.0) + value
    return sorted(values.items())


def _linear_slope_per_hour(samples):
    if len(samples) < 2:
        return None
    origin = samples[0][0]
    xs = [(timestamp - origin) / 3600 for timestamp, _value in samples]
    ys = [value for _timestamp_value, value in samples]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((value - mean_x) ** 2 for value in xs)
    if denominator <= 0:
        return None
    return (
        sum((x_value - mean_x) * (y_value - mean_y) for x_value, y_value in zip(xs, ys))
        / denominator
    )


def _endpoint_growth(samples):
    if len(samples) < 2:
        return None
    window = max(1, min(len(samples) // 5, max(3, len(samples) // 10)))
    first = sorted(value for _timestamp_value, value in samples[:window])
    last = sorted(value for _timestamp_value, value in samples[-window:])

    def median(values):
        middle = len(values) // 2
        if len(values) % 2:
            return values[middle]
        return (values[middle - 1] + values[middle]) / 2

    return median(last) - median(first)


def load_policy(matrix_path):
    matrix_path = Path(matrix_path).resolve()
    matrix = _load(matrix_path)
    if matrix.get("schemaVersion") != 1 or matrix.get("environment") != "staging":
        raise ValueError("soak matrix must be schema 1 and staging-only")
    policy = matrix.get("soakPolicy") or {}
    if policy.get("schemaVersion") != 1:
        raise ValueError("soakPolicy.schemaVersion must be 1")
    sequence = policy.get("requiredSequence") or []
    if sequence != ["15m", "01h", "04h", "12h"]:
        raise ValueError("soakPolicy must use the exact 15m/01h/04h/12h ladder")
    if set(policy.get("stages") or {}) != set(sequence):
        raise ValueError("soakPolicy stages must exactly match requiredSequence")
    for field in (
        "minimumMeasurementCoverageRatio",
        "minimumTelemetryCoverageRatio",
    ):
        value = _finite_number(policy.get(field), field)
        if value <= 0 or value > 1:
            raise ValueError(f"{field} must be greater than zero and at most one")
    maximum_watch_gap = _finite_number(
        policy.get("maximumWatchRenewalGapSeconds"),
        "maximumWatchRenewalGapSeconds",
    )
    if maximum_watch_gap <= 0 or maximum_watch_gap >= 60:
        raise ValueError("maximumWatchRenewalGapSeconds must be between 0 and 60")
    maximum_output_stall = policy.get("maxConsecutiveNonadvancingOutputIntervals")
    if (
        not isinstance(maximum_output_stall, int)
        or isinstance(maximum_output_stall, bool)
        or maximum_output_stall < 0
    ):
        raise ValueError(
            "maxConsecutiveNonadvancingOutputIntervals must be a nonnegative integer"
        )
    for field in ("maxFrameLatencyP95Ms", "maxFrameLatencyP99Ms"):
        if _finite_number(policy.get(field), field) <= 0:
            raise ValueError(f"{field} must be greater than zero")
    required_counters = policy.get("requiredCounters") or []
    if not isinstance(required_counters, list) or not {
        "captured",
        "decoded",
        "inferred",
        "rendered",
        "published",
    }.issubset(required_counters):
        raise ValueError("requiredCounters must include the complete output path")
    corpus_path = (
        matrix_path.parent / str(policy.get("corpusManifestPath") or "")
    ).resolve()
    if matrix_path.parent.parent not in corpus_path.parents:
        raise ValueError("corpusManifestPath must stay inside the benchmark directory")
    corpus = _load_corpus_bundle(corpus_path)
    if corpus["sha256"] != policy.get("corpusBundleSha256"):
        raise ValueError("corpus bundle digest does not match soakPolicy")
    stage_threshold_fields = (
        "durationSeconds",
        "maxProcessorMemoryGrowthMiB",
        "maxRelayMemoryGrowthMiB",
        "maxProcessorMemorySlopeMiBPerHour",
        "maxRelayMemorySlopeMiBPerHour",
        "maxGpuFramebufferGrowthMiB",
        "maxGpuFramebufferSlopeMiBPerHour",
    )
    for stage, stage_policy in policy["stages"].items():
        for field in stage_threshold_fields:
            value = _finite_number(stage_policy.get(field), f"{stage}.{field}")
            if value < 0 or (field == "durationSeconds" and value == 0):
                raise ValueError(f"{stage}.{field} must be nonnegative")
    scenarios = {}
    seen = set()
    for raw in matrix.get("scenarios") or []:
        family = raw.get("soakFamily")
        stage = raw.get("soakStage")
        if family not in {"cpu", "gpu"} or stage not in sequence:
            raise ValueError(
                "every soak scenario needs a cpu/gpu family and valid stage"
            )
        key = (family, stage)
        if key in seen:
            raise ValueError(f"duplicate soak scenario for {family}/{stage}")
        seen.add(key)
        stage_policy = policy["stages"][stage]
        if raw.get("durationSeconds") != stage_policy.get("durationSeconds"):
            raise ValueError(f"scenario {raw.get('name')} duration differs from policy")
        if raw.get("publishOutput") is not True:
            raise ValueError(f"scenario {raw.get('name')} must publish output")
        mode = raw.get("mode") or matrix.get("defaults", {}).get("mode") or "stream"
        if mode != "stream":
            raise ValueError(f"scenario {raw.get('name')} must use stream mode")
        scenarios[raw["name"]] = {**raw, "policy": stage_policy}
    expected = {(family, stage) for family in ("cpu", "gpu") for stage in sequence}
    if seen != expected:
        raise ValueError("soak matrix must contain one cpu and gpu scenario per stage")
    return {
        "matrix": matrix,
        "path": matrix_path,
        "sha256": _sha256(matrix_path),
        "policy": policy,
        "scenarios": scenarios,
        "corpus": corpus,
    }


def _expected_profile_contracts(matrix_config, scenario):
    workloads = []
    for index, value in enumerate(scenario["workloads"]):
        profile_id, raw_count = value.split("=", 1)
        raw_count, separator, raw_delay = raw_count.partition("@")
        workloads.append(
            {
                "index": index,
                "profile": profile_id,
                "count": int(raw_count),
                "startAfterSeconds": float(raw_delay) if separator else 0.0,
            }
        )
    expected = []
    ordinal = 0
    defaults = matrix_config["matrix"]["defaults"]
    for workload in sorted(
        workloads, key=lambda item: (item["startAfterSeconds"], item["index"])
    ):
        profile = matrix_config["corpus"]["profiles"].get(workload["profile"])
        if not profile:
            raise ValueError(f"unknown soak corpus profile {workload['profile']}")
        for copy_index in range(workload["count"]):
            ordinal += 1
            specification = copy.deepcopy(profile["specification"])
            metadata = dict(specification.get("metadata") or {})
            metadata["benchmark"] = {
                "profile": workload["profile"],
                "instance": ordinal,
            }
            specification["metadata"] = metadata
            expected.append(
                {
                    "ordinal": ordinal,
                    "copy": copy_index + 1,
                    "profile": workload["profile"],
                    "provisionalClass": profile["provisionalClass"],
                    "tier": profile["tier"],
                    "mode": scenario.get("mode") or defaults.get("mode") or "stream",
                    "imageOutput": profile.get("imageOutput"),
                    "maxFps": scenario.get("maxFps", defaults.get("maxFps")),
                    "startAfterSeconds": workload["startAfterSeconds"],
                    "workflowSpecificationSha256": _canonical_sha256(specification),
                }
            )
    return expected


def _job_series(report, planned_jobs):
    starts = report.get("starts") or []
    job_ids = [((item.get("job") or {}).get("id")) for item in starts]
    if len(job_ids) != planned_jobs or any(not value for value in job_ids):
        raise ValueError("report does not retain every planned start identity")
    if len(set(job_ids)) != planned_jobs:
        raise ValueError("report start identities are not distinct")
    expected = set(job_ids)
    measurement = [
        sample
        for sample in report.get("samples") or []
        if sample.get("phase") == "measurement"
    ]
    by_job = {job_id: [] for job_id in job_ids}
    exact_coverage = True
    for sample in measurement:
        jobs = sample.get("jobs") or []
        actual = {job.get("id") for job in jobs}
        if actual != expected or len(jobs) != planned_jobs:
            exact_coverage = False
            continue
        for job in jobs:
            by_job[job["id"]].append(job)
    return job_ids, measurement, by_job, exact_coverage


def _counter_checks(by_job, required_counters):
    resets = []
    missing = []
    nonadvancing = []
    frame_resets = []
    processor_migrations = []
    attempt_changes = []
    nonrunning = []
    missing_latency_histograms = []
    stream_summaries = []
    for job_id, jobs in by_job.items():
        if not jobs:
            missing.append(f"{job_id}:samples")
            continue
        processors = [job.get("processorId") for job in jobs]
        attempts = [job.get("attempts") for job in jobs]
        states = [job.get("state") for job in jobs]
        if any(state != "running" for state in states):
            nonrunning.append(job_id)
        if not processors[0] or any(value != processors[0] for value in processors):
            processor_migrations.append(job_id)
        if attempts[0] is None or any(value != attempts[0] for value in attempts):
            attempt_changes.append(job_id)
        frame_values = [(job.get("stats") or {}).get("frames") for job in jobs]
        if any(not isinstance(value, (int, float)) for value in frame_values):
            missing.append(f"{job_id}:frames")
        else:
            if any(
                current < previous
                for previous, current in zip(frame_values, frame_values[1:])
            ):
                frame_resets.append(job_id)
            if frame_values[-1] <= frame_values[0]:
                nonadvancing.append(f"{job_id}:frames")
        deltas = {}
        maximum_output_stall = 0
        for counter in required_counters:
            values = [
                ((job.get("stats") or {}).get("counters") or {}).get(counter)
                for job in jobs
            ]
            if any(not isinstance(value, (int, float)) for value in values):
                missing.append(f"{job_id}:{counter}")
                continue
            if any(current < previous for previous, current in zip(values, values[1:])):
                resets.append(f"{job_id}:{counter}")
            delta = values[-1] - values[0]
            deltas[counter] = delta
            if delta <= 0:
                nonadvancing.append(f"{job_id}:{counter}")
            if counter == "published":
                streak = 0
                for previous, current in zip(values, values[1:]):
                    streak = streak + 1 if current <= previous else 0
                    maximum_output_stall = max(maximum_output_stall, streak)
        latency = _histogram_latency_delta([{"job": job} for job in jobs])
        if not latency["frameLatencyHistogramCount"]:
            missing_latency_histograms.append(job_id)
        stream_summaries.append(
            {
                "jobId": job_id,
                "processorId": processors[0],
                "attempt": attempts[0],
                "sampleCount": len(jobs),
                "frameDelta": (
                    frame_values[-1] - frame_values[0]
                    if all(isinstance(value, (int, float)) for value in frame_values)
                    else None
                ),
                "counterDeltas": deltas,
                "maximumConsecutiveNonadvancingOutputIntervals": maximum_output_stall,
                "latency": latency,
            }
        )
    return {
        "streams": stream_summaries,
        "counterResets": resets,
        "frameResets": frame_resets,
        "missingCounters": missing,
        "nonadvancingCounters": nonadvancing,
        "processorMigrations": processor_migrations,
        "attemptChanges": attempt_changes,
        "nonrunningJobs": nonrunning,
        "missingLatencyHistograms": missing_latency_histograms,
    }


def _watch_checks(report, job_ids, policy, start, end):
    leases = report.get("watchLeases") or {}
    exact_jobs = set(leases) == set(job_ids)
    details = {}
    for job_id in job_ids:
        item = leases.get(job_id) or {}
        try:
            first = _timestamp(item.get("firstRequestedAt"))
            last = _timestamp(item.get("lastRequestedAt"))
        except ValueError:
            first = None
            last = None
        maximum_gap = item.get("maximumRenewalGapSeconds")
        maximum_gap = (
            _finite_number(maximum_gap, "maximumRenewalGapSeconds")
            if maximum_gap is not None
            else None
        )
        renewal_count = item.get("renewalCount")
        minimum_renewal_count = max(
            1, math.ceil((end - start) / policy["maximumWatchRenewalGapSeconds"])
        )
        requested_through_measurement = (
            first is not None
            and last is not None
            and first <= start
            and last >= end - policy["maximumWatchRenewalGapSeconds"]
        )
        bounded_gap = (maximum_gap is None and renewal_count == 1) or (
            maximum_gap is not None
            and maximum_gap <= policy["maximumWatchRenewalGapSeconds"]
        )
        details[job_id] = {
            "renewalCount": renewal_count,
            "minimumRenewalCount": minimum_renewal_count,
            "firstRequestedAt": item.get("firstRequestedAt"),
            "lastRequestedAt": item.get("lastRequestedAt"),
            "maximumRenewalGapSeconds": maximum_gap,
            "requestedThroughMeasurement": requested_through_measurement,
            "boundedRenewalGap": bounded_gap,
            "noRenewalErrors": not (item.get("errors") or []),
            "outputSelected": bool(item.get("output")),
        }
    return {
        "checks": {
            "exactJobCoverage": exact_jobs,
            "allLeasesRenewed": exact_jobs
            and all(
                isinstance(item["renewalCount"], int)
                and not isinstance(item["renewalCount"], bool)
                and item["renewalCount"] >= item["minimumRenewalCount"]
                and item["requestedThroughMeasurement"]
                and item["boundedRenewalGap"]
                and item["noRenewalErrors"]
                and item["outputSelected"]
                for item in details.values()
            ),
        },
        "jobs": details,
    }


def _cluster_identity_checks(resources, start, end):
    cluster_identity = resources.get("clusterIdentity") or {}
    approved = cluster_identity.get("approved") or {}
    observed = cluster_identity.get("observed") or {}
    if not cluster_identity.get("approvedPath"):
        raise ValueError("resource artifact has no approved cluster identity path")
    approved_path = Path(str(cluster_identity["approvedPath"])).resolve()
    approved_bytes = approved_path.read_bytes()
    approved_from_file = json.loads(approved_bytes)
    approved_at = _timestamp(approved.get("approvedAt"))
    valid_until = _timestamp(approved.get("validUntil"))
    shape_valid = (
        approved.get("schemaVersion") == 1
        and approved.get("environment") == "staging"
        and approved.get("context") == "ck8s-stg"
        and resources.get("environment") == "staging"
        and resources.get("clusterContext") == "ck8s-stg"
        and isinstance(approved.get("approvedBy"), str)
        and bool(approved.get("approvedBy").strip())
        and isinstance(approved.get("apiServer"), str)
        and approved.get("apiServer").startswith("https://")
        and isinstance(approved.get("kubeSystemNamespaceUid"), str)
        and bool(approved.get("kubeSystemNamespaceUid").strip())
    )
    digest_valid = cluster_identity.get("approvedSha256") == _canonical_sha256(approved)
    file_binding_valid = (
        approved_from_file == approved
        and cluster_identity.get("approvedFileSha256")
        == hashlib.sha256(approved_bytes).hexdigest()
    )
    observed_matches = observed.get("apiServer") == approved.get(
        "apiServer"
    ) and observed.get("kubeSystemNamespaceUid") == approved.get(
        "kubeSystemNamespaceUid"
    )
    validity_covers_measurement = (
        approved_at <= start < end <= valid_until
        and valid_until - approved_at <= MAX_CLUSTER_IDENTITY_VALIDITY_SECONDS
    )
    return {
        "checks": {
            "clusterIdentityShape": shape_valid,
            "clusterIdentityDigest": digest_valid,
            "clusterIdentityFileBinding": file_binding_valid,
            "clusterIdentityObserved": observed_matches,
            "clusterIdentityValidity": validity_covers_measurement,
        },
        "approvedSha256": cluster_identity.get("approvedSha256"),
        "approvedFileSha256": cluster_identity.get("approvedFileSha256"),
        "approvedPath": str(approved_path),
        "approvedBy": approved.get("approvedBy"),
        "apiServer": approved.get("apiServer"),
        "kubeSystemNamespaceUid": approved.get("kubeSystemNamespaceUid"),
        "approvedAt": approved.get("approvedAt"),
        "validUntil": approved.get("validUntil"),
    }


def _resource_checks(resources, scenario, policy, start, end):
    metrics = resources.get("metrics") or {}
    required = set(COMMON_METRICS)
    if scenario["soakFamily"] == "gpu":
        required.update(GPU_METRICS)
    missing_metrics = sorted(required - set(metrics))
    step = _finite_number(resources.get("sampleStepSeconds"), "sampleStepSeconds")
    expected_count = max(2, math.floor((end - start) / step) + 1)
    minimum_count = max(
        2,
        math.ceil(expected_count * float(policy["minimumTelemetryCoverageRatio"])),
    )
    coverage = {}
    samples_by_metric = {}
    for name in sorted(required & set(metrics)):
        samples = _metric_samples(metrics[name])
        samples_by_metric[name] = samples
        coverage[name] = {
            "sampleCount": len(samples),
            "minimumSampleCount": minimum_count,
            "covered": len(samples) >= minimum_count,
            "firstAt": samples[0][0] if samples else None,
            "lastAt": samples[-1][0] if samples else None,
            "spansMeasurementWindow": bool(samples)
            and samples[0][0] <= start + step * 2
            and samples[-1][0] >= end - step * 2,
            "timestampsWithinQueryWindow": bool(samples)
            and samples[0][0] >= start - step
            and samples[-1][0] <= end + step,
        }
    memory = {}
    for name, prefix in (
        ("processorMemoryWorkingSetBytes", "processor"),
        ("relayMemoryWorkingSetBytes", "relay"),
    ):
        samples = samples_by_metric.get(name) or []
        growth = _endpoint_growth(samples)
        slope = _linear_slope_per_hour(samples)
        memory[prefix] = {
            "endpointGrowthMiB": None if growth is None else round(growth / MIB, 3),
            "linearSlopeMiBPerHour": None if slope is None else round(slope / MIB, 3),
        }
    gpu_memory_samples = samples_by_metric.get("gpuFramebufferUsedMiB") or []
    gpu_memory_growth = _endpoint_growth(gpu_memory_samples)
    gpu_memory_slope = _linear_slope_per_hour(gpu_memory_samples)
    memory["gpuFramebuffer"] = {
        "endpointGrowthMiB": (
            None if gpu_memory_growth is None else round(gpu_memory_growth, 3)
        ),
        "linearSlopeMiBPerHour": (
            None if gpu_memory_slope is None else round(gpu_memory_slope, 3)
        ),
    }
    readers = [value for _time, value in samples_by_metric.get("relayReaders", [])]
    ingress = [
        value
        for _time, value in samples_by_metric.get("relayIngressBytesPerSecond", [])
    ]
    egress = [
        value for _time, value in samples_by_metric.get("relayEgressBytesPerSecond", [])
    ]
    output_paths = [
        value for _time, value in samples_by_metric.get("relayOutputPathCount", [])
    ]
    output_ingress = [
        value
        for _time, value in samples_by_metric.get(
            "relayOutputIngressBytesPerSecond", []
        )
    ]
    lost = [
        value
        for _time, value in samples_by_metric.get("relayRtspPacketsLostPerSecond", [])
    ]
    errors = [
        value
        for _time, value in samples_by_metric.get(
            "relayRtspPacketsInErrorPerSecond", []
        )
    ]
    processor_restarts = [
        value
        for _time, value in samples_by_metric.get("processorContainerRestarts", [])
    ]
    relay_restarts = [
        value for _time, value in samples_by_metric.get("relayContainerRestarts", [])
    ]
    thresholds = scenario["policy"]
    identity_series = (metrics.get("relayPodIdentity") or {}).get("series") or []
    relay_identities = []
    for series in identity_series:
        labels = series.get("metric") or {}
        samples = _metric_samples({"series": [series]})
        relay_identities.append(
            {
                "pod": labels.get("pod"),
                "uid": labels.get("uid"),
                "sampleCount": len(samples),
                "firstAt": samples[0][0] if samples else None,
                "lastAt": samples[-1][0] if samples else None,
                "spansMeasurementWindow": bool(samples)
                and samples[0][0] <= start + step * 2
                and samples[-1][0] >= end - step * 2,
            }
        )
    restart_series = (metrics.get("relayContainerRestarts") or {}).get("series") or []
    restart_pods = [
        (series.get("metric") or {}).get("pod") for series in restart_series
    ]
    output_metric_pods = {
        name: [
            (series.get("metric") or {}).get("pod")
            for series in ((metrics.get(name) or {}).get("series") or [])
        ]
        for name in ("relayOutputPathCount", "relayOutputIngressBytesPerSecond")
    }
    relay_identity_stable = (
        len(relay_identities) == 1
        and bool(relay_identities[0]["pod"])
        and bool(relay_identities[0]["uid"])
        and relay_identities[0]["spansMeasurementWindow"]
        and restart_pods == [relay_identities[0]["pod"]]
        and all(
            pods == [relay_identities[0]["pod"]] for pods in output_metric_pods.values()
        )
    )
    checks = {
        "allRequiredMetricsPresent": not missing_metrics,
        "telemetryCoverage": bool(coverage)
        and all(
            item["covered"]
            and item["spansMeasurementWindow"]
            and item["timestampsWithinQueryWindow"]
            for item in coverage.values()
        ),
        "processorMemoryGrowth": memory["processor"]["endpointGrowthMiB"] is not None
        and memory["processor"]["endpointGrowthMiB"]
        <= thresholds["maxProcessorMemoryGrowthMiB"],
        "relayMemoryGrowth": memory["relay"]["endpointGrowthMiB"] is not None
        and memory["relay"]["endpointGrowthMiB"]
        <= thresholds["maxRelayMemoryGrowthMiB"],
        "processorMemorySlope": memory["processor"]["linearSlopeMiBPerHour"] is not None
        and memory["processor"]["linearSlopeMiBPerHour"]
        <= thresholds["maxProcessorMemorySlopeMiBPerHour"],
        "relayMemorySlope": memory["relay"]["linearSlopeMiBPerHour"] is not None
        and memory["relay"]["linearSlopeMiBPerHour"]
        <= thresholds["maxRelayMemorySlopeMiBPerHour"],
        "relayReaderFloor": bool(readers) and min(readers) >= scenario["plannedJobs"],
        "relayTrafficAdvanced": bool(ingress)
        and bool(egress)
        and max(ingress) > 0
        and max(egress) > 0,
        "relayOutputPathsPresent": bool(output_paths)
        and min(output_paths) >= scenario["plannedJobs"],
        "relayOutputIngressAdvanced": bool(output_ingress) and min(output_ingress) > 0,
        "relayPacketLossZero": bool(lost) and min(lost) == 0 and max(lost) == 0,
        "relayPacketErrorsZero": bool(errors) and min(errors) == 0 and max(errors) == 0,
        "processorRestartCounterStable": bool(processor_restarts)
        and max(processor_restarts) == min(processor_restarts),
        "relayRestartCounterStable": bool(relay_restarts)
        and max(relay_restarts) == min(relay_restarts),
        "relayPodIdentityStable": relay_identity_stable,
    }
    if scenario["soakFamily"] == "gpu":
        checks.update(
            {
                "gpuFramebufferGrowth": memory["gpuFramebuffer"]["endpointGrowthMiB"]
                is not None
                and memory["gpuFramebuffer"]["endpointGrowthMiB"]
                <= thresholds["maxGpuFramebufferGrowthMiB"],
                "gpuFramebufferSlope": memory["gpuFramebuffer"]["linearSlopeMiBPerHour"]
                is not None
                and memory["gpuFramebuffer"]["linearSlopeMiBPerHour"]
                <= thresholds["maxGpuFramebufferSlopeMiBPerHour"],
            }
        )
    return {
        "checks": checks,
        "missingMetrics": missing_metrics,
        "coverage": coverage,
        "memory": memory,
        "relay": {
            "minimumReaders": min(readers) if readers else None,
            "maximumPacketLossPerSecond": max(lost) if lost else None,
            "maximumPacketErrorsPerSecond": max(errors) if errors else None,
            "restartCounterRange": (
                [min(relay_restarts), max(relay_restarts)] if relay_restarts else None
            ),
            "podIdentities": relay_identities,
        },
        "processorRestartCounterRange": (
            [min(processor_restarts), max(processor_restarts)]
            if processor_restarts
            else None
        ),
    }


def certify_run(report_path, resource_path, matrix_config, scenario_name):
    report_path = Path(report_path).resolve()
    resource_path = Path(resource_path).resolve()
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    resources = _load(resource_path)
    scenario = matrix_config["scenarios"].get(scenario_name)
    if not scenario:
        raise ValueError(f"unknown soak scenario {scenario_name}")
    expected_profiles = _expected_profile_contracts(matrix_config, scenario)
    planned_jobs = len(expected_profiles)
    scenario = {**scenario, "plannedJobs": planned_jobs}
    start = _timestamp(report.get("measurementStartedAt"))
    end = _timestamp(report.get("measurementEndedAt"))
    measured_seconds = end - start
    if measured_seconds <= 0:
        raise ValueError("measurement timestamps are not ordered")
    job_ids, samples, by_job, exact_coverage = _job_series(report, planned_jobs)
    expected_samples = max(
        1,
        math.floor(
            scenario["durationSeconds"]
            / float(
                scenario.get("pollIntervalSeconds")
                or matrix_config["matrix"]["defaults"]["pollIntervalSeconds"]
            )
        ),
    )
    minimum_samples = math.ceil(
        expected_samples
        * float(matrix_config["policy"]["minimumMeasurementCoverageRatio"])
    )
    counters = _counter_checks(by_job, matrix_config["policy"]["requiredCounters"])
    watch = _watch_checks(report, job_ids, matrix_config["policy"], start, end)
    cluster_identity = _cluster_identity_checks(resources, start, end)
    expected_pods = sorted(
        {
            ((job.get("stats") or {}).get("runtime") or {}).get("hostname")
            for job in report.get("jobs") or []
            if ((job.get("stats") or {}).get("runtime") or {}).get("hostname")
        }
    )
    measured_processors = sorted(
        {
            stream["processorId"]
            for stream in counters["streams"]
            if stream["processorId"]
        }
    )
    binding_checks = {
        "resourceSourceHash": resources.get("sourceReportSha256")
        == hashlib.sha256(report_bytes).hexdigest(),
        "runId": bool(report.get("runId"))
        and resources.get("runId") == report.get("runId"),
        "measurementWindow": resources.get("measurementStartedAt")
        == report.get("measurementStartedAt")
        and resources.get("measurementEndedAt") == report.get("measurementEndedAt"),
        "processorPods": bool(expected_pods)
        and sorted(resources.get("processorPods") or []) == expected_pods,
        "processorRuntimeIdentity": measured_processors == expected_pods,
        "corpusBundle": report.get("corpusBundleSha256")
        == matrix_config["corpus"]["sha256"],
        "workspace": report.get("workspace")
        == matrix_config["matrix"]["defaults"].get("workspace"),
        "apiBase": str(report.get("apiBase") or "").rstrip("/")
        == str(matrix_config["matrix"]["defaults"].get("apiBase") or "").rstrip("/"),
    }
    report_checks = {
        "reportSucceeded": report.get("success") is True,
        "reportComplete": (report.get("checkpoint") or {}).get("phase") == "complete"
        and bool(report.get("endedAt")),
        "noRunErrors": not (report.get("errors") or [])
        and not (report.get("cancelErrors") or []),
        "noRecoveryEvents": not (report.get("recoveries") or []),
        "plannedConcurrency": report.get("plannedConcurrency") == planned_jobs,
        "exactWorkloadProfiles": report.get("profiles") == expected_profiles,
        "sourceIdentity": (report.get("source") or {}).get("id")
        == matrix_config["matrix"]["defaults"].get("sourceId"),
        "measurementDuration": measured_seconds
        >= scenario["durationSeconds"]
        * float(matrix_config["policy"]["minimumMeasurementCoverageRatio"]),
        "measurementSampleCoverage": len(samples) >= minimum_samples,
        "exactJobCoverage": exact_coverage
        and all(len(by_job[job_id]) == len(samples) for job_id in job_ids),
        "allJobsRunning": not counters["nonrunningJobs"],
        "stableProcessorIdentity": not counters["processorMigrations"],
        "stableAttemptIdentity": not counters["attemptChanges"],
        "singleProcessorPlacement": len(measured_processors) == 1
        and len(expected_pods) == 1,
        "noFrameResets": not counters["frameResets"],
        "noCounterResets": not counters["counterResets"],
        "allCountersPresent": not counters["missingCounters"],
        "framesAndOutputAdvanced": not counters["nonadvancingCounters"],
        "outputNeverStalled": all(
            item["maximumConsecutiveNonadvancingOutputIntervals"]
            <= matrix_config["policy"]["maxConsecutiveNonadvancingOutputIntervals"]
            for item in counters["streams"]
        ),
        "frameLatencyHistogramPresent": not counters["missingLatencyHistograms"],
        "frameLatencyWithinSlo": all(
            item["latency"]["frameLatencyP95ApproxMs"]
            <= matrix_config["policy"]["maxFrameLatencyP95Ms"]
            and item["latency"]["frameLatencyP99ApproxMs"]
            <= matrix_config["policy"]["maxFrameLatencyP99Ms"]
            for item in counters["streams"]
            if item["latency"]["frameLatencyHistogramCount"]
        )
        and not counters["missingLatencyHistograms"],
        "streamProfiles": all(
            item.get("mode") == "stream" and bool(item.get("imageOutput"))
            for item in report.get("profiles") or []
        )
        and len(report.get("profiles") or []) == planned_jobs,
        "cleanupCancelled": len(report.get("jobs") or []) == planned_jobs
        and all(job.get("state") == "cancelled" for job in report.get("jobs") or []),
    }
    resources_result = _resource_checks(
        resources, scenario, matrix_config["policy"], start, end
    )
    checks = {
        **{f"binding.{key}": value for key, value in binding_checks.items()},
        **{
            f"binding.{key}": value for key, value in cluster_identity["checks"].items()
        },
        **{f"report.{key}": value for key, value in report_checks.items()},
        **{f"watch.{key}": value for key, value in watch["checks"].items()},
        **{
            f"resources.{key}": value
            for key, value in resources_result["checks"].items()
        },
    }
    return {
        "schemaVersion": 1,
        "environment": "staging",
        "scenario": scenario_name,
        "soakFamily": scenario["soakFamily"],
        "soakStage": scenario["soakStage"],
        "runId": report.get("runId"),
        "passed": all(checks.values()),
        "checks": checks,
        "measurement": {
            "observedSeconds": round(measured_seconds, 3),
            "requiredSeconds": scenario["durationSeconds"],
            "sampleCount": len(samples),
            "minimumSampleCount": minimum_samples,
        },
        "jobEvidence": counters,
        "watchEvidence": watch,
        "clusterIdentityEvidence": cluster_identity,
        "resourceEvidence": resources_result,
        "artifacts": {
            "matrix": {
                "path": str(matrix_config["path"]),
                "sha256": matrix_config["sha256"],
            },
            "report": {"path": str(report_path), "sha256": _sha256(report_path)},
            "resources": {
                "path": str(resource_path),
                "sha256": _sha256(resource_path),
            },
        },
    }


def certify_stage(
    report_path,
    resource_path,
    matrix_config,
    scenario_name,
    prior_certification_paths=None,
):
    """Recompute this stage and every required predecessor from raw artifacts."""

    result = certify_run(report_path, resource_path, matrix_config, scenario_name)
    sequence = matrix_config["policy"]["requiredSequence"]
    family = result["soakFamily"]
    stage_index = sequence.index(result["soakStage"])
    required_prior_stages = sequence[:stage_index]
    priors = {}
    descriptors = []
    for path in prior_certification_paths or []:
        path = Path(path).resolve()
        certification = _load(path)
        prior_stage = certification.get("soakStage")
        if certification.get("soakFamily") != family:
            raise ValueError("prior soak certification belongs to another family")
        if prior_stage in priors:
            raise ValueError(f"duplicate prior soak certification for {prior_stage}")
        if prior_stage not in required_prior_stages:
            raise ValueError(f"unexpected prior soak certification for {prior_stage}")
        artifacts = certification.get("artifacts") or {}
        if (artifacts.get("matrix") or {}).get("sha256") != matrix_config["sha256"]:
            raise ValueError("prior soak certification uses a different matrix")
        prior_scenario = certification.get("scenario")
        report_artifact = artifacts.get("report") or {}
        resource_artifact = artifacts.get("resources") or {}
        recomputed = certify_run(
            report_artifact.get("path"),
            resource_artifact.get("path"),
            matrix_config,
            prior_scenario,
        )
        if (
            not recomputed["passed"]
            or certification.get("passed") is not True
            or certification.get("predecessorsPassed") is not True
            or recomputed["soakStage"] != prior_stage
            or recomputed["soakFamily"] != family
            or recomputed["artifacts"] != artifacts
        ):
            raise ValueError(
                f"prior {prior_stage} soak certification is not reproducible"
            )
        priors[prior_stage] = certification
        descriptors.append({"path": str(path), "sha256": _sha256(path)})
    if set(priors) != set(required_prior_stages):
        missing = sorted(set(required_prior_stages) - set(priors))
        raise ValueError(
            "missing required prior soak certifications: " + ", ".join(missing)
        )
    result["evidencePassed"] = result["passed"]
    result["predecessorsPassed"] = True
    result["passed"] = result["evidencePassed"] and result["predecessorsPassed"]
    result["priorCertifications"] = descriptors
    result["nextStage"] = (
        sequence[stage_index + 1] if stage_index + 1 < len(sequence) else None
    )
    return result


def certify_ladder(matrix_path, suite_path, results_dir):
    config = load_policy(matrix_path)
    suite_path = Path(suite_path).resolve()
    suite = _load(suite_path)
    if (
        suite.get("schemaVersion") != 2
        or suite.get("environment") != "staging"
        or suite.get("execute") is not True
        or suite.get("matrixSha256") != config["sha256"]
    ):
        raise ValueError("suite is not an executed run of the exact soak matrix")
    results_dir = Path(results_dir).resolve()
    run_by_scenario = {}
    for run in suite.get("runs") or []:
        scenario = run.get("scenario")
        if scenario in run_by_scenario:
            raise ValueError(f"suite repeats soak scenario {scenario}")
        run_by_scenario[scenario] = run
    families = {}
    for family in ("gpu", "cpu"):
        stages = []
        predecessors_passed = True
        for stage in config["policy"]["requiredSequence"]:
            scenario_name = next(
                name
                for name, scenario in config["scenarios"].items()
                if scenario["soakFamily"] == family and scenario["soakStage"] == stage
            )
            run = run_by_scenario.get(scenario_name)
            result = None
            error = None
            if run is None:
                error = "scenario has not run"
            elif run.get("status") != "completed" or run.get("returnCode") != 0:
                error = "scenario did not complete successfully"
            else:
                run_id = run.get("runId")
                report_path = results_dir / f"api-corpus-{run_id}.json"
                resource_path = results_dir / f"api-corpus-{run_id}-resources.json"
                try:
                    result = certify_run(
                        report_path, resource_path, config, scenario_name
                    )
                except (OSError, ValueError, json.JSONDecodeError) as exception:
                    error = str(exception)
            evidence_passed = bool(result and result["passed"])
            stage_passed = predecessors_passed and evidence_passed
            stages.append(
                {
                    "stage": stage,
                    "scenario": scenario_name,
                    "predecessorsPassed": predecessors_passed,
                    "evidencePassed": evidence_passed,
                    "passed": stage_passed,
                    "error": error,
                    "certification": result,
                }
            )
            predecessors_passed = stage_passed
        next_stage = next(
            (item["stage"] for item in stages if not item["passed"]), None
        )
        families[family] = {
            "passed": all(item["passed"] for item in stages),
            "nextRequiredStage": next_stage,
            "stages": stages,
        }
    return {
        "schemaVersion": 1,
        "environment": "staging",
        "suiteId": suite.get("suiteId"),
        "passed": all(item["passed"] for item in families.values()),
        "families": families,
        "artifacts": {
            "matrix": {"path": str(config["path"]), "sha256": config["sha256"]},
            "suite": {"path": str(suite_path), "sha256": _sha256(suite_path)},
        },
    }


def _write_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--scenario")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--resources", type=Path)
    parser.add_argument("--prior-certification", action="append", default=[], type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    single_stage = any((args.scenario, args.report, args.resources))
    full_ladder = any((args.suite, args.results_dir))
    if single_stage == full_ladder:
        parser.error(
            "use either --scenario/--report/--resources or --suite/--results-dir"
        )
    if single_stage:
        if not all((args.scenario, args.report, args.resources)):
            parser.error(
                "single-stage mode requires --scenario, --report, and --resources"
            )
        result = certify_stage(
            args.report,
            args.resources,
            load_policy(args.matrix),
            args.scenario,
            args.prior_certification,
        )
    else:
        if not all((args.suite, args.results_dir)):
            parser.error("ladder mode requires --suite and --results-dir")
        if args.prior_certification:
            parser.error("--prior-certification is only valid in single-stage mode")
        result = certify_ladder(args.matrix, args.suite, args.results_dir)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        _write_atomic(args.output, result)
    else:
        print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
