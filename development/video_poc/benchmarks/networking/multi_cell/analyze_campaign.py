#!/usr/bin/env python3
"""Certify paired multi-cell benchmark evidence, failing closed on omissions."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

REQUIRED_METRICS = {
    "relayBytesReceived": "prometheus",
    "relayBytesSent": "prometheus",
    "lbBytesIn": "crusoe-metrics",
    "lbBytesOut": "crusoe-metrics",
    "publisherVpcRxBytes": "crusoe-metrics",
    "publisherVpcTxBytes": "crusoe-metrics",
    "relayVpcRxBytes": "crusoe-metrics",
    "relayVpcTxBytes": "crusoe-metrics",
    "readerVpcRxBytes": "crusoe-metrics",
    "readerVpcTxBytes": "crusoe-metrics",
    "rtpPacketsReceived": "prometheus",
    "rtpPacketsLost": "prometheus",
    "rtpPacketsInError": "prometheus",
    "publisherNodePacketDrops": "prometheus",
    "relayNodePacketDrops": "prometheus",
    "readerNodePacketDrops": "prometheus",
    "publisherTcpRetransmits": "prometheus",
    "relayTcpRetransmits": "prometheus",
    "readerTcpRetransmits": "prometheus",
}


class EvidenceError(ValueError):
    pass


def _load(path):
    with Path(path).open() as source:
        return json.load(source)


def load_artifacts(evidence, artifact_root):
    root = Path(artifact_root).resolve()
    artifacts = {}
    for item in evidence.get("artifactIndex") or []:
        artifact_id = str(item.get("id") or "")
        if not artifact_id or artifact_id in artifacts:
            raise EvidenceError("artifact index has a duplicate or missing id")
        relative = Path(str(item.get("relativePath") or ""))
        if not relative.parts or relative.is_absolute():
            raise EvidenceError("artifact %s path must be relative" % artifact_id)
        path = (root / relative).resolve()
        if root != path and root not in path.parents:
            raise EvidenceError("artifact %s escapes artifact root" % artifact_id)
        try:
            payload_bytes = path.read_bytes()
        except OSError as error:
            raise EvidenceError("artifact %s cannot be read" % artifact_id) from error
        digest = hashlib.sha256(payload_bytes).hexdigest()
        _exact(digest, item.get("sha256"), "artifact %s sha256" % artifact_id)
        if len(payload_bytes) != item.get("bytes"):
            raise EvidenceError("artifact %s byte length does not match" % artifact_id)
        kind = str(item.get("kind") or "")
        if not kind:
            raise EvidenceError("artifact %s kind is missing" % artifact_id)
        try:
            payload = json.loads(payload_bytes)
        except json.JSONDecodeError as error:
            raise EvidenceError("artifact %s is not JSON" % artifact_id) from error
        artifacts[artifact_id] = {
            "kind": kind,
            "payload": payload,
            "sha256": digest,
            "relativePath": str(relative),
        }
    if not artifacts:
        raise EvidenceError("artifact index is empty")
    return artifacts


def _artifact(artifacts, artifact_id, kind, field):
    if not artifact_id or artifact_id not in artifacts:
        raise EvidenceError("%s references a missing raw artifact" % field)
    artifact = artifacts[artifact_id]
    _exact(artifact["kind"], kind, field + ".kind")
    return artifact["payload"]


def _contains(actual, expected, field):
    if not isinstance(actual, dict):
        raise EvidenceError("%s raw artifact must be an object" % field)
    for key, value in expected.items():
        if actual.get(key) != value:
            raise EvidenceError("%s raw artifact disagrees on %s" % (field, key))


def _finite(value, field, minimum=None):
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise EvidenceError("%s must be numeric" % field) from error
    if not math.isfinite(value):
        raise EvidenceError("%s must be finite" % field)
    if minimum is not None and value < minimum:
        raise EvidenceError("%s must be >= %s" % (field, minimum))
    return value


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def _exact(actual, expected, field):
    if actual != expected:
        raise EvidenceError("%s does not match immutable campaign identity" % field)


def _validate_metric(
    metric_name, raw, window, expected_identity, expected_scope, artifacts
):
    if not isinstance(raw, dict):
        raise EvidenceError("metrics.%s is missing" % metric_name)
    raw_payload = _artifact(
        artifacts,
        raw.get("rawArtifactId"),
        "metric-range-response",
        "metrics.%s.rawArtifactId" % metric_name,
    )
    _contains(
        raw_payload,
        {key: value for key, value in raw.items() if key != "rawArtifactId"},
        "metrics.%s" % metric_name,
    )
    _exact(
        raw.get("source"),
        REQUIRED_METRICS[metric_name],
        "metrics.%s.source" % metric_name,
    )
    metric = str(raw.get("metric") or "")
    if not metric:
        raise EvidenceError("metrics.%s.metric is missing" % metric_name)
    _exact(
        raw.get("identitySha256"),
        expected_identity,
        "metrics.%s.identitySha256" % metric_name,
    )
    _exact(
        raw.get("scopeSha256"),
        expected_scope,
        "metrics.%s.scopeSha256" % metric_name,
    )
    query_sha256 = str(raw.get("querySha256") or "")
    if len(query_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in query_sha256
    ):
        raise EvidenceError("metrics.%s.querySha256 is invalid" % metric_name)
    start = _finite(
        raw.get("startUnixSeconds"), "metrics.%s.startUnixSeconds" % metric_name
    )
    end = _finite(raw.get("endUnixSeconds"), "metrics.%s.endUnixSeconds" % metric_name)
    if start > window[0] or end < window[1]:
        raise EvidenceError(
            "metrics.%s does not cover the measurement window" % metric_name
        )
    delta = _finite(raw.get("delta"), "metrics.%s.delta" % metric_name, 0)
    if raw.get("counterReset") is not False:
        raise EvidenceError("metrics.%s must prove counterReset=false" % metric_name)
    samples = int(raw.get("samples") or 0)
    if samples < 2:
        raise EvidenceError("metrics.%s requires at least two samples" % metric_name)
    result = {"metric": metric, "delta": delta, "samples": samples}
    if raw.get("peakBitsPerSecond") is not None:
        result["peakBitsPerSecond"] = _finite(
            raw["peakBitsPerSecond"],
            "metrics.%s.peakBitsPerSecond" % metric_name,
            0,
        )
    return result


def _expected_run_index(manifest):
    runs = {}
    for run in manifest.get("sequence") or []:
        run_id = run.get("runId")
        if not run_id or run_id in runs:
            raise EvidenceError("campaign manifest has duplicate or missing runId")
        runs[run_id] = run
    if not runs:
        raise EvidenceError("campaign manifest contains no runs")
    return runs


def _validate_observed_location(
    observed, expected_location, expected_cell, cells, field
):
    if not isinstance(observed, dict):
        raise EvidenceError("%s placement is missing" % field)
    cell = cells[expected_cell]
    _exact(observed.get("cell"), expected_cell, field + ".cell")
    _exact(
        observed.get("clusterContext"),
        cell["clusterContext"],
        field + ".clusterContext",
    )
    _exact(
        observed.get("clusterIdentity"),
        cell["clusterIdentity"],
        field + ".clusterIdentity",
    )
    _exact(observed.get("nodeName"), expected_location["nodeName"], field + ".nodeName")
    _exact(observed.get("nodeUid"), expected_location["nodeUid"], field + ".nodeUid")
    _exact(
        observed.get("instanceType"),
        expected_location["instanceType"],
        field + ".instanceType",
    )
    _exact(
        observed.get("networkIdentity"),
        expected_location["networkIdentity"],
        field + ".networkIdentity",
    )
    image_id = str(observed.get("resolvedImageId") or "")
    if "@sha256:" not in image_id:
        raise EvidenceError(
            "%s.resolvedImageId must be an immutable runtime digest" % field
        )


def _metric_identity(metric_name, expected, relay, locations):
    if metric_name.startswith("relayVpc"):
        location = relay["location"]
        return locations[location]["networkIdentity"]["sha256"]
    if metric_name.startswith("relayNode") or metric_name.startswith("relayTcp"):
        location = relay["location"]
        return locations[location]["networkIdentity"]["sha256"]
    if metric_name.startswith("relay") or metric_name.startswith("rtp"):
        return relay["workloadIdentity"]["sha256"]
    if metric_name.startswith("lb"):
        return relay["loadBalancer"]["configSha256"]
    if (
        metric_name.startswith("publisherVpc")
        or metric_name.startswith("publisherNode")
        or metric_name.startswith("publisherTcp")
    ):
        location = expected["publisherLocation"]
        return locations[location]["networkIdentity"]["sha256"]
    location = expected["readerLocation"]
    return locations[location]["networkIdentity"]["sha256"]


def _validate_processor(manifest, expected, observed, prefix, window):
    if not isinstance(observed, dict):
        raise EvidenceError(prefix + ".processor evidence is missing")
    _exact(
        observed.get("workflow"),
        manifest["processorWorkflow"],
        prefix + ".processor.workflow",
    )
    _exact(
        observed.get("runtimeIdentity"),
        expected["processorRuntimeIdentity"],
        prefix + ".processor.runtimeIdentity",
    )
    _exact(
        observed.get("executionCell"),
        expected["executionCell"],
        prefix + ".processor.executionCell",
    )
    _exact(
        observed.get("sourcePlacementGeneration"),
        manifest["sourcePlacements"][expected["source"]]["placementGeneration"],
        prefix + ".processor.sourcePlacementGeneration",
    )
    location = manifest["locations"][expected["readerLocation"]]
    _exact(
        observed.get("nodeName"), location["nodeName"], prefix + ".processor.nodeName"
    )
    _exact(observed.get("nodeUid"), location["nodeUid"], prefix + ".processor.nodeUid")
    transport = {
        "same-cell": "cluster-internal",
        "cross-cell-ingest": "cluster-internal",
        "cross-cell-direct": "public-direct",
        "cross-cell-replica": "local-replica",
        "cross-cell-ingest-and-execution": "public-direct",
    }[expected["topology"]]
    _exact(
        observed.get("sourceTransport"),
        transport,
        prefix + ".processor.sourceTransport",
    )
    if observed.get("claimCount") != 1 or observed.get("attempt") != 1:
        raise EvidenceError(
            prefix + ".processor must prove exactly one first-attempt claim"
        )
    if observed.get("state") not in ("running", "completed", "cancelled"):
        raise EvidenceError(
            prefix + ".processor state is not a measured terminal/running state"
        )
    if (
        _finite(
            observed.get("framesProcessed"), prefix + ".processor.framesProcessed", 0
        )
        <= 0
    ):
        raise EvidenceError(prefix + ".processor must prove post-claim frame progress")
    first_result = _finite(
        observed.get("firstResultUnixSeconds"),
        prefix + ".processor.firstResultUnixSeconds",
    )
    if (
        first_result < manifest["identitySnapshot"]["capturedAtUnixSeconds"]
        or first_result > window[1]
    ):
        raise EvidenceError(
            prefix + ".processor first result is outside the run window"
        )
    for field in ("jobIdentitySha256", "claimResponseSha256"):
        value = str(observed.get(field) or "")
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise EvidenceError(prefix + ".processor.%s is invalid" % field)


def _validate_run_artifacts(observed, run_id, artifacts, prefix):
    publisher = _artifact(
        artifacts,
        observed.get("publisherReportArtifactId"),
        "publisher-termination-report",
        prefix + ".publisherReportArtifactId",
    )
    _contains(
        publisher,
        {"role": "publish-clock", "success": True},
        prefix + ".publisherReport",
    )
    if (publisher.get("identity") or {}).get("runId") != run_id:
        raise EvidenceError(prefix + ".publisherReport has the wrong run identity")

    probe = _artifact(
        artifacts,
        observed.get("probeReportArtifactId"),
        "probe-termination-report",
        prefix + ".probeReportArtifactId",
    )
    _contains(probe, {"role": "probe-clock", "success": True}, prefix + ".probeReport")
    if (probe.get("identity") or {}).get("runId") != run_id:
        raise EvidenceError(prefix + ".probeReport has the wrong run identity")
    measurement = observed.get("measurement") or {}
    _contains(
        probe,
        {
            key: measurement.get(key)
            for key in (
                "connectStartMonotonicNs",
                "firstDecodedFrameMonotonicNs",
                "decodedFrameArrivalMonotonicNs",
                "pixelClockLatencyMs",
                "latencySource",
                "encodedPayloadBytes",
                "encodedPayloadBytesSource",
                "measurementStartMonotonicNs",
                "measurementEndMonotonicNs",
            )
        },
        prefix + ".probeReport",
    )

    processor = _artifact(
        artifacts,
        observed.get("processorApiArtifactId"),
        "processor-api-response",
        prefix + ".processorApiArtifactId",
    )
    _contains(processor, observed.get("processor") or {}, prefix + ".processorApi")
    placement = _artifact(
        artifacts,
        observed.get("placementApiArtifactId"),
        "placement-api-response",
        prefix + ".placementApiArtifactId",
    )
    _contains(placement, observed.get("placement") or {}, prefix + ".placementApi")


def _validate_run(manifest, expected, observed, artifacts):
    run_id = expected["runId"]
    prefix = "runs.%s" % run_id
    if observed.get("observationStatus") != "collected":
        raise EvidenceError(prefix + " is not marked observationStatus=collected")
    for key in (
        "pairId",
        "repetition",
        "arm",
        "topology",
        "source",
        "relayCell",
        "executionCell",
    ):
        _exact(observed.get(key), expected.get(key), "%s.%s" % (prefix, key))
    cells = manifest["cells"]
    locations = manifest["locations"]
    source = manifest["sourcePlacements"][expected["source"]]
    relay = cells[expected["relayCell"]]["relay"]
    placement = observed.get("placement") or {}
    _exact(
        placement.get("sourceIdentitySha256"),
        source["sourceIdentitySha256"],
        prefix + ".placement.sourceIdentitySha256",
    )
    _exact(
        placement.get("homeCell"), source["homeCell"], prefix + ".placement.homeCell"
    )
    _exact(
        placement.get("relayShard"),
        source["relayShard"],
        prefix + ".placement.relayShard",
    )
    _exact(
        placement.get("placementGeneration"),
        source["placementGeneration"],
        prefix + ".placement.placementGeneration",
    )
    _exact(
        placement.get("recordRevision"),
        source["recordRevision"],
        prefix + ".placement.recordRevision",
    )
    _exact(
        placement.get("executionCell"),
        expected["executionCell"],
        prefix + ".placement.executionCell",
    )
    if placement.get("remoteExecution") is not (
        expected["executionCell"] != expected["relayCell"]
    ):
        raise EvidenceError(prefix + ".placement.remoteExecution is incorrect")

    observed_relay = observed.get("relay") or {}
    for key in ("shard", "workloadIdentity", "configSha256", "image", "loadBalancer"):
        _exact(observed_relay.get(key), relay[key], prefix + ".relay." + key)
    if expected["topology"] == "cross-cell-replica":
        _exact(
            observed.get("replicaIdentity"),
            expected["replicaIdentity"],
            prefix + ".replicaIdentity",
        )
    elif observed.get("replicaIdentity") is not None:
        raise EvidenceError(prefix + ".replicaIdentity must be absent")
    _exact(
        observed.get("processorRuntimeIdentity"),
        expected["processorRuntimeIdentity"],
        prefix + ".processorRuntimeIdentity",
    )

    _validate_observed_location(
        observed.get("publisher"),
        locations[expected["publisherLocation"]],
        locations[expected["publisherLocation"]]["cell"],
        cells,
        prefix + ".publisher",
    )
    _validate_observed_location(
        observed.get("reader"),
        locations[expected["readerLocation"]],
        locations[expected["readerLocation"]]["cell"],
        cells,
        prefix + ".reader",
    )

    window = observed.get("measurementWindow") or {}
    start = _finite(
        window.get("startUnixSeconds"), prefix + ".measurementWindow.startUnixSeconds"
    )
    end = _finite(
        window.get("endUnixSeconds"), prefix + ".measurementWindow.endUnixSeconds"
    )
    duration = end - start
    snapshot = manifest.get("identitySnapshot") or {}
    if start < snapshot.get(
        "capturedAtUnixSeconds", float("inf")
    ) or end > snapshot.get("expiresAtUnixSeconds", float("-inf")):
        raise EvidenceError(
            prefix + ".measurementWindow falls outside identity snapshot validity"
        )
    if duration < manifest["timing"]["measureSeconds"]:
        raise EvidenceError(
            prefix + ".measurementWindow is shorter than the planned measurement"
        )
    if window.get("warmupComplete") is not True:
        raise EvidenceError(
            prefix + ".measurementWindow must prove warmupComplete=true"
        )
    _validate_processor(
        manifest,
        expected,
        observed.get("processor"),
        prefix,
        (start, end),
    )

    measurement = observed.get("measurement") or {}
    connect_ns = _finite(
        measurement.get("connectStartMonotonicNs"),
        prefix + ".connectStartMonotonicNs",
        0,
    )
    first_ns = _finite(
        measurement.get("firstDecodedFrameMonotonicNs"),
        prefix + ".firstDecodedFrameMonotonicNs",
        0,
    )
    if first_ns < connect_ns:
        raise EvidenceError(prefix + " first frame precedes connection start")
    ttff_ms = (first_ns - connect_ns) / 1e6
    arrivals = [
        _finite(value, prefix + ".arrivalMonotonicNs", 0)
        for value in measurement.get("decodedFrameArrivalMonotonicNs") or []
    ]
    if len(arrivals) < 2 or arrivals != sorted(arrivals):
        raise EvidenceError(prefix + " requires ordered per-frame monotonic arrivals")
    if arrivals[0] < first_ns:
        raise EvidenceError(
            prefix + " arrival series begins before first decoded frame"
        )
    measurement_start_ns = _finite(
        measurement.get("measurementStartMonotonicNs"),
        prefix + ".measurementStartMonotonicNs",
        0,
    )
    measurement_end_ns = _finite(
        measurement.get("measurementEndMonotonicNs"),
        prefix + ".measurementEndMonotonicNs",
        0,
    )
    if (
        measurement_end_ns <= measurement_start_ns
        or arrivals[0] < measurement_start_ns
        or arrivals[-1] > measurement_end_ns
    ):
        raise EvidenceError(prefix + " monotonic measurement bounds are invalid")
    monotonic_duration = (measurement_end_ns - measurement_start_ns) / 1e9
    if abs(monotonic_duration - duration) / duration > 0.02:
        raise EvidenceError(
            prefix + " wall and monotonic measurement durations disagree"
        )
    latency = [
        _finite(value, prefix + ".pixelClockLatencyMs", 0)
        for value in measurement.get("pixelClockLatencyMs") or []
    ]
    if len(latency) != len(arrivals):
        raise EvidenceError(
            prefix + " needs one pixel-clock latency sample per decoded arrival"
        )
    _exact(
        measurement.get("latencySource"), "pixel-clock-v1", prefix + ".latencySource"
    )
    clock = measurement.get("clockSync") or {}
    if clock.get("source") not in ("chrony", "ntp", "ptp"):
        raise EvidenceError(prefix + ".clockSync.source is not authoritative")
    clock_uncertainty = max(
        _finite(
            clock.get("publisherUncertaintyMs"), prefix + ".publisherUncertaintyMs", 0
        ),
        _finite(clock.get("readerUncertaintyMs"), prefix + ".readerUncertaintyMs", 0),
    )
    if clock.get("checkedBeforeAndAfter") is not True:
        raise EvidenceError(
            prefix + " clock synchronization must be checked before and after"
        )

    expected_interval_ms = 1000.0 / manifest["fixture"]["fps"]
    bounded_arrivals = [measurement_start_ns, *arrivals, measurement_end_ns]
    excess_gaps = [
        max(0.0, (right - left) / 1e6 - expected_interval_ms)
        for left, right in zip(bounded_arrivals, bounded_arrivals[1:])
    ]
    stalls = [gap for gap in excess_gaps if gap > manifest["thresholds"]["maxStallMs"]]
    measured_frames = len(arrivals)
    delivered_fps = measured_frames / monotonic_duration
    delivered_ratio = delivered_fps / manifest["fixture"]["fps"]
    encoded_bytes = _finite(
        measurement.get("encodedPayloadBytes"), prefix + ".encodedPayloadBytes", 0
    )
    _exact(
        measurement.get("encodedPayloadBytesSource"),
        "pyav-packet-size",
        prefix + ".encodedPayloadBytesSource",
    )
    _validate_run_artifacts(observed, run_id, artifacts, prefix)
    throughput_bps = encoded_bytes * 8 / duration

    metrics = {}
    for metric_name in REQUIRED_METRICS:
        identity = _metric_identity(metric_name, expected, relay, locations)
        metrics[metric_name] = _validate_metric(
            metric_name,
            (observed.get("metrics") or {}).get(metric_name),
            (start, end),
            identity,
            expected["streamIdentitySha256"],
            artifacts,
        )
    received_packets = metrics["rtpPacketsReceived"]["delta"]
    lost_packets = metrics["rtpPacketsLost"]["delta"]
    packet_loss_ratio = (
        lost_packets / (received_packets + lost_packets)
        if received_packets + lost_packets
        else 0.0
    )
    if received_packets <= 0:
        raise EvidenceError(prefix + " did not observe authoritative RTP packets")
    publisher_vpc_peak = max(
        metrics["publisherVpcRxBytes"].get("peakBitsPerSecond", 0),
        metrics["publisherVpcTxBytes"].get("peakBitsPerSecond", 0),
    )
    relay_vpc_peak = max(
        metrics["relayVpcRxBytes"].get("peakBitsPerSecond", 0),
        metrics["relayVpcTxBytes"].get("peakBitsPerSecond", 0),
    )
    reader_vpc_peak = max(
        metrics["readerVpcRxBytes"].get("peakBitsPerSecond", 0),
        metrics["readerVpcTxBytes"].get("peakBitsPerSecond", 0),
    )
    lb_peak = max(
        metrics["lbBytesIn"].get("peakBitsPerSecond", 0),
        metrics["lbBytesOut"].get("peakBitsPerSecond", 0),
    )
    publisher_vpc_limit = locations[expected["publisherLocation"]][
        "documentedVpcBandwidthBps"
    ]
    relay_vpc_limit = locations[relay["location"]]["documentedVpcBandwidthBps"]
    reader_vpc_limit = locations[expected["readerLocation"]][
        "documentedVpcBandwidthBps"
    ]
    lb_limit = relay["loadBalancer"]["bandwidthCeilingBps"]

    thresholds = manifest["thresholds"]
    gates = {
        "agentSuccess": observed.get("success") is True,
        "ttff": ttff_ms <= thresholds["maxTtffMs"],
        "stalls": len(stalls) <= thresholds["maxStalls"],
        "deliveredFps": delivered_ratio >= thresholds["minDeliveredFpsRatio"],
        "packetLoss": packet_loss_ratio <= thresholds["maxPacketLossRatio"],
        "inputErrors": metrics["rtpPacketsInError"]["delta"] == 0,
        "nodeDrops": all(
            metrics[name]["delta"] == 0
            for name in (
                "publisherNodePacketDrops",
                "relayNodePacketDrops",
                "readerNodePacketDrops",
            )
        ),
        "clockSync": clock_uncertainty <= thresholds["maxClockUncertaintyMs"],
        "latency": _percentile(latency, 0.95) <= thresholds["maxLatencyP95Ms"],
        "publisherVpcHeadroom": publisher_vpc_peak / publisher_vpc_limit
        <= thresholds["maxVpcUtilizationRatio"],
        "relayVpcHeadroom": relay_vpc_peak / relay_vpc_limit
        <= thresholds["maxVpcUtilizationRatio"],
        "readerVpcHeadroom": reader_vpc_peak / reader_vpc_limit
        <= thresholds["maxVpcUtilizationRatio"],
        "lbHeadroom": lb_peak / lb_limit <= thresholds["maxLbUtilizationRatio"],
    }
    return {
        "runId": run_id,
        "pairId": expected["pairId"],
        "repetition": expected["repetition"],
        "arm": expected["arm"],
        "topology": expected["topology"],
        "durationSeconds": duration,
        "ttffMs": round(ttff_ms, 3),
        "deliveredFps": round(delivered_fps, 3),
        "deliveredFpsRatio": round(delivered_ratio, 5),
        "encodedThroughputBps": round(throughput_bps, 3),
        "latencyP50Ms": round(_percentile(latency, 0.50), 3),
        "latencyP95Ms": round(_percentile(latency, 0.95), 3),
        "latencyP99Ms": round(_percentile(latency, 0.99), 3),
        "clockUncertaintyMs": round(clock_uncertainty, 3),
        "stallCount": len(stalls),
        "maxExcessInterFrameGapMs": round(max(excess_gaps) if excess_gaps else 0, 3),
        "packetLossRatio": round(packet_loss_ratio, 8),
        "tcpRetransmits": sum(
            metrics[name]["delta"]
            for name in (
                "publisherTcpRetransmits",
                "relayTcpRetransmits",
                "readerTcpRetransmits",
            )
        ),
        "publisherVpcPeakBitsPerSecond": publisher_vpc_peak,
        "publisherVpcUtilizationRatio": round(
            publisher_vpc_peak / publisher_vpc_limit, 5
        ),
        "relayVpcPeakBitsPerSecond": relay_vpc_peak,
        "relayVpcUtilizationRatio": round(relay_vpc_peak / relay_vpc_limit, 5),
        "readerVpcPeakBitsPerSecond": reader_vpc_peak,
        "readerVpcUtilizationRatio": round(reader_vpc_peak / reader_vpc_limit, 5),
        "lbPeakBitsPerSecond": lb_peak,
        "lbUtilizationRatio": round(lb_peak / lb_limit, 5),
        "gates": gates,
        "passed": all(gates.values()),
    }


def _validate_placement_cases(manifest, evidence, artifacts):
    expected_cases = {case["id"]: case for case in manifest.get("placementCases") or []}
    observed_cases = {}
    for case in evidence.get("placementCases") or []:
        case_id = case.get("id")
        if not case_id or case_id in observed_cases:
            raise EvidenceError("placement evidence has duplicate or missing case id")
        observed_cases[case_id] = case
    _exact(set(observed_cases), set(expected_cases), "placement case coverage")
    results = []
    for case_id, expected in expected_cases.items():
        observed = observed_cases[case_id]
        if observed.get("observationStatus") != "collected":
            raise EvidenceError(
                "placement case %s is not marked observationStatus=collected" % case_id
            )
        _exact(
            observed.get("source"),
            expected["source"],
            "placementCases.%s.source" % case_id,
        )
        source = manifest["sourcePlacements"][expected["source"]]
        observations = observed.get("observations") or []
        if len(observations) != len(expected["actions"]):
            raise EvidenceError(
                "placement case %s has incomplete action coverage" % case_id
            )
        generation = source["placementGeneration"]
        for action, actual in zip(expected["actions"], observations):
            raw_payload = _artifact(
                artifacts,
                actual.get("rawArtifactId"),
                "placement-api-response",
                "placementCases.%s.rawArtifactId" % case_id,
            )
            _contains(
                raw_payload,
                {key: value for key, value in actual.items() if key != "rawArtifactId"},
                "placementCases.%s" % case_id,
            )
            for key in ("ordinal", "action", "expectedOutcome"):
                expected_value = action[key]
                actual_key = "outcome" if key == "expectedOutcome" else key
                _exact(
                    actual.get(actual_key),
                    expected_value,
                    "placementCases.%s.%s" % (case_id, actual_key),
                )
            _exact(
                actual.get("requestedExecutionCell"),
                action["requestedExecutionCell"],
                "placement requested cell",
            )
            if action["expectedOutcome"] == "rejected":
                _exact(
                    actual.get("reasonCode"),
                    action["expectedReasonCode"],
                    "placement rejection reason",
                )
                if (
                    actual.get("executionCell") is not None
                    or actual.get("jobClaimed") is not False
                ):
                    raise EvidenceError(
                        "rejected placement must prove no execution and no claim"
                    )
            else:
                _exact(
                    actual.get("executionCell"),
                    action["expectedExecutionCell"],
                    "placement execution cell",
                )
                _exact(
                    actual.get("homeCell"), source["homeCell"], "placement home cell"
                )
                _exact(
                    actual.get("relayShard"),
                    source["relayShard"],
                    "placement relay shard",
                )
                _exact(
                    actual.get("placementGeneration"),
                    generation,
                    "sticky placement generation",
                )
                if (
                    actual.get("jobClaimed") is not True
                    and action["action"] != "activate-preview"
                ):
                    raise EvidenceError("accepted job placement must prove a claim")
        results.append({"id": case_id, "passed": True, "actions": len(observations)})
    return results


def analyze(manifest, evidence, artifacts):
    if manifest.get("schemaVersion") != 1 or evidence.get("schemaVersion") != 1:
        raise EvidenceError("manifest and evidence schemaVersion must be 1")
    _exact(evidence.get("environment"), "staging", "evidence.environment")
    _exact(evidence.get("campaign"), manifest.get("campaign"), "evidence.campaign")
    _exact(
        evidence.get("campaignSha256"),
        manifest.get("campaignSha256"),
        "evidence.campaignSha256",
    )
    _exact(
        evidence.get("executionId"),
        manifest.get("executionId"),
        "evidence.executionId",
    )
    expected_runs = _expected_run_index(manifest)
    observed_runs = {}
    for run in evidence.get("runs") or []:
        run_id = run.get("runId")
        if not run_id or run_id in observed_runs:
            raise EvidenceError("evidence has duplicate or missing runId")
        observed_runs[run_id] = run
    _exact(set(observed_runs), set(expected_runs), "run coverage")
    runs = [
        _validate_run(manifest, expected_runs[run_id], observed_runs[run_id], artifacts)
        for run_id in expected_runs
    ]
    placement = _validate_placement_cases(manifest, evidence, artifacts)

    paired = []
    for pair_id in sorted({run["pairId"] for run in runs}):
        pair_runs = [run for run in runs if run["pairId"] == pair_id]
        repetitions = sorted({run["repetition"] for run in pair_runs})
        if repetitions != list(range(1, manifest["repetitions"] + 1)):
            raise EvidenceError("pair %s has incomplete repetitions" % pair_id)
        deltas = []
        for repetition in repetitions:
            rep = [run for run in pair_runs if run["repetition"] == repetition]
            by_arm = {run["arm"]: run for run in rep}
            if set(by_arm) != {"control", "variant"} or len(rep) != 2:
                raise EvidenceError(
                    "pair %s repetition %d is not a complete pair"
                    % (pair_id, repetition)
                )
            control = by_arm["control"]
            variant = by_arm["variant"]
            deltas.append(
                {
                    "repetition": repetition,
                    "latencyP95DeltaMs": variant["latencyP95Ms"]
                    - control["latencyP95Ms"],
                    "latencyP95RegressionRatio": (
                        variant["latencyP95Ms"] / control["latencyP95Ms"] - 1
                        if control["latencyP95Ms"]
                        else float("inf")
                    ),
                    "ttffDeltaMs": variant["ttffMs"] - control["ttffMs"],
                    "fpsRatioDelta": variant["deliveredFpsRatio"]
                    - control["deliveredFpsRatio"],
                }
            )
        regression = statistics.median(
            item["latencyP95RegressionRatio"] for item in deltas
        )
        gate = regression <= manifest["thresholds"]["maxPairedLatencyRegressionRatio"]
        paired.append(
            {
                "pairId": pair_id,
                "repetitions": len(deltas),
                "medianLatencyP95RegressionRatio": round(regression, 5),
                "medianLatencyP95DeltaMs": round(
                    statistics.median(item["latencyP95DeltaMs"] for item in deltas), 3
                ),
                "medianTtffDeltaMs": round(
                    statistics.median(item["ttffDeltaMs"] for item in deltas), 3
                ),
                "medianFpsRatioDelta": round(
                    statistics.median(item["fpsRatioDelta"] for item in deltas), 5
                ),
                "pairedLatencyGate": gate,
                "passed": gate and all(run["passed"] for run in pair_runs),
                "deltas": deltas,
            }
        )
    passed = all(run["passed"] for run in runs) and all(
        item["passed"] for item in paired
    )
    return {
        "schemaVersion": 1,
        "campaign": manifest["campaign"],
        "campaignSha256": manifest["campaignSha256"],
        "runs": runs,
        "placementCases": placement,
        "pairs": paired,
        "certified": passed,
        "certificationBoundary": (
            "paired staging evidence only; no production or untested topology claim"
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        evidence = _load(args.evidence)
        report = analyze(
            _load(args.manifest),
            evidence,
            load_artifacts(evidence, args.artifact_root),
        )
        output = Path(args.output).resolve()
        if output.exists():
            raise EvidenceError("refusing to overwrite %s" % output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(output)
    return 0 if report["certified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
