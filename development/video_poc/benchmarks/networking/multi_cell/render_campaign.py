#!/usr/bin/env python3
"""Render a fail-closed, staging-only multi-cell media benchmark campaign.

The renderer is deliberately read-only.  It validates immutable cell, relay,
network, source-placement, and workload identities, then writes Kubernetes Job
lists and a credential-free campaign manifest for operator review.  It never
invokes kubectl, a cloud API, or the video control plane.
"""

import argparse
import hashlib
import json
import platform
import re
import sys
import time
from pathlib import Path

SHA256 = re.compile(r"^[0-9a-f]{64}$")
IMAGE_DIGEST = re.compile(r"^\S+@sha256:[0-9a-f]{64}$")
DNS_LABEL = re.compile(r"^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$")
STAGING_CONTEXT = re.compile(r"(?:^|[-_.])(stg|staging)(?:$|[-_.])", re.I)
PLACEHOLDER = re.compile(r"(?:replace|todo|tbd|unknown|example|0{12})", re.I)
ROLES = ("publish-clock", "probe-clock", "probe-copy")
MANAGED_BY = "video-multi-cell-benchmark"
MAX_RENDERED_JOBS = 2000


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _required_string(value, field, allow_placeholder=False):
    value = str(value or "").strip()
    if not value:
        raise ValueError("%s must be non-empty" % field)
    if not allow_placeholder and PLACEHOLDER.search(value):
        raise ValueError("%s contains a placeholder" % field)
    return value


def _dns(value, field):
    value = _required_string(value, field)
    if len(value) > 63 or not DNS_LABEL.fullmatch(value):
        raise ValueError("%s must be a lowercase Kubernetes DNS label" % field)
    return value


def _digest(value, field):
    value = _required_string(value, field)
    if not SHA256.fullmatch(value) or value == "0" * 64:
        raise ValueError("%s must be a non-zero sha256 hex digest" % field)
    return value


def _positive(value, field, integer=False):
    try:
        parsed = int(value) if integer else float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("%s must be numeric" % field) from error
    if parsed <= 0:
        raise ValueError("%s must be positive" % field)
    return parsed


def _ratio(value, field, allow_zero=False):
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("%s must be numeric" % field) from error
    lower_ok = parsed >= 0 if allow_zero else parsed > 0
    if not lower_ok or parsed > 1:
        raise ValueError(
            "%s must be in %s" % (field, "[0, 1]" if allow_zero else "(0, 1]")
        )
    return parsed


def _secret_ref(value, field):
    if not isinstance(value, dict):
        raise ValueError("%s must be a secretKeyRef object" % field)
    return {
        "name": _dns(value.get("name"), field + ".name"),
        "key": _required_string(value.get("key"), field + ".key"),
    }


def _staging_context(value, field):
    value = _required_string(value, field)
    if not STAGING_CONTEXT.search(value) or re.search(r"prod(?:uction)?", value, re.I):
        raise ValueError("%s must be recognizably staging and not production" % field)
    return value


def _identity(raw, field):
    if not isinstance(raw, dict):
        raise ValueError("%s must be an object" % field)
    return {
        "uid": _required_string(raw.get("uid"), field + ".uid"),
        "revision": _required_string(raw.get("revision"), field + ".revision"),
        "sha256": _digest(raw.get("sha256"), field + ".sha256"),
    }


def _location(raw, field, cells):
    if not isinstance(raw, dict):
        raise ValueError("%s must be an object" % field)
    cell = _dns(raw.get("cell"), field + ".cell")
    if cell not in cells:
        raise ValueError("%s references unknown cell %s" % (field, cell))
    # Selectors are deliberately insufficient for certification: a mutable
    # selector can silently land on another node class between paired arms.
    node_name = _required_string(raw.get("nodeName"), field + ".nodeName")
    return {
        "cell": cell,
        "nodeName": node_name,
        "nodeUid": _required_string(raw.get("nodeUid"), field + ".nodeUid"),
        "instanceType": _required_string(
            raw.get("instanceType"), field + ".instanceType"
        ),
        "documentedVpcBandwidthBps": _positive(
            raw.get("documentedVpcBandwidthBps"),
            field + ".documentedVpcBandwidthBps",
            integer=True,
        ),
        "networkIdentity": _identity(
            raw.get("networkIdentity"), field + ".networkIdentity"
        ),
        "tolerations": raw.get("tolerations") or [],
    }


def load_campaign(path):
    campaign_path = Path(path).resolve()
    with campaign_path.open() as source:
        raw = json.load(source)
    if raw.get("schemaVersion") != 1:
        raise ValueError("campaign schemaVersion must be 1")
    if raw.get("environment") != "staging":
        raise ValueError("multi-cell campaigns are restricted to staging")
    snapshot = raw.get("identitySnapshot") or {}
    if snapshot.get("status") != "captured":
        raise ValueError(
            "identitySnapshot.status must be captured; committed templates are not runnable"
        )
    captured_at = _positive(
        snapshot.get("capturedAtUnixSeconds"),
        "identitySnapshot.capturedAtUnixSeconds",
    )
    expires_at = _positive(
        snapshot.get("expiresAtUnixSeconds"),
        "identitySnapshot.expiresAtUnixSeconds",
    )
    now = time.time()
    if captured_at > now + 60 or expires_at <= now:
        raise ValueError("identitySnapshot must be current and unexpired")
    if expires_at - captured_at > 6 * 60 * 60:
        raise ValueError("identitySnapshot validity cannot exceed six hours")
    normalized_snapshot = {
        "status": "captured",
        "capturedAtUnixSeconds": captured_at,
        "expiresAtUnixSeconds": expires_at,
        "sourceSha256": _digest(
            snapshot.get("sourceSha256"), "identitySnapshot.sourceSha256"
        ),
        "capturedBy": _identity(
            snapshot.get("capturedBy"), "identitySnapshot.capturedBy"
        ),
    }
    name = _dns(raw.get("name"), "name")
    namespace = _dns(raw.get("namespace"), "namespace")
    if "bench" not in namespace:
        raise ValueError("namespace must be a dedicated benchmark namespace")
    image = _required_string(raw.get("agentImage"), "agentImage")
    if not IMAGE_DIGEST.fullmatch(image):
        raise ValueError("agentImage must use an immutable sha256 digest")
    agent_command = raw.get("agentCommand") or [
        "python",
        "/opt/roboflow/multi_cell_probe.py",
    ]
    if not isinstance(agent_command, list) or not agent_command:
        raise ValueError("agentCommand must be a non-empty argument vector")

    cells = {}
    contexts = set()
    for key, value in (raw.get("cells") or {}).items():
        key = _dns(key, "cell key")
        context = _staging_context(
            value.get("clusterContext"), "cells.%s.clusterContext" % key
        )
        if context in contexts:
            raise ValueError("each cell must use a distinct clusterContext")
        contexts.add(context)
        namespace_value = _dns(
            value.get("namespace", namespace), "cells.%s.namespace" % key
        )
        if "bench" not in namespace_value:
            raise ValueError("cell namespace must be dedicated to benchmarks")
        relay = value.get("relay") or {}
        lb = relay.get("loadBalancer") or {}
        cells[key] = {
            "id": _required_string(value.get("id"), "cells.%s.id" % key),
            "exclusiveBenchmarkCell": value.get("exclusiveBenchmarkCell") is True,
            "provider": _required_string(
                value.get("provider"), "cells.%s.provider" % key
            ),
            "region": _required_string(value.get("region"), "cells.%s.region" % key),
            "clusterContext": context,
            "clusterIdentity": _identity(
                value.get("clusterIdentity"), "cells.%s.clusterIdentity" % key
            ),
            "namespace": namespace_value,
            "relay": {
                "shard": _required_string(
                    relay.get("shard"), "cells.%s.relay.shard" % key
                ),
                "location": _dns(
                    relay.get("location"), "cells.%s.relay.location" % key
                ),
                "workloadIdentity": _identity(
                    relay.get("workloadIdentity"),
                    "cells.%s.relay.workloadIdentity" % key,
                ),
                "configSha256": _digest(
                    relay.get("configSha256"), "cells.%s.relay.configSha256" % key
                ),
                "image": _required_string(
                    relay.get("image"), "cells.%s.relay.image" % key
                ),
                "loadBalancer": {
                    "id": _required_string(
                        lb.get("id"), "cells.%s.relay.loadBalancer.id" % key
                    ),
                    "configSha256": _digest(
                        lb.get("configSha256"),
                        "cells.%s.relay.loadBalancer.configSha256" % key,
                    ),
                    "bandwidthCeilingBps": _positive(
                        lb.get("bandwidthCeilingBps"),
                        "cells.%s.relay.loadBalancer.bandwidthCeilingBps" % key,
                        integer=True,
                    ),
                    "limitSource": _required_string(
                        lb.get("limitSource"),
                        "cells.%s.relay.loadBalancer.limitSource" % key,
                    ),
                },
                "publishUrl": _secret_ref(
                    relay.get("publishUrl"), "cells.%s.relay.publishUrl" % key
                ),
                "internalReadUrl": _secret_ref(
                    relay.get("internalReadUrl"),
                    "cells.%s.relay.internalReadUrl" % key,
                ),
                "externalReadUrl": _secret_ref(
                    relay.get("externalReadUrl"),
                    "cells.%s.relay.externalReadUrl" % key,
                ),
            },
        }
        if not cells[key]["exclusiveBenchmarkCell"]:
            raise ValueError(
                "cells.%s.exclusiveBenchmarkCell must be true for aggregate network evidence"
                % key
            )
        if not IMAGE_DIGEST.fullmatch(cells[key]["relay"]["image"]):
            raise ValueError(
                "cells.%s.relay.image must use an immutable sha256 digest" % key
            )
    if len(cells) < 2:
        raise ValueError("campaign must define at least two staging cells")

    locations = {
        key: _location(value, "locations.%s" % key, cells)
        for key, value in (raw.get("locations") or {}).items()
    }
    if not locations:
        raise ValueError("campaign must define immutable locations")
    for key, cell in cells.items():
        relay_location = cell["relay"]["location"]
        if relay_location not in locations:
            raise ValueError("cell %s relay references unknown location" % key)
        if locations[relay_location]["cell"] != key:
            raise ValueError("cell %s relay location must be in that cell" % key)

    fixture = raw.get("fixture") or {}
    fixture_source = fixture.get("source") or {}
    if "value" in fixture_source:
        source_value = _required_string(fixture_source["value"], "fixture.source.value")
        if "://" in source_value and ("?" in source_value or "@" in source_value):
            raise ValueError("credentialed fixture URLs must use a secretKeyRef")
        normalized_source = {"value": source_value}
    else:
        normalized_source = {
            "secretKeyRef": _secret_ref(
                fixture_source.get("secretKeyRef"), "fixture.source.secretKeyRef"
            )
        }
    normalized_fixture = {
        "id": _required_string(fixture.get("id"), "fixture.id"),
        "sha256": _digest(fixture.get("sha256"), "fixture.sha256"),
        "codec": _required_string(fixture.get("codec"), "fixture.codec"),
        "width": _positive(fixture.get("width"), "fixture.width", integer=True),
        "height": _positive(fixture.get("height"), "fixture.height", integer=True),
        "fps": _positive(fixture.get("fps"), "fixture.fps"),
        "bitrateBps": _positive(fixture.get("bitrateBps"), "fixture.bitrateBps"),
        "clockEncoding": _required_string(
            fixture.get("clockEncoding"), "fixture.clockEncoding"
        ),
        "source": normalized_source,
    }
    if normalized_fixture["clockEncoding"] != "pixel-clock-v1":
        raise ValueError("fixture.clockEncoding must be pixel-clock-v1")

    workflow = raw.get("processorWorkflow") or {}
    tier = str(workflow.get("tier") or "")
    if tier not in ("cpu", "gpu"):
        raise ValueError("processorWorkflow.tier must be cpu or gpu")
    normalized_workflow = {
        "id": _required_string(workflow.get("id"), "processorWorkflow.id"),
        "specificationSha256": _digest(
            workflow.get("specificationSha256"),
            "processorWorkflow.specificationSha256",
        ),
        "runtimeInputsSha256": _digest(
            workflow.get("runtimeInputsSha256"),
            "processorWorkflow.runtimeInputsSha256",
        ),
        "tier": tier,
        "targetFps": _positive(
            workflow.get("targetFps"), "processorWorkflow.targetFps"
        ),
        "outputPublishing": bool(workflow.get("outputPublishing", False)),
    }

    policies = {}
    for key, value in (raw.get("workspacePolicies") or {}).items():
        key = _dns(key, "workspace policy key")
        allowed = [
            _dns(cell, "workspace policy allowed cell")
            for cell in value.get("allowedCells") or []
        ]
        if not allowed or any(cell not in cells for cell in allowed):
            raise ValueError("workspace policy %s has invalid allowedCells" % key)
        mode = str(value.get("mode") or "")
        if mode not in ("shared", "dedicated"):
            raise ValueError("workspace policy mode must be shared or dedicated")
        policies[key] = {
            "mode": mode,
            "allowedCells": allowed,
            "preferredCells": [
                _dns(cell, "workspace policy preferred cell")
                for cell in value.get("preferredCells") or []
            ],
            "allowRemoteExecution": bool(value.get("allowRemoteExecution", False)),
            "allowSharedFallback": bool(value.get("allowSharedFallback", False)),
            "revision": _required_string(
                value.get("revision"), "workspace policy revision"
            ),
            "sha256": _digest(value.get("sha256"), "workspace policy sha256"),
        }
        if any(cell not in allowed for cell in policies[key]["preferredCells"]):
            raise ValueError("preferredCells must be a subset of allowedCells")
        if mode == "dedicated" and policies[key]["allowSharedFallback"]:
            raise ValueError(
                "dedicated policy must fail closed without shared fallback"
            )
    if not policies:
        raise ValueError("campaign must define workspace placement policies")

    sources = {}
    for key, value in (raw.get("sourcePlacements") or {}).items():
        key = _dns(key, "source placement key")
        policy = _dns(value.get("workspacePolicy"), "source workspacePolicy")
        if policy not in policies:
            raise ValueError("source %s references unknown workspace policy" % key)
        home_cell = _dns(value.get("homeCell"), "source homeCell")
        if home_cell not in policies[policy]["allowedCells"]:
            raise ValueError("source homeCell is not allowed by workspace policy")
        sources[key] = {
            "sourceIdentitySha256": _digest(
                value.get("sourceIdentitySha256"), "source sourceIdentitySha256"
            ),
            "workspacePolicy": policy,
            "homeCell": home_cell,
            "relayShard": _required_string(
                value.get("relayShard"), "source relayShard"
            ),
            "placementGeneration": _positive(
                value.get("placementGeneration"),
                "source placementGeneration",
                integer=True,
            ),
            "recordRevision": _required_string(
                value.get("recordRevision"), "source recordRevision"
            ),
            "connectorIdentity": _identity(
                value.get("connectorIdentity"), "source connectorIdentity"
            ),
        }
        if sources[key]["relayShard"] != cells[home_cell]["relay"]["shard"]:
            raise ValueError("source relayShard must match its home cell")
    if not sources:
        raise ValueError("campaign must define source placements")

    placement_cases = []
    placement_case_ids = set()
    for case in raw.get("placementCases") or []:
        case_id = _dns(case.get("id"), "placementCase.id")
        if case_id in placement_case_ids:
            raise ValueError("placement case ids must be unique")
        placement_case_ids.add(case_id)
        source = _dns(case.get("source"), "placementCase.source")
        if source not in sources:
            raise ValueError("placement case references unknown source")
        source_placement = sources[source]
        source_policy = policies[source_placement["workspacePolicy"]]
        actions = []
        for ordinal, action in enumerate(case.get("actions") or [], 1):
            kind = str(action.get("action") or "")
            if kind not in (
                "activate-preview",
                "activate-job",
                "reactivate-after-idle",
                "request-remote-job",
            ):
                raise ValueError("unsupported placement action: %s" % kind)
            outcome = str(action.get("expectedOutcome") or "")
            if outcome not in ("assigned", "reused", "rejected"):
                raise ValueError("unsupported placement outcome: %s" % outcome)
            requested_cell = action.get("requestedExecutionCell")
            if requested_cell is not None:
                requested_cell = _dns(
                    requested_cell, "placementCase.requestedExecutionCell"
                )
                if requested_cell not in cells:
                    raise ValueError("placement action requests unknown cell")
            expected_cell = action.get("expectedExecutionCell")
            if expected_cell is not None:
                expected_cell = _dns(
                    expected_cell, "placementCase.expectedExecutionCell"
                )
                if expected_cell not in cells:
                    raise ValueError("placement action expects unknown cell")
            reason = str(action.get("expectedReasonCode") or "") or None
            if outcome == "rejected" and (expected_cell is not None or not reason):
                raise ValueError(
                    "rejected placement action requires only a reason code"
                )
            if outcome != "rejected" and expected_cell is None:
                raise ValueError(
                    "accepted placement action requires expectedExecutionCell"
                )
            if kind in (
                "activate-preview",
                "activate-job",
                "reactivate-after-idle",
            ):
                if (
                    outcome == "rejected"
                    or expected_cell != source_placement["homeCell"]
                ):
                    raise ValueError(
                        "ordinary activation must reuse the source home cell"
                    )
                if kind == "reactivate-after-idle" and outcome != "reused":
                    raise ValueError("idle reactivation must prove sticky reuse")
            else:
                if requested_cell is None:
                    raise ValueError(
                        "remote placement action requires a requested cell"
                    )
                allowed = requested_cell in source_policy["allowedCells"] and (
                    requested_cell == source_placement["homeCell"]
                    or source_policy["allowRemoteExecution"]
                )
                if allowed != (outcome != "rejected"):
                    raise ValueError(
                        "placement action outcome disagrees with workspace policy"
                    )
                if allowed and expected_cell != requested_cell:
                    raise ValueError(
                        "accepted remote action must execute in the requested cell"
                    )
            actions.append(
                {
                    "ordinal": ordinal,
                    "action": kind,
                    "requestedExecutionCell": requested_cell,
                    "expectedOutcome": outcome,
                    "expectedExecutionCell": expected_cell,
                    "expectedReasonCode": reason,
                }
            )
        if not actions:
            raise ValueError("placement case must define actions")
        placement_cases.append({"id": case_id, "source": source, "actions": actions})
    if not placement_cases:
        raise ValueError("campaign must define source placement contract cases")

    repetitions = _positive(raw.get("repetitions"), "repetitions", integer=True)
    if repetitions < 3:
        raise ValueError("paired campaigns require at least three repetitions")
    pairs = []
    pair_ids = set()
    for pair in raw.get("pairs") or []:
        pair_id = _dns(pair.get("id"), "pair.id")
        if pair_id in pair_ids:
            raise ValueError("pair ids must be unique")
        pair_ids.add(pair_id)
        source = _dns(pair.get("source"), "pair.source")
        if source not in sources:
            raise ValueError("pair references unknown source")
        arms = pair.get("arms") or {}
        if set(arms) != {"control", "variant"}:
            raise ValueError("pair arms must contain exactly control and variant")
        normalized_arms = {}
        for arm_name in ("control", "variant"):
            arm = arms[arm_name]
            publisher_location = _dns(arm.get("publisherLocation"), "publisherLocation")
            reader_location = _dns(arm.get("readerLocation"), "readerLocation")
            if publisher_location not in locations or reader_location not in locations:
                raise ValueError("pair arm references unknown location")
            relay_cell = _dns(arm.get("relayCell"), "relayCell")
            execution_cell = _dns(arm.get("executionCell"), "executionCell")
            if relay_cell not in cells or execution_cell not in cells:
                raise ValueError("pair arm references unknown cell")
            placement = sources[source]
            policy = policies[placement["workspacePolicy"]]
            if relay_cell != placement["homeCell"]:
                raise ValueError("relayCell must equal the persisted source homeCell")
            remote = execution_cell != relay_cell
            if remote and not policy["allowRemoteExecution"]:
                raise ValueError("remote execution is forbidden by workspace policy")
            if execution_cell not in policy["allowedCells"]:
                raise ValueError("executionCell is forbidden by workspace policy")
            topology = str(arm.get("topology") or "")
            if topology not in (
                "same-cell",
                "cross-cell-direct",
                "cross-cell-replica",
                "cross-cell-ingest",
                "cross-cell-ingest-and-execution",
            ):
                raise ValueError("unsupported topology: %s" % topology)
            publisher_remote = locations[publisher_location]["cell"] != relay_cell
            expected_locality = {
                "same-cell": (False, False),
                "cross-cell-direct": (False, True),
                "cross-cell-replica": (False, True),
                "cross-cell-ingest": (True, False),
                "cross-cell-ingest-and-execution": (True, True),
            }[topology]
            if (publisher_remote, remote) != expected_locality:
                raise ValueError(
                    "topology disagrees with connector/relay/execution locality"
                )
            replica_read_url = None
            if topology == "cross-cell-replica":
                replica_read_url = _secret_ref(
                    arm.get("replicaReadUrl"), "replicaReadUrl"
                )
            normalized_arms[arm_name] = {
                "topology": topology,
                "publisherLocation": publisher_location,
                "readerLocation": reader_location,
                "relayCell": relay_cell,
                "executionCell": execution_cell,
                "replicaIdentity": (
                    _identity(arm.get("replicaIdentity"), "replicaIdentity")
                    if topology == "cross-cell-replica"
                    else None
                ),
                "replicaReadUrl": replica_read_url,
                "processorRuntimeIdentity": _identity(
                    arm.get("processorRuntimeIdentity"), "processorRuntimeIdentity"
                ),
            }
        if normalized_arms["control"]["topology"] != "same-cell":
            raise ValueError("control arm must be same-cell")
        pairs.append({"id": pair_id, "source": source, "arms": normalized_arms})
    if not pairs:
        raise ValueError("campaign must define at least one paired topology")

    thresholds = raw.get("thresholds") or {}
    normalized_thresholds = {
        "maxTtffMs": _positive(thresholds.get("maxTtffMs"), "thresholds.maxTtffMs"),
        "maxStallMs": _positive(thresholds.get("maxStallMs"), "thresholds.maxStallMs"),
        "maxStalls": int(thresholds.get("maxStalls", 0)),
        "minDeliveredFpsRatio": _ratio(
            thresholds.get("minDeliveredFpsRatio"), "thresholds.minDeliveredFpsRatio"
        ),
        "maxPacketLossRatio": _ratio(
            thresholds.get("maxPacketLossRatio", 0),
            "thresholds.maxPacketLossRatio",
            allow_zero=True,
        ),
        "maxLatencyP95Ms": _positive(
            thresholds.get("maxLatencyP95Ms"), "thresholds.maxLatencyP95Ms"
        ),
        "maxClockUncertaintyMs": _positive(
            thresholds.get("maxClockUncertaintyMs", 5),
            "thresholds.maxClockUncertaintyMs",
        ),
        "maxVpcUtilizationRatio": _ratio(
            thresholds.get("maxVpcUtilizationRatio", 0.7),
            "thresholds.maxVpcUtilizationRatio",
        ),
        "maxLbUtilizationRatio": _ratio(
            thresholds.get("maxLbUtilizationRatio", 0.7),
            "thresholds.maxLbUtilizationRatio",
        ),
        "maxPairedLatencyRegressionRatio": _ratio(
            thresholds.get("maxPairedLatencyRegressionRatio", 0.5),
            "thresholds.maxPairedLatencyRegressionRatio",
        ),
    }
    if normalized_thresholds["maxStalls"] < 0:
        raise ValueError("thresholds.maxStalls cannot be negative")

    timing = raw.get("timing") or {}
    normalized_timing = {
        "warmupSeconds": _positive(timing.get("warmupSeconds"), "timing.warmupSeconds"),
        "measureSeconds": _positive(
            timing.get("measureSeconds"), "timing.measureSeconds"
        ),
        "startupGraceSeconds": _positive(
            timing.get("startupGraceSeconds"), "timing.startupGraceSeconds"
        ),
        "ttlSecondsAfterFinished": _positive(
            timing.get("ttlSecondsAfterFinished", 3600),
            "timing.ttlSecondsAfterFinished",
            integer=True,
        ),
        "shutdownMarginSeconds": _positive(
            timing.get("shutdownMarginSeconds", 30),
            "timing.shutdownMarginSeconds",
        ),
    }

    return {
        "schemaVersion": 1,
        "campaignPath": str(campaign_path),
        "name": name,
        "environment": "staging",
        "identitySnapshot": normalized_snapshot,
        "namespace": namespace,
        "serviceAccountName": _dns(
            raw.get("serviceAccountName", "video-multi-cell-benchmark"),
            "serviceAccountName",
        ),
        "agentImage": image,
        "agentCommand": [str(value) for value in agent_command],
        "maxRenderedJobs": _positive(
            raw.get("maxRenderedJobs", 100), "maxRenderedJobs", integer=True
        ),
        "cells": cells,
        "locations": locations,
        "fixture": normalized_fixture,
        "processorWorkflow": normalized_workflow,
        "workspacePolicies": policies,
        "sourcePlacements": sources,
        "placementCases": placement_cases,
        "repetitions": repetitions,
        "pairs": pairs,
        "thresholds": normalized_thresholds,
        "timing": normalized_timing,
    }


def _run_id(campaign_name, pair_id, repetition, arm):
    value = "%s-%s-r%d-%s" % (campaign_name, pair_id, repetition, arm)
    if len(value) <= 63:
        return value
    return value[:54].rstrip("-") + "-" + hashlib.sha256(value.encode()).hexdigest()[:8]


def _limited_dns(*parts):
    raw = "-".join(str(part) for part in parts).lower()
    raw = re.sub(r"[^a-z0-9-]", "-", raw).strip("-")
    if len(raw) <= 63:
        return _dns(raw, "generated name")
    return _dns(
        raw[:54].rstrip("-") + "-" + hashlib.sha256(raw.encode()).hexdigest()[:8],
        "generated name",
    )


def build_sequence(campaign, execution_id):
    """Counterbalance AB/BA order while retaining exact within-rep pairing."""
    sequence = []
    for pair in campaign["pairs"]:
        for repetition in range(1, campaign["repetitions"] + 1):
            order = ("control", "variant") if repetition % 2 else ("variant", "control")
            for ordinal, arm in enumerate(order, 1):
                run_id = _run_id(execution_id, pair["id"], repetition, arm)
                sequence.append(
                    {
                        "runId": run_id,
                        "streamIdentitySha256": _hash(
                            {
                                "campaign": campaign["name"],
                                "runId": run_id,
                                "stream": "bench-" + run_id,
                            }
                        ),
                        "pairId": pair["id"],
                        "repetition": repetition,
                        "arm": arm,
                        "withinPairOrder": ordinal,
                        "source": pair["source"],
                        **pair["arms"][arm],
                    }
                )
    return sequence


def _env(name, value):
    return {"name": name, "value": str(value)}


def _secret_env(name, ref):
    return {"name": name, "valueFrom": {"secretKeyRef": ref}}


def _job(campaign, campaign_hash, execution_id, run, role, location_name):
    location = campaign["locations"][location_name]
    cell = campaign["cells"][location["cell"]]
    relay = campaign["cells"][run["relayCell"]]["relay"]
    source = campaign["sourcePlacements"][run["source"]]
    name = _limited_dns(run["runId"], role)
    labels = {
        "app.kubernetes.io/name": "video-multi-cell-probe",
        "app.kubernetes.io/managed-by": MANAGED_BY,
        "benchmark.roboflow.com/campaign": campaign["name"],
        "benchmark.roboflow.com/execution-id": execution_id,
        "benchmark.roboflow.com/run-id": run["runId"],
        "benchmark.roboflow.com/pair-id": run["pairId"],
        "benchmark.roboflow.com/arm": run["arm"],
        "benchmark.roboflow.com/role": role,
        "benchmark.roboflow.com/cell": location["cell"],
    }
    if role == "publish-clock":
        url_env = _secret_env("BENCH_URL_TEMPLATE", relay["publishUrl"])
        fixture_env = (
            _env("BENCH_FIXTURE_SOURCE", campaign["fixture"]["source"]["value"])
            if "value" in campaign["fixture"]["source"]
            else _secret_env(
                "BENCH_FIXTURE_SOURCE",
                campaign["fixture"]["source"]["secretKeyRef"],
            )
        )
    else:
        read_ref = (
            run["replicaReadUrl"]
            if run["topology"] == "cross-cell-replica"
            else (
                relay["externalReadUrl"]
                if run["executionCell"] != run["relayCell"]
                else relay["internalReadUrl"]
            )
        )
        url_env = _secret_env("BENCH_URL_TEMPLATE", read_ref)
        fixture_env = None
    env = [
        url_env,
        _env("BENCH_RUN_ID", run["runId"]),
        _env("BENCH_ROLE", role),
        _env("BENCH_STREAM", "bench-" + run["runId"]),
        _env("BENCH_EXPECTED_CELL", location["cell"]),
        _env("BENCH_EXPECTED_NODE", location["nodeName"]),
        _env("BENCH_EXPECTED_NODE_UID", location["nodeUid"]),
        _env("BENCH_EXPECTED_NETWORK_SHA256", location["networkIdentity"]["sha256"]),
        _env("BENCH_SOURCE_PLACEMENT_GENERATION", source["placementGeneration"]),
        _env("BENCH_FIXTURE_SHA256", campaign["fixture"]["sha256"]),
        _env("BENCH_EXPECTED_FPS", campaign["fixture"]["fps"]),
        _env("BENCH_WIDTH", campaign["fixture"]["width"]),
        _env("BENCH_HEIGHT", campaign["fixture"]["height"]),
        _env("BENCH_BITRATE_BPS", campaign["fixture"]["bitrateBps"]),
        _env("BENCH_WARMUP_SECONDS", campaign["timing"]["warmupSeconds"]),
        _env("BENCH_MEASURE_SECONDS", campaign["timing"]["measureSeconds"]),
        _env(
            "BENCH_STARTUP_GRACE_SECONDS",
            campaign["timing"]["startupGraceSeconds"],
        ),
        _env(
            "BENCH_SHUTDOWN_MARGIN_SECONDS",
            campaign["timing"]["shutdownMarginSeconds"],
        ),
        {
            "name": "BENCH_OBSERVED_NODE",
            "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}},
        },
        {
            "name": "BENCH_POD_UID",
            "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}},
        },
    ]
    if fixture_env:
        env.append(fixture_env)
    args = [
        "--role",
        role,
        "--report-path",
        "/dev/termination-log",
    ]
    total = (
        campaign["timing"]["warmupSeconds"]
        + campaign["timing"]["measureSeconds"]
        + campaign["timing"]["startupGraceSeconds"]
        + campaign["timing"]["shutdownMarginSeconds"]
    )
    pod_spec = {
        "serviceAccountName": campaign["serviceAccountName"],
        "restartPolicy": "Never",
        "nodeName": location["nodeName"],
        "tolerations": location["tolerations"],
        "terminationGracePeriodSeconds": 15,
        "securityContext": {"seccompProfile": {"type": "RuntimeDefault"}},
        "containers": [
            {
                "name": "probe",
                "image": campaign["agentImage"],
                "imagePullPolicy": "IfNotPresent",
                "command": campaign["agentCommand"],
                "args": args,
                "env": env,
                "resources": {
                    "requests": {"cpu": "500m", "memory": "256Mi"},
                    "limits": {"cpu": "2", "memory": "1Gi"},
                },
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "capabilities": {"drop": ["ALL"]},
                    "readOnlyRootFilesystem": True,
                    "runAsNonRoot": True,
                },
                "terminationMessagePath": "/dev/termination-log",
                "terminationMessagePolicy": "File",
            }
        ],
    }
    return cell["clusterContext"], {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": cell["namespace"],
            "labels": labels,
            "annotations": {
                "benchmark.roboflow.com/campaign-sha256": campaign_hash,
                "benchmark.roboflow.com/expected-node-uid": location["nodeUid"],
                "benchmark.roboflow.com/expected-network-sha256": location[
                    "networkIdentity"
                ]["sha256"],
            },
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": int(total),
            "ttlSecondsAfterFinished": campaign["timing"]["ttlSecondsAfterFinished"],
            "template": {"metadata": {"labels": labels}, "spec": pod_spec},
        },
    }


def render(campaign, execution_id):
    execution_id = _dns(execution_id, "executionId")
    normalized = {
        key: value for key, value in campaign.items() if key != "campaignPath"
    }
    campaign_hash = _hash(normalized)
    sequence = build_sequence(campaign, execution_id)
    artifacts = {}
    execution = []
    for run in sequence:
        publisher_location = campaign["locations"][run["publisherLocation"]]
        reader_location = campaign["locations"][run["readerLocation"]]
        run_artifacts = {}
        for role, location in (
            ("publish-clock", run["publisherLocation"]),
            ("probe-clock", run["readerLocation"]),
        ):
            context, job = _job(
                campaign, campaign_hash, execution_id, run, role, location
            )
            filename = "jobs.%s.%s.%s.json" % (
                run["runId"],
                role,
                _hash(context)[:12],
            )
            artifacts[filename] = {
                "context": context,
                "document": {"apiVersion": "v1", "kind": "List", "items": [job]},
            }
            run_artifacts[role] = {"filename": filename, "clusterContext": context}
        if reader_location["cell"] != run["executionCell"]:
            raise ValueError("reader location must be in the execution cell")
        execution.append(
            {
                "runId": run["runId"],
                "steps": [
                    {
                        "ordinal": 1,
                        "action": "apply-publisher",
                        "clusterContext": run_artifacts["publish-clock"][
                            "clusterContext"
                        ],
                        "command": [
                            "kubectl",
                            "--context",
                            run_artifacts["publish-clock"]["clusterContext"],
                            "apply",
                            "-f",
                            run_artifacts["publish-clock"]["filename"],
                        ],
                    },
                    {
                        "ordinal": 2,
                        "action": "readiness-barrier",
                        "requirements": [
                            "publisher pod is Ready and exact node UID revalidated",
                            "MediaMTX path is ready on the expected relay workload UID",
                            "source home cell, shard, and placement generation still match",
                            "Prometheus and Crusoe metric windows have baseline samples",
                        ],
                    },
                    {
                        "ordinal": 3,
                        "action": "create-processor-job",
                        "requirements": [
                            "staging-only service API identity is authenticated",
                            "job request binds source placement generation and execution cell",
                            "workflow and runtime-input hashes match the campaign",
                            "idempotency key equals the run ID",
                        ],
                    },
                    {
                        "ordinal": 4,
                        "action": "processor-claim-barrier",
                        "requirements": [
                            "exact processor runtime, pod UID, node UID, and execution cell match",
                            "one claim at attempt one and post-claim frame progress are observed",
                            "processor source transport matches the planned topology",
                        ],
                    },
                    {
                        "ordinal": 5,
                        "action": "apply-probe",
                        "clusterContext": run_artifacts["probe-clock"][
                            "clusterContext"
                        ],
                        "command": [
                            "kubectl",
                            "--context",
                            run_artifacts["probe-clock"]["clusterContext"],
                            "apply",
                            "-f",
                            run_artifacts["probe-clock"]["filename"],
                        ],
                    },
                    {
                        "ordinal": 6,
                        "action": "collect-and-clean-before-next-run",
                        "requirements": [
                            "both terminal reports collected",
                            "all authoritative metric range queries collected",
                            "run-scoped Jobs deleted in every involved context",
                            "relay path and active-flow counts returned to baseline",
                        ],
                    },
                ],
            }
        )
    count = len(artifacts)
    if campaign["maxRenderedJobs"] > MAX_RENDERED_JOBS:
        raise ValueError(
            "maxRenderedJobs exceeds hard safety cap %d" % MAX_RENDERED_JOBS
        )
    if count > campaign["maxRenderedJobs"]:
        raise ValueError(
            "campaign expands to %d Jobs, exceeding maxRenderedJobs=%d"
            % (count, campaign["maxRenderedJobs"])
        )
    manifest = {
        "schemaVersion": 1,
        "campaign": campaign["name"],
        "executionId": execution_id,
        "campaignSha256": campaign_hash,
        "environment": "staging",
        "identitySnapshot": campaign["identitySnapshot"],
        "agentImage": campaign["agentImage"],
        "fixture": {
            key: value for key, value in campaign["fixture"].items() if key != "source"
        },
        "processorWorkflow": campaign["processorWorkflow"],
        "cells": campaign["cells"],
        "locations": campaign["locations"],
        "workspacePolicies": campaign["workspacePolicies"],
        "sourcePlacements": campaign["sourcePlacements"],
        "placementCases": campaign["placementCases"],
        "thresholds": campaign["thresholds"],
        "timing": campaign["timing"],
        "repetitions": campaign["repetitions"],
        "sequence": sequence,
        "execution": execution,
        "renderedJobCount": count,
        "cleanup": [
            {
                "clusterContext": context,
                "command": [
                    "kubectl",
                    "--context",
                    context,
                    "--namespace",
                    next(
                        cell["namespace"]
                        for cell in campaign["cells"].values()
                        if cell["clusterContext"] == context
                    ),
                    "delete",
                    "jobs",
                    "--selector",
                    "benchmark.roboflow.com/execution-id=%s" % execution_id,
                    "--wait=false",
                ],
            }
            for context in sorted(
                {cell["clusterContext"] for cell in campaign["cells"].values()}
            )
        ],
        "renderedBy": {
            "contractVersion": 1,
            "renderedAt": time.time(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
    return artifacts, manifest


def build_evidence_template(manifest):
    """Return a non-certifying skeleton keyed to every planned observation."""
    runs = []
    for run in manifest["sequence"]:
        source = manifest["sourcePlacements"][run["source"]]
        relay = manifest["cells"][run["relayCell"]]["relay"]
        runs.append(
            {
                "runId": run["runId"],
                "pairId": run["pairId"],
                "repetition": run["repetition"],
                "arm": run["arm"],
                "topology": run["topology"],
                "source": run["source"],
                "relayCell": run["relayCell"],
                "executionCell": run["executionCell"],
                "observationStatus": "not-collected",
                "success": None,
                "publisherReportArtifactId": None,
                "probeReportArtifactId": None,
                "processorApiArtifactId": None,
                "placementApiArtifactId": None,
                "placement": {
                    "sourceIdentitySha256": source["sourceIdentitySha256"],
                    "homeCell": source["homeCell"],
                    "relayShard": source["relayShard"],
                    "placementGeneration": source["placementGeneration"],
                    "recordRevision": source["recordRevision"],
                    "executionCell": run["executionCell"],
                    "remoteExecution": run["executionCell"] != run["relayCell"],
                },
                "relay": relay,
                "replicaIdentity": run["replicaIdentity"],
                "processorRuntimeIdentity": run["processorRuntimeIdentity"],
                "processor": None,
                "publisher": None,
                "reader": None,
                "measurementWindow": None,
                "measurement": None,
                "metrics": {
                    key: {"rawArtifactId": None}
                    for key in (
                        "relayBytesReceived",
                        "relayBytesSent",
                        "lbBytesIn",
                        "lbBytesOut",
                        "publisherVpcRxBytes",
                        "publisherVpcTxBytes",
                        "relayVpcRxBytes",
                        "relayVpcTxBytes",
                        "readerVpcRxBytes",
                        "readerVpcTxBytes",
                        "rtpPacketsReceived",
                        "rtpPacketsLost",
                        "rtpPacketsInError",
                        "publisherNodePacketDrops",
                        "relayNodePacketDrops",
                        "readerNodePacketDrops",
                        "publisherTcpRetransmits",
                        "relayTcpRetransmits",
                        "readerTcpRetransmits",
                    )
                },
            }
        )
    cases = []
    for case in manifest["placementCases"]:
        cases.append(
            {
                "id": case["id"],
                "source": case["source"],
                "observationStatus": "not-collected",
                "observations": [
                    {
                        "ordinal": action["ordinal"],
                        "action": action["action"],
                        "requestedExecutionCell": action["requestedExecutionCell"],
                        "outcome": None,
                        "reasonCode": None,
                        "executionCell": None,
                        "homeCell": None,
                        "relayShard": None,
                        "placementGeneration": None,
                        "jobClaimed": None,
                        "rawArtifactId": None,
                    }
                    for action in case["actions"]
                ],
            }
        )
    return {
        "schemaVersion": 1,
        "environment": "staging",
        "campaign": manifest["campaign"],
        "executionId": manifest["executionId"],
        "campaignSha256": manifest["campaignSha256"],
        "runs": runs,
        "placementCases": cases,
        "artifactIndex": [],
    }


def write(output_dir, artifacts, manifest):
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, artifact in sorted(artifacts.items()):
        path = output / filename
        if path.exists():
            raise ValueError("refusing to overwrite %s" % path)
        path.write_text(
            json.dumps(artifact["document"], indent=2, sort_keys=True) + "\n"
        )
        paths.append(path)
    manifest_path = output / "campaign-manifest.json"
    if manifest_path.exists():
        raise ValueError("refusing to overwrite %s" % manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    evidence_path = output / "evidence-template.json"
    if evidence_path.exists():
        raise ValueError("refusing to overwrite %s" % evidence_path)
    evidence_path.write_text(
        json.dumps(build_evidence_template(manifest), indent=2, sort_keys=True) + "\n"
    )
    return paths, manifest_path, evidence_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--execution-id", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    try:
        campaign = load_campaign(args.campaign)
        artifacts, manifest = render(campaign, args.execution_id)
        paths, manifest_path, evidence_path = write(
            args.output_dir, artifacts, manifest
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    for path in paths:
        print(path)
    print(manifest_path)
    print(evidence_path)
    print("No cluster, cloud, API, DNS, or production reads/writes were performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
