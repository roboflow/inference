#!/usr/bin/env python3
"""Render a staging-only, two-cell video-routing validation bundle.

The renderer is deliberately offline. It creates one Kubernetes Job list per
cell, a redacted evidence-collection manifest, and an empty report template. It
never invokes Kubernetes, a control-plane API, Prometheus, or a credential
issuer.
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path, PurePosixPath

from render_distributed_relay_benchmark import (
    IMMUTABLE_IMAGE,
    STAGING_CONTEXT,
    _canonical_hash,
    _limited_name,
    _require_dns_label,
    _secret_key_ref,
    render_job,
    resolve_run_id,
)

ENDPOINT_KEYS = (
    "ingestTemplate",
    "consumeInternalTemplate",
    "consumePublicTemplate",
    "previewBase",
    "processorApiBase",
    "prometheusBase",
)
ASSIGNMENT_CASE_KINDS = (
    "first-activation",
    "reconnect-stickiness",
    "dedicated-workspace",
    "preferred-workspace",
    "cell-loss-recovery",
)
PATH_KINDS = ("same-cell", "explicit-cross-cell")
FAILURE_TARGETS = ("mediamtx", "gateway", "processor-pool", "cell-network")
CASE_ARTIFACT_KEYS = {
    "first-activation": {
        "registeredIdle",
        "activated",
        "connectorPublishEndpoint",
        "previewProof",
    },
    "reconnect-stickiness": {
        "beforeReconnect",
        "afterReconnect",
        "connectorPublishEndpoint",
        "previewProof",
    },
    "dedicated-workspace": {
        "policySnapshot",
        "activated",
        "connectorPublishEndpoint",
        "previewProof",
    },
    "preferred-workspace": {"policySnapshot", "activated"},
    "cell-loss-recovery": {
        "cellLossStarted",
        "unavailableObserved",
        "reassignmentDecision",
        "recovered",
        "connectorPublishAfterRecovery",
        "previewAfterRecovery",
    },
}
PATH_EVIDENCE_KEYS = {
    "processorClaimedCell",
    "relaySessions",
    "relayReaders",
    "podPlacement",
    "nodePlacement",
    "metricsWindow",
}


def _positive(value, field, integer=False):
    try:
        normalized = int(value) if integer else float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("%s must be numeric" % field) from error
    if normalized <= 0:
        raise ValueError("%s must be positive" % field)
    return normalized


def _non_negative(value, field):
    try:
        normalized = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("%s must be numeric" % field) from error
    if normalized < 0:
        raise ValueError("%s cannot be negative" % field)
    return normalized


def _artifact_ref(value, field):
    """Accept only local, non-secret evidence paths in rendered reports."""

    value = str(value or "")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "://" in value
        or "?" in value
        or "@" in value
    ):
        raise ValueError("%s must be a relative non-URL artifact path" % field)
    return value


def _resources(value, field):
    value = value or {}
    if not isinstance(value, dict):
        raise ValueError("%s must be an object" % field)
    return value


def _strict_secret_ref(value, field):
    if not isinstance(value, dict) or set(value) != {"name", "key"}:
        raise ValueError(
            "%s must contain only a Kubernetes Secret name and key" % field
        )
    return _secret_key_ref(value, field)


def _load_cells(raw):
    cells = {}
    cell_ids = set()
    for alias, value in (raw or {}).items():
        alias = _require_dns_label(alias, "cell alias")
        if not isinstance(value, dict):
            raise ValueError("cell %s must be an object" % alias)
        cell_id = _require_dns_label(value.get("cellId"), "cell.cellId")
        if cell_id in cell_ids:
            raise ValueError("cellId values must be unique")
        context = str(value.get("clusterContext") or "")
        if not STAGING_CONTEXT.search(context):
            raise ValueError("every cell clusterContext must be recognizably staging")
        namespace = _require_dns_label(value.get("namespace"), "cell.namespace")
        if "bench" not in namespace:
            raise ValueError(
                "every cell namespace must be a dedicated benchmark namespace"
            )
        endpoints = value.get("endpoints") or {}
        normalized_endpoints = {
            key: _strict_secret_ref(endpoints.get(key), "cell.endpoints.%s" % key)
            for key in ENDPOINT_KEYS
        }
        cells[alias] = {
            "cellId": cell_id,
            "provider": str(value.get("provider") or ""),
            "region": str(value.get("region") or ""),
            "clusterContext": context,
            "namespace": namespace,
            "serviceAccountName": _require_dns_label(
                value.get("serviceAccountName", "video-relay-benchmark"),
                "cell.serviceAccountName",
            ),
            "endpoints": normalized_endpoints,
            "versions": value.get("versions") or {},
        }
        cell_ids.add(cell_id)
    if len(cells) != 2:
        raise ValueError("two-cell validation requires exactly two staging cells")
    return cells


def _load_locations(raw, cells):
    locations = {}
    for name, value in (raw or {}).items():
        name = _require_dns_label(name, "location name")
        if not isinstance(value, dict):
            raise ValueError("location %s must be an object" % name)
        cell = _require_dns_label(value.get("cell"), "location.cell")
        if cell not in cells:
            raise ValueError("location references unknown cell: %s" % cell)
        node_selector = value.get("nodeSelector") or {}
        node_name = value.get("nodeName")
        if not node_selector and not node_name:
            raise ValueError("location %s must declare nodeSelector or nodeName" % name)
        locations[name] = {
            "cell": cell,
            "nodeSelector": node_selector,
            "nodeName": node_name,
            "tolerations": value.get("tolerations") or [],
            "affinity": value.get("affinity"),
            "annotations": value.get("annotations") or {},
            "metadata": value.get("metadata") or {},
        }
    if not locations:
        raise ValueError("scenario must define explicit locations")
    return locations


def _load_sources(raw, cells, locations):
    sources = {}
    for value in raw or []:
        name = _require_dns_label(value.get("name"), "source.name")
        if name in sources:
            raise ValueError("source names must be unique")
        home_cell = _require_dns_label(
            value.get("expectedHomeCell"), "source.expectedHomeCell"
        )
        location = _require_dns_label(
            value.get("publisherLocation"), "source.publisherLocation"
        )
        if home_cell not in cells:
            raise ValueError("source references unknown expectedHomeCell")
        if location not in locations or locations[location]["cell"] != home_cell:
            raise ValueError(
                "source publisherLocation must be in its expected home cell"
            )
        sources[name] = {
            "name": name,
            "expectedHomeCell": home_cell,
            "publisherLocation": location,
            "resources": _resources(value.get("resources"), "source.resources"),
            "startAfterSeconds": _non_negative(
                value.get("startAfterSeconds", 0), "source.startAfterSeconds"
            ),
        }
    if len(sources) < 2:
        raise ValueError("two-cell validation requires at least two sources")
    return sources


def _load_paths(raw, cells, locations, sources):
    paths = []
    names = set()
    covered = set()
    for value in raw or []:
        name = _require_dns_label(value.get("name"), "mediaPath.name")
        if name in names:
            raise ValueError("mediaPath names must be unique")
        kind = str(value.get("kind") or "")
        if kind not in PATH_KINDS:
            raise ValueError("mediaPath.kind must be same-cell or explicit-cross-cell")
        source_name = _require_dns_label(value.get("source"), "mediaPath.source")
        execution_cell = _require_dns_label(
            value.get("executionCell"), "mediaPath.executionCell"
        )
        location = _require_dns_label(
            value.get("readerLocation"), "mediaPath.readerLocation"
        )
        if source_name not in sources or execution_cell not in cells:
            raise ValueError("mediaPath references an unknown source or cell")
        if location not in locations or locations[location]["cell"] != execution_cell:
            raise ValueError("mediaPath readerLocation must be in executionCell")
        source_cell = sources[source_name]["expectedHomeCell"]
        remote = value.get("remoteExecution") is True
        endpoint = str(value.get("consumeEndpoint") or "")
        if kind == "same-cell":
            if execution_cell != source_cell or remote:
                raise ValueError(
                    "same-cell paths must be local and remoteExecution=false"
                )
            if endpoint != "consumeInternalTemplate":
                raise ValueError("same-cell paths must use consumeInternalTemplate")
        else:
            if execution_cell == source_cell or not remote:
                raise ValueError(
                    "explicit-cross-cell paths require a different execution cell and remoteExecution=true"
                )
            if endpoint != "consumePublicTemplate":
                raise ValueError("cross-cell paths must use consumePublicTemplate")
        copies = _positive(
            value.get("copyReaders", 1), "mediaPath.copyReaders", integer=True
        )
        decode = _positive(
            value.get("decodeReaders", 1), "mediaPath.decodeReaders", integer=True
        )
        evidence_artifacts = {
            str(key): _artifact_ref(path, "mediaPath evidence artifact")
            for key, path in (value.get("evidenceArtifacts") or {}).items()
        }
        if not PATH_EVIDENCE_KEYS.issubset(evidence_artifacts):
            raise ValueError(
                "mediaPath evidence must include processor cell, relay sessions/readers, pod/node placement, and metrics window artifacts"
            )
        paths.append(
            {
                "name": name,
                "kind": kind,
                "source": source_name,
                "sourceCell": source_cell,
                "executionCell": execution_cell,
                "remoteExecution": remote,
                "consumeEndpoint": endpoint,
                "readerLocation": location,
                "copyReaders": copies,
                "decodeReaders": decode,
                "resources": _resources(value.get("resources"), "mediaPath.resources"),
                "evidenceArtifacts": evidence_artifacts,
                "startAfterSeconds": _non_negative(
                    value.get("startAfterSeconds", 5), "mediaPath.startAfterSeconds"
                ),
            }
        )
        names.add(name)
        covered.add(kind)
    if covered != set(PATH_KINDS):
        raise ValueError(
            "scenario must include same-cell and explicit-cross-cell paths"
        )
    return paths


def _load_control_plane_evidence(raw, cells, sources):
    raw = raw or {}
    endpoint = _strict_secret_ref(raw.get("endpoint"), "controlPlaneEvidence.endpoint")
    auth = _strict_secret_ref(raw.get("auth"), "controlPlaneEvidence.auth")
    registry = _artifact_ref(raw.get("cellRegistrySnapshot"), "cellRegistrySnapshot")
    cases = []
    covered = set()
    for value in raw.get("cases") or []:
        name = _require_dns_label(value.get("name"), "controlPlane case.name")
        kind = str(value.get("kind") or "")
        if kind not in ASSIGNMENT_CASE_KINDS:
            raise ValueError("unknown control-plane evidence case kind")
        source = _require_dns_label(value.get("source"), "controlPlane case.source")
        expected_cell = _require_dns_label(
            value.get("expectedCell"), "controlPlane case.expectedCell"
        )
        if source not in sources or expected_cell not in cells:
            raise ValueError("control-plane case references unknown source or cell")
        artifacts = {
            str(key): _artifact_ref(path, "controlPlane case artifact")
            for key, path in (value.get("artifacts") or {}).items()
        }
        if not CASE_ARTIFACT_KEYS[kind].issubset(artifacts):
            raise ValueError(
                "control-plane case is missing required connector, preview, policy, or recovery artifacts"
            )
        recovery_cell = value.get("recoveryCell")
        if kind == "cell-loss-recovery":
            recovery_cell = _require_dns_label(
                recovery_cell, "controlPlane case.recoveryCell"
            )
            if recovery_cell not in cells:
                raise ValueError("cell-loss recovery references unknown recoveryCell")
        elif recovery_cell is not None:
            raise ValueError("recoveryCell is valid only for cell-loss-recovery")
        cases.append(
            {
                "name": name,
                "kind": kind,
                "source": source,
                "workspaceAlias": _require_dns_label(
                    value.get("workspaceAlias"), "controlPlane case.workspaceAlias"
                ),
                "expectedCell": expected_cell,
                "recoveryCell": recovery_cell,
                "artifacts": artifacts,
            }
        )
        covered.add(kind)
    required = {
        "first-activation",
        "reconnect-stickiness",
        "dedicated-workspace",
        "cell-loss-recovery",
    }
    if not required.issubset(covered):
        raise ValueError(
            "control-plane evidence must cover first activation, reconnect stickiness, dedicated workspace policy, and cell-loss recovery"
        )
    return {
        "endpoint": endpoint,
        "auth": auth,
        "cellRegistrySnapshot": registry,
        "cases": cases,
    }


def _load_measurement_plan(raw, cells):
    raw = raw or {}
    profiles = []
    for value in raw.get("networkProfiles") or []:
        name = _require_dns_label(value.get("name"), "networkProfile.name")
        profiles.append(
            {
                "name": name,
                "latencyMs": _non_negative(value.get("latencyMs", 0), "latencyMs"),
                "lossPercent": _non_negative(
                    value.get("lossPercent", 0), "lossPercent"
                ),
                "bandwidthBps": _positive(value.get("bandwidthBps"), "bandwidthBps"),
                "egressAccounting": bool(value.get("egressAccounting", True)),
                "application": "operator-approved",
            }
        )
    if not profiles or not any(
        profile["latencyMs"] == 0 and profile["lossPercent"] == 0
        for profile in profiles
    ):
        raise ValueError("measurementPlan must include an unimpaired baseline profile")
    failures = []
    for value in raw.get("failureExperiments") or []:
        target_cell = _require_dns_label(value.get("targetCell"), "failure.targetCell")
        target = str(value.get("target") or "")
        if target_cell not in cells or target not in FAILURE_TARGETS:
            raise ValueError("failure experiment references an unknown target")
        if value.get("requiresApproval") is not True:
            raise ValueError("failure experiments must explicitly require approval")
        failures.append(
            {
                "name": _require_dns_label(value.get("name"), "failure.name"),
                "targetCell": target_cell,
                "target": target,
                "requiresApproval": True,
                "maxUnavailableSeconds": _positive(
                    value.get("maxUnavailableSeconds"),
                    "failure.maxUnavailableSeconds",
                ),
                "reassignmentPolicy": str(value.get("reassignmentPolicy") or ""),
                "timelineArtifact": _artifact_ref(
                    value.get("timelineArtifact"), "failure.timelineArtifact"
                ),
                "rollbackArtifact": _artifact_ref(
                    value.get("rollbackArtifact"), "failure.rollbackArtifact"
                ),
                "requiredEvents": [
                    "baseline-healthy",
                    "failure-started",
                    "cell-marked-unavailable",
                    "reassignment-decision",
                    "recovery-started",
                    "routing-recovered",
                ],
            }
        )
        if failures[-1]["reassignmentPolicy"] not in (
            "retain-home-cell",
            "explicit-migration",
        ):
            raise ValueError(
                "failure.reassignmentPolicy must be retain-home-cell or explicit-migration"
            )
    if not failures:
        raise ValueError("measurementPlan must define a failure/recovery experiment")
    return {"networkProfiles": profiles, "failureExperiments": failures}


def load_scenario(path):
    scenario_path = Path(path).resolve()
    with scenario_path.open() as source:
        raw = json.load(source)
    if raw.get("schemaVersion") != 2:
        raise ValueError("two-cell scenario schemaVersion must be 2")
    if raw.get("environment") != "staging" or raw.get("executionMode") != "render-only":
        raise ValueError("two-cell validation is staging-only and render-only")
    name = _require_dns_label(raw.get("name"), "name")
    image = str(raw.get("agentImage") or "")
    if not IMMUTABLE_IMAGE.fullmatch(image):
        raise ValueError("agentImage must use an immutable sha256 digest")
    fixture = raw.get("fixture") or {}
    fixture_ref = _strict_secret_ref(
        fixture.get("secretKeyRef"), "fixture.secretKeyRef"
    )
    fixture_metadata = fixture.get("metadata") or {}
    expected_fps = _positive(fixture_metadata.get("fps"), "fixture.metadata.fps")
    cells = _load_cells(raw.get("cells"))
    locations = _load_locations(raw.get("locations"), cells)
    sources = _load_sources(raw.get("sources"), cells, locations)
    paths = _load_paths(raw.get("mediaPaths"), cells, locations, sources)
    control_plane = _load_control_plane_evidence(
        raw.get("controlPlaneEvidence"), cells, sources
    )
    measurement_plan = _load_measurement_plan(raw.get("measurementPlan"), cells)
    timing = raw.get("timing") or {}
    thresholds = raw.get("stopThresholds") or {}
    return {
        "schemaVersion": 2,
        "scenarioPath": str(scenario_path),
        "name": name,
        "environment": "staging",
        "executionMode": "render-only",
        "agentImage": image,
        "agentCommand": [
            str(value)
            for value in raw.get("agentCommand")
            or ["python", "/opt/roboflow/relay_agent.py"]
        ],
        "maxAgents": _positive(raw.get("maxAgents", 100), "maxAgents", integer=True),
        "fixture": {
            "source": {"secretKeyRef": fixture_ref},
            "metadata": fixture_metadata,
            "expectedFps": expected_fps,
        },
        "cells": cells,
        "locations": locations,
        "sources": sources,
        "mediaPaths": paths,
        "controlPlaneEvidence": control_plane,
        "measurementPlan": measurement_plan,
        "timing": {
            "durationSeconds": _positive(
                timing.get("durationSeconds", 60), "durationSeconds"
            ),
            "startupGraceSeconds": _positive(
                timing.get("startupGraceSeconds", 60), "startupGraceSeconds"
            ),
            "ttlSecondsAfterFinished": _positive(
                timing.get("ttlSecondsAfterFinished", 600),
                "ttlSecondsAfterFinished",
                integer=True,
            ),
        },
        "stopThresholds": {
            "maxStartupSeconds": _positive(
                thresholds.get("maxStartupSeconds", 30), "maxStartupSeconds"
            ),
            "maxProgressStallSeconds": _positive(
                thresholds.get("maxProgressStallSeconds", 15), "maxProgressStallSeconds"
            ),
            "maxReconnects": int(thresholds.get("maxReconnects", 1)),
            "minDeliveredFpsRatio": float(thresholds.get("minDeliveredFpsRatio", 0.95)),
        },
    }


def _base_scenario(scenario, cell_alias, endpoint_key, location_name):
    cell = scenario["cells"][cell_alias]
    return {
        "name": scenario["name"],
        "namespace": cell["namespace"],
        "serviceAccountName": cell["serviceAccountName"],
        "agentImage": scenario["agentImage"],
        "agentCommand": scenario["agentCommand"],
        "fixture": scenario["fixture"],
        "mediaUrlSecret": {
            "publishTemplate": cell["endpoints"].get(endpoint_key),
            "readTemplate": cell["endpoints"].get(endpoint_key),
        },
        "locations": {location_name: scenario["locations"][location_name]},
        "timing": scenario["timing"],
        "stopThresholds": scenario["stopThresholds"],
    }


def _annotate_job(job, scenario, cell_alias, source_cell, path_name, path_kind):
    cell = scenario["cells"][cell_alias]
    labels = job["metadata"]["labels"]
    labels.update(
        {
            "benchmark.roboflow.com/expected-cell": cell["cellId"],
            "benchmark.roboflow.com/source-cell": scenario["cells"][source_cell][
                "cellId"
            ],
            "benchmark.roboflow.com/media-path": path_name,
            "benchmark.roboflow.com/path-kind": path_kind,
        }
    )
    job["spec"]["template"]["metadata"]["labels"] = dict(labels)
    env = job["spec"]["template"]["spec"]["containers"][0]["env"]
    env.extend(
        [
            {"name": "EXPECTED_CELL_ID", "value": cell["cellId"]},
            {"name": "EXPECTED_CLUSTER_CONTEXT", "value": cell["clusterContext"]},
            {"name": "MEDIA_PATH", "value": path_name},
            {"name": "MEDIA_PATH_KIND", "value": path_kind},
        ]
    )


def render(scenario, run_id):
    scenario_hash = _canonical_hash(
        {key: value for key, value in scenario.items() if key != "scenarioPath"}
    )
    bundles = {alias: [] for alias in scenario["cells"]}
    agents = []
    streams = {}
    for source in scenario["sources"].values():
        stream = _limited_name("bench", run_id, source["name"])
        streams[source["name"]] = stream
        agent = {
            "name": _limited_name(run_id, source["name"], "publisher"),
            "group": source["name"],
            "role": "publish-copy",
            "location": source["publisherLocation"],
            "stream": stream,
            "startAfterSeconds": source["startAfterSeconds"],
            "resources": source["resources"],
        }
        base = _base_scenario(
            scenario,
            source["expectedHomeCell"],
            "ingestTemplate",
            source["publisherLocation"],
        )
        job = render_job(base, run_id, scenario_hash, agent)
        _annotate_job(
            job,
            scenario,
            source["expectedHomeCell"],
            source["expectedHomeCell"],
            _limited_name(source["name"], "ingest"),
            "same-cell",
        )
        bundles[source["expectedHomeCell"]].append(job)
        agents.append(
            {**agent, "cell": source["expectedHomeCell"], "source": source["name"]}
        )

    for path in scenario["mediaPaths"]:
        roles = (
            ("read-copy", path["copyReaders"]),
            ("read-decode", path["decodeReaders"]),
        )
        for role, count in roles:
            for index in range(1, count + 1):
                agent = {
                    "name": _limited_name(run_id, path["name"], role, index),
                    "group": path["name"],
                    "role": role,
                    "location": path["readerLocation"],
                    "stream": streams[path["source"]],
                    "startAfterSeconds": path["startAfterSeconds"],
                    "resources": path["resources"],
                }
                base = _base_scenario(
                    scenario,
                    path["sourceCell"],
                    path["consumeEndpoint"],
                    path["readerLocation"],
                )
                # The Job runs in executionCell, while its read endpoint belongs
                # to sourceCell. Keep namespace/service account local to the Job.
                execution = scenario["cells"][path["executionCell"]]
                base["namespace"] = execution["namespace"]
                base["serviceAccountName"] = execution["serviceAccountName"]
                job = render_job(base, run_id, scenario_hash, agent)
                _annotate_job(
                    job,
                    scenario,
                    path["executionCell"],
                    path["sourceCell"],
                    path["name"],
                    path["kind"],
                )
                bundles[path["executionCell"]].append(job)
                agents.append(
                    {
                        **agent,
                        "cell": path["executionCell"],
                        "source": path["source"],
                        "mediaPath": path["name"],
                        "pathKind": path["kind"],
                    }
                )
    if len(agents) > scenario["maxAgents"] or scenario["maxAgents"] > 5000:
        raise ValueError("scenario agent expansion exceeds its safety limit")

    job_lists = {
        alias: {"apiVersion": "v1", "kind": "List", "items": jobs}
        for alias, jobs in bundles.items()
    }
    cleanup = {}
    for alias, cell in scenario["cells"].items():
        selector = "benchmark.roboflow.com/run-id=%s" % run_id
        cleanup[alias] = {
            "command": [
                "kubectl",
                "--context",
                cell["clusterContext"],
                "--namespace",
                cell["namespace"],
                "delete",
                "jobs",
                "--selector",
                selector,
                "--wait=false",
            ],
            "credentialRevocation": "operator-owned; no credential contents are rendered",
        }

    manifest = {
        "schemaVersion": 2,
        "runId": run_id,
        "scenario": scenario["name"],
        "scenarioSha256": scenario_hash,
        "environment": "staging",
        "executionMode": "render-only",
        "agentImage": scenario["agentImage"],
        "cells": scenario["cells"],
        "sources": list(scenario["sources"].values()),
        "mediaPaths": scenario["mediaPaths"],
        "controlPlaneEvidence": scenario["controlPlaneEvidence"],
        "measurementPlan": scenario["measurementPlan"],
        "agents": agents,
        "collectionContract": _collection_contract(scenario, run_id),
        "cleanup": cleanup,
        "rollback": {
            "order": list(reversed(list(scenario["cells"]))),
            "scope": "benchmark Jobs and separately issued run credentials only",
            "cellResourcesChanged": False,
            "failureExperimentRollbackRequiresApproval": True,
        },
        "renderedBy": {
            "contractVersion": 2,
            "renderedAt": time.time(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
    report = _report_template(manifest)
    return job_lists, manifest, report


def _collection_contract(scenario, run_id):
    placement = {}
    for alias, cell in scenario["cells"].items():
        selector = "benchmark.roboflow.com/run-id=%s" % run_id
        placement[alias] = {
            "command": [
                "kubectl",
                "--context",
                cell["clusterContext"],
                "--namespace",
                cell["namespace"],
                "get",
                "pods",
                "--selector",
                selector,
                "-o",
                "json",
            ],
            "requiredObservedFields": [
                "metadata.name",
                "metadata.uid",
                "spec.nodeName",
                "status.podIP",
                "status.containerStatuses[].imageID",
                "node.metadata.labels.instance-type",
                "clusterContext",
                "expectedCell",
                "observedCell",
            ],
        }
    return {
        "placement": placement,
        "agentReports": {
            "source": "termination message or BENCHMARK_FINAL_JSON log line",
            "required": ["status", "progress", "attempts", "placement", "resources"],
        },
        "controlPlaneRouting": {
            "requiredPerSource": [
                "persisted home cell, assignment state/version, and lease/last-active metadata",
                "actual connector publish endpoint resolved from the activation response",
                "preview request and successful response from the assigned cell endpoint",
                "job sourceCell/executionCell/remoteExecution placement fields",
                "processor claim/status cell assertion",
            ]
        },
        "prometheus": {
            "window": "measured run start/end plus two scrape intervals",
            "endpointSource": "each cell endpoints.prometheusBase secretKeyRef",
            "requiredEvidence": [
                "MediaMTX paths/readers/sessions/bytes/RTP loss/input errors",
                "processor cell info/claims/rejections/jobs/CPU/memory/network",
                "pod and node bytes/packets/drops/errors/retransmits",
                "Cilium drops/map pressure/conntrack and connectivity latency",
                "agent startup/frames/reconnects",
            ],
        },
        "network": {
            "requiredPerPath": [
                "latency p50/p95/p99",
                "packet or RTP loss and input errors",
                "delivered and encoded bandwidth",
                "source-cell egress bytes",
                "execution-cell ingress bytes",
                "documented and observed node/LB bandwidth limits",
            ]
        },
        "failureTimeline": {
            "clock": "UTC RFC3339 with monotonic offsets",
            "requiredFields": [
                "event",
                "observedAt",
                "offsetSeconds",
                "cell",
                "evidenceArtifact",
            ],
        },
    }


def _report_template(manifest):
    return {
        "schemaVersion": 2,
        "runId": manifest["runId"],
        "scenarioSha256": manifest["scenarioSha256"],
        "status": "pending",
        "secretsRecorded": False,
        "assertions": {
            "firstActivationAssignment": "pending",
            "reconnectStickiness": "pending",
            "dedicatedWorkspacePolicy": "pending",
            "connectorUsesAssignedIngest": "pending",
            "sameCellProcessing": "pending",
            "noAccidentalCrossCellPath": "pending",
            "explicitCrossCellExperiment": "pending",
            "failureRecovery": "pending",
            "cellLossUnavailableBounded": "pending",
            "reassignmentDecisionHonored": "pending",
            "previewFromAssignedCell": "pending",
            "processorClaimedExpectedCell": "pending",
            "relaySessionsAndReadersMatch": "pending",
        },
        "controlPlaneEvidence": [],
        "observedPlacement": [],
        "agentReports": [],
        "mediaPathMeasurements": [],
        "prometheusEvidence": [],
        "failureRecoveryTimeline": [],
        "cleanup": {"completed": False, "credentialRevoked": False, "evidence": []},
        "remainingRisks": [],
    }


def write_rendered(output_dir, job_lists, manifest, report):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "validation-manifest.json": manifest,
        "validation-report.template.json": report,
    }
    for alias, jobs in job_lists.items():
        outputs["jobs-%s.json" % alias] = jobs
    for filename in outputs:
        if (output_dir / filename).exists():
            raise ValueError("refusing to overwrite an existing rendered run")
    paths = []
    for filename, payload in outputs.items():
        path = output_dir / filename
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        paths.append(path)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    try:
        scenario = load_scenario(args.scenario)
        run_id = resolve_run_id(args.run_id)
        job_lists, manifest, report = render(scenario, run_id)
        paths = write_rendered(args.output_dir, job_lists, manifest, report)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    for path in paths:
        print(path)
    print("No cluster, control-plane, DNS, or credential writes were performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
