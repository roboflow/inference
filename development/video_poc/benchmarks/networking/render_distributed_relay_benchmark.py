#!/usr/bin/env python3
"""Render staging-only Kubernetes Jobs for a distributed relay benchmark.

This command never calls Kubernetes. It produces a Kubernetes JSON List and a
redacted run manifest for inspection before an operator applies anything.
"""

import argparse
import hashlib
import json
import platform
import re
import sys
import time
import uuid
from pathlib import Path


DNS_LABEL = re.compile(r"^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$")
STAGING_CONTEXT = re.compile(r"(?:^|[-_.])(stg|staging)(?:$|[-_.])", re.IGNORECASE)
IMMUTABLE_IMAGE = re.compile(r"^\S+@sha256:[0-9a-f]{64}$")
ROLES = ("publish-copy", "read-copy", "read-decode")
MANAGED_BY = "video-relay-benchmark"


def _require_dns_label(value, field):
    value = str(value or "")
    if len(value) > 63 or not DNS_LABEL.fullmatch(value):
        raise ValueError("%s must be a lowercase Kubernetes DNS label" % field)
    return value


def _positive(value, field, integer=False):
    try:
        value = int(value) if integer else float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("%s must be numeric" % field) from error
    if value <= 0:
        raise ValueError("%s must be positive" % field)
    return value


def _non_negative(value, field):
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("%s must be numeric" % field) from error
    if value < 0:
        raise ValueError("%s cannot be negative" % field)
    return value


def resolve_run_id(value=None):
    run_id = value or "%s-%s" % (
        time.strftime("%Y%m%d-%H%M%S", time.gmtime()),
        uuid.uuid4().hex[:8],
    )
    return _require_dns_label(run_id, "runId")


def _canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _secret_key_ref(raw, field):
    if not isinstance(raw, dict):
        raise ValueError("%s must be a secretKeyRef object" % field)
    name = _require_dns_label(raw.get("name"), field + ".name")
    key = str(raw.get("key") or "")
    if not key or any(character.isspace() for character in key):
        raise ValueError("%s.key must be non-empty and contain no whitespace" % field)
    return {"name": name, "key": key}


def load_scenario(path):
    scenario_path = Path(path).resolve()
    with scenario_path.open() as source:
        raw = json.load(source)
    if raw.get("schemaVersion") != 1:
        raise ValueError("scenario schemaVersion must be 1")
    if raw.get("environment") != "staging":
        raise ValueError("distributed relay benchmarks are restricted to staging")
    cluster_context = str(raw.get("clusterContext") or "")
    if not STAGING_CONTEXT.search(cluster_context):
        raise ValueError("clusterContext must be recognizably staging")
    namespace = _require_dns_label(raw.get("namespace"), "namespace")
    if "bench" not in namespace:
        raise ValueError("namespace must be a dedicated benchmark namespace")
    name = _require_dns_label(raw.get("name"), "name")
    image = str(raw.get("agentImage") or "")
    if not IMMUTABLE_IMAGE.fullmatch(image):
        raise ValueError("agentImage must use an immutable sha256 digest")
    agent_command = raw.get("agentCommand") or [
        "python",
        "/opt/roboflow/relay_agent.py",
    ]
    if not isinstance(agent_command, list) or not agent_command:
        raise ValueError("agentCommand must be a non-empty list")

    locations = {}
    for location_name, location in (raw.get("locations") or {}).items():
        location_name = _require_dns_label(location_name, "location name")
        if not isinstance(location, dict):
            raise ValueError("location %s must be an object" % location_name)
        node_selector = location.get("nodeSelector") or {}
        node_name = location.get("nodeName")
        if not node_selector and not node_name:
            raise ValueError(
                "location %s must declare nodeSelector or nodeName" % location_name
            )
        locations[location_name] = {
            "nodeSelector": node_selector,
            "nodeName": node_name,
            "tolerations": location.get("tolerations") or [],
            "affinity": location.get("affinity"),
            "annotations": location.get("annotations") or {},
            "metadata": location.get("metadata") or {},
        }
    if not locations:
        raise ValueError("scenario must define at least one explicit location")

    fixture = raw.get("fixture") or {}
    fixture_source = {}
    if fixture.get("value") is not None:
        value = str(fixture["value"])
        if "://" in value and ("?" in value or "@" in value):
            raise ValueError("credentialed fixture URLs must use fixture.secretKeyRef")
        fixture_source["value"] = value
    elif fixture.get("secretKeyRef"):
        fixture_source["secretKeyRef"] = _secret_key_ref(
            fixture["secretKeyRef"], "fixture.secretKeyRef"
        )
    else:
        raise ValueError("fixture must define value or secretKeyRef")
    fixture_metadata = fixture.get("metadata") or {}
    expected_fps = fixture_metadata.get("fps")
    if expected_fps is not None:
        expected_fps = _positive(expected_fps, "fixture.metadata.fps")

    media_secret = raw.get("mediaUrlSecret") or {}
    publish_ref = _secret_key_ref(
        media_secret.get("publishTemplate"), "mediaUrlSecret.publishTemplate"
    )
    read_ref = _secret_key_ref(
        media_secret.get("readTemplate"), "mediaUrlSecret.readTemplate"
    )

    timing = raw.get("timing") or {}
    normalized_timing = {
        "durationSeconds": _positive(
            timing.get("durationSeconds", 60), "timing.durationSeconds"
        ),
        "startupGraceSeconds": _positive(
            timing.get("startupGraceSeconds", 60), "timing.startupGraceSeconds"
        ),
        "ttlSecondsAfterFinished": _positive(
            timing.get("ttlSecondsAfterFinished", 600),
            "timing.ttlSecondsAfterFinished",
            integer=True,
        ),
    }
    thresholds = raw.get("stopThresholds") or {}
    normalized_thresholds = {
        "maxStartupSeconds": _positive(
            thresholds.get("maxStartupSeconds", 30), "stopThresholds.maxStartupSeconds"
        ),
        "maxProgressStallSeconds": _positive(
            thresholds.get("maxProgressStallSeconds", 15),
            "stopThresholds.maxProgressStallSeconds",
        ),
        "maxReconnects": int(thresholds.get("maxReconnects", 0)),
        "minDeliveredFpsRatio": float(thresholds.get("minDeliveredFpsRatio", 0.95)),
        "maxFailedAgents": int(thresholds.get("maxFailedAgents", 0)),
        "maxRelayCpuRatio": float(thresholds.get("maxRelayCpuRatio", 0.8)),
        "maxRelayMemoryRatio": float(thresholds.get("maxRelayMemoryRatio", 0.8)),
        "maxRelayVpcBandwidthRatio": float(
            thresholds.get("maxRelayVpcBandwidthRatio", 0.7)
        ),
        "maxConntrackPressureRatio": float(
            thresholds.get("maxConntrackPressureRatio", 0.5)
        ),
    }
    if (
        normalized_thresholds["maxReconnects"] < 0
        or normalized_thresholds["maxFailedAgents"] < 0
    ):
        raise ValueError("reconnect and failed-agent thresholds cannot be negative")
    for key in (
        "minDeliveredFpsRatio",
        "maxRelayCpuRatio",
        "maxRelayMemoryRatio",
        "maxRelayVpcBandwidthRatio",
        "maxConntrackPressureRatio",
    ):
        if not 0 < normalized_thresholds[key] <= 1:
            raise ValueError("stopThresholds.%s must be in (0, 1]" % key)

    publisher_names = set()
    publishers = []
    for item in raw.get("publishers") or []:
        publisher = {
            "name": _require_dns_label(item.get("name"), "publisher.name"),
            "location": _require_dns_label(item.get("location"), "publisher.location"),
            "count": _positive(item.get("count", 1), "publisher.count", integer=True),
            "streamStart": int(item.get("streamStart", 1)),
            "startAfterSeconds": _non_negative(
                item.get("startAfterSeconds", 0), "publisher.startAfterSeconds"
            ),
            "resources": item.get("resources") or {},
        }
        if publisher["name"] in publisher_names:
            raise ValueError("publisher names must be unique")
        if publisher["location"] not in locations:
            raise ValueError(
                "publisher references unknown location: %s" % publisher["location"]
            )
        if publisher["streamStart"] < 1:
            raise ValueError("publisher.streamStart must be positive")
        publisher_names.add(publisher["name"])
        publishers.append(publisher)
    if not publishers:
        raise ValueError("scenario must define at least one publisher group")

    readers = []
    reader_names = set()
    for item in raw.get("readers") or []:
        role = str(item.get("role") or "")
        if role not in ("read-copy", "read-decode"):
            raise ValueError("reader role must be read-copy or read-decode")
        groups = item.get("publisherGroups") or sorted(publisher_names)
        if not groups or any(group not in publisher_names for group in groups):
            raise ValueError("reader references an unknown publisher group")
        reader = {
            "name": _require_dns_label(item.get("name"), "reader.name"),
            "role": role,
            "location": _require_dns_label(item.get("location"), "reader.location"),
            "publisherGroups": list(groups),
            "copiesPerStream": _positive(
                item.get("copiesPerStream", 1), "reader.copiesPerStream", integer=True
            ),
            "startAfterSeconds": _non_negative(
                item.get("startAfterSeconds", 5), "reader.startAfterSeconds"
            ),
            "resources": item.get("resources") or {},
        }
        if reader["name"] in reader_names:
            raise ValueError("reader names must be unique")
        if reader["location"] not in locations:
            raise ValueError(
                "reader references unknown location: %s" % reader["location"]
            )
        reader_names.add(reader["name"])
        readers.append(reader)

    return {
        "schemaVersion": 1,
        "scenarioPath": str(scenario_path),
        "name": name,
        "environment": "staging",
        "clusterContext": cluster_context,
        "namespace": namespace,
        "serviceAccountName": _require_dns_label(
            raw.get("serviceAccountName", "video-relay-benchmark"),
            "serviceAccountName",
        ),
        "agentImage": image,
        "agentCommand": [str(part) for part in agent_command],
        "maxAgents": _positive(raw.get("maxAgents", 500), "maxAgents", integer=True),
        "environmentMetadata": raw.get("environmentMetadata") or {},
        "locations": locations,
        "fixture": {
            "source": fixture_source,
            "metadata": fixture_metadata,
            "expectedFps": expected_fps,
        },
        "mediaUrlSecret": {"publishTemplate": publish_ref, "readTemplate": read_ref},
        "timing": normalized_timing,
        "stopThresholds": normalized_thresholds,
        "publishers": publishers,
        "readers": readers,
    }


def _limited_name(*parts):
    raw = "-".join(str(part) for part in parts).lower()
    raw = re.sub(r"[^a-z0-9-]", "-", raw).strip("-")
    if len(raw) <= 63:
        return raw
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    return raw[:54].rstrip("-") + "-" + digest


def _value_or_secret(name, source):
    result = {"name": name}
    if "value" in source:
        result["value"] = source["value"]
    else:
        result["valueFrom"] = {"secretKeyRef": source["secretKeyRef"]}
    return result


def _secret_env(name, secret_ref):
    return {"name": name, "valueFrom": {"secretKeyRef": secret_ref}}


def _downward_env():
    return [
        {"name": "POD_NAME", "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}}},
        {"name": "POD_UID", "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}}},
        {
            "name": "POD_NAMESPACE",
            "valueFrom": {"fieldRef": {"fieldPath": "metadata.namespace"}},
        },
        {
            "name": "NODE_NAME",
            "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}},
        },
    ]


def _streams(scenario, run_id):
    result = {}
    for publisher in scenario["publishers"]:
        streams = []
        for offset in range(publisher["count"]):
            streams.append(
                _limited_name(
                    "bench",
                    run_id,
                    publisher["name"],
                    publisher["streamStart"] + offset,
                )
            )
        result[publisher["name"]] = streams
    return result


def expand_agents(scenario, run_id):
    streams_by_group = _streams(scenario, run_id)
    agents = []
    for publisher in scenario["publishers"]:
        for index, stream in enumerate(streams_by_group[publisher["name"]], 1):
            agents.append(
                {
                    "name": _limited_name(run_id, publisher["name"], "p", index),
                    "group": publisher["name"],
                    "role": "publish-copy",
                    "location": publisher["location"],
                    "stream": stream,
                    "startAfterSeconds": publisher["startAfterSeconds"],
                    "resources": publisher["resources"],
                }
            )
    for reader in scenario["readers"]:
        reader_index = 0
        for publisher_group in reader["publisherGroups"]:
            for stream in streams_by_group[publisher_group]:
                for _copy in range(reader["copiesPerStream"]):
                    reader_index += 1
                    agents.append(
                        {
                            "name": _limited_name(
                                run_id, reader["name"], "r", reader_index
                            ),
                            "group": reader["name"],
                            "role": reader["role"],
                            "location": reader["location"],
                            "stream": stream,
                            "startAfterSeconds": reader["startAfterSeconds"],
                            "resources": reader["resources"],
                        }
                    )
    return agents


def _agent_args(scenario, run_id, agent):
    thresholds = scenario["stopThresholds"]
    args = [
        "--role",
        agent["role"],
        "--run-id",
        run_id,
        "--location",
        agent["location"],
        "--stream",
        agent["stream"],
        "--duration-seconds",
        str(scenario["timing"]["durationSeconds"]),
        "--start-delay-seconds",
        str(agent["startAfterSeconds"]),
        "--max-startup-seconds",
        str(thresholds["maxStartupSeconds"]),
        "--max-progress-stall-seconds",
        str(thresholds["maxProgressStallSeconds"]),
        "--max-reconnects",
        str(thresholds["maxReconnects"]),
        "--min-delivered-fps-ratio",
        str(thresholds["minDeliveredFpsRatio"]),
        "--report-path",
        "/dev/termination-log",
        "--metrics-port",
        "9091",
    ]
    if agent["role"] == "read-decode" and scenario["fixture"]["expectedFps"]:
        args.extend(["--expected-fps", str(scenario["fixture"]["expectedFps"])])
    return args


def render_job(scenario, run_id, scenario_hash, agent):
    location = scenario["locations"][agent["location"]]
    labels = {
        "app.kubernetes.io/name": "video-relay-benchmark-agent",
        "app.kubernetes.io/managed-by": MANAGED_BY,
        "benchmark.roboflow.com/run-id": run_id,
        "benchmark.roboflow.com/scenario": scenario["name"],
        "benchmark.roboflow.com/role": agent["role"],
        "benchmark.roboflow.com/location": agent["location"],
        "benchmark.roboflow.com/group": agent["group"],
    }
    env = [
        _value_or_secret("BENCH_INPUT_URL", scenario["fixture"]["source"])
        if agent["role"] == "publish-copy"
        else _secret_env("BENCH_INPUT_URL", scenario["mediaUrlSecret"]["readTemplate"]),
        {"name": "AGENT_IMAGE_REFERENCE", "value": scenario["agentImage"]},
        {
            "name": "REQUESTED_NODE_INSTANCE_TYPE",
            "value": str(location["metadata"].get("instanceType") or ""),
        },
    ]
    if agent["role"] == "publish-copy":
        env.append(
            _secret_env(
                "BENCH_OUTPUT_URL",
                scenario["mediaUrlSecret"]["publishTemplate"],
            )
        )
    env.extend(_downward_env())
    pod_spec = {
        "serviceAccountName": scenario["serviceAccountName"],
        "restartPolicy": "Never",
        "terminationGracePeriodSeconds": 15,
        "securityContext": {"seccompProfile": {"type": "RuntimeDefault"}},
        "containers": [
            {
                "name": "agent",
                "image": scenario["agentImage"],
                "imagePullPolicy": "IfNotPresent",
                "command": scenario["agentCommand"],
                "args": _agent_args(scenario, run_id, agent),
                "env": env,
                "ports": [{"name": "metrics", "containerPort": 9091}],
                "resources": agent["resources"],
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "capabilities": {"drop": ["ALL"]},
                    "runAsNonRoot": True,
                },
                "terminationMessagePath": "/dev/termination-log",
                "terminationMessagePolicy": "File",
            }
        ],
    }
    if location["nodeSelector"]:
        pod_spec["nodeSelector"] = location["nodeSelector"]
    if location["nodeName"]:
        pod_spec["nodeName"] = location["nodeName"]
    if location["tolerations"]:
        pod_spec["tolerations"] = location["tolerations"]
    if location["affinity"]:
        pod_spec["affinity"] = location["affinity"]
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": agent["name"],
            "namespace": scenario["namespace"],
            "labels": labels,
            "annotations": {
                "benchmark.roboflow.com/scenario-sha256": scenario_hash,
                **location["annotations"],
            },
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": int(
                agent["startAfterSeconds"]
                + scenario["timing"]["durationSeconds"]
                + scenario["timing"]["startupGraceSeconds"]
            ),
            "ttlSecondsAfterFinished": scenario["timing"]["ttlSecondsAfterFinished"],
            "template": {"metadata": {"labels": labels}, "spec": pod_spec},
        },
    }


def render(scenario, run_id):
    scenario_for_hash = {
        key: value for key, value in scenario.items() if key != "scenarioPath"
    }
    scenario_hash = _canonical_hash(scenario_for_hash)
    agents = expand_agents(scenario, run_id)
    if scenario["maxAgents"] > 5000:
        raise ValueError("maxAgents cannot exceed the hard safety cap of 5000")
    if len(agents) > scenario["maxAgents"]:
        raise ValueError(
            "scenario expands to %d agents, exceeding maxAgents=%d"
            % (len(agents), scenario["maxAgents"])
        )
    jobs = [render_job(scenario, run_id, scenario_hash, agent) for agent in agents]
    selector = "benchmark.roboflow.com/run-id=%s" % run_id
    run_manifest = {
        "schemaVersion": 1,
        "runId": run_id,
        "scenario": scenario["name"],
        "scenarioSha256": scenario_hash,
        "environment": {
            "name": scenario["environment"],
            "clusterContext": scenario["clusterContext"],
            "namespace": scenario["namespace"],
            "metadata": scenario["environmentMetadata"],
        },
        "agentImage": scenario["agentImage"],
        "maxAgents": scenario["maxAgents"],
        "fixture": {
            "metadata": scenario["fixture"]["metadata"],
            "source": (
                "secretKeyRef"
                if "secretKeyRef" in scenario["fixture"]["source"]
                else "non-secret-value"
            ),
        },
        "requestedPlacement": scenario["locations"],
        "stopThresholds": scenario["stopThresholds"],
        "timing": scenario["timing"],
        "agents": agents,
        "cleanup": {
            "labelSelector": selector,
            "command": [
                "kubectl",
                "--context",
                scenario["clusterContext"],
                "--namespace",
                scenario["namespace"],
                "delete",
                "jobs",
                "--selector",
                selector,
                "--wait=false",
            ],
            "automatic": {
                "maximumActiveDeadlineSeconds": max(
                    job["spec"]["activeDeadlineSeconds"] for job in jobs
                ),
                "ttlSecondsAfterFinished": scenario["timing"]
                ["ttlSecondsAfterFinished"],
            },
        },
        "renderedBy": {
            "controllerContractVersion": 1,
            "renderedAt": time.time(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
    return {"apiVersion": "v1", "kind": "List", "items": jobs}, run_manifest


def write_rendered(output_dir, jobs, run_manifest):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs_path = output_dir / "jobs.json"
    run_path = output_dir / "run-manifest.json"
    if jobs_path.exists() or run_path.exists():
        raise ValueError("refusing to overwrite an existing rendered run")
    jobs_path.write_text(json.dumps(jobs, indent=2, sort_keys=True) + "\n")
    run_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")
    return jobs_path, run_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, help="versioned scenario JSON")
    parser.add_argument("--run-id", help="lowercase Kubernetes-safe run ID")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    try:
        scenario = load_scenario(args.scenario)
        run_id = resolve_run_id(args.run_id)
        jobs, run_manifest = render(scenario, run_id)
        jobs_path, run_path = write_rendered(args.output_dir, jobs, run_manifest)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(jobs_path)
    print(run_path)
    print("No cluster writes were performed. Inspect both files before applying.")
    print("Cleanup:", " ".join(run_manifest["cleanup"]["command"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
