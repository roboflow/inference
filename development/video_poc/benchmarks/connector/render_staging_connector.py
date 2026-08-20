#!/usr/bin/env python3
"""Render a locked-down, headless video connector for the Crusoe staging cell.

This module only validates configuration and writes JSON. It never invokes
kubectl, a Kubernetes client, or the Roboflow API.
"""

import argparse
import json
import os
import re
import shlex
import sys
import threading
import urllib.parse
from pathlib import Path


SCHEMA_VERSION = 1
ALLOWED_CLUSTER_IDENTITY = {
    "context": "ck8s-stg",
    "cluster": "ck8s-stg",
    "server": (
        "https://ck8s-stg-83c07ac7.us-east1-a."
        "cmk.crusoecloudcompute.com"
    ),
}
ALLOWED_API_BASES = {
    "https://api.roboflow.one",
    "https://us-central1-roboflow-staging.cloudfunctions.net/light-v2-device",
}
ALLOWED_FIXTURE_HOSTS = {"media.roboflow.com"}
DNS_LABEL = re.compile(r"^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$")
SECRET_KEY = re.compile(r"^[A-Za-z0-9._-]+$")
SAFE_FILENAME = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
IMMUTABLE_IMAGE = re.compile(
    r"^[A-Za-z0-9._:/-]+@sha256:(?P<digest>[0-9a-f]{64})$"
)
QUANTITY = re.compile(r"^[0-9]+(?:\.[0-9]+)?(?:m|Ki|Mi|Gi|Ti)?$")

ROOT_FIELDS = {
    "schemaVersion",
    "environment",
    "clusterIdentity",
    "namespace",
    "serviceAccountName",
    "connectorId",
    "connectorName",
    "connectorImage",
    "fixtureInitImage",
    "apiBase",
    "apiKeySecretRef",
    "fixtureHostAllowlist",
    "fixtures",
    "resources",
    "nodePlacement",
}


def _require_fields(value, expected, field):
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    unknown = sorted(set(value) - set(expected))
    missing = sorted(set(expected) - set(value))
    if unknown:
        raise ValueError(f"{field} has unknown fields: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"{field} is missing fields: {', '.join(missing)}")


def _dns_label(value, field):
    value = str(value or "")
    if len(value) > 63 or not DNS_LABEL.fullmatch(value):
        raise ValueError(f"{field} must be a DNS label of at most 63 characters")
    return value


def _immutable_image(value, field):
    value = str(value or "")
    match = IMMUTABLE_IMAGE.fullmatch(value)
    if not match or match.group("digest") == "0" * 64:
        raise ValueError(f"{field} must use a non-placeholder sha256 image digest")
    return value


def _sha256(value, field):
    value = str(value or "")
    if not SHA256.fullmatch(value) or value == "0" * 64:
        raise ValueError(f"{field} must be an exact, verified lowercase SHA-256")
    return value


def _resource_map(value, field):
    _require_fields(value, {"cpu", "memory"}, field)
    normalized = {}
    for name in ("cpu", "memory"):
        quantity = str(value[name] or "")
        if not QUANTITY.fullmatch(quantity):
            raise ValueError(f"{field}.{name} must be an explicit resource quantity")
        normalized[name] = quantity
    return normalized


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(
        path.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp"
    )
    with temporary.open("w") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
    temporary.replace(path)


def load_config(path):
    path = Path(path).resolve()
    with path.open() as source:
        raw = json.load(source)
    _require_fields(raw, ROOT_FIELDS, "config")
    if raw["schemaVersion"] != SCHEMA_VERSION:
        raise ValueError("schemaVersion must be 1")
    if raw["environment"] != "staging":
        raise ValueError("environment must be staging")

    _require_fields(
        raw["clusterIdentity"],
        {"context", "cluster", "server"},
        "clusterIdentity",
    )
    if raw["clusterIdentity"] != ALLOWED_CLUSTER_IDENTITY:
        raise ValueError("clusterIdentity must exactly match the Crusoe staging cell")

    namespace = _dns_label(raw["namespace"], "namespace")
    if "bench" not in namespace:
        raise ValueError("namespace must be a dedicated namespace containing 'bench'")
    service_account = _dns_label(raw["serviceAccountName"], "serviceAccountName")
    connector_id = _dns_label(raw["connectorId"], "connectorId")
    if not connector_id.startswith("bench-"):
        raise ValueError(
            "connectorId must start with 'bench-' for a benchmark identity"
        )
    connector_name = str(raw["connectorName"] or "")
    if not connector_name or len(connector_name) > 128:
        raise ValueError("connectorName must be non-empty and at most 128 characters")

    api_base = str(raw["apiBase"] or "").rstrip("/")
    if api_base not in ALLOWED_API_BASES or api_base != raw["apiBase"]:
        raise ValueError("apiBase must exactly match an allowlisted staging API base")

    _require_fields(raw["apiKeySecretRef"], {"name", "key"}, "apiKeySecretRef")
    secret_ref = {
        "name": _dns_label(raw["apiKeySecretRef"]["name"], "apiKeySecretRef.name"),
        "key": str(raw["apiKeySecretRef"]["key"] or ""),
    }
    if not SECRET_KEY.fullmatch(secret_ref["key"]):
        raise ValueError("apiKeySecretRef.key is not a valid Secret key")

    fixture_hosts = raw["fixtureHostAllowlist"]
    if (
        not isinstance(fixture_hosts, list)
        or not fixture_hosts
        or len(fixture_hosts) != len(set(fixture_hosts))
        or not set(fixture_hosts).issubset(ALLOWED_FIXTURE_HOSTS)
    ):
        raise ValueError(
            "fixtureHostAllowlist must contain only hard-allowlisted hosts"
        )

    fixtures = []
    filenames = set()
    if not isinstance(raw["fixtures"], list) or not raw["fixtures"]:
        raise ValueError("fixtures must be a non-empty list")
    for index, fixture in enumerate(raw["fixtures"]):
        field = f"fixtures[{index}]"
        _require_fields(fixture, {"url", "filename", "sha256"}, field)
        parsed = urllib.parse.urlparse(str(fixture["url"] or ""))
        if (
            parsed.scheme != "https"
            or parsed.hostname not in fixture_hosts
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port not in (None, 443)
            or not parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                f"{field}.url must be a plain HTTPS URL on an allowlisted host"
            )
        filename = str(fixture["filename"] or "")
        if (
            len(filename) > 128
            or not SAFE_FILENAME.fullmatch(filename)
            or filename in {".", ".."}
            or ".." in filename
            or filename in filenames
        ):
            raise ValueError(f"{field}.filename must be unique and path-safe")
        filenames.add(filename)
        fixtures.append(
            {
                "url": urllib.parse.urlunparse(parsed),
                "filename": filename,
                "sha256": _sha256(fixture["sha256"], f"{field}.sha256"),
            }
        )

    _require_fields(raw["resources"], {"requests", "limits"}, "resources")
    resources = {
        "requests": _resource_map(
            raw["resources"]["requests"], "resources.requests"
        ),
        "limits": _resource_map(raw["resources"]["limits"], "resources.limits"),
    }
    _require_fields(
        raw["nodePlacement"],
        {"nodeSelector", "tolerations"},
        "nodePlacement",
    )
    node_selector = raw["nodePlacement"]["nodeSelector"]
    if not isinstance(node_selector, dict) or not node_selector:
        raise ValueError("nodePlacement.nodeSelector must be a non-empty object")
    if node_selector.get("kubernetes.io/os") != "linux":
        raise ValueError("nodePlacement.nodeSelector must explicitly select Linux")
    if not all(
        isinstance(key, str)
        and key
        and isinstance(value, str)
        and value
        for key, value in node_selector.items()
    ):
        raise ValueError("nodePlacement.nodeSelector entries must be non-empty strings")
    tolerations = raw["nodePlacement"]["tolerations"]
    if not isinstance(tolerations, list):
        raise ValueError("nodePlacement.tolerations must be a list")
    for item in tolerations:
        if not isinstance(item, dict):
            raise ValueError("nodePlacement.tolerations entries must be objects")

    return {
        "path": path,
        "environment": "staging",
        "clusterIdentity": dict(ALLOWED_CLUSTER_IDENTITY),
        "namespace": namespace,
        "serviceAccountName": service_account,
        "connectorId": connector_id,
        "connectorName": connector_name,
        "connectorImage": _immutable_image(raw["connectorImage"], "connectorImage"),
        "fixtureInitImage": _immutable_image(
            raw["fixtureInitImage"], "fixtureInitImage"
        ),
        "apiBase": api_base,
        "apiKeySecretRef": secret_ref,
        "fixtureHostAllowlist": list(fixture_hosts),
        "fixtures": fixtures,
        "resources": resources,
        "nodePlacement": {
            "nodeSelector": dict(node_selector),
            "tolerations": tolerations,
        },
    }


def _labels(config):
    return {
        "app.kubernetes.io/name": "rf-video-connector",
        "app.kubernetes.io/instance": config["connectorId"],
        "app.kubernetes.io/part-of": "video-proc-benchmark",
        "roboflow.com/environment": "staging",
    }


def _init_container(config, fixture, ordinal):
    target = f"/fixtures/{fixture['filename']}"
    script = (
        "set -eu\n"
        "curl --fail --location --proto '=https' --tlsv1.2 "
        "--output \"$2\" \"$1\"\n"
        "printf '%s  %s\\n' \"$3\" \"$2\" | sha256sum -c -\n"
    )
    return {
        "name": f"fixture-{ordinal:02d}",
        "image": config["fixtureInitImage"],
        "imagePullPolicy": "IfNotPresent",
        "command": ["/bin/sh", "-ec"],
        "args": [script, "fixture-download", fixture["url"], target, fixture["sha256"]],
        "resources": {
            "requests": {"cpu": "25m", "memory": "32Mi"},
            "limits": {"cpu": "250m", "memory": "128Mi"},
        },
        "securityContext": {
            "allowPrivilegeEscalation": False,
            "capabilities": {"drop": ["ALL"]},
            "readOnlyRootFilesystem": True,
            "runAsNonRoot": True,
            "runAsUser": 10001,
            "runAsGroup": 10001,
        },
        "volumeMounts": [{"name": "fixtures", "mountPath": "/fixtures"}],
    }


def build_manifest(config):
    labels = _labels(config)
    namespace = config["namespace"]
    pod_spec = {
        "serviceAccountName": config["serviceAccountName"],
        "automountServiceAccountToken": False,
        "restartPolicy": "Always",
        "terminationGracePeriodSeconds": 30,
        "hostNetwork": False,
        "hostPID": False,
        "hostIPC": False,
        "dnsPolicy": "ClusterFirst",
        "securityContext": {
            "runAsNonRoot": True,
            "runAsUser": 10001,
            "runAsGroup": 10001,
            "fsGroup": 10001,
            "fsGroupChangePolicy": "OnRootMismatch",
            "seccompProfile": {"type": "RuntimeDefault"},
        },
        "initContainers": [
            _init_container(config, fixture, index)
            for index, fixture in enumerate(config["fixtures"], 1)
        ],
        "containers": [
            {
                "name": "connector",
                "image": config["connectorImage"],
                "imagePullPolicy": "IfNotPresent",
                "args": [
                    "--api-url",
                    config["apiBase"],
                    "--id",
                    config["connectorId"],
                    "--name",
                    config["connectorName"],
                    "--files-dir",
                    "/fixtures",
                    "--config",
                    "/state/connector.json",
                    "--ui-addr=",
                    "--net-discovery=false",
                ],
                "env": [
                    {
                        "name": "ROBOFLOW_API_KEY",
                        "valueFrom": {"secretKeyRef": dict(config["apiKeySecretRef"])},
                    }
                ],
                "resources": config["resources"],
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "capabilities": {"drop": ["ALL"]},
                    "readOnlyRootFilesystem": True,
                    "runAsNonRoot": True,
                    "runAsUser": 10001,
                    "runAsGroup": 10001,
                },
                "volumeMounts": [
                    {"name": "fixtures", "mountPath": "/fixtures", "readOnly": True},
                    {"name": "state", "mountPath": "/state"},
                    {"name": "scratch", "mountPath": "/tmp"},
                ],
            }
        ],
        "volumes": [
            {"name": "fixtures", "emptyDir": {"sizeLimit": "4Gi"}},
            {"name": "state", "emptyDir": {"sizeLimit": "64Mi"}},
            {"name": "scratch", "emptyDir": {"sizeLimit": "512Mi"}},
        ],
        "nodeSelector": config["nodePlacement"]["nodeSelector"],
        "tolerations": config["nodePlacement"]["tolerations"],
    }
    return {
        "apiVersion": "v1",
        "kind": "List",
        "items": [
            {
                "apiVersion": "v1",
                "kind": "Namespace",
                "metadata": {
                    "name": namespace,
                    "labels": {
                        "pod-security.kubernetes.io/enforce": "restricted",
                        "pod-security.kubernetes.io/audit": "restricted",
                        "pod-security.kubernetes.io/warn": "restricted",
                    },
                },
            },
            {
                "apiVersion": "v1",
                "kind": "ServiceAccount",
                "metadata": {
                    "name": config["serviceAccountName"],
                    "namespace": namespace,
                },
                "automountServiceAccountToken": False,
            },
            {
                "apiVersion": "networking.k8s.io/v1",
                "kind": "NetworkPolicy",
                "metadata": {
                    "name": f"{config['connectorId']}-outbound-only",
                    "namespace": namespace,
                },
                "spec": {
                    "podSelector": {"matchLabels": labels},
                    "policyTypes": ["Ingress", "Egress"],
                    "ingress": [],
                    "egress": [
                        {
                            "ports": [
                                {"protocol": "UDP", "port": 53},
                                {"protocol": "TCP", "port": 53},
                            ]
                        },
                        {
                            "ports": [
                                {"protocol": "TCP", "port": 443},
                                {"protocol": "TCP", "port": 8554},
                            ]
                        },
                    ],
                },
            },
            {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": config["connectorId"],
                    "namespace": namespace,
                    "labels": labels,
                },
                "spec": {
                    "replicas": 1,
                    "strategy": {"type": "Recreate"},
                    "selector": {"matchLabels": labels},
                    "template": {"metadata": {"labels": labels}, "spec": pod_spec},
                },
            },
        ],
    }


def build_plan(config, manifest_path):
    context = config["clusterIdentity"]["context"]
    namespace = config["namespace"]
    manifest_path = str(Path(manifest_path).resolve())
    return {
        "schemaVersion": 1,
        "kind": "staging-headless-connector-render",
        "environment": "staging",
        "clusterIdentity": dict(config["clusterIdentity"]),
        "namespace": namespace,
        "connectorId": config["connectorId"],
        "manifest": manifest_path,
        "credential": {
            "source": "SecretKeyRef",
            "name": config["apiKeySecretRef"]["name"],
            "key": config["apiKeySecretRef"]["key"],
            "value": "[redacted]",
        },
        "commands": {
            "apply": shlex.join(
                ["kubectl", "--context", context, "apply", "-f", manifest_path]
            ),
            "observe": shlex.join(
                [
                    "kubectl",
                    "--context",
                    context,
                    "--namespace",
                    namespace,
                    "get",
                    "deployment,pod",
                    "-l",
                    f"app.kubernetes.io/instance={config['connectorId']}",
                ]
            ),
            "cleanup": shlex.join(
                [
                    "kubectl",
                    "--context",
                    context,
                    "--namespace",
                    namespace,
                    "delete",
                    f"deployment/{config['connectorId']}",
                    f"networkpolicy/{config['connectorId']}-outbound-only",
                    f"serviceaccount/{config['serviceAccountName']}",
                    "--ignore-not-found=true",
                ]
            ),
            "cleanupNamespace": shlex.join(
                [
                    "kubectl",
                    "--context",
                    context,
                    "delete",
                    "namespace",
                    namespace,
                    "--ignore-not-found=true",
                ]
            ),
        },
        "performedClusterWrites": False,
    }


def render(config_path, output_dir):
    config = load_config(config_path)
    output_dir = Path(output_dir).resolve()
    manifest_path = output_dir / f"{config['connectorId']}.manifest.json"
    plan_path = output_dir / f"{config['connectorId']}.plan.json"
    _atomic_json(manifest_path, build_manifest(config))
    _atomic_json(plan_path, build_plan(config, manifest_path))
    return manifest_path, plan_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    try:
        manifest_path, plan_path = render(args.config, args.output_dir)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(manifest_path)
    print(plan_path)
    print("No Kubernetes or API calls were performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
