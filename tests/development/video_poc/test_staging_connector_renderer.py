import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
CONNECTOR_DIR = ROOT / "development" / "video_poc" / "benchmarks" / "connector"
MODULE_PATH = CONNECTOR_DIR / "render_staging_connector.py"
SPEC = importlib.util.spec_from_file_location("staging_connector_renderer", MODULE_PATH)
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)

NONZERO_DIGEST_A = "1" * 64
NONZERO_DIGEST_B = "2" * 64
FIXTURE_SHA256 = "ac81100d9310bd4e9c02bc0b13b6492781d009742ced347766b2601be3c44ad4"


def valid_document():
    document = json.loads((CONNECTOR_DIR / "staging.example.json").read_text())
    document["connectorImage"] = (
        "us-central1-docker.pkg.dev/roboflow-staging/video-proc/"
        f"rf-video-connector@sha256:{NONZERO_DIGEST_A}"
    )
    document["fixtureInitImage"] = (
        f"docker.io/curlimages/curl@sha256:{NONZERO_DIGEST_B}"
    )
    assert document["fixtures"][0]["sha256"] == FIXTURE_SHA256
    return document


def write_config(tmp_path, document=None):
    path = tmp_path / "connector.json"
    path.write_text(json.dumps(document or valid_document()))
    return path


def objects_by_kind(manifest):
    return {item["kind"]: item for item in manifest["items"]}


def test_example_is_fully_pinned_and_renderable():
    config = renderer.load_config(CONNECTOR_DIR / "staging.example.json")

    assert config["connectorImage"].endswith(
        "@sha256:5a81d357bab2490db782a409203da70dd89d0df642ad142780d86b05b50911c4"
    )
    assert config["fixtureInitImage"].endswith(
        "@sha256:5a91ea0c9c3ee27b4abe657b68cf6bf0676afa13b236b3bda34283cb3924d4f6"
    )


def test_render_is_headless_redacted_and_bound_to_exact_staging_identity(tmp_path):
    config = renderer.load_config(write_config(tmp_path))
    manifest = renderer.build_manifest(config)
    objects = objects_by_kind(manifest)
    deployment = objects["Deployment"]
    pod = deployment["spec"]["template"]["spec"]
    connector = pod["containers"][0]

    assert config["clusterIdentity"] == renderer.ALLOWED_CLUSTER_IDENTITY
    assert set(objects) == {
        "Namespace",
        "ServiceAccount",
        "NetworkPolicy",
        "Deployment",
    }
    assert deployment["spec"]["replicas"] == 1
    assert deployment["spec"]["strategy"] == {"type": "Recreate"}
    assert "--ui-addr=" in connector["args"]
    assert "--net-discovery=false" in connector["args"]
    assert (
        connector["args"][connector["args"].index("--id") + 1]
        == "bench-fixture-connector"
    )
    assert connector["env"] == [
        {
            "name": "ROBOFLOW_API_KEY",
            "valueFrom": {
                "secretKeyRef": {
                    "name": "video-connector-benchmark-api-key",
                    "key": "api-key",
                }
            },
        }
    ]
    serialized = json.dumps(manifest)
    assert "apiKey\"" not in serialized
    assert "ROBOFLOW_API_KEY" in serialized
    assert "kind\": \"Service\"" not in serialized
    assert pod["automountServiceAccountToken"] is False
    assert pod["nodeSelector"] == {
        "kubernetes.io/os": "linux",
        "kubernetes.io/arch": "amd64",
    }
    assert connector["securityContext"]["readOnlyRootFilesystem"] is True
    assert connector["securityContext"]["runAsNonRoot"] is True
    assert connector["securityContext"]["capabilities"] == {"drop": ["ALL"]}
    mounts = {item["mountPath"]: item for item in connector["volumeMounts"]}
    assert mounts["/fixtures"]["readOnly"] is True
    assert mounts["/state"].get("readOnly") is not True


def test_fixture_init_is_digest_checked_into_empty_dir(tmp_path):
    config = renderer.load_config(write_config(tmp_path))
    pod = objects_by_kind(renderer.build_manifest(config))["Deployment"]["spec"][
        "template"
    ]["spec"]
    init = pod["initContainers"][0]

    assert init["image"].endswith(f"@sha256:{NONZERO_DIGEST_B}")
    assert "sha256sum -c -" in init["args"][0]
    assert (
        init["args"][2]
        == "https://media.roboflow.com/supervision/video-examples/vehicles.mp4"
    )
    assert init["args"][3] == "/fixtures/vehicles.mp4"
    assert init["args"][4] == FIXTURE_SHA256
    assert init["volumeMounts"] == [{"name": "fixtures", "mountPath": "/fixtures"}]
    assert {item["name"] for item in pod["volumes"]} == {
        "fixtures",
        "state",
        "scratch",
    }


def test_network_policy_denies_ingress_and_limits_outbound_ports(tmp_path):
    config = renderer.load_config(write_config(tmp_path))
    policy = objects_by_kind(renderer.build_manifest(config))["NetworkPolicy"]["spec"]

    assert policy["policyTypes"] == ["Ingress", "Egress"]
    assert policy["ingress"] == []
    ports = {
        (port["protocol"], port["port"])
        for rule in policy["egress"]
        for port in rule["ports"]
    }
    assert ports == {("UDP", 53), ("TCP", 53), ("TCP", 443), ("TCP", 8554)}


def test_plan_contains_only_literal_commands_and_redacted_secret(tmp_path):
    config = renderer.load_config(write_config(tmp_path))
    manifest_path = tmp_path / "rendered.json"
    plan = renderer.build_plan(config, manifest_path)

    assert plan["performedClusterWrites"] is False
    assert plan["credential"]["value"] == "[redacted]"
    assert plan["clusterIdentity"] == renderer.ALLOWED_CLUSTER_IDENTITY
    assert all(isinstance(command, str) for command in plan["commands"].values())
    assert plan["commands"]["apply"].startswith("kubectl --context ck8s-stg apply")
    assert "delete" in plan["commands"]["cleanup"]
    assert "delete namespace" not in plan["commands"]["cleanup"]
    assert "delete namespace" in plan["commands"]["cleanupNamespace"]
    assert "secret" not in " ".join(plan["commands"].values()).lower()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(environment="production"), "environment"),
        (
            lambda value: value["clusterIdentity"].update(context="prod"),
            "clusterIdentity",
        ),
        (lambda value: value.update(namespace="video-proc"), "namespace"),
        (lambda value: value.update(connectorId="connector"), "connectorId"),
        (lambda value: value.update(apiBase="https://api.roboflow.com"), "apiBase"),
        (lambda value: value.update(connectorImage="repo:latest"), "digest"),
        (
            lambda value: value["nodePlacement"].update(
                nodeSelector={"kubernetes.io/arch": "amd64"}
            ),
            "Linux",
        ),
    ],
)
def test_renderer_rejects_non_staging_or_unbounded_runtime_config(
    tmp_path, mutate, message
):
    document = valid_document()
    mutate(document)
    with pytest.raises(ValueError, match=message):
        renderer.load_config(write_config(tmp_path, document))


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"url": "http://media.roboflow.com/vehicles.mp4"}, "HTTPS"),
        ({"url": "https://evil.example/vehicles.mp4"}, "allowlisted"),
        ({"url": "https://media.roboflow.com/vehicles.mp4?token=value"}, "plain HTTPS"),
        ({"filename": "../vehicles.mp4"}, "path-safe"),
        ({"sha256": "not-a-digest"}, "SHA-256"),
        ({"sha256": "0" * 64}, "SHA-256"),
    ],
)
def test_renderer_rejects_untrusted_or_unverified_fixtures(tmp_path, update, message):
    document = valid_document()
    document["fixtures"][0].update(update)
    with pytest.raises(ValueError, match=message):
        renderer.load_config(write_config(tmp_path, document))


def test_renderer_rejects_inline_key_material_and_unknown_fields(tmp_path):
    document = valid_document()
    document["apiKey"] = "do-not-serialize"
    with pytest.raises(ValueError, match="unknown fields"):
        renderer.load_config(write_config(tmp_path, document))


def test_render_writes_only_manifest_and_plan_without_cluster_calls(tmp_path):
    output = tmp_path / "out"
    manifest_path, plan_path = renderer.render(write_config(tmp_path), output)

    assert {path.name for path in output.iterdir()} == {
        "bench-fixture-connector.manifest.json",
        "bench-fixture-connector.plan.json",
    }
    assert manifest_path == output / "bench-fixture-connector.manifest.json"
    assert plan_path == output / "bench-fixture-connector.plan.json"
    assert json.loads(plan_path.read_text())["performedClusterWrites"] is False
    source = MODULE_PATH.read_text()
    assert "import subprocess" not in source
    assert "import kubernetes" not in source
