import copy
import json
import sys
from pathlib import Path

import pytest

NETWORKING_DIR = (
    Path(__file__).resolve().parents[3]
    / "development"
    / "video_poc"
    / "benchmarks"
    / "networking"
)
sys.path.insert(0, str(NETWORKING_DIR))

from render_distributed_relay_benchmark import (  # noqa: E402
    expand_agents,
    load_scenario,
    render,
    resolve_run_id,
)


def scenario_dict():
    return {
        "schemaVersion": 1,
        "name": "calibration",
        "environment": "staging",
        "clusterContext": "ck8s-stg",
        "namespace": "video-proc-bench",
        "serviceAccountName": "video-relay-benchmark",
        "agentImage": "registry/relay-agent@sha256:" + "a" * 64,
        "fixture": {
            "value": "/fixtures/input.mp4",
            "metadata": {"fps": 15, "sha256": "fixture-hash"},
        },
        "mediaUrlSecret": {
            "publishTemplate": {"name": "run-media", "key": "publish"},
            "readTemplate": {"name": "run-media", "key": "read"},
        },
        "locations": {
            "generator": {"nodeSelector": {"pool": "generator"}},
            "processor": {"nodeSelector": {"pool": "gpu"}},
        },
        "publishers": [
            {"name": "sources", "location": "generator", "count": 2}
        ],
        "readers": [
            {
                "name": "copies",
                "role": "read-copy",
                "location": "processor",
                "publisherGroups": ["sources"],
                "copiesPerStream": 2,
            },
            {
                "name": "decode",
                "role": "read-decode",
                "location": "processor",
                "publisherGroups": ["sources"],
                "copiesPerStream": 1,
            },
        ],
    }


def load(tmp_path, raw=None):
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(raw or scenario_dict()))
    return load_scenario(path)


def test_staging_and_dedicated_namespace_are_hard_guards(tmp_path):
    raw = scenario_dict()
    raw["environment"] = "production"
    with pytest.raises(ValueError, match="restricted to staging"):
        load(tmp_path, raw)

    raw = scenario_dict()
    raw["clusterContext"] = "production"
    with pytest.raises(ValueError, match="recognizably staging"):
        load(tmp_path, raw)

    raw = scenario_dict()
    raw["namespace"] = "video-proc"
    with pytest.raises(ValueError, match="dedicated benchmark namespace"):
        load(tmp_path, raw)


def test_locations_require_explicit_placement_and_image_is_immutable(tmp_path):
    raw = scenario_dict()
    raw["locations"]["generator"] = {}
    with pytest.raises(ValueError, match="nodeSelector or nodeName"):
        load(tmp_path, raw)

    raw = scenario_dict()
    raw["agentImage"] = "registry/relay-agent:latest"
    with pytest.raises(ValueError, match="immutable sha256"):
        load(tmp_path, raw)


def test_credentialed_fixture_urls_must_use_a_secret(tmp_path):
    raw = scenario_dict()
    raw["fixture"]["value"] = "https://user:secret@example.test/input.mp4?token=x"
    with pytest.raises(ValueError, match="must use fixture.secretKeyRef"):
        load(tmp_path, raw)


def test_secret_fixture_is_referenced_without_value_in_rendered_output(tmp_path):
    raw = scenario_dict()
    raw["fixture"].pop("value")
    raw["fixture"]["secretKeyRef"] = {
        "name": "run-media",
        "key": "fixture",
    }
    scenario = load(tmp_path, raw)
    jobs, manifest = render(scenario, "relay-secret-fixture-001")
    publish_job = next(
        item
        for item in jobs["items"]
        if item["metadata"]["labels"]["benchmark.roboflow.com/role"]
        == "publish-copy"
    )
    env = publish_job["spec"]["template"]["spec"]["containers"][0]["env"]
    input_env = next(item for item in env if item["name"] == "BENCH_INPUT_URL")
    assert input_env == {
        "name": "BENCH_INPUT_URL",
        "valueFrom": {
            "secretKeyRef": {"name": "run-media", "key": "fixture"}
        },
    }
    assert manifest["fixture"]["source"] == "secretKeyRef"


def test_expansion_makes_one_job_per_agent_and_maps_readers_to_streams(tmp_path):
    scenario = load(tmp_path)
    agents = expand_agents(scenario, "relay-001")

    assert len(agents) == 8
    assert [agent["role"] for agent in agents].count("publish-copy") == 2
    assert [agent["role"] for agent in agents].count("read-copy") == 4
    assert [agent["role"] for agent in agents].count("read-decode") == 2
    assert len({agent["stream"] for agent in agents}) == 2
    assert all(
        agent["startAfterSeconds"] == 5
        for agent in agents
        if agent["role"].startswith("read-")
    )


def test_render_uses_secret_refs_deadlines_placement_and_redacted_manifest(tmp_path):
    scenario = load(tmp_path)
    jobs, manifest = render(scenario, "relay-001")

    assert jobs["kind"] == "List"
    assert len(jobs["items"]) == 8
    publish_job = next(
        item
        for item in jobs["items"]
        if item["metadata"]["labels"]["benchmark.roboflow.com/role"]
        == "publish-copy"
    )
    container = publish_job["spec"]["template"]["spec"]["containers"][0]
    env = {item["name"]: item for item in container["env"]}
    assert env["BENCH_OUTPUT_URL"]["valueFrom"]["secretKeyRef"] == {
        "name": "run-media",
        "key": "publish",
    }
    assert env["REQUESTED_NODE_INSTANCE_TYPE"]["value"] == ""
    assert publish_job["spec"]["template"]["spec"]["nodeSelector"] == {
        "pool": "generator"
    }
    assert publish_job["spec"]["backoffLimit"] == 0
    assert publish_job["spec"]["activeDeadlineSeconds"] == 120
    assert publish_job["spec"]["ttlSecondsAfterFinished"] == 600
    assert container["terminationMessagePath"] == "/dev/termination-log"
    rendered = json.dumps({"jobs": jobs, "manifest": manifest})
    assert "user:password" not in rendered
    assert "run-media" in rendered
    assert manifest["fixture"]["source"] == "non-secret-value"
    assert manifest["cleanup"]["labelSelector"].endswith("relay-001")


def test_unknown_reader_group_is_rejected(tmp_path):
    raw = copy.deepcopy(scenario_dict())
    raw["readers"][0]["publisherGroups"] = ["missing"]
    with pytest.raises(ValueError, match="unknown publisher group"):
        load(tmp_path, raw)


def test_render_rejects_expansion_over_declared_agent_limit(tmp_path):
    raw = scenario_dict()
    raw["maxAgents"] = 7
    scenario = load(tmp_path, raw)
    with pytest.raises(ValueError, match="exceeding maxAgents=7"):
        render(scenario, "relay-over-limit-001")


def test_run_id_is_kubernetes_safe():
    assert resolve_run_id("relay-calibration-001") == "relay-calibration-001"
    with pytest.raises(ValueError, match="Kubernetes DNS label"):
        resolve_run_id("Relay_Invalid")
