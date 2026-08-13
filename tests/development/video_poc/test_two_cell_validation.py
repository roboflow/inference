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

from render_two_cell_validation import (  # noqa: E402
    load_scenario,
    render,
    write_rendered,
)

EXAMPLE = NETWORKING_DIR / "two-cell-south-east.staging.example.json"


def example_dict():
    return json.loads(EXAMPLE.read_text())


def load(tmp_path, raw=None):
    path = tmp_path / "two-cell.json"
    path.write_text(json.dumps(raw or example_dict()))
    return load_scenario(path)


def test_example_renders_separate_staging_bundles_with_explicit_cells(tmp_path):
    scenario = load(tmp_path)
    bundles, manifest, report = render(scenario, "two-cell-001")

    assert set(bundles) == {"east", "south"}
    assert len(bundles["east"]["items"]) == 6
    assert len(bundles["south"]["items"]) == 3
    assert manifest["environment"] == "staging"
    assert manifest["executionMode"] == "render-only"
    assert manifest["agentImage"].endswith("@sha256:" + "0" * 64)
    assert manifest["cells"]["east"]["cellId"] == "crusoe-use1"
    assert manifest["cells"]["south"]["cellId"] == "crusoe-ussc1"
    assert manifest["cells"]["east"]["clusterContext"] == "ck8s-stg"
    assert manifest["cells"]["south"]["clusterContext"] == ("ck8s-stg-us-southcentral1")
    assert all(
        location["nodeSelector"] == {"node.kubernetes.io/instance-type": "c1a.16x"}
        for location in scenario["locations"].values()
    )
    assert "l40s" not in EXAMPLE.read_text().lower()
    assert report["scenarioSha256"] == manifest["scenarioSha256"]
    assert report["secretsRecorded"] is False


def test_cross_cell_job_runs_on_east_cpu_but_reads_south_public_endpoint(tmp_path):
    bundles, manifest, _report = render(load(tmp_path), "two-cell-001")
    cross_job = next(
        job
        for job in bundles["east"]["items"]
        if job["metadata"]["labels"]["benchmark.roboflow.com/media-path"]
        == "south-origin-east-processor"
    )
    labels = cross_job["metadata"]["labels"]
    assert labels["benchmark.roboflow.com/expected-cell"] == "crusoe-use1"
    assert labels["benchmark.roboflow.com/source-cell"] == "crusoe-ussc1"
    assert labels["benchmark.roboflow.com/path-kind"] == "explicit-cross-cell"
    assert cross_job["metadata"]["namespace"] == "video-proc-bench"
    assert cross_job["spec"]["template"]["spec"]["nodeSelector"] == {
        "node.kubernetes.io/instance-type": "c1a.16x"
    }
    container = cross_job["spec"]["template"]["spec"]["containers"][0]
    env = {value["name"]: value for value in container["env"]}
    assert env["BENCH_INPUT_URL"]["valueFrom"]["secretKeyRef"] == {
        "name": "replace-south-cell-endpoints",
        "key": "consume-public-template",
    }
    assert env["EXPECTED_CELL_ID"]["value"] == "crusoe-use1"
    assert env["EXPECTED_CLUSTER_CONTEXT"]["value"] == "ck8s-stg"
    assert env["MEDIA_PATH_KIND"]["value"] == "explicit-cross-cell"
    rendered = json.dumps({"bundles": bundles, "manifest": manifest})
    assert "rtsp://" not in rendered
    assert "https://" not in rendered


def test_same_cell_path_requires_internal_endpoint_and_remote_false(tmp_path):
    raw = example_dict()
    raw["mediaPaths"][0]["remoteExecution"] = True
    with pytest.raises(ValueError, match="same-cell paths must be local"):
        load(tmp_path, raw)

    raw = example_dict()
    raw["mediaPaths"][0]["consumeEndpoint"] = "consumePublicTemplate"
    with pytest.raises(ValueError, match="consumeInternalTemplate"):
        load(tmp_path, raw)


def test_cross_cell_path_must_be_explicit_and_use_public_endpoint(tmp_path):
    raw = example_dict()
    cross = raw["mediaPaths"][2]
    cross["remoteExecution"] = False
    with pytest.raises(ValueError, match="remoteExecution=true"):
        load(tmp_path, raw)

    raw = example_dict()
    raw["mediaPaths"][2]["consumeEndpoint"] = "consumeInternalTemplate"
    with pytest.raises(ValueError, match="consumePublicTemplate"):
        load(tmp_path, raw)


def test_staging_render_only_and_secret_reference_guards_are_hard(tmp_path):
    raw = example_dict()
    raw["environment"] = "production"
    with pytest.raises(ValueError, match="staging-only"):
        load(tmp_path, raw)

    raw = example_dict()
    raw["cells"]["south"]["clusterContext"] = "production"
    with pytest.raises(ValueError, match="recognizably staging"):
        load(tmp_path, raw)

    raw = example_dict()
    raw["executionMode"] = "apply"
    with pytest.raises(ValueError, match="render-only"):
        load(tmp_path, raw)

    raw = example_dict()
    raw["cells"]["east"]["endpoints"]["ingestTemplate"] = {
        "value": "rtsp://credential@example.test/{stream}"
    }
    with pytest.raises(ValueError, match="Secret name and key"):
        load(tmp_path, raw)


def test_control_plane_evidence_requires_assignment_reconnect_and_policy(tmp_path):
    raw = example_dict()
    raw["controlPlaneEvidence"]["cases"] = raw["controlPlaneEvidence"]["cases"][:3]
    with pytest.raises(ValueError, match="cell-loss recovery"):
        load(tmp_path, raw)

    raw = example_dict()
    raw["controlPlaneEvidence"]["cases"][0]["artifacts"][
        "activated"
    ] = "https://signed.example/evidence?token=secret"
    with pytest.raises(ValueError, match="relative non-URL artifact path"):
        load(tmp_path, raw)


def test_report_contract_covers_placement_metrics_network_and_recovery(tmp_path):
    _bundles, manifest, report = render(load(tmp_path), "two-cell-001")
    collection = manifest["collectionContract"]

    assert set(collection["placement"]) == {"east", "south"}
    assert "observedCell" in collection["placement"]["east"]["requiredObservedFields"]
    assert any(
        "MediaMTX" in item for item in collection["prometheus"]["requiredEvidence"]
    )
    assert any(
        "processor cell" in item
        for item in collection["prometheus"]["requiredEvidence"]
    )
    assert "source-cell egress bytes" in collection["network"]["requiredPerPath"]
    assert collection["failureTimeline"]["clock"].startswith("UTC RFC3339")
    assert "actual connector publish endpoint" in " ".join(
        collection["controlPlaneRouting"]["requiredPerSource"]
    )
    assert "preview request" in " ".join(
        collection["controlPlaneRouting"]["requiredPerSource"]
    )
    assert (
        "processor claim/status cell assertion"
        in collection["controlPlaneRouting"]["requiredPerSource"]
    )
    assert report["assertions"]["firstActivationAssignment"] == "pending"
    assert report["assertions"]["explicitCrossCellExperiment"] == "pending"
    assert report["assertions"]["failureRecovery"] == "pending"
    assert report["assertions"]["cellLossUnavailableBounded"] == "pending"
    assert report["assertions"]["previewFromAssignedCell"] == "pending"
    assert report["assertions"]["processorClaimedExpectedCell"] == "pending"
    assert report["assertions"]["relaySessionsAndReadersMatch"] == "pending"
    failure = manifest["measurementPlan"]["failureExperiments"][0]
    assert failure["targetCell"] == "south"
    assert failure["maxUnavailableSeconds"] == 120
    assert failure["reassignmentPolicy"] == "retain-home-cell"
    assert "cell-marked-unavailable" in failure["requiredEvents"]
    assert "reassignment-decision" in failure["requiredEvents"]
    assert manifest["rollback"]["cellResourcesChanged"] is False
    assert manifest["rollback"]["failureExperimentRollbackRequiresApproval"] is True


def test_scenario_hash_is_stable_and_outputs_refuse_overwrite(tmp_path):
    scenario = load(tmp_path)
    first = render(scenario, "two-cell-001")
    second = render(scenario, "two-cell-001")
    assert first[1]["scenarioSha256"] == second[1]["scenarioSha256"]

    output_dir = tmp_path / "rendered"
    paths = write_rendered(output_dir, *first)
    assert {path.name for path in paths} == {
        "jobs-east.json",
        "jobs-south.json",
        "validation-manifest.json",
        "validation-report.template.json",
    }
    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_rendered(output_dir, *first)


def test_failure_experiment_requires_approval_and_recovery_artifacts(tmp_path):
    raw = copy.deepcopy(example_dict())
    raw["measurementPlan"]["failureExperiments"][0]["requiresApproval"] = False
    with pytest.raises(ValueError, match="explicitly require approval"):
        load(tmp_path, raw)
