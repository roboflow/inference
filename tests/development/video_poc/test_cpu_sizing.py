import base64
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = REPOSITORY / "development" / "video_poc" / "benchmarks"
CPU_DIR = BENCHMARK_DIR / "cpu_sizing"
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(CPU_DIR))

from build_processor_jobs import load_corpus  # noqa: E402
from run_api_experiment_matrix import load_matrix  # noqa: E402
from run_api_workflow_corpus import parse_workload  # noqa: E402


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


renderer = load_module("cpu_sizing_renderer", CPU_DIR / "render_variant.py")
collector = load_module("cpu_sizing_collector", CPU_DIR / "collect_cpu_telemetry.py")
arrival = load_module("cpu_sizing_arrival", CPU_DIR / "analyze_arrival.py")
builder = load_module("cpu_process_builder", CPU_DIR / "build_process_image.py")
preflight = load_module("cpu_sizing_preflight", CPU_DIR / "verify_rollout_patch.py")
gate = load_module("cpu_process_gate", CPU_DIR / "validate_process_gate.py")


def process_provenance(catalog):
    image = catalog["imagePolicy"]["repository"] + "@sha256:" + "a" * 64
    revision = "b" * 40
    manifest = {
        "development/video_poc/processor/processor.py": "c" * 64,
        "development/video_poc/processor/Dockerfile.overlay": "d" * 64,
    }
    build = {
        "status": "SUCCESS",
        "substitutions": {
            "_BASE_PROCESSOR_IMAGE": catalog["observedBaseline"]["image"],
            "_OUTPUT": f'{catalog["imagePolicy"]["repository"]}:{revision}',
            "_GIT_SHA": revision,
            "_RUNTIME_VARIANT": "cpu-process",
        },
        "sourceProvenance": {
            "resolvedStorageSource": {"bucket": "source", "object": "object"},
            "fileHashes": {
                path.removeprefix("development/video_poc/processor/"): {
                    "fileHash": [
                        {
                            "type": "SHA256",
                            "value": base64.b64encode(bytes.fromhex(digest)).decode(),
                        }
                    ]
                }
                for path, digest in manifest.items()
            },
        },
        "results": {
            "images": [
                {
                    "name": f'{catalog["imagePolicy"]["repository"]}:{revision}',
                    "digest": "sha256:" + "a" * 64,
                }
            ]
        },
    }
    evidence = {
        "build": build,
        "registry": {"image_summary": {"fully_qualified_digest": image}},
        "smoke": {"status": "SUCCESS", "substitutions": {"_IMAGE": image}},
    }
    provenance = {
        "schemaVersion": 1,
        "environment": "staging",
        "image": image,
        "baseImage": catalog["observedBaseline"]["image"],
        "sourceRevision": revision,
        "sourceFilesSha256": manifest,
        "sourceManifestSha256": renderer._manifest_hash(manifest),
        "evidence": {
            name: {"path": f"{name}.json", "sha256": "e" * 64} for name in evidence
        },
    }
    return provenance, evidence


def test_cpu_size_catalog_is_staging_only_and_uses_guaranteed_classes():
    catalog = renderer.load_catalog(CPU_DIR / "size_classes.json")

    assert catalog["clusterContext"] == "ck8s-stg"
    assert catalog["namespace"] == "video-proc"
    assert "@sha256:" in catalog["observedBaseline"]["image"]
    assert catalog["observedBaseline"]["qosClass"] == "Burstable"
    assert {
        name: (item["cpu"], item["memory"], item["maxJobs"])
        for name, item in catalog["sizeClasses"].items()
    } == {
        "cpu-2": ("2", "4Gi", 1),
        "cpu-4": ("4", "8Gi", 2),
        "cpu-8": ("8", "16Gi", 4),
    }


@pytest.mark.parametrize("size", ("cpu-2", "cpu-4", "cpu-8"))
def test_thread_patch_is_exact_immutable_and_guaranteed(size):
    catalog = renderer.load_catalog(CPU_DIR / "size_classes.json")
    patch = renderer.render_patch(catalog, size, "thread")
    container = patch["spec"]["template"]["spec"]["containers"][0]
    expected = catalog["sizeClasses"][size]
    environment = {item["name"]: item["value"] for item in container["env"]}

    assert patch["metadata"] == {
        "name": "video-processor-pool-cpu",
        "namespace": "video-proc",
    }
    assert set(patch["spec"]) == {"template"}
    assert "selector" not in patch["spec"]
    assert "replicas" not in patch["spec"]
    assert "nodeSelector" not in patch["spec"]["template"]["spec"]
    assert container["image"] == catalog["observedBaseline"]["image"]
    assert container["resources"]["requests"] == container["resources"]["limits"]
    assert container["resources"]["limits"] == {
        "cpu": expected["cpu"],
        "memory": expected["memory"],
    }
    assert environment["MAX_CONCURRENT_JOBS"] == str(expected["maxJobs"])
    assert environment["ONNX_INTRA_OP_THREADS"] == "2"
    assert environment["OMP_NUM_THREADS"] == "2"
    assert environment["PROCESSOR_JOB_EXECUTION_MODE"] == "thread"
    assert environment["PROCESSOR_EXECUTION_DOMAIN_MODE"] == "in_process"
    assert environment["VIDEO_PROC_IMAGE"] == container["image"]
    assert (
        environment["VIDEO_PROC_GIT_SHA"]
        == catalog["observedBaseline"]["sourceRevision"]
    )
    assert (
        json.loads((CPU_DIR / "patches" / f"{size}-thread.json").read_text()) == patch
    )


def test_process_patch_requires_validated_build_and_smoke_provenance():
    catalog = renderer.load_catalog(CPU_DIR / "size_classes.json")
    with pytest.raises(ValueError, match="validated process provenance"):
        renderer.render_patch(catalog, "cpu-4", "process")

    provenance, evidence = process_provenance(catalog)
    validated = renderer.validate_process_provenance(
        catalog,
        provenance,
        ancestor_check=lambda _base, _revision: True,
        source_manifest_loader=lambda _revision: provenance["sourceFilesSha256"],
        evidence_loader=lambda descriptor: evidence[descriptor["path"].split(".")[0]],
    )
    original_validator = renderer.validate_process_provenance
    renderer.validate_process_provenance = (
        lambda _catalog, _value, _path=None: validated
    )
    try:
        patch = renderer.render_patch(catalog, "cpu-4", "process", provenance)
    finally:
        renderer.validate_process_provenance = original_validator
    container = patch["spec"]["template"]["spec"]["containers"][0]
    environment = {item["name"]: item["value"] for item in container["env"]}
    assert container["image"] == provenance["image"]
    assert environment["PROCESSOR_JOB_EXECUTION_MODE"] == "process"
    assert environment["VIDEO_PROC_RUNTIME_VARIANT"] == "cpu-4-process"
    assert environment["VIDEO_PROC_GIT_SHA"] == "b" * 40


def test_process_provenance_rejects_baseline_failed_smoke_and_old_source():
    catalog = renderer.load_catalog(CPU_DIR / "size_classes.json")
    provenance, evidence = process_provenance(catalog)
    provenance["image"] = catalog["observedBaseline"]["image"]
    with pytest.raises(ValueError, match="thread-only baseline"):
        renderer.validate_process_provenance(
            catalog,
            provenance,
            ancestor_check=lambda _base, _revision: True,
            source_manifest_loader=lambda _revision: provenance["sourceFilesSha256"],
            evidence_loader=lambda descriptor: evidence[
                descriptor["path"].split(".")[0]
            ],
        )

    provenance["image"] = catalog["imagePolicy"]["repository"] + "@sha256:" + "a" * 64
    evidence["smoke"]["status"] = "FAILURE"
    with pytest.raises(ValueError, match="smoke did not succeed"):
        renderer.validate_process_provenance(
            catalog,
            provenance,
            ancestor_check=lambda _base, _revision: True,
            source_manifest_loader=lambda _revision: provenance["sourceFilesSha256"],
            evidence_loader=lambda descriptor: evidence[
                descriptor["path"].split(".")[0]
            ],
        )
    evidence["smoke"]["status"] = "SUCCESS"
    with pytest.raises(ValueError, match="descendant"):
        renderer.validate_process_provenance(
            catalog,
            provenance,
            ancestor_check=lambda _base, _revision: False,
            source_manifest_loader=lambda _revision: provenance["sourceFilesSha256"],
            evidence_loader=lambda descriptor: evidence[
                descriptor["path"].split(".")[0]
            ],
        )


def test_process_provenance_rejects_cloud_build_source_hash_drift():
    catalog = renderer.load_catalog(CPU_DIR / "size_classes.json")
    provenance, evidence = process_provenance(catalog)
    evidence["build"]["sourceProvenance"]["fileHashes"]["processor.py"]["fileHash"][0][
        "value"
    ] = base64.b64encode(bytes.fromhex("f" * 64)).decode()
    with pytest.raises(ValueError, match="input hashes differ"):
        renderer.validate_process_provenance(
            catalog,
            provenance,
            ancestor_check=lambda _base, _revision: True,
            source_manifest_loader=lambda _revision: provenance["sourceFilesSha256"],
            evidence_loader=lambda descriptor: evidence[
                descriptor["path"].split(".")[0]
            ],
        )


def test_process_provenance_loader_requires_committed_cpu_record(tmp_path):
    outside = tmp_path / "process.json"
    outside.write_text("{}")
    with pytest.raises(ValueError, match="must live"):
        renderer.load_tracked_process_provenance(
            outside, tracked_check=lambda _path: True
        )

    candidate = renderer.PROVENANCE_DIR / "future-process.json"
    candidate.write_text("{}")
    try:
        with pytest.raises(ValueError, match="must be committed"):
            renderer.load_tracked_process_provenance(
                candidate,
                tracked_check=lambda _path: False,
                committed_content_loader=lambda _path: b"{}",
            )
        path, document = renderer.load_tracked_process_provenance(
            candidate,
            tracked_check=lambda _path: True,
            committed_content_loader=lambda _path: b"{}",
        )
        assert path == candidate.resolve()
        assert document == {}
        candidate.write_text('{"modified": true}')
        with pytest.raises(ValueError, match="differs from the committed"):
            renderer.load_tracked_process_provenance(
                candidate,
                tracked_check=lambda _path: True,
                committed_content_loader=lambda _path: b"{}",
            )
    finally:
        candidate.unlink()


def test_process_builder_materializes_only_exact_revision_bytes(tmp_path):
    revision = (
        __import__("subprocess")
        .run(
            ["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
    )
    destination = tmp_path / "context"
    destination.mkdir()
    manifest = builder.materialize_exact_context(REPOSITORY, revision, destination)

    assert manifest == renderer._exact_revision_source_manifest(REPOSITORY, revision)
    assert not any(path.name == "__pycache__" for path in destination.rglob("*"))
    assert (destination / "cloudbuild.process-overlay.yaml").is_file()
    assert (destination / "job_process.py").is_file()


def test_cpu_matrix_and_cpu_workflow_corpus_are_complete_and_credential_free():
    matrix_path = CPU_DIR / "cpu-capacity.staging.json"
    document = json.loads(matrix_path.read_text())
    matrix = load_matrix(matrix_path)
    corpus = load_corpus(BENCHMARK_DIR / "workflows" / "manifest.json")

    assert document["environment"] == "staging"
    assert document["defaults"]["sourceId"] == "d5XmPQAZssPpE3clCmcY"
    assert len(matrix["scenarios"]) == 27
    assert all(item["repetitions"] == 2 for item in matrix["scenarios"])
    assert all(item["requireSingleProcessor"] for item in matrix["scenarios"])
    mixed = {
        item["name"]: item for item in document["scenarios"] if "-mix-" in item["name"]
    }
    assert mixed
    assert all(
        any("@60" in workload for workload in item["workloads"])
        for item in mixed.values()
    )
    serialized = json.dumps(document).lower()
    assert "authorization" not in serialized
    assert "bearer " not in serialized

    expected_cpu_profiles = {
        "cpu-blur",
        "single-detection-cpu",
        "detection-tracking-cpu",
        "dual-detection-cpu",
        "instance-segmentation-cpu",
    }
    assert expected_cpu_profiles <= set(corpus)
    for scenario in document["scenarios"]:
        for workload in scenario["workloads"]:
            profile = parse_workload(workload)["profile"]
            assert profile in corpus
            assert corpus[profile]["tier"] == "cpu"
    for cpu_profile in expected_cpu_profiles:
        assert corpus[cpu_profile]["tier"] == "cpu"
    for base, cpu in (
        ("single-detection", "single-detection-cpu"),
        ("detection-tracking", "detection-tracking-cpu"),
        ("dual-detection", "dual-detection-cpu"),
        ("instance-segmentation", "instance-segmentation-cpu"),
    ):
        assert corpus[base]["specification"] == corpus[cpu]["specification"]
        assert corpus[base]["models"] == corpus[cpu]["models"]


def test_cpu_telemetry_queries_are_bounded_and_include_throttling():
    queries = collector.cpu_queries(["video-processor-pool-cpu-abc12"])

    assert {
        "processorCpuThrottledPeriodsRatio",
        "processorThreads",
        "processorRestarts",
        "processorCpuRequests",
        "processorMemoryRequestsBytes",
        "processorCpuLimits",
        "processorMemoryLimitsBytes",
        "processorGuaranteedQos",
        "processorOomKilled",
        "processorEvicted",
        "processorActiveJobs",
        "processorImageInfo",
    } == set(queries)
    serialized = json.dumps(queries)
    assert 'namespace=\\"video-proc\\"' in serialized
    assert 'container=\\"processor\\"' in serialized
    assert "apiKey" not in serialized
    assert "secret" not in serialized
    node_query = collector.node_cpu_query(["np-cpu-1.example.internal"])
    assert "kube_node_status_allocatable" in node_query
    assert 'resource="cpu"' in node_query


def test_cpu_certification_fails_closed_on_missing_metric_coverage():
    catalog = renderer.load_catalog(CPU_DIR / "size_classes.json")
    report = {
        "measurementStartedAt": "2026-08-13T00:00:00Z",
        "measurementEndedAt": "2026-08-13T00:03:00Z",
        "plannedConcurrency": 1,
        "jobs": [
            {
                "stats": {
                    "runtime": {
                        "variant": "cpu-2-thread",
                        "image": catalog["observedBaseline"]["image"],
                        "revision": catalog["observedBaseline"]["sourceRevision"],
                        "processId": 10,
                    }
                }
            }
        ],
    }
    patch = renderer.render_patch(catalog, "cpu-2", "thread")
    expected = collector.expected_runtime_from_patch(patch, catalog)
    evidence = collector.certification_evidence({}, report, catalog, expected)

    assert evidence["evidenceComplete"] is False
    assert evidence["checks"]["allRequiredMetricsCovered"] is False
    assert evidence["coverage"]["processorCpuThrottledPeriodsRatio"]["covered"] is False


def test_cpu_metric_coverage_requires_duration_cardinality_and_raw_samples():
    metric = {
        "summary": {"count": 999, "p95": 0, "max": 0},
        "series": [
            {
                "metric": {"pod": "cpu-a"},
                "values": [[0, "0"], [180, "1"]],
            }
        ],
    }
    sparse = collector._temporal_coverage(metric, 0, 180, 15, {"cpu-a"})
    assert sparse["covered"] is False
    assert sparse["sampleCount"] == 2
    wrong_pod = collector._temporal_coverage(metric, 0, 180, 180, {"cpu-b"})
    assert wrong_pod["covered"] is False
    assert wrong_pod["podsMatch"] is False
    duplicate = copy.deepcopy(metric)
    duplicate["series"].append(copy.deepcopy(duplicate["series"][0]))
    duplicate_coverage = collector._temporal_coverage(duplicate, 0, 180, 180, {"cpu-a"})
    assert duplicate_coverage["covered"] is False
    assert duplicate_coverage["cardinalityMatches"] is False
    asymmetric = copy.deepcopy(metric)
    asymmetric["series"].append({"metric": {"pod": "cpu-b"}, "values": [[0, "0"]]})
    asymmetric_coverage = collector._temporal_coverage(
        asymmetric, 0, 180, 180, {"cpu-a", "cpu-b"}
    )
    assert asymmetric_coverage["podsMatch"] is True
    assert asymmetric_coverage["cardinalityMatches"] is True
    assert asymmetric_coverage["covered"] is False
    assert [item["covered"] for item in asymmetric_coverage["perSeries"]] == [
        True,
        False,
    ]
    assert collector._series_delta(metric) == 1


def test_live_patch_validator_rejects_uid_drift_and_working_pods():
    catalog = renderer.load_catalog(CPU_DIR / "size_classes.json")
    patch = renderer.render_patch(catalog, "cpu-2", "thread")
    processor = {
        "name": "processor",
        "image": catalog["observedBaseline"]["image"],
        "env": patch["spec"]["template"]["spec"]["containers"][0]["env"],
        "resources": {
            "requests": {"cpu": "2", "memory": "4Gi"},
            "limits": {"cpu": "2", "memory": "4Gi"},
        },
    }
    live = {
        "metadata": {
            "uid": "wrong",
            "namespace": "video-proc",
            "name": "video-processor-pool-cpu",
            "generation": 7,
        },
        "spec": {
            "replicas": 1,
            "template": {
                "metadata": {},
                "spec": {
                    "nodeSelector": {"pool": "default"},
                    "containers": [processor],
                },
            },
        },
        "status": {
            "observedGeneration": 7,
            "replicas": 1,
            "updatedReplicas": 1,
            "readyReplicas": 1,
            "availableReplicas": 1,
        },
    }
    with pytest.raises(ValueError, match="UID"):
        preflight.validate_live_and_rendered(
            catalog, patch, live, live, [], processor["image"]
        )
    live["metadata"]["uid"] = catalog["observedBaseline"]["deploymentUid"]
    with pytest.raises(ValueError, match="working"):
        preflight.validate_live_and_rendered(
            catalog, patch, live, live, [{}], processor["image"]
        )
    rendered = copy.deepcopy(live)
    rendered["spec"]["template"]["metadata"]["annotations"] = patch["spec"]["template"][
        "metadata"
    ]["annotations"]
    assert (
        preflight.validate_live_and_rendered(
            catalog, patch, live, rendered, [], processor["image"]
        )
        is True
    )


def test_patch_validator_rejects_extra_metadata_and_live_replica_drift():
    catalog = renderer.load_catalog(CPU_DIR / "size_classes.json")
    patch = renderer.render_patch(catalog, "cpu-2", "thread")
    malformed = copy.deepcopy(patch)
    malformed["metadata"]["uid"] = "unsafe-target"
    with pytest.raises(ValueError, match="metadata keys"):
        preflight.validate_patch_document(catalog, malformed)

    live = {
        "metadata": {
            "uid": catalog["observedBaseline"]["deploymentUid"],
            "namespace": catalog["namespace"],
            "name": catalog["deployment"],
            "generation": 7,
        },
        "spec": {
            "replicas": 2,
            "template": {
                "metadata": {},
                "spec": {
                    "nodeSelector": catalog["nodeSelector"],
                    "containers": [patch["spec"]["template"]["spec"]["containers"][0]],
                },
            },
        },
        "status": {
            "observedGeneration": 7,
            "replicas": 2,
            "updatedReplicas": 2,
            "readyReplicas": 2,
            "availableReplicas": 2,
        },
    }
    with pytest.raises(ValueError, match="exactly one desired"):
        preflight.validate_live_and_rendered(
            catalog,
            patch,
            live,
            live,
            [],
            patch["spec"]["template"]["spec"]["containers"][0]["image"],
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda container: container["env"][3].update({"value": "99"}),
            "environment differs",
        ),
        (
            lambda container: container["resources"]["limits"].update(
                {"memory": "5Gi"}
            ),
            "requests and limits",
        ),
        (
            lambda container: container["env"][1].update({"value": "cpu-2-unknown"}),
            "known CPU size/topology",
        ),
    ),
)
def test_patch_schema_rejects_semantic_size_topology_drift(mutation, message):
    catalog = renderer.load_catalog(CPU_DIR / "size_classes.json")
    patch = renderer.render_patch(catalog, "cpu-2", "thread")
    mutation(patch["spec"]["template"]["spec"]["containers"][0])
    with pytest.raises(ValueError, match=message):
        preflight.validate_patch_document(catalog, patch)


def test_delayed_arrival_analysis_requires_real_baseline_and_retention():
    report = {
        "runId": "cpu-mix",
        "success": True,
        "profiles": [
            {"ordinal": 1, "profile": "single-detection-cpu", "startAfterSeconds": 0},
            {
                "ordinal": 2,
                "profile": "detection-tracking-cpu",
                "startAfterSeconds": 60,
            },
        ],
        "starts": [
            {"ordinal": 1, "job": {"id": "incumbent"}},
            {"ordinal": 2, "job": {"id": "arrival"}},
        ],
        "samples": [],
    }
    for index in range(12):
        report["samples"].append(
            {
                "phase": "baseline",
                "elapsedSeconds": index * 2,
                "jobs": [{"id": "incumbent", "stats": {"frames": index * 16}}],
            }
        )
        report["samples"].append(
            {
                "phase": "measurement",
                "elapsedSeconds": 100 + index * 2,
                "jobs": [{"id": "incumbent", "stats": {"frames": 500 + index * 15}}],
            }
        )

    result = arrival.analyze(report)
    assert result["passed"] is True
    assert result["incumbents"][0]["retentionRatio"] == pytest.approx(0.9375)

    report["samples"] = [
        item for item in report["samples"] if item["phase"] != "baseline"
    ]
    result = arrival.analyze(report)
    assert result["passed"] is False
    assert result["incumbents"][0]["checks"]["baselineCoverage"] is False


def process_gate_fixture(tmp_path):
    catalog = renderer.load_catalog(CPU_DIR / "size_classes.json")
    provenance, _ = process_provenance(catalog)
    validated = {
        "image": provenance["image"],
        "sourceRevision": provenance["sourceRevision"],
    }
    original = renderer.validate_process_provenance
    renderer.validate_process_provenance = lambda *_args, **_kwargs: validated
    try:
        patch = renderer.render_patch(catalog, "cpu-4", "process", provenance)
    finally:
        renderer.validate_process_provenance = original
    runtime = {
        "image": provenance["image"],
        "revision": provenance["sourceRevision"],
        "variant": "cpu-4-process",
        "podUid": "pod-1",
    }
    artifacts = {
        "topologyReport": {
            "plannedConcurrency": 2,
            "jobs": [
                {"id": "job-a", "stats": {"runtime": {**runtime, "processId": 11}}},
                {"id": "job-b", "stats": {"runtime": {**runtime, "processId": 12}}},
            ],
        },
        "processSnapshot": {
            "runtime": runtime,
            "supervisorProcessId": 10,
            "processes": [
                {"pid": 10, "ppid": 1, "argv": ["python", "processor.py"]},
                {"pid": 11, "ppid": 10, "argv": ["python", "job_process.py"]},
                {"pid": 12, "ppid": 10, "argv": ["python", "job_process.py"]},
            ],
        },
        "cancellationObservation": {
            "runtime": runtime,
            "targetJobId": "job-a",
            "siblingJobId": "job-b",
            "action": {
                "type": "cancel",
                "targetProcessId": 11,
                "responseStatus": 200,
            },
            "observedExit": {"processId": 11, "observed": True, "exitCode": 0},
            "beforeProcesses": [
                {"pid": 10, "ppid": 1, "argv": ["python", "processor.py"]},
                {"pid": 11, "ppid": 10, "argv": ["python", "job_process.py"]},
                {"pid": 12, "ppid": 10, "argv": ["python", "job_process.py"]},
            ],
            "afterProcesses": [
                {"pid": 10, "ppid": 1, "argv": ["python", "processor.py"]},
                {"pid": 12, "ppid": 10, "argv": ["python", "job_process.py"]},
            ],
            "before": {
                "jobs": [
                    {"id": "job-a", "state": "running", "frames": 10},
                    {"id": "job-b", "state": "running", "frames": 10},
                ]
            },
            "after": {
                "jobs": [
                    {"id": "job-a", "state": "cancelled", "frames": 10},
                    {"id": "job-b", "state": "running", "frames": 20},
                ]
            },
        },
        "crashObservation": {
            "runtime": runtime,
            "targetJobId": "job-a",
            "siblingJobId": "job-b",
            "action": {
                "type": "signal",
                "signal": "SIGKILL",
                "targetProcessId": 11,
                "deliveryExitCode": 0,
            },
            "observedExit": {
                "processId": 11,
                "observed": True,
                "signal": "SIGKILL",
            },
            "beforeProcesses": [
                {"pid": 10, "ppid": 1, "argv": ["python", "processor.py"]},
                {"pid": 11, "ppid": 10, "argv": ["python", "job_process.py"]},
                {"pid": 12, "ppid": 10, "argv": ["python", "job_process.py"]},
            ],
            "afterProcesses": [
                {"pid": 10, "ppid": 1, "argv": ["python", "processor.py"]},
                {"pid": 12, "ppid": 10, "argv": ["python", "job_process.py"]},
            ],
            "before": {
                "jobs": [
                    {"id": "job-a", "state": "running", "frames": 10},
                    {"id": "job-b", "state": "running", "frames": 20},
                ]
            },
            "after": {
                "jobs": [
                    {"id": "job-a", "state": "error", "frames": 10},
                    {"id": "job-b", "state": "running", "frames": 30},
                ]
            },
            "failure": {"message": "child exited with status 9"},
        },
        "cleanupStatus": {
            "runtime": runtime,
            "activeJobs": 0,
            "processes": [{"pid": 10, "ppid": 1, "argv": ["python", "processor.py"]}],
        },
    }
    files = {}
    for name, document in artifacts.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(document))
        files[name] = path
    patch_path = tmp_path / "patch.json"
    patch_path.write_text(json.dumps(patch))
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog))
    return files, patch_path, catalog_path


def test_process_containment_gate_recomputes_hashed_raw_artifacts(tmp_path):
    files, patch, catalog = process_gate_fixture(tmp_path)
    bundle_path = tmp_path / "bundle.json"
    bundle = gate.generate(files, patch, catalog, bundle_path)

    assert gate.validate(bundle_path) is True
    assert bundle["derived"]["topology"]["childProcessIds"] == [11, 12]
    crash = json.loads(files["crashObservation"].read_text())
    crash["after"]["jobs"][1]["frames"] = 20
    files["crashObservation"].write_text(json.dumps(crash))
    with pytest.raises(ValueError, match="hash mismatch"):
        gate.validate(bundle_path)


def test_process_containment_gate_rejects_credentials_and_false_survival(tmp_path):
    files, patch, catalog = process_gate_fixture(tmp_path)
    crash = json.loads(files["crashObservation"].read_text())
    crash["failure"]["message"] = "Authorization: Bearer abc.def"
    files["crashObservation"].write_text(json.dumps(crash))
    with pytest.raises(ValueError, match="credential-shaped"):
        gate.generate(files, patch, catalog, tmp_path / "bundle.json")

    files, patch, catalog = process_gate_fixture(tmp_path)
    snapshot = json.loads(files["processSnapshot"].read_text())
    snapshot["processes"][1]["argv"].extend(["--api-key", "leaked"])
    files["processSnapshot"].write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="process argument"):
        gate.generate(files, patch, catalog, tmp_path / "bundle-argv.json")

    files, patch, catalog = process_gate_fixture(tmp_path)
    cancellation = json.loads(files["cancellationObservation"].read_text())
    cancellation["after"]["jobs"][1]["frames"] = 10
    files["cancellationObservation"].write_text(json.dumps(cancellation))
    with pytest.raises(ValueError, match="frames did not advance"):
        gate.generate(files, patch, catalog, tmp_path / "bundle-2.json")

    files, patch, catalog = process_gate_fixture(tmp_path)
    crash = json.loads(files["crashObservation"].read_text())
    crash["action"]["targetProcessId"] = 12
    files["crashObservation"].write_text(json.dumps(crash))
    with pytest.raises(ValueError, match="bind the target child PID"):
        gate.generate(files, patch, catalog, tmp_path / "bundle-wrong-pid.json")

    files, patch, catalog = process_gate_fixture(tmp_path)
    cleanup = json.loads(files["cleanupStatus"].read_text())
    cleanup["processes"].append(
        {"pid": 12, "ppid": 10, "argv": ["python", "job_process.py"]}
    )
    files["cleanupStatus"].write_text(json.dumps(cleanup))
    with pytest.raises(ValueError, match="remove every child"):
        gate.generate(files, patch, catalog, tmp_path / "bundle-leak.json")

    files, patch, catalog = process_gate_fixture(tmp_path)
    cancellation = json.loads(files["cancellationObservation"].read_text())
    cancellation["afterProcesses"][1]["ppid"] = 1
    files["cancellationObservation"].write_text(json.dumps(cancellation))
    with pytest.raises(ValueError, match="sibling is not parented"):
        gate.generate(files, patch, catalog, tmp_path / "bundle-reparent.json")

    files, patch, catalog = process_gate_fixture(tmp_path)
    cleanup = json.loads(files["cleanupStatus"].read_text())
    cleanup["processes"].append(
        {"pid": 99, "ppid": 10, "argv": ["python", "job_process.py"]}
    )
    files["cleanupStatus"].write_text(json.dumps(cleanup))
    with pytest.raises(ValueError, match="remove every child"):
        gate.generate(files, patch, catalog, tmp_path / "bundle-respawn.json")
