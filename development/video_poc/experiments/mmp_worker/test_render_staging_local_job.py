import pytest

from render_staging_local_job import render


IMAGE = (
    "us-central1-docker.pkg.dev/roboflow-staging/video-proc/"
    "video-processor-mmp@sha256:" + "a" * 64
)


def test_render_is_staging_bounded_and_credential_free():
    manifest = render(
        IMAGE, "smoke-001", "rf-inference-benchmark", "video-mmp-smoke-001"
    )
    config_map, pod_manifest = manifest["items"]
    pod = pod_manifest["spec"]
    worker = pod["containers"][0]

    assert pod_manifest["metadata"]["namespace"] == "video-proc"
    assert pod["activeDeadlineSeconds"] == 900
    assert pod["automountServiceAccountToken"] is False
    assert pod["nodeSelector"] == {"gpu_type": "L40S"}
    assert worker["image"] == IMAGE
    assert worker["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert pod["volumes"][0]["emptyDir"] == {
        "medium": "Memory",
        "sizeLimit": "2Gi",
    }
    rendered = str(manifest)
    assert "apiKey" not in rendered
    assert "Ocx" not in rendered
    job = __import__("json").loads(config_map["data"]["job-1.json"])
    assert job["workspace"] == "rf-inference-benchmark"
    assert job["mode"] == "batch"
    assert job["workflowSpecification"]["steps"][0]["model_id"] == (
        "microsoft-coco-obj-det/8"
    )
    secret_ref = worker["env"][-1]["valueFrom"]["secretKeyRef"]
    assert secret_ref == {"key": "api-key", "name": "video-mmp-smoke-001"}
    assert pod["volumes"][1]["configMap"]["name"] == "video-mmp-smoke-001-job"


def test_render_concurrent_legacy_control():
    manifest = render(
        IMAGE,
        "legacy-c4",
        "rf-inference-benchmark",
        "video-mmp-smoke-001",
        backend="legacy",
        concurrency=4,
        max_fps=5,
    )
    config_map, pod_manifest = manifest["items"]
    worker, relay = pod_manifest["spec"]["containers"]
    assert sorted(key for key in config_map["data"] if key.startswith("job-")) == [
        "job-1.json",
        "job-2.json",
        "job-3.json",
        "job-4.json",
    ]
    assert worker["args"].count("--job-file") == 4
    assert worker["args"][-4:] == ["--max-jobs", "4", "--tier", "gpu"]
    env = {item["name"]: item.get("value") for item in worker["env"]}
    assert env["PROCESSOR_MODEL_MANAGER_MODE"] == "legacy"
    assert env["LEGACY_MMP_ADAPTER_MODE"] == "off"
    jobs = [
        __import__("json").loads(value)
        for key, value in config_map["data"].items()
        if key.startswith("job-")
    ]
    assert {job["maxFps"] for job in jobs} == {5.0}
    assert len({job["id"] for job in jobs}) == 4
    assert relay["image"].endswith(
        "@sha256:59aaad04627c7c8f40ceb01a5ff1c43f91e01939da147c3419f1aaa0c78d6cf5"
    )


def test_render_stream_jobs_have_unique_internal_relay_paths():
    manifest = render(
        IMAGE,
        "mmp-c2",
        "rf-inference-benchmark",
        "video-mmp-smoke-001",
        backend="subprocess",
        concurrency=2,
        mode="stream",
    )
    jobs = [
        __import__("json").loads(value)
        for key, value in manifest["items"][0]["data"].items()
        if key.startswith("job-")
    ]
    assert {job["mode"] for job in jobs} == {"stream"}
    assert len({job["simPublishUrl"] for job in jobs}) == 2
    assert all(
        job["simPublishUrl"].startswith(
            "rtsp://127.0.0.1:8554/sim-mmp-c2-"
        )
        for job in jobs
    )


@pytest.mark.parametrize(
    "image",
    [
        "video-processor-mmp:latest",
        "us-central1-docker.pkg.dev/roboflow-platform/video-proc/"
        "video-processor-mmp@sha256:" + "a" * 64,
        "us-central1-docker.pkg.dev/roboflow-staging/video-proc/"
        "video-processor-mmp:mutable",
    ],
)
def test_render_rejects_non_staging_or_mutable_images(image):
    with pytest.raises(ValueError, match="immutable staging"):
        render(
            image,
            "smoke-001",
            "rf-inference-benchmark",
            "video-mmp-smoke-001",
        )


def test_render_rejects_unsafe_names_and_unknown_backend():
    with pytest.raises(ValueError, match="run ID"):
        render(
            IMAGE,
            "Production_Smoke",
            "rf-inference-benchmark",
            "video-mmp-smoke-001",
        )
    with pytest.raises(ValueError, match="API key Secret"):
        render(IMAGE, "smoke-001", "rf-inference-benchmark", "../secret")
    with pytest.raises(ValueError, match="workspace"):
        render(IMAGE, "smoke-001", "", "video-mmp-smoke-001")
    with pytest.raises(ValueError, match="backend"):
        render(
            IMAGE,
            "smoke-001",
            "rf-inference-benchmark",
            "video-mmp-smoke-001",
            "external",
        )
    with pytest.raises(ValueError, match="concurrency"):
        render(
            IMAGE,
            "smoke-001",
            "rf-inference-benchmark",
            "video-mmp-smoke-001",
            concurrency=9,
        )
