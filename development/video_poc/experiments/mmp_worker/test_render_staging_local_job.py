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
    job = __import__("json").loads(config_map["data"]["job.json"])
    assert job["workspace"] == "rf-inference-benchmark"
    assert job["mode"] == "batch"
    assert job["workflowSpecification"]["steps"][0]["model_id"] == (
        "microsoft-coco-obj-det/8"
    )
    secret_ref = worker["env"][-1]["valueFrom"]["secretKeyRef"]
    assert secret_ref == {"key": "api-key", "name": "video-mmp-smoke-001"}
    assert pod["volumes"][1]["configMap"]["name"] == "video-mmp-smoke-001-job"


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
