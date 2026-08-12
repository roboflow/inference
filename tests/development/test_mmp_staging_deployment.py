import importlib.util
from pathlib import Path

import pytest


MODULE = (
    Path(__file__).resolve().parents[2]
    / "development"
    / "mmp_staging_benchmark"
    / "render_staging_deployment.py"
)
SPEC = importlib.util.spec_from_file_location("mmp_deployment", MODULE)
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


IMAGE = (
    "us-central1-docker.pkg.dev/roboflow-staging/video-proc/"
    "mmp-benchmark@sha256:" + "a" * 64
)
REVISION = "b" * 40


def deployment(document):
    return [item for item in document["items"] if item["kind"] == "Deployment"][0]


def test_renders_dedicated_digest_pinned_l40s_deployment():
    document = renderer.render(IMAGE, REVISION, "mmp-benchmark-api-keys")
    workload = deployment(document)
    pod = workload["spec"]["template"]["spec"]
    container = pod["containers"][0]

    assert document["items"][0]["metadata"]["name"] == "video-proc-bench-mmp"
    assert workload["spec"]["strategy"] == {"type": "Recreate"}
    assert pod["nodeSelector"] == {"gpu_type": "L40S"}
    assert container["image"] == IMAGE
    assert container["resources"]["limits"]["nvidia.com/gpu"] == 1
    assert {item["name"]: item["emptyDir"] for item in pod["volumes"]}[
        "dshm"
    ] == {"medium": "Memory", "sizeLimit": "4Gi"}
    env = {item["name"]: item for item in container["env"]}
    assert env["NVIDIA_MPS"]["value"] == "0"
    assert "value" not in env["RF_BENCH_TENANT_A_KEY"]
    assert env["RF_BENCH_TENANT_A_KEY"]["valueFrom"]["secretKeyRef"] == {
        "name": "mmp-benchmark-api-keys",
        "key": "tenant-a",
    }


def test_mps_changes_only_the_explicit_runtime_mode():
    document = renderer.render(IMAGE, REVISION, "mmp-benchmark-api-keys", mps=True)
    workload = deployment(document)
    container = workload["spec"]["template"]["spec"]["containers"][0]
    env = {item["name"]: item for item in container["env"]}

    assert env["NVIDIA_MPS"]["value"] == "1"
    assert workload["metadata"]["annotations"]["roboflow.com/mps"] == "enabled"


@pytest.mark.parametrize(
    "image",
    [
        "us-central1-docker.pkg.dev/roboflow-staging/video-proc/mmp-benchmark:tag",
        "us-central1-docker.pkg.dev/roboflow-platform/video-proc/"
        "mmp-benchmark@sha256:" + "a" * 64,
        "example.com/mmp-benchmark@sha256:" + "a" * 64,
    ],
)
def test_rejects_mutable_or_nonstaging_images(image):
    with pytest.raises(ValueError, match="staging mmp-benchmark"):
        renderer.render(image, REVISION, "mmp-benchmark-api-keys")
