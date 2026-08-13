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


def render(**kwargs):
    return renderer.render(
        IMAGE,
        REVISION,
        run_id="mmp-smoke-001",
        **kwargs,
    )


def test_renders_dedicated_digest_pinned_l40s_deployment():
    document = render()
    workload = deployment(document)
    pod = workload["spec"]["template"]["spec"]
    container = pod["containers"][0]

    assert document["items"][0]["metadata"]["name"] == "video-proc-bench-mmp"
    assert workload["spec"]["strategy"] == {"type": "Recreate"}
    assert pod["nodeSelector"] == {"gpu_type": "L40S"}
    assert pod["automountServiceAccountToken"] is False
    assert (
        workload["spec"]["template"]["metadata"]["annotations"]
        == workload["metadata"]["annotations"]
    )
    assert container["image"] == IMAGE
    assert container["resources"]["limits"]["nvidia.com/gpu"] == 1
    assert {item["name"]: item["emptyDir"] for item in pod["volumes"]}["dshm"] == {
        "medium": "Memory",
        "sizeLimit": "4Gi",
    }
    env = {item["name"]: item for item in container["env"]}
    assert env["NVIDIA_MPS"]["value"] == "0"
    assert env["INFERENCE_DECODER"]["value"] == "imagecodecs"
    assert env["MMP_BENCHMARK_RUN_ID"]["value"] == "mmp-smoke-001"
    assert env["POD_NAME"]["valueFrom"]["fieldRef"] == {"fieldPath": "metadata.name"}
    assert env["NODE_NAME"]["valueFrom"]["fieldRef"] == {"fieldPath": "spec.nodeName"}
    assert "RF_BENCH_TENANT_A_KEY" not in env
    assert "RF_BENCH_TENANT_B_KEY" not in env


def test_mps_changes_only_the_explicit_runtime_mode():
    document = render(mps=True, active_thread_percentage=50)
    workload = deployment(document)
    container = workload["spec"]["template"]["spec"]["containers"][0]
    env = {item["name"]: item for item in container["env"]}

    assert env["NVIDIA_MPS"]["value"] == "1"
    assert env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"]["value"] == "50"
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
        renderer.render(
            image,
            REVISION,
            run_id="mmp-smoke-001",
        )


@pytest.mark.parametrize("run_id", ["", "UPPER", "a" * 64])
def test_rejects_invalid_run_id(run_id):
    with pytest.raises(ValueError, match="run ID"):
        renderer.render(
            IMAGE,
            REVISION,
            run_id=run_id,
        )


def test_rejects_mps_partition_without_mps():
    with pytest.raises(ValueError, match="requires MPS"):
        render(active_thread_percentage=50)


def test_pins_benchmark_to_capability_node_when_requested():
    document = render(node_name="gpu-l40s-001")
    assert deployment(document)["spec"]["template"]["spec"]["nodeSelector"] == {
        "gpu_type": "L40S",
        "kubernetes.io/hostname": "gpu-l40s-001",
    }
