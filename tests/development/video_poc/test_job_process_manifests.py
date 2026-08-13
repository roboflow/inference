from pathlib import Path

import pytest

yaml = pytest.importorskip(
    "yaml",
    reason="manifest validation requires the CLI PyYAML dependency",
)


REPO_ROOT = Path(__file__).parents[3]
RUNTIME_VARIANTS = (
    REPO_ROOT / "development" / "video_poc" / "benchmarks" / "runtime_variants"
)
SMOKE = (
    REPO_ROOT
    / "development"
    / "video_poc"
    / "experiments"
    / "process_isolation"
    / "smoke"
)

SOURCE_SHA = "008d5e64b27d19c7c5da6334ec9497ba756827ad"
D_IMAGE = (
    "us-central1-docker.pkg.dev/roboflow-staging/video-proc/"
    "video-processor-process@sha256:"
    "0e12efc9321dc495540dfa1fda0a2413286df468f2b6c5e8dd869aaf52f1a1bd"
)
EF_IMAGE = (
    "us-central1-docker.pkg.dev/roboflow-staging/video-proc/"
    "video-processor-process@sha256:"
    "4f1767d45ec3d90e07215f377ebbbba21b7c8b1a42ffa8acedf4b6217c06a70c"
)

PATCH_CASES = (
    (
        "l40s-capacity-d-legacy-process.yaml",
        D_IMAGE,
        "legacy-process-pyav",
        "pyav",
        False,
    ),
    (
        "l40s-capacity-e-v14-process-pyav.yaml",
        EF_IMAGE,
        "v14-process-pyav",
        "pyav",
        True,
    ),
    (
        "l40s-capacity-f-v14-process-nvdec.yaml",
        EF_IMAGE,
        "v14-process-nvdec",
        "gstreamer_cuda",
        True,
    ),
)


def _load(path: Path):
    return yaml.safe_load(path.read_text())


def _env_by_name(container):
    env = container["env"]
    names = [item["name"] for item in env]
    assert len(names) == len(set(names))
    return {item["name"]: item for item in env}


@pytest.mark.parametrize("filename,image,variant,ingest,tensor", PATCH_CASES)
def test_process_variant_patch_is_exact_and_atomic(
    filename, image, variant, ingest, tensor
):
    patch = _load(RUNTIME_VARIANTS / filename)
    assert set(patch) == {"spec"}
    template = patch["spec"]["template"]
    assert set(template) == {"metadata", "spec"}
    assert set(template["metadata"]) == {"annotations"}
    assert set(template["metadata"]["annotations"]) == {
        "roboflow.com/staging-rollout"
    }
    assert set(template["spec"]) == {"containers"}
    assert len(template["spec"]["containers"]) == 1

    container = template["spec"]["containers"][0]
    assert set(container) == {"name", "image", "env"}
    assert container["name"] == "processor"
    assert container["image"] == image
    env = _env_by_name(container)

    assert env["VIDEO_PROC_IMAGE"]["value"] == image
    assert env["VIDEO_PROC_GIT_SHA"]["value"] == SOURCE_SHA
    assert env["VIDEO_PROC_RUNTIME_VARIANT"]["value"] == variant
    assert env["PROCESSOR_JOB_EXECUTION_MODE"]["value"] == "process"
    assert env["PROCESSOR_EXECUTION_DOMAIN_MODE"]["value"] == "in_process"
    assert env["PROCESSOR_VIDEO_INGEST_MODE"]["value"] == ingest
    assert env["VIDEO_SOURCE_ADAPTIVE_BACKPRESSURE"]["value"] == "true"
    assert env["ROBOFLOW_RTSP_LATENCY_MS"]["value"] == "200"
    assert env["ROBOFLOW_RTSP_PROTOCOLS"]["value"] == "tcp"
    assert env["ROBOFLOW_RTSP_VIDEO_CODEC"]["value"] == "h264"
    assert env["ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING"] == {
        "$patch": "delete",
        "name": "ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING",
    }

    tensor_names = (
        "ENABLE_TENSOR_DATA_REPRESENTATION",
        "USE_INFERENCE_MODELS",
        "WORKFLOWS_IMAGE_TENSOR_DEVICE",
    )
    if tensor:
        assert env["ENABLE_TENSOR_DATA_REPRESENTATION"]["value"] == "true"
        assert env["USE_INFERENCE_MODELS"]["value"] == "true"
        assert env["WORKFLOWS_IMAGE_TENSOR_DEVICE"]["value"] == "cuda"
    else:
        for name in tensor_names:
            assert env[name] == {"$patch": "delete", "name": name}


SMOKE_CASES = (
    (
        "l40s-d-legacy-process.yaml",
        "video-process-smoke-d-0e12efc9",
        D_IMAGE,
        "pyav",
        False,
    ),
    (
        "l40s-e-v14-process-pyav.yaml",
        "video-process-smoke-e-4f1767d4",
        EF_IMAGE,
        "pyav",
        True,
    ),
    (
        "l40s-f-v14-process-nvdec.yaml",
        "video-process-smoke-f-4f1767d4",
        EF_IMAGE,
        "gstreamer_cuda",
        True,
    ),
)


@pytest.mark.parametrize("filename,pod_name,image,ingest,tensor", SMOKE_CASES)
def test_l40s_smoke_is_credential_free_and_asserts_process_topology(
    filename, pod_name, image, ingest, tensor
):
    pod = _load(SMOKE / filename)
    assert pod["apiVersion"] == "v1"
    assert pod["kind"] == "Pod"
    assert pod["metadata"]["name"] == pod_name
    assert pod["metadata"]["namespace"] == "video-proc"

    spec = pod["spec"]
    assert spec["automountServiceAccountToken"] is False
    assert spec["activeDeadlineSeconds"] == 180
    assert spec["terminationGracePeriodSeconds"] == 10
    assert spec["restartPolicy"] == "Never"
    assert "serviceAccountName" not in spec
    assert "volumes" not in spec
    assert spec["nodeSelector"] == {"gpu_type": "L40S"}
    assert spec["imagePullSecrets"] == [{"name": "gcp-ar-pull-secret"}]
    assert len(spec["containers"]) == 1

    container = spec["containers"][0]
    assert container["image"] == image
    assert container["command"] == ["python", "-c"]
    assert len(container["args"]) == 1
    script = container["args"][0]
    assert 'mp.get_context("spawn")' in script
    assert "child.daemon = True" in script
    assert "child.pid != os.getpid()" in script
    assert "child.exitcode == 0" in script
    assert "torch.cuda.is_available()" in script
    assert '"L40S" in torch.cuda.get_device_name(0)' in script

    env = _env_by_name(container)
    assert set(env) <= {
        "PROJECT",
        "PROCESSOR_JOB_EXECUTION_MODE",
        "PROCESSOR_EXECUTION_DOMAIN_MODE",
        "PROCESSOR_VIDEO_INGEST_MODE",
        "ENABLE_TENSOR_DATA_REPRESENTATION",
        "USE_INFERENCE_MODELS",
        "WORKFLOWS_IMAGE_TENSOR_DEVICE",
    }
    assert env["PROJECT"]["value"] == "roboflow-staging"
    assert env["PROCESSOR_JOB_EXECUTION_MODE"]["value"] == "process"
    assert env["PROCESSOR_EXECUTION_DOMAIN_MODE"]["value"] == "in_process"
    assert env["PROCESSOR_VIDEO_INGEST_MODE"]["value"] == ingest
    assert all(
        fragment not in str(container).lower()
        for fragment in ("api_key", "apikey", "token", "secret", "password")
    )

    if tensor:
        assert env["ENABLE_TENSOR_DATA_REPRESENTATION"]["value"] == "true"
        assert env["USE_INFERENCE_MODELS"]["value"] == "true"
        assert env["WORKFLOWS_IMAGE_TENSOR_DEVICE"]["value"] == "cuda"
    else:
        assert "ENABLE_TENSOR_DATA_REPRESENTATION" not in env
        assert "USE_INFERENCE_MODELS" not in env
        assert "WORKFLOWS_IMAGE_TENSOR_DEVICE" not in env

    if ingest == "gstreamer_cuda":
        assert "GstreamerCudaVideoFrameProducer" in script
    else:
        assert "GstreamerCudaVideoFrameProducer" not in script
