from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def test_jetson_images_using_triton_set_a_writable_cache_dir():
    # RF-DETR preprocessing JIT-compiles Triton kernels. The documented Jetson
    # command runs the container --read-only with only /tmp writable, so any
    # image that can take the Triton path must move Triton's cache off ~/.triton.
    triton_images = {}
    for image in (ROOT / "docker/dockerfiles").glob("Dockerfile.onnx.jetson.*"):
        source = image.read_text()
        if "TRITON_VERSION" in source and (
            "INFERENCE_MODELS_RFDETR_PREPROCESSOR=base" not in source
        ):
            triton_images[image.name] = source
    assert set(triton_images) >= {
        "Dockerfile.onnx.jetson.6.2.0",
        "Dockerfile.onnx.jetson.7.2.0",
    }
    for name, source in triton_images.items():
        assert "TRITON_CACHE_DIR=/tmp/triton-cache" in source, name


def test_x86_gpu_images_using_triton_set_a_writable_cache_dir():
    # Same reason as Jetson: the documented docker run command is
    # --read-only with only /tmp writable.
    toolchain_images = {
        image.name
        for image in (ROOT / "docker/dockerfiles").glob("Dockerfile.*")
        if "verify_triton_jit_toolchain.py" in image.read_text()
    }
    assert toolchain_images >= {"Dockerfile.onnx.gpu", "Dockerfile.onnx.cu13.gpu"}
    # The slim image skips the toolchain check but still takes the Triton path.
    for name in sorted(toolchain_images | {"Dockerfile.onnx.gpu.slim"}):
        source = (ROOT / "docker/dockerfiles" / name).read_text()
        assert "TRITON_CACHE_DIR=/tmp/triton-cache" in source, name
