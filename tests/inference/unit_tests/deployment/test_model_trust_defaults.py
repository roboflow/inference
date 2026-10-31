from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def test_all_current_jetson_images_retain_trust_filter():
    images = list((ROOT / "docker/dockerfiles").glob("Dockerfile.onnx.jetson.*"))
    assert {p.name for p in images} >= {
        "Dockerfile.onnx.jetson.5.1.1",
        "Dockerfile.onnx.jetson.6.2.0",
        "Dockerfile.onnx.jetson.7.2.0",
    }
    for image in images:
        source = image.read_text()
        assert "ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES=True" not in source
        assert "ALLOW_INFERENCE_EXP_UNTRUSTED_MODELS=True" not in source
