from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def test_all_current_jetson_images_retain_trust_filter_and_disable_stream_default():
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
        assert "ENABLE_STREAM_API=True" not in source


def test_all_first_party_images_and_desktop_bundles_boot_without_stream_token():
    import ast
    import re

    images = list((ROOT / "docker/dockerfiles").glob("Dockerfile*"))
    for image in images:
        assert not re.search(
            r"\bENABLE_STREAM_API(?:=|\s+)[\"']?(?:true|1|yes)\b",
            image.read_text(),
            re.I,
        ), image
    for platform in ["osx", "windows"]:
        path = ROOT / "app_bundles" / platform / "run_inference.py"
        defaults = [
            node
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "setdefault"
            and len(node.args) == 2
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "ENABLE_STREAM_API"
        ]
        assert len(defaults) == 1, path
        assert ast.literal_eval(defaults[0].args[1]) == "False", path
