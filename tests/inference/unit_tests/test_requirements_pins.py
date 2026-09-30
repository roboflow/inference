import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[3]

INFERENCE_MODELS_PIN_PATTERN = re.compile(
    r"^inference-models(\[[^\]]*\])?[~=]=([^\s#]+)"
)
MODEL_MANAGER_PIN_PATTERN = re.compile(
    r'^inference-model-manager(\[[^\]]*\])?[~=]=([^\s#"]+)'
)

INFERENCE_MODELS_REQUIREMENTS_FILES = [
    "requirements.cpu.txt",
    "requirements.gpu.txt",
    "requirements.gpu.cu13.txt",
    "requirements.jetson.txt",
    "requirements.vino.txt",
]


def _load_version(pyproject_path: Path) -> str:
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]


def _extract_pin(text: str, pattern: re.Pattern) -> str:
    for line in text.splitlines():
        match = pattern.match(line.strip())
        if match:
            return match.group(2)
    raise AssertionError(f"No pin found matching {pattern.pattern!r}")


def _extract_dependency_pin(pyproject_path: Path, pattern: re.Pattern) -> str:
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)
    for dependency in data["project"]["dependencies"]:
        match = pattern.match(dependency)
        if match:
            return match.group(2)
    raise AssertionError(f"No pin found matching {pattern.pattern!r}")


def test_inference_models_requirement_pins_match_package_version():
    inference_models_version = _load_version(
        REPO_ROOT / "inference_models" / "pyproject.toml"
    )
    for filename in INFERENCE_MODELS_REQUIREMENTS_FILES:
        text = (REPO_ROOT / "requirements" / filename).read_text()
        pin = _extract_pin(text, INFERENCE_MODELS_PIN_PATTERN)
        assert pin == inference_models_version, filename


def test_inference_server_model_manager_pin_matches_package_version():
    model_manager_version = _load_version(
        REPO_ROOT / "inference_model_manager" / "pyproject.toml"
    )
    pin = _extract_dependency_pin(
        REPO_ROOT / "inference_server" / "pyproject.toml", MODEL_MANAGER_PIN_PATTERN
    )
    assert pin == model_manager_version
