"""Load and run native Core ML (.mlpackage) model packages on macOS."""

import hashlib
import os
import shutil
import tempfile
import threading
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from filelock import FileLock

from inference_models.configuration import (
    INFERENCE_MODELS_COREML_COMPUTE_UNITS,
    OFFLINE_MODE,
)
from inference_models.errors import (
    CorruptedModelPackageError,
    InvalidEnvVariable,
    MissingDependencyError,
)
from inference_models.models.common.model_packages import COREML_CACHE_DIR_NAME

MLPACKAGE_NAME = "weights.mlpackage"
MLPACKAGE_ARCHIVE_NAME = f"{MLPACKAGE_NAME}.zip"
NATIVE_MLPACKAGE_CACHE_DIR = "native"
MLPACKAGE_MANIFEST = "Manifest.json"
COMPUTE_UNITS = {
    "CPUAndGPU": "CPU_AND_GPU",
    "ALL": "ALL",
    "CPUAndNeuralEngine": "CPU_AND_NE",
    "CPUOnly": "CPU_ONLY",
}


@dataclass(frozen=True)
class CoreMLModelSignature:
    """Input / output layout read from a Core ML model's spec."""

    input_name: str
    image_input: bool
    input_height: int
    input_width: int
    output_names: Tuple[str, ...]


class CoreMLModel:
    """A loaded Core ML model plus its signature; ``predict`` is serialized across threads."""

    def __init__(self, model: Any, signature: CoreMLModelSignature):
        self._model = model
        self.signature = signature
        self._lock = threading.Lock()

    def predict(self, feed: Mapping[str, Any]) -> Dict[str, Any]:
        with self._lock:
            return self._model.predict(dict(feed))


def load_coreml_model(
    mlpackage_path: str,
    compute_units: str = INFERENCE_MODELS_COREML_COMPUTE_UNITS,
) -> CoreMLModel:
    if compute_units not in COMPUTE_UNITS:
        raise InvalidEnvVariable(
            message=f"Core ML compute units must be one of {sorted(COMPUTE_UNITS)}, got '{compute_units}' "
            f"(configured with INFERENCE_MODELS_COREML_COMPUTE_UNITS).",
            help_url="https://inference-models.roboflow.com/errors/runtime-environment/#invalidenvvariable",
        )
    # Imported here rather than at module level so package resolution and signature parsing stay importable
    # (and testable) without the macOS-only `coreml` extra.
    try:
        import coremltools
    except ImportError as import_error:
        raise MissingDependencyError(
            message="Running a model with the Core ML backend requires coremltools, which is brought with the "
            "`coreml` extra of `inference-models` (macOS only). If you see this error running locally, please "
            "follow our installation guide: https://inference-models.roboflow.com/getting-started/installation/",
            help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror",
        ) from import_error
    model = coremltools.models.MLModel(
        mlpackage_path,
        compute_units=getattr(coremltools.ComputeUnit, COMPUTE_UNITS[compute_units]),
    )
    return CoreMLModel(model=model, signature=read_signature(model.get_spec()))


def read_signature(spec: Any) -> CoreMLModelSignature:
    model_input = spec.description.input[0]
    input_type = model_input.type.WhichOneof("Type")
    if input_type == "imageType":
        height, width = (
            model_input.type.imageType.height,
            model_input.type.imageType.width,
        )
    elif input_type == "multiArrayType":
        height, width = (int(d) for d in model_input.type.multiArrayType.shape[-2:])
    else:
        raise CorruptedModelPackageError(
            message=f"Core ML model input '{model_input.name}' has unsupported type '{input_type}'; "
            f"expected an image or a multi-array.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    return CoreMLModelSignature(
        input_name=model_input.name,
        image_input=input_type == "imageType",
        input_height=int(height),
        input_width=int(width),
        output_names=tuple(o.name for o in spec.description.output),
    )


def resolve_mlpackage(model_package_dir: str) -> str:
    """Return the path of the package's ``.mlpackage`` bundle, extracting the zipped form once if needed.

    Registry packages ship the bundle as ``weights.mlpackage.zip`` (a directory bundle does not fit a flat
    artefact list). It is extracted into the package's ``coreml_cache`` directory, which the inference cache
    watchdog purges as a whole, so a partially deleted bundle can never be loaded. Offline or read-only
    packages are extracted into a temporary directory instead of being written to.
    """
    bundle = os.path.join(model_package_dir, MLPACKAGE_NAME)
    if os.path.isfile(os.path.join(bundle, MLPACKAGE_MANIFEST)):
        return bundle
    archive = os.path.join(model_package_dir, MLPACKAGE_ARCHIVE_NAME)
    if not os.path.isfile(archive):
        raise CorruptedModelPackageError(
            message=f"Core ML model package at {model_package_dir} contains neither {MLPACKAGE_NAME} nor "
            f"{MLPACKAGE_ARCHIVE_NAME}.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    target = os.path.join(
        _extraction_root(model_package_dir), NATIVE_MLPACKAGE_CACHE_DIR, MLPACKAGE_NAME
    )
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with FileLock(f"{target}.lock"):
        if not os.path.isfile(os.path.join(target, MLPACKAGE_MANIFEST)):
            _extract_bundle(archive=archive, target=target)
    return target


def _extraction_root(model_package_dir: str) -> str:
    if not OFFLINE_MODE and os.access(model_package_dir, os.W_OK):
        return os.path.join(model_package_dir, COREML_CACHE_DIR_NAME)
    key = hashlib.sha256(os.path.abspath(model_package_dir).encode()).hexdigest()[:16]
    return os.path.join(tempfile.gettempdir(), "inference-models-coreml", key)


def _extract_bundle(archive: str, target: str) -> None:
    staging = tempfile.mkdtemp(dir=os.path.dirname(target), prefix=".extract-")
    try:
        with zipfile.ZipFile(archive) as zip_file:
            staging_root = os.path.realpath(staging)
            for member in zip_file.namelist():
                destination = os.path.realpath(os.path.join(staging_root, member))
                if os.path.commonpath([staging_root, destination]) != staging_root:
                    raise CorruptedModelPackageError(
                        message=f"Core ML archive {archive} contains an entry outside the bundle: {member}",
                        help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
                    )
            zip_file.extractall(staging)
        bundle_root = _find_bundle_root(staging)
        if bundle_root is None:
            raise CorruptedModelPackageError(
                message=f"Core ML archive {archive} does not contain an .mlpackage bundle ({MLPACKAGE_MANIFEST} "
                f"not found).",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        shutil.rmtree(target, ignore_errors=True)
        os.replace(bundle_root, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _find_bundle_root(directory: str):
    # Train has zipped the bundle both with its contents at the archive root and inside a top-level folder.
    if os.path.isfile(os.path.join(directory, MLPACKAGE_MANIFEST)):
        return directory
    for entry in sorted(os.listdir(directory)):
        candidate = os.path.join(directory, entry)
        if os.path.isfile(os.path.join(candidate, MLPACKAGE_MANIFEST)):
            return candidate
    return None
