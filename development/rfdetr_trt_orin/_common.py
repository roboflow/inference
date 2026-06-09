"""Shared constants and helpers for RF-DETR Jetson Orin TRT package workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_MODEL_ID = "rfdetr-seg-nano"
DEFAULT_PROD_API_HOST = "https://api.roboflow.com"
DEFAULT_STAGING_API_HOST = "https://api.roboflow.one"

CLASS_NAMES_FILE = "class_names.txt"
INFERENCE_CONFIG_FILE = "inference_config.json"
TRT_CONFIG_FILE = "trt_config.json"
ENGINE_PLAN_FILE = "engine.plan"
WEIGHTS_ONNX_FILE = "weights.onnx"
REGISTRATION_MANIFEST_FILE = "registration_manifest.json"

TRT_PACKAGE_FILE_HANDLES = [
    CLASS_NAMES_FILE,
    INFERENCE_CONFIG_FILE,
    TRT_CONFIG_FILE,
    ENGINE_PLAN_FILE,
]


def read_json(path: Path) -> dict:
    with path.open() as file:
        return json.load(file)


def write_json(path: Path, contents: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(contents, file, indent=4)


def prepare_adjusted_inference_config(
    *,
    inference_config_path: Path,
    target_path: Path,
) -> None:
    inference_config = read_json(inference_config_path)
    network_input = inference_config["network_input"]
    network_input["dynamic_spatial_size_supported"] = False
    network_input["dynamic_spatial_size_mode"] = None
    write_json(target_path, inference_config)


def get_training_input_size(*, inference_config_path: Path) -> tuple[int, int]:
    inference_config = read_json(inference_config_path)
    dimensions = inference_config["network_input"]["training_input_size"]
    return dimensions["height"], dimensions["width"]


def build_registration_manifest(
    *,
    model_id: str,
    source_model_id: str,
    model_architecture: str,
    task_type: Optional[str],
    model_variant: Optional[str],
    package_manifest: dict,
    trt_config: dict,
    precision: str,
) -> dict:
    return {
        "modelId": model_id,
        "sourceModelId": source_model_id,
        "modelArchitecture": model_architecture,
        "taskType": task_type,
        "modelVariant": model_variant,
        "precision": precision,
        "packageManifest": package_manifest,
        "trtConfig": trt_config,
        "fileHandles": list(TRT_PACKAGE_FILE_HANDLES),
    }


def load_registration_manifest(*, trt_package_dir: Path) -> Dict[str, Any]:
    manifest_path = trt_package_dir / REGISTRATION_MANIFEST_FILE
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {REGISTRATION_MANIFEST_FILE} in {trt_package_dir}. "
            "Run fetch_and_compile_rfdetr_trt_orin.py first."
        )
    return read_json(manifest_path)


def validate_trt_package_dir(*, trt_package_dir: Path) -> None:
    missing_files = [
        file_name
        for file_name in TRT_PACKAGE_FILE_HANDLES
        if not (trt_package_dir / file_name).is_file()
    ]
    if missing_files:
        missing = ", ".join(missing_files)
        raise FileNotFoundError(
            f"TRT package directory {trt_package_dir} is missing: {missing}"
        )
