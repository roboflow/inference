import json
import os.path
import shutil
from typing import List, Literal, Optional, Tuple, Union

import onnxruntime
from inference_models.configuration import ROBOFLOW_API_KEY
from inference_models.logger import LOGGER
from inference_models.models.auto_loaders.auto_negotiation import negotiate_model_packages
from inference_models.runtime_introspection.core import x_ray_runtime_environment
from inference_models.utils.download import download_files_to_directory
from inference_models.weights_providers.core import get_model_from_provider
from inference_models.weights_providers.entities import (
    ModelPackageMetadata,
    Quantization,
)
from inference_models.models.auto_loaders.entities import BackendType

from .engine_builder import EngineBuilder

WEIGHTS_FILE_NAME = "weights.onnx"


def compile_model(
    model_id: str,
    target_path: str,
    precision: Literal["fp32", "fp16", "int8"],
    min_batch_size: Optional[int] = None,
    opt_batch_size: Optional[int] = None,
    max_batch_size: Optional[int] = None,
    model_input_size: Optional[Union[int, Tuple[int, int]]] = None,
    roboflow_api_key: Optional[str] = None,
    workspace_size_gb: int = 8,
    trt_version_compatible: bool = False,
    same_compute_compatibility: bool = False,
) -> None:
    if not roboflow_api_key:
        roboflow_api_key = ROBOFLOW_API_KEY
    dynamic_dimensions_in_use = all(
        e is not None for e in [min_batch_size, opt_batch_size, max_batch_size]
    )
    matching_model_packages = select_matching_model_packages(
        model_id=model_id,
        roboflow_api_key=roboflow_api_key,
        dynamic_dimensions_in_use=dynamic_dimensions_in_use,
    )
    if not matching_model_packages:
        raise ValueError("Cannot find matching model packages.")
    model_packages_dirs = download_model_packages(
        matching_model_packages=matching_model_packages,
        target_path=target_path,
    )
    models_matching_size = find_model_packages_matching_by_size(
        model_packages_dirs=model_packages_dirs,
        model_input_size=model_input_size,
    )
    to_remove = [p for p in model_packages_dirs if p not in models_matching_size]
    for p in to_remove:
        shutil.rmtree(p)
    if not models_matching_size:
        raise ValueError("Cannot find model packages matching input size.")
    for matching_model_dir in models_matching_size:
        try:
            compile_model_to_trt(
                model_dir=matching_model_dir,
                precision=precision,
                min_batch_size=min_batch_size,
                opt_batch_size=opt_batch_size,
                max_batch_size=max_batch_size,
                model_input_size=model_input_size,
                workspace_size_gb=workspace_size_gb,
                trt_version_compatible=trt_version_compatible,
                same_compute_compatibility=same_compute_compatibility,
            )
        except Exception as e:
            LOGGER.exception(f"Failed to create engine: {e}")


def select_matching_model_packages(
    model_id: str,
    roboflow_api_key: Optional[str],
    dynamic_dimensions_in_use: bool,
) -> List[ModelPackageMetadata]:
    model_metadata = get_model_from_provider(
        provider="roboflow",
        model_id=model_id,
        api_key=roboflow_api_key,
    )
    matching_model_packages = negotiate_model_packages(
        model_packages=model_metadata.model_packages,
        requested_backends=BackendType.ONNX,
        requested_quantization=Quantization.FP32,
        allow_untrusted_packages=True,
        verbose=True,
    )
    if dynamic_dimensions_in_use:
        return [p for p in matching_model_packages if p.dynamic_batch_size_supported]
    return [p for p in matching_model_packages if not p.dynamic_batch_size_supported]


def download_model_packages(
    matching_model_packages: List[ModelPackageMetadata],
    target_path: str,
) -> List[str]:
    result = []
    for model_package in matching_model_packages:
        print(f"Downloading package: {model_package.package_id}")
        package_dir = os.path.join(target_path, model_package.package_id)
        os.makedirs(package_dir, exist_ok=True)
        files_specs = [
            (a.file_handle, a.download_url, a.md5_hash)
            for a in model_package.package_artefacts
        ]
        download_files_to_directory(
            target_dir=package_dir,
            files_specs=files_specs,
            verbose=True,
        )
        result.append(package_dir)
    return result


def find_model_packages_matching_by_size(
    model_packages_dirs: List[str],
    model_input_size: Optional[Union[int, Tuple[int, int]]],
) -> List[str]:
    if model_input_size is None:
        return model_packages_dirs
    if isinstance(model_input_size, int):
        model_input_size = (model_input_size, model_input_size)
    return [
        p
        for p in model_packages_dirs
        if model_matches_input_size(
            onnx_path=os.path.join(p, WEIGHTS_FILE_NAME),
            model_input_size=model_input_size,
        )
    ]


def model_matches_input_size(
    onnx_path: str,
    model_input_size: Tuple[int, int],
) -> bool:
    session = onnxruntime.InferenceSession(onnx_path)
    input_shape = session.get_inputs()[0].shape
    h, w = input_shape[2], input_shape[3]
    return dimension_matches(
        actual=h, expected=model_input_size[0]
    ) and dimension_matches(actual=w, expected=model_input_size[1])


def dimension_matches(actual: Union[str, int], expected: int) -> bool:
    if isinstance(actual, str):
        return True
    return actual == expected


def compile_model_to_trt(
    model_dir: str,
    precision: Literal["fp32", "fp16", "int8"],
    min_batch_size: Optional[int] = None,
    opt_batch_size: Optional[int] = None,
    max_batch_size: Optional[int] = None,
    model_input_size: Optional[Union[int, Tuple[int, int]]] = None,
    workspace_size_gb: int = 8,
    trt_version_compatible: bool = False,
    same_compute_compatibility: bool = False,
) -> None:
    print(f"Compiling model in {model_dir}")
    runtime_xray = x_ray_runtime_environment()
    xray_path = os.path.join(model_dir, "env-x-ray.json")
    dump_json(
        path=xray_path,
        contents={
            "gpu_available": runtime_xray.gpu_available,
            "gpu_devices": runtime_xray.gpu_devices,
            "gpu_devices_cc": [str(e) for e in runtime_xray.gpu_devices_cc],
            "driver_version": (
                str(runtime_xray.driver_version)
                if runtime_xray.driver_version
                else None
            ),
            "cuda_version": (
                str(runtime_xray.cuda_version) if runtime_xray.cuda_version else None
            ),
            "trt_version": (
                str(runtime_xray.trt_version) if runtime_xray.trt_version else None
            ),
            "jetson_type": runtime_xray.jetson_type,
            "l4t_version": (
                str(runtime_xray.l4t_version) if runtime_xray.l4t_version else None
            ),
            "os_version": runtime_xray.os_version,
            "torch_available": runtime_xray.torch_available,
            "onnxruntime_version": (
                str(runtime_xray.onnxruntime_version)
                if runtime_xray.onnxruntime_version
                else None
            ),
            "available_onnx_execution_providers": (
                list(runtime_xray.available_onnx_execution_providers)
                if runtime_xray.available_onnx_execution_providers
                else None
            ),
            "hf_transformers_available": runtime_xray.hf_transformers_available,
            "ultralytics_available": runtime_xray.ultralytics_available,
            "trt_python_package_available": runtime_xray.trt_python_package_available,
        },
    )
    onnx_path = os.path.join(model_dir, WEIGHTS_FILE_NAME)
    session = onnxruntime.InferenceSession(onnx_path)
    if model_input_size is not None:
        if isinstance(model_input_size, int):
            model_input_size = (model_input_size, model_input_size)
    else:
        model_input_size = session.get_inputs()[0].shape[2:4]
    dynamic_batch_sizes = None
    dynamic_dimensions_in_use = all(
        e is not None for e in [min_batch_size, opt_batch_size, max_batch_size]
    )
    static_batch_size = None
    if dynamic_dimensions_in_use:
        dynamic_batch_sizes = min_batch_size, opt_batch_size, max_batch_size
    else:
        static_batch_size = session.get_inputs()[0].shape[0]

    engine_name_postfix = ""
    if trt_version_compatible:
        engine_name_postfix += "-trt-version-compatible"
    if same_compute_compatibility:
        engine_name_postfix += "-same-cc"
    engine_path = os.path.join(
        model_dir, f"engine-{precision}{engine_name_postfix}.plan"
    )
    if os.path.exists(engine_path):
        return None
    trt_config_path = os.path.join(
        model_dir, f"trt-config-{precision}{engine_name_postfix}.json"
    )
    dump_json(
        path=trt_config_path,
        contents={
            "static_batch_size": static_batch_size,
            "dynamic_batch_size_min": min_batch_size,
            "dynamic_batch_size_opt": opt_batch_size,
            "dynamic_batch_size_max": max_batch_size,
            "trt_version_compatible": trt_version_compatible,
            "same_compute_compatibility": same_compute_compatibility,
            "precision": precision,
        },
    )
    engine_builder = EngineBuilder(workspace=workspace_size_gb)
    engine_builder.create_network(onnx_path=onnx_path)
    engine_builder.create_engine(
        engine_path=engine_path,
        input_name=session.get_inputs()[0].name,
        precision=precision,
        input_size=model_input_size,
        dynamic_batch_sizes=dynamic_batch_sizes,
        trt_version_compatible=trt_version_compatible,
        same_compute_compatibility=same_compute_compatibility,
    )


def dump_json(path: str, contents: dict) -> None:
    with open(path, "w") as f:
        json.dump(contents, f, indent=4)
