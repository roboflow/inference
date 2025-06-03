import os
import platform
import re
import subprocess
from dataclasses import dataclass
from functools import cache
from typing import List, Optional, Tuple

import torch
from packaging.version import Version

from inference.v1.configuration import JETPACK_VERSION, RUNNING_ON_JETSON
from inference.v1.errors import JetsonTypeResolutionError
from inference.v1.utils.environment import str2bool

JETSON_DEVICES_TABLE = [
    "NVIDIA Jetson Orin Nano",
    "NVIDIA Jetson Orin NX",
    "NVIDIA Jetson AGX Orin",
    "NVIDIA Jetson IGX Orin",
    "NVIDIA Jetson Xavier NX",
    "NVIDIA Jetson AGX Xavier Industrial",
    "NVIDIA Jetson AGX Xavier",
    "NVIDIA Jetson Nano",
    "NVIDIA Jetson TX2",
]


@dataclass(frozen=True)
class RuntimeXRayResult:
    gpu_available: bool
    gpu_devices: List[str]
    driver_version: Optional[Version]
    cuda_version: Optional[Version]
    trt_version: Optional[Version]
    jetson_type: Optional[str]
    jetpack_version: Optional[str]
    os_version: Optional[str]
    torch_available: bool
    onnxruntime_available: bool
    hf_transformers_available: bool
    ultralytics_available: bool
    trt_python_package_available: bool

    def __str__(self) -> str:
        gpu_devices_str = ", ".join(self.gpu_devices)
        return (
            f"Runtime X-Ray: gpu_available={self.gpu_available} gpu_devices=[{gpu_devices_str}] "
            f"gpu_driver={self.driver_version} cuda_version={self.cuda_version} trt_version={self.trt_version} "
            f"jetson_type={self.jetson_type} jetpack_version={self.jetpack_version} os_version={self.os_version} "
            f"torch_available={self.torch_available} onnxruntime_available={self.onnxruntime_available} "
            f"hf_transformers_available={self.hf_transformers_available} "
            f"ultralytics_available={self.ultralytics_available} "
            f"trt_python_package_available={self.trt_python_package_available}"
        )


@cache
def x_ray_runtime_environment(verbose: bool = False) -> RuntimeXRayResult:
    trt_version = get_trt_version(verbose=verbose)
    cuda_version = get_cuda_version(verbose=verbose)
    jetson_type, jetpack_version, os_version, driver_version = None, None, None, None
    if is_running_on_jetson():
        jetson_type = get_jetson_type(verbose=verbose)
        jetpack_version = get_jetpack_version(verbose=verbose)
        gpu_devices = get_available_gpu_devices(verbose=verbose)
    else:
        os_version = get_os_version(verbose=verbose)
        driver_version = get_driver_version(verbose=verbose)
        gpu_devices = get_available_gpu_devices(verbose=verbose)
    torch_available = is_torch_available()
    onnxruntime_available = is_onnxruntime_available()
    hf_transformers_available = is_hf_transformers_available()
    ultralytics_available = is_ultralytics_available()
    trt_python_package_available = is_trt_python_package_available()
    result = RuntimeXRayResult(
        gpu_available=len(gpu_devices) > 0,
        gpu_devices=gpu_devices,
        driver_version=driver_version,
        cuda_version=cuda_version,
        trt_version=trt_version,
        jetson_type=jetson_type,
        jetpack_version=jetpack_version,
        os_version=os_version,
        torch_available=torch_available,
        onnxruntime_available=onnxruntime_available,
        hf_transformers_available=hf_transformers_available,
        ultralytics_available=ultralytics_available,
        trt_python_package_available=trt_python_package_available,
    )
    if verbose:
        print(result)
    return result


@cache
def is_running_on_jetson() -> bool:
    if RUNNING_ON_JETSON is not None:
        return str2bool(value=RUNNING_ON_JETSON)
    return get_jetson_type() is not None


@cache
def get_available_gpu_devices() -> List[str]:
    num_devices = torch.cuda.device_count()
    result = []
    for i in range(num_devices):
        device_name = torch.cuda.get_device_name(i)
        result.append(device_name.replace(" ", "-").lower())
    return result


@cache
def get_cuda_version() -> Optional[Version]:
    _, cuda_version = get_trt_and_cuda_version_from_libnvinfer()
    if cuda_version:
        return cuda_version
    return get_cuda_version_from_nvcc()


def get_cuda_version_from_nvcc() -> Optional[Version]:
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode()
        for line in output.splitlines():
            match = re.search(r"release\s+([0-9.]+),", line)
            if match:
                return Version(match.group(1))
    except Exception:
        return None


@cache
def get_trt_version() -> Optional[Version]:
    trt_version, _ = get_trt_and_cuda_version_from_libnvinfer()
    if trt_version:
        return trt_version
    try:
        import tensorrt as trt

        return Version(trt.__version__)
    except Exception:
        return None


@cache
def get_trt_and_cuda_version_from_libnvinfer() -> (
    Tuple[Optional[Version], Optional[Version]]
):
    try:
        result = subprocess.run(
            "dpkg -l | grep libnvinfer-bin", shell=True, capture_output=True, text=True
        )
        if result.returncode != 0:
            return None, None
        result_chunks = result.stdout.strip().split()
        libraries_versions = result_chunks[2].split("+cuda")
        if len(libraries_versions) != 2:
            return Version(libraries_versions[0]), None
        return Version(libraries_versions[0]), Version(libraries_versions[1])
    except Exception:
        return None, None


@cache
def get_jetson_type() -> Optional[str]:
    declared_json_module = os.getenv("JETSON_MODULE")
    if declared_json_module:
        return resolve_jetson_type(jetson_module_name=declared_json_module)
    return get_jetson_type_from_device_tree()


def get_jetson_type_from_device_tree() -> Optional[str]:
    try:
        with open("/proc/device-tree/model") as f:
            model_headline = f.read().strip().split("\n")[0]
            return resolve_jetson_type(jetson_module_name=model_headline)
    except Exception:
        return None


def resolve_jetson_type(jetson_module_name: str) -> str:
    for jetson_device in JETSON_DEVICES_TABLE:
        if jetson_module_name.startswith(jetson_device):
            return jetson_device.replace(" ", "-").lower()
    raise JetsonTypeResolutionError(
        f"Could not resolve jetson type. Value found in environment: {jetson_module_name}"
    )


@cache
def get_jetpack_version() -> Optional[str]:
    if JETPACK_VERSION:
        return JETPACK_VERSION
    return get_jetpack_version_from_tegra_release()


def get_jetpack_version_from_tegra_release() -> Optional[str]:
    try:
        with open("/etc/nv_tegra_release") as f:
            file_header = f.readline()
            match = re.search(r"R(\d+).*REVISION:\s*([\d.]+)", file_header)
            if match:
                major = match.group(1)
                minor_patch = match.group(2)
                return f"{major}.{minor_patch}"
            return None
    except Exception:
        return None


@cache
def get_os_version() -> Optional[str]:
    system = platform.system()
    if system == "Linux":
        return get_linux_os_version()
    elif system == "Darwin":
        return get_mac_os_version()
    elif system == "Windows":
        return get_windows_os_version()
    return None


def get_linux_os_version() -> Optional[str]:
    os_version_from_os_release = get_linux_os_from_os_release()
    if os_version_from_os_release:
        return os_version_from_os_release
    return platform.platform().lower()


def get_linux_os_from_os_release() -> Optional[str]:
    try:
        with open("/etc/os-release") as f:
            data = dict(line.strip().split("=", 1) for line in f if "=" in line)
        distro_name = data["NAME"].strip('"')
        version_id = data["VERSION_ID"].strip('"')
        return f"{distro_name}-{version_id}".lower()
    except Exception:
        return None


def get_mac_os_version() -> Optional[str]:
    return platform.platform().lower()


def get_windows_os_version() -> Optional[str]:
    return platform.platform().lower()


@cache
def get_driver_version() -> Optional[Version]:
    try:
        with open("/proc/driver/nvidia/version") as f:
            head_line = f.readline()
        match = re.search(r"\b(\d+(\.\d+){1,2})\b", head_line)
        if match:
            return Version(match.group(1))
    except Exception:
        return None


@cache
def is_trt_python_package_available() -> bool:
    try:
        import tensorrt

        return True
    except ImportError:
        return False


@cache
def is_torch_available() -> bool:
    try:
        import torch

        return True
    except ImportError:
        return False


@cache
def is_onnxruntime_available() -> bool:
    try:
        import onnxruntime

        return True
    except ImportError:
        return False


@cache
def is_hf_transformers_available() -> bool:
    try:
        import transformers

        return True
    except ImportError:
        return False


@cache
def is_ultralytics_available() -> bool:
    try:
        import ultralytics

        return True
    except ImportError:
        return False
