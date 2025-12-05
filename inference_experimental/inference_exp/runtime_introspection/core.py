import os
import platform
import re
import subprocess
from dataclasses import dataclass
from functools import cache
from typing import List, Optional, Set, Tuple

import torch
from inference_exp.configuration import L4T_VERSION, RUNNING_ON_JETSON
from inference_exp.errors import JetsonTypeResolutionError
from inference_exp.logger import LOGGER
from inference_exp.utils.environment import str2bool
from packaging.version import InvalidVersion, Version

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
    gpu_devices_cc: List[Version]
    driver_version: Optional[Version]
    cuda_version: Optional[Version]
    trt_version: Optional[Version]
    jetson_type: Optional[str]
    l4t_version: Optional[Version]
    os_version: Optional[str]
    torch_available: bool
    torch_version: Optional[Version]
    torchvision_version: Optional[Version]
    onnxruntime_version: Optional[Version]
    available_onnx_execution_providers: Optional[Set[str]]
    hf_transformers_available: bool
    ultralytics_available: bool
    trt_python_package_available: bool
    mediapipe_available: bool

    def __str__(self) -> str:
        gpu_devices_str = ", ".join(self.gpu_devices)
        return (
            f"RuntimeXRayResult(gpu_available={self.gpu_available}, gpu_devices=[{gpu_devices_str}], "
            f"gpu_devices_cc={self.gpu_devices_cc}, gpu_driver={self.driver_version}, "
            f"cuda_version={self.cuda_version}, trt_version={self.trt_version}, "
            f"jetson_type={self.jetson_type}, l4t_version={self.l4t_version}, os_version={self.os_version}, "
            f"torch_available={self.torch_available}, onnxruntime_version={self.onnxruntime_version}, "
            f"available_onnx_execution_providers={self.available_onnx_execution_providers}, hf_transformers_available={self.hf_transformers_available}, "
            f"ultralytics_available={self.ultralytics_available}, "
            f"trt_python_package_available={self.trt_python_package_available}, torch_version={self.torch_version}, "
            f"torchvision_version={self.torchvision_version}, mediapipe_available={self.mediapipe_available})"
        )


@cache
def x_ray_runtime_environment() -> RuntimeXRayResult:
    trt_version = get_trt_version()
    cuda_version = get_cuda_version()
    jetson_type, l4t_version, os_version, driver_version = None, None, None, None
    if is_running_on_jetson():
        jetson_type = get_jetson_type()
        l4t_version = get_l4t_version()
    else:
        os_version = get_os_version()
    driver_version = get_driver_version()
    gpu_devices = get_available_gpu_devices()
    gpu_devices_cc = get_available_gpu_devices_cc()
    torch_available = is_torch_available()
    torch_version = get_torch_version()
    torchvision_version = get_torchvision_version()
    onnx_info = get_onnxruntime_info()
    if onnx_info:
        onnxruntime_version, available_onnx_execution_providers = onnx_info
    else:
        onnxruntime_version, available_onnx_execution_providers = None, None
    hf_transformers_available = is_hf_transformers_available()
    ultralytics_available = is_ultralytics_available()
    trt_python_package_available = is_trt_python_package_available()
    mediapipe_available = is_mediapipe_available()
    return RuntimeXRayResult(
        gpu_available=len(gpu_devices) > 0,
        gpu_devices=gpu_devices,
        gpu_devices_cc=gpu_devices_cc,
        driver_version=driver_version,
        cuda_version=cuda_version,
        trt_version=trt_version,
        jetson_type=jetson_type,
        l4t_version=l4t_version,
        os_version=os_version,
        torch_available=torch_available,
        torch_version=torch_version,
        torchvision_version=torchvision_version,
        onnxruntime_version=onnxruntime_version,
        available_onnx_execution_providers=available_onnx_execution_providers,
        hf_transformers_available=hf_transformers_available,
        ultralytics_available=ultralytics_available,
        trt_python_package_available=trt_python_package_available,
        mediapipe_available=mediapipe_available,
    )


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
def get_available_gpu_devices_cc() -> List[Version]:
    num_devices = torch.cuda.device_count()
    result = []
    for i in range(num_devices):
        device_cc_raw = torch.cuda.get_device_capability(i)
        result.append(Version(f"{device_cc_raw[0]}.{device_cc_raw[1]}"))
    return result


@cache
def get_cuda_version() -> Optional[Version]:
    try:
        result = subprocess.run(
            "dpkg -l | grep cuda-cudart", shell=True, capture_output=True, text=True
        )
        if result.returncode != 0:
            return None
        result_chunks = result.stdout.strip().split(os.linesep)[0].split()
        return Version(result_chunks[2])
    except Exception:
        return None


@cache
def get_trt_version() -> Optional[Version]:
    trt_version = get_trt_version_from_libnvinfer()
    if trt_version is not None:
        return trt_version
    try:
        import tensorrt as trt

        return Version(trt.__version__)
    except Exception:
        return None


@cache
def get_trt_version_from_libnvinfer() -> Optional[Version]:
    try:
        result = subprocess.run(
            "dpkg -l | grep libnvinfer-bin", shell=True, capture_output=True, text=True
        )
        if result.returncode != 0:
            return None
        result_chunks = result.stdout.strip().split()
        libraries_versions = result_chunks[2].split("+cuda")
        return Version(libraries_versions[0])
    except Exception:
        return None


@cache
def get_jetson_type() -> Optional[str]:
    declared_json_module = os.getenv("JETSON_MODULE")
    if declared_json_module:
        return resolve_jetson_type(jetson_module_name=declared_json_module)
    jetson_type_from_hardware_inspection = get_jetson_type_from_hardware_inspection()
    if jetson_type_from_hardware_inspection:
        return jetson_type_from_hardware_inspection
    return get_jetson_type_from_device_tree()


def get_jetson_type_from_hardware_inspection() -> Optional[str]:
    try:
        result = subprocess.run(
            "lshw | grep 'product: NVIDIA Jetson'",
            shell=True,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        for result_line in result.stdout.strip().split("\n"):
            start_idx = result_line.find("NVIDIA")
            end_idx = result_line.find("HDA")
            if start_idx < 0 or end_idx < 0:
                continue
            jetson_type = result_line[start_idx:end_idx].strip()
            return resolve_jetson_type(jetson_module_name=jetson_type)
        return None
    except Exception:
        return None


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
        message=f"Could not resolve jetson type. Value found in environment: {jetson_module_name}",
        help_url="https://todo",
    )


@cache
def get_l4t_version() -> Optional[Version]:
    if L4T_VERSION:
        return Version(L4T_VERSION)
    return get_l4t_version_from_tegra_release()


def get_l4t_version_from_tegra_release() -> Optional[Version]:
    try:
        with open("/etc/nv_tegra_release") as f:
            file_header = f.readline()
            match = re.search(r"R(\d+).*REVISION:\s*([\d.]+)", file_header)
            if match:
                major = match.group(1)
                minor_patch = match.group(2)
                return Version(f"{major}.{minor_patch}")
            return None
    except Exception:
        return None


@cache
def get_os_version() -> Optional[str]:
    system = platform.system().lower()
    if system == "linux":
        return get_linux_os_version()
    elif system == "darwin":
        return platform.platform().lower()
    elif system == "windows":
        return platform.platform().lower()
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
def is_mediapipe_available() -> bool:
    try:
        import mediapipe

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
def get_torch_version() -> Optional[Version]:
    try:
        import torch

        version_str = torch.__version__
        if "+" in version_str:
            version_str = version_str.split("+")[0]
        return Version(version_str)
    except ImportError:
        return None
    except InvalidVersion as e:
        LOGGER.warning(f"Could not parse torch version: {e}")
        return None


@cache
def get_torchvision_version() -> Optional[Version]:
    try:
        import torchvision

        version_str = torchvision.__version__
        if "+" in version_str:
            version_str = version_str.split("+")[0]
        return Version(version_str)
    except ImportError:
        return None
    except InvalidVersion as e:
        LOGGER.warning(f"Could not parse torch version: {e}")
        return None


@cache
def get_onnxruntime_info() -> Optional[Tuple[Version, Set[str]]]:
    try:
        import onnxruntime

        available_providers = onnxruntime.get_available_providers()
        return Version(onnxruntime.__version__), available_providers
    except ImportError:
        return None


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
