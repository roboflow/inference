from cpuinfo import get_cpu_info
from pynvml import (
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetName,
    nvmlInit,
)


def retrieve_platform_specifics() -> dict:
    gpus_count = 0
    gpu_names = []
    try:
        nvmlInit()
        gpus_count = nvmlDeviceGetCount()
        for i in range(gpus_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            gpu_name = nvmlDeviceGetName(handle)
            gpu_names.append(gpu_name)
    except Exception:
        # No GPUs
        pass

    cpu_info = get_cpu_info()
    return {
        "python_version": cpu_info["python_version"],
        "architecture": cpu_info["arch_string_raw"],
        "bits": cpu_info["bits"],
        "cpu_count": cpu_info["count"],
        "cpu_model": cpu_info.get("brand_raw"),
        "gpu_count": len(gpu_names),
        "gpu_names": gpu_names,
    }
