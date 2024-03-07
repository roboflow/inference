from cpuinfo import get_cpu_info
from GPUtil import GPUtil


def retrieve_platform_specifics() -> dict:
    cpu_info = get_cpu_info()
    gpus = GPUtil.getGPUs()
    gpu_names = list({gpu.name for gpu in gpus})
    return {
        "python_version": cpu_info["python_version"],
        "architecture": cpu_info["arch_string_raw"],
        "bits": cpu_info["bits"],
        "cpu_count": cpu_info["count"],
        "cpu_model": cpu_info.get("brand_raw"),
        "gpu_count": len(gpu_names),
        "gpu_names": gpu_names,
    }
