import os
import platform
import random
import string
import uuid

from inference.core.env import DEVICE_ID, INFERENCE_SERVER_ID


def is_running_in_docker():
    """Checks if the current process is running inside a Docker container.

    Returns:
        bool: True if running inside a Docker container, False otherwise.
    """
    return os.path.exists("/.dockerenv")


def get_gpu_id():
    """Fetches the GPU ID if a GPU is present.

    Tries to import and use the `pynvml` (delivered by nvidia-ml-py) module to retrieve the GPU information.

    Returns:
        Optional[int]: GPU ID if available, None otherwise.
    """
    try:
        from pynvml import nvmlDeviceGetCount, nvmlInit

        nvmlInit()
        gpus_count = nvmlDeviceGetCount()
        if gpus_count:
            return 0
    except ImportError:
        return None
    except Exception:
        return None


def get_cpu_id():
    """Fetches the CPU ID based on the operating system.

    Attempts to get the CPU ID for Windows, Linux, and MacOS.
    In case of any error or an unsupported OS, returns None.

    Returns:
        Optional[str]: CPU ID string if available, None otherwise.
    """
    try:
        if platform.system() == "Windows":
            return os.popen("wmic cpu get ProcessorId").read().strip()
        elif platform.system() == "Linux":
            return (
                open("/proc/cpuinfo").read().split("processor")[0].split(":")[1].strip()
            )
        elif platform.system() == "Darwin":
            import subprocess

            return (
                subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"])
                .strip()
                .decode()
            )
    except Exception as e:
        return None


def get_jetson_id():
    """Fetches the Jetson device's serial number.

    Attempts to read the serial number from the device tree.
    In case of any error, returns None.

    Returns:
        Optional[str]: Jetson device serial number if available, None otherwise.
    """
    try:
        # Fetch the device's serial number
        if not os.path.exists("/proc/device-tree/serial-number"):
            return None
        serial_number = os.popen("cat /proc/device-tree/serial-number").read().strip()
        if serial_number == "":
            return None
        return serial_number
    except Exception as e:
        return None


def get_container_id():
    if is_running_in_docker():
        return (
            os.popen(
                "cat /proc/self/cgroup | grep 'docker' | sed 's/^.*\///' | tail -n1"
            )
            .read()
            .strip()
        )
    else:
        return str(uuid.uuid4())


def random_string(length):
    letters = string.ascii_letters + string.digits
    return "".join(random.choice(letters) for i in range(length))


def get_device_hostname():
    """Fetches the device's hostname.

    Returns:
        str: The device's hostname.
    """
    return platform.node()


def get_inference_server_id():
    """Fetches a unique device ID.

    Tries to get the GPU ID first, then falls back to CPU ID.
    If the application is running inside Docker, the Docker container ID is appended to the hostname.

    Returns:
        str: A unique string representing the device. If unable to determine, returns "UNKNOWN".
    """
    try:
        if INFERENCE_SERVER_ID is not None:
            return INFERENCE_SERVER_ID
        id = random_string(6)
        gpu_id = get_gpu_id()
        if gpu_id is not None:
            return f"{id}-GPU-{gpu_id}"
        jetson_id = get_jetson_id()
        if jetson_id is not None:
            return f"{id}-JETSON-{jetson_id}"
        return id
    except Exception as e:
        return "UNKNOWN"


GLOBAL_INFERENCE_SERVER_ID = get_inference_server_id()
GLOBAL_DEVICE_ID = DEVICE_ID if DEVICE_ID is not None else get_device_hostname()
