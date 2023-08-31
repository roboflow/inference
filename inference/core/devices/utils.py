import os
import platform
import socket

from inference.core.env import DEVICE_ID


def is_running_in_docker():
    """Checks if the current process is running inside a Docker container.

    Returns:
        bool: True if running inside a Docker container, False otherwise.
    """
    return os.path.exists("/.dockerenv")


def get_gpu_id():
    """Fetches the GPU ID if a GPU is present.

    Tries to import and use the `GPUtil` module to retrieve the GPU information.

    Returns:
        Optional[int]: GPU ID if available, None otherwise.
    """
    try:
        import GPUtil

        GPUs = GPUtil.getGPUs()
        if GPUs:
            return GPUs[0].id
    except ImportError:
        return None
    except Exception as e:
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
        serial_number = os.popen("cat /proc/device-tree/serial-number").read().strip()
        return serial_number
    except Exception as e:
        return None


def get_device_id():
    """Fetches a unique device ID.

    Tries to get the GPU ID first, then falls back to CPU ID.
    If the application is running inside Docker, the Docker container ID is appended to the hostname.

    Returns:
        str: A unique string representing the device. If unable to determine, returns "UNKNOWN".
    """
    try:
        if DEVICE_ID is not None:
            return DEVICE_ID
        id = get_gpu_id()
        if id is not None:
            return f"GPU-{id}"

        id = get_cpu_id()
        if id is not None:
            return f"CPU-{id}"

        # Fallback to hostname
        hostname = socket.gethostname()

        if is_running_in_docker():
            # Append Docker container ID to the hostname
            container_id = (
                os.popen(
                    "cat /proc/self/cgroup | grep 'docker' | sed 's/^.*\///' | tail -n1"
                )
                .read()
                .strip()
            )
            hostname = f"{hostname}-DOCKER-{container_id}"

        return hostname
    except Exception as e:
        return "UNKNOWN"
