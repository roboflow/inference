import socket
import platform
import os

from inference.core.env import DEVICE_ID


def is_running_in_docker():
    return os.path.exists("/.dockerenv")


def get_gpu_id():
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
    try:
        # Fetch the device's serial number
        serial_number = os.popen("cat /proc/device-tree/serial-number").read().strip()
        return serial_number
    except Exception as e:
        return None


def get_device_id():
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
