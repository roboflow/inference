import json
import platform
import re
import socket
import time
import uuid

from inference.core.cache import cache
from inference.core.logger import logger
from inference.core.version import __version__


def get_model_metrics(
    inverence_server_id: str, model_id: str, min: float = -1, max: float = float("inf")
) -> dict:
    """
    Gets the metrics for a given model between a specified time range.

    Args:
        device_id (str): The identifier of the device.
        model_id (str): The identifier of the model.
        start (float, optional): The starting timestamp of the time range. Defaults to -1.
        stop (float, optional): The ending timestamp of the time range. Defaults to float("inf").

    Returns:
        dict: A dictionary containing the metrics of the model:
              - num_inferences (int): The number of inferences made.
              - avg_inference_time (float): The average inference time.
              - num_errors (int): The number of errors occurred.
    """
    now = time.time()
    inferences_with_times = cache.zrangebyscore(
        f"inference:{inverence_server_id}:{model_id}", min=min, max=max, withscores=True
    )
    num_inferences = len(inferences_with_times)
    inference_times = []
    for inference, t in inferences_with_times:
        response = inference["response"]
        if isinstance(response, list):
            times = [r["time"] for r in response if "time" in r]
            inference_times.extend(times)
        else:
            if "time" in response:
                inference_times.append(response["time"])
    avg_inference_time = (
        sum(inference_times) / len(inference_times) if len(inference_times) > 0 else 0
    )
    errors_with_times = cache.zrangebyscore(
        f"error:{inverence_server_id}:{model_id}", min=min, max=max, withscores=True
    )
    num_errors = len(errors_with_times)
    return {
        "num_inferences": num_inferences,
        "avg_inference_time": avg_inference_time,
        "num_errors": num_errors,
    }


def get_system_info():
    """Collects system information such as platform, architecture, hostname, IP address, MAC address, and processor details.

    Returns:
        dict: A dictionary containing detailed system information.
    """
    info = {}
    try:
        info["platform"] = platform.system()
        info["platform_release"] = platform.release()
        info["platform_version"] = platform.version()
        info["architecture"] = platform.machine()
        info["hostname"] = socket.gethostname()
        info["ip_address"] = socket.gethostbyname(socket.gethostname())
        info["mac_address"] = ":".join(re.findall("..", "%012x" % uuid.getnode()))
        info["processor"] = platform.processor()
        return json.dumps(info)
    except Exception as e:
        logger.exception(e)
    finally:
        return info
