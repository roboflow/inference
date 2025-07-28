import http.client
import json
import platform
import re
import socket
import time
import uuid

from inference.core.cache import cache
from inference.core.logger import logger


def get_model_metrics(
    inference_server_id: str, model_id: str, min: float = -1, max: float = float("inf")
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
        f"inference:{inference_server_id}:{model_id}", min=min, max=max, withscores=True
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
        f"error:{inference_server_id}:{model_id}", min=min, max=max, withscores=True
    )
    num_errors = len(errors_with_times)
    return {
        "num_inferences": num_inferences,
        "avg_inference_time": avg_inference_time,
        "num_errors": num_errors,
    }


def get_system_info() -> dict:
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
    except Exception as e:
        logger.exception(e)
    finally:
        return info


def get_inference_results_for_model(
    inference_server_id: str, model_id: str, min: float = -1, max: float = float("inf")
):
    inferences_with_times = cache.zrangebyscore(
        f"inference:{inference_server_id}:{model_id}", min=min, max=max, withscores=True
    )
    inference_results = []
    for result, score in inferences_with_times:
        # Don't send large image files
        if result.get("request", {}).get("image"):
            del result["request"]["image"]
        responses = result.get("response")
        if responses:
            if not isinstance(responses, list):
                responses = [responses]
            for resp in responses:
                if resp.get("image"):
                    del resp["image"]
        inference_results.append({"request_time": score, "inference": result})

    return inference_results


def get_container_stats(docker_socket_path: str) -> dict:
    """
    Gets the container stats.
    Returns:
        dict: A dictionary containing the container stats.
    """

    try:
        container_id = socket.gethostname()
        connection = http.client.HTTPConnection("localhost")
        connection.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.sock.connect(docker_socket_path)
        connection.request(
            "GET",
            f"/containers/{container_id}/stats?stream=false",
            headers={"Host": "localhost"},
        )
        response = connection.getresponse()
        data = response.read()
        connection.close()
        if response.status != 200:
            raise Exception(data.decode())
        stats = json.loads(data.decode())
        return {"stats": stats}
    except Exception as e:
        logger.exception(e)
        raise Exception("An error occurred while fetching container stats.")
