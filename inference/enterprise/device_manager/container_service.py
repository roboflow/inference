import time
import sys
import base64
from dataclasses import dataclass
from datetime import datetime
from PIL import Image
from io import BytesIO

import requests

import docker
from inference.core.logger import logger
from inference.core.env import METRICS_INTERVAL
from inference.core.cache import cache
from inference.core.utils.image_utils import load_image
from inference.enterprise.device_manager.helpers import get_cache_model_items


@dataclass
class InferServerContainer:
    status: str
    alias: str
    container_id: str
    port: int
    host: str
    startup_time: float
    version: str

    def __init__(self, docker_container, details):
        self.container = docker_container
        self.status = details.get("status")
        self.alias = details.get("alias")
        self.container_id = details.get("short_id")
        self.port = details.get("port")
        self.host = details.get("host")
        self.version = details.get("version")
        t = details.get("startup_time_ts").split(".")[0]
        self.startup_time = (
            datetime.strptime(t, "%Y-%m-%dT%H:%M:%S").timestamp()
            if t is not None
            else datetime.now().timestamp()
        )

    def kill(self):
        try:
            self.container.kill()
            return True, None
        except Exception as e:
            logger.error(e)
            return False, None

    def restart(self):
        try:
            self.container.restart()
            return True, None
        except Exception as e:
            logger.error(e)
            return False, None

    def stop(self):
        try:
            self.container.stop()
            return True, None
        except Exception as e:
            logger.error(e)
            return False, None

    def start(self):
        try:
            self.container.start()
            return True, None
        except Exception as e:
            logger.error(e)
            return False, None

    def inspect(self):
        try:
            info = requests.get(f"http://{self.host}:{self.port}/info").json()
            return True, info
        except Exception as e:
            logger.error(e)
            return False, None

    def get_startup_config(self):
        """
        Get the startup configuration for this container.

        Returns:
            dict: A dictionary containing the startup configuration for this container.
        """
        env_vars = self.container.attrs.get("Config", {}).get("Env", {})
        port_bindings = self.container.attrs.get("HostConfig", {}).get(
            "PortBindings", {}
        )
        detached = self.container.attrs.get("HostConfig", {}).get("Detached", False)
        image = self.container.attrs.get("Config", {}).get("Image", "")
        privileged = self.container.attrs.get("HostConfig", {}).get("Privileged", False)
        labels = self.container.attrs.get("Config", {}).get("Labels", {})
        env = []
        for var in env_vars:
            name, value = var.split("=")
            env.append(f"{name}={value}")
        return {
            "env": env,
            "port_bindings": port_bindings,
            "detach": detached,
            "image": image,
            "privileged": privileged,
            "labels": labels,
            # TODO: add device requests
        }


def is_inference_server_container(container):
    """
    Checks if a container is an inference server container

    Args:
        container (any): A container object from the Docker SDK

    Returns:
        boolean: True if the container is an inference server container, False otherwise
    """
    image_tags = container.image.tags
    for t in image_tags:
        if t.startswith("roboflow/roboflow-inference-server"):
            return True
    return False


def get_inference_containers():
    """
    Discovers inference server containers running on the host
    and parses their information into a list of InferServerContainer objects
    """
    client = docker.from_env()
    containers = client.containers.list()
    inference_containers = []
    for c in containers:
        if is_inference_server_container(c):
            details = parse_container_info(c)
            info = {}
            try:
                info = requests.get(
                    f"http://{details['host']}:{details['port']}/info", timeout=3
                ).json()
            except Exception as e:
                logger.error(f"Failed to get info from container {details} {e}")
            details.update(info)
            details["alias"] = details.get("uuid")
            infer_container = InferServerContainer(c, details)
            if len(inference_containers) == 0:
                inference_containers.append(infer_container)
                continue
            for ic in inference_containers:
                if ic.alias == infer_container.alias:
                    continue
                inference_containers.append(infer_container)
    return inference_containers


def parse_container_info(c):
    """
    Parses the container information into a dictionary

    Args:
        c (any): Docker SDK Container object

    Returns:
        dict: A dictionary containing the container information
    """
    env = c.attrs.get("Config", {}).get("Env", {})
    info = {"short_id": c.short_id, "port": 9001, "host": "0.0.0.0"}
    for var in env:
        if var.startswith("PORT="):
            info["port"] = var.split("=")[1]
        elif var.startswith("HOST="):
            info["host"] = var.split("=")[1]
    status = c.attrs.get("State", {}).get("Status")
    if status:
        info["status"] = status
    container_name = c.attrs.get("Name")
    if container_name:
        info["container_name_on_host"] = container_name
    startup_time = c.attrs.get("State", {}).get("StartedAt")
    if startup_time:
        info["startup_time_ts"] = startup_time
    return info


def get_container_by_id(alias, default_to_first=False):
    """
    Gets an inference server container by its alias

    Args:
        id (string): The id of the container

    Returns:
        container: The container object if found, None otherwise
    """
    containers = get_inference_containers()
    for c in containers:
        if c.alias == alias:
            return c
    if default_to_first and len(containers) > 0:
        return containers[0]
    return None


def get_container_ids():
    """
    Gets the ids of the inference server containers

    Returns:
        list: A list of container ids
    """
    containers = get_inference_containers()
    return [c.alias for c in containers]


def _get_cached_model_ids(container_name, now=None):
    if now is None:
        now = time.time()
    cached_models = get_cache_model_items()
    api_keys = cached_models.get(container_name, {}).keys()
    model_ids = []
    for api_key in api_keys:
        mids = cached_models.get(container_name, {}).get(api_key, [])
        model_ids.extend(mids)

    return model_ids


def _get_formatted_image_value(image):
    value = None
    if image["type"] == "base64" or image["type"] == "url":
        logger.info(f"returning value {image['value']}")
        return image["value"]
    loaded_image, is_bgr = load_image(image)
    image = Image.fromarray(loaded_image)
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    value = base64_image
    return f"data:image/jpeg;base64, {value}"


def get_latest_inferences(container_id=None, max=1):
    container = get_container_by_id(container_id, default_to_first=True)
    if container is None:
        return {}

    now = time.time()
    start = now - (METRICS_INTERVAL * 2)
    model_ids = _get_cached_model_ids(container.alias, now=now)

    num_images = 0
    latest_inferred_images = {}
    for model_id in model_ids:
        if num_images >= max:
            break
        latest_reqs = cache.zrangebyscore(
            f"inference:{container.alias}:{model_id}", min=start, max=now
        )
        latest_reqs = filter(lambda x: isinstance(x, dict), latest_reqs)
        for req in latest_reqs:
            images = req.get("request", {}).get("image")
            response = req.get("response", [])[0]
            if response is None:
                continue
            image_dims = response.get("image")
            predictions = response.get("predictions", [])
            if images is None or len(images) == 0:
                continue
            if type(images) is not list:
                images = [images]
            for image in images:
                value = _get_formatted_image_value(image)
                if latest_inferred_images.get(model_id) is None:
                    latest_inferred_images[model_id] = []
                latest_inferred_images[model_id].append(
                    {
                        "image": value,
                        "type": image["type"],
                        "dimensions": image_dims,
                        "predictions": predictions,
                    }
                )
                if value:
                    num_images += 1

    return latest_inferred_images


def check_for_duplicate_aliases():
    containers = get_inference_containers()
    aliases = [c.alias for c in containers]
    counts = {}
    for a in aliases:
        counts[a] = counts.get(a, 0) + 1
        if counts[a] > 1:
            logger.fatal(
                f"Multiple containers found with alias {a}.\n Please restart the inference server and specify a unique value for the `INFERENCE_SERVER_ID` environment variable."
            )
            sys.exit(1)
