import base64
import time
from dataclasses import dataclass
from datetime import datetime

import requests

import docker
from inference.core.cache import cache
from inference.core.env import METRICS_INTERVAL
from inference.core.logger import logger
from inference.core.utils.image_utils import load_image_rgb
from inference.enterprise.device_manager.helpers import get_cache_model_items


@dataclass
class InferServerContainer:
    status: str
    id: str
    port: int
    host: str
    startup_time: float
    version: str

    def __init__(self, docker_container, details):
        self.container = docker_container
        self.status = details.get("status")
        self.id = details.get("uuid")
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

    def snapshot(self):
        try:
            snapshot = self.get_latest_inferred_images()
            snapshot.update({"container_id": self.id})
            return True, snapshot
        except Exception as e:
            logger.error(e)
            return False, None

    def get_latest_inferred_images(self, max=4):
        """
        Retrieve the latest inferred images and associated information for this container.

        This method fetches the most recent inferred images within the time interval defined by METRICS_INTERVAL.

        Args:
            max (int, optional): The maximum number of inferred images to retrieve.
                Defaults to 4.

        Returns:
            dict: A dictionary where each key represents a model ID associated with this
            container, and the corresponding value is a list of dictionaries containing
            information about the latest inferred images. Each dictionary has the following keys:
            - "image" (str): The base64-encoded image data.
            - "dimensions" (dict): Image dimensions (width and height).
            - "predictions" (list): A list of predictions or results associated with the image.

        Notes:
            - This method uses the global constant METRICS_INTERVAL to specify the time interval.
        """

        now = time.time()
        start = now - METRICS_INTERVAL
        api_keys = get_cache_model_items().get(self.id, dict()).keys()
        model_ids = []
        for api_key in api_keys:
            mids = get_cache_model_items().get(self.id, dict()).get(api_key, [])
            model_ids.extend(mids)
        num_images = 0
        latest_inferred_images = dict()
        for model_id in model_ids:
            if num_images >= max:
                break
            latest_reqs = cache.zrangebyscore(
                f"inference:{self.id}:{model_id}", min=start, max=now
            )
            for req in latest_reqs:
                images = req["request"]["image"]
                image_dims = req.get("response", {}).get("image", dict())
                predictions = req.get("response", {}).get("predictions", [])
                if images is None or len(images) == 0:
                    continue
                if type(images) is not list:
                    images = [images]
                for image in images:
                    value = None
                    if image["type"] == "base64":
                        value = image["value"]
                    else:
                        loaded_image = load_image_rgb(image)
                        image_bytes = loaded_image.tobytes()
                        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                        value = image_base64
                    if latest_inferred_images.get(model_id) is None:
                        latest_inferred_images[model_id] = []
                    inference = dict(
                        image=value, dimensions=image_dims, predictions=predictions
                    )
                    latest_inferred_images[model_id].append(inference)
                    num_images += 1
        return latest_inferred_images

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
                logger.error(f"Failed to get info from container {c.id} {details} {e}")
            details.update(info)
            infer_container = InferServerContainer(c, details)
            if len(inference_containers) == 0:
                inference_containers.append(infer_container)
                continue
            for ic in inference_containers:
                if ic.id == infer_container.id:
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
    info = {"container_id": c.id, "port": 9001, "host": "0.0.0.0"}
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


def get_container_by_id(id):
    """
    Gets an inference server container by its id

    Args:
        id (string): The id of the container

    Returns:
        container: The container object if found, None otherwise
    """
    containers = get_inference_containers()
    for c in containers:
        if c.id == id:
            return c
    return None


def get_container_ids():
    """
    Gets the ids of the inference server containers

    Returns:
        list: A list of container ids
    """
    containers = get_inference_containers()
    return [c.id for c in containers]
