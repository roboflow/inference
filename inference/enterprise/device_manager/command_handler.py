from typing import Literal, Optional

import requests
from pydantic import BaseModel

import docker
from inference.core.devices.utils import GLOBAL_DEVICE_ID
from inference.core.env import API_BASE_URL, API_KEY
from inference.core.logger import logger
from inference.core.utils.url_utils import wrap_url
from inference.enterprise.device_manager.container_service import get_container_by_id


class Command(BaseModel):
    id: str
    containerId: str
    command: Literal["restart", "stop", "ping", "snapshot", "update_version"]
    deviceId: str
    requested_on: Optional[int] = None


def fetch_commands():
    url = wrap_url(
        f"{API_BASE_URL}/devices/{GLOBAL_DEVICE_ID}/commands?api_key={API_KEY}"
    )
    resp = requests.get(url).json()
    for cmd in resp.get("data", []):
        handle_command(cmd)


def handle_command(cmd_payload: dict):
    was_processed = False
    container_id = cmd_payload.get("containerId")
    container = get_container_by_id(container_id)
    if not container:
        logger.warn(f"Container with id {container_id} not found")
        ack_command(cmd_payload.get("id"), was_processed)
        return
    cmd = cmd_payload.get("command")
    data = None
    match cmd:
        case "restart":
            was_processed, data = container.restart()
        case "stop":
            was_processed, data = container.stop()
        case "ping":
            was_processed, data = container.ping()
        case "snapshot":
            was_processed, data = container.snapshot()
        case "start":
            was_processed, data = container.start()
        case "update_version":
            was_processed, data = handle_version_update(container)
        case _:
            logger.error("Unknown command: {}".format(cmd))
    return ack_command(cmd_payload.get("id"), was_processed, data=data)


def ack_command(command_id, was_processed, data=None):
    post_body = dict()
    post_body["api_key"] = API_KEY
    post_body["commandId"] = command_id
    post_body["wasProcessed"] = was_processed
    if data:
        post_body["data"] = data
    url = wrap_url(f"{API_BASE_URL}/devices/{GLOBAL_DEVICE_ID}/commands/ack")
    requests.post(url, json=post_body)


def handle_version_update(container):
    try:
        config = container.get_startup_config()
        image_name = config["image"].split(":")[0]
        container.kill()
        client = docker.from_env()
        new_container = client.containers.run(
            image=f"{image_name}:latest",
            detach=config["detach"],
            privileged=config["privileged"],
            labels=config["labels"],
            ports=config["port_bindings"],
            environment=config["env"],
            network="host",
        )
        logger.info(f"New container started {new_container}")
        return True, None
    except Exception as e:
        logger.error(e)
        return False, None
