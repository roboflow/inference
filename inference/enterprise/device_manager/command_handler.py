from typing import Literal, Optional

import requests
from pydantic import BaseModel

from inference.core.devices.utils import GLOBAL_DEVICE_ID
from inference.core.env import API_BASE_URL, API_KEY
from inference.core.logger import logger
from inference.core.utils.url_utils import ApiUrl
from inference.enterprise.device_manager.container_service import container_service


class Command(BaseModel):
    id: str
    containerId: str
    command: Literal["restart", "stop", "ping", "snapshot"]
    deviceId: str
    requested_on: Optional[int] = None


def fetch_commands():
    url = ApiUrl(
        f"{API_BASE_URL}/devices/{GLOBAL_DEVICE_ID}/commands?api_key={API_KEY}"
    )
    resp = requests.get(url).json()
    for cmd in resp.get("data", []):
        handle_command(cmd)


def handle_command(remote_command: Command):
    was_processed = False
    cmd_payload = remote_command
    container_id = cmd_payload.get("containerId")
    container = container_service.get_container_by_id(container_id)
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
    url = ApiUrl(f"{API_BASE_URL}/devices/{GLOBAL_DEVICE_ID}/commands/ack")
    requests.post(url, json=post_body)
