import requests
from pydantic import BaseModel
from typing import Literal

from inference.core.logger import logger
from inference.core.devices.utils import GLOBAL_DEVICE_ID
from inference.enterprise.device_manager.container_service import container_service
from inference.core.env import API_BASE_URL


class Command(BaseModel):
    id: str
    containerId: str
    command: Literal["restart", "stop", "ping", "snapshot"]
    deviceId: str
    requested_on: int


def fetch_commands():
    resp = requests.get(f"{API_BASE_URL}/devices/{GLOBAL_DEVICE_ID}/commands").json()
    for cmd in resp.get("data", []):
        handle_command(cmd)


def handle_command(remote_command: Command):
    was_processed = False
    cmd_payload = remote_command.dict()
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
        case _:
            logger.error("Unknown command: {}".format(cmd))
    return ack_command(cmd_payload.get("id"), was_processed, data=data)


def ack_command(command_id, was_processed, data=None):
    post_body = dict()
    post_body["commandId"] = command_id
    post_body["wasProcessed"] = was_processed
    if data:
        post_body["data"] = data
    requests.post(
        f"{API_BASE_URL}/devices/{GLOBAL_DEVICE_ID}/commands/ack", json=post_body
    )
