import requests
from pydantic import BaseModel
from typing import Literal

from inference.core.logger import logger
from inference.core.devices.utils import GLOBAL_DEVICE_ID
from inference.enterprise.device_manager.container_service import ContainerService
from inference.core.env import API_BASE_URL


class Command(BaseModel):
    containerId: str
    command: Literal["restart", "stop", "ping"]
    deviceId: str
    requested_on: int


class RemoteCommandHandler:
    def __init__(self):
        self.container_service = ContainerService()

    def fetch_commands(self):
        resp = requests.get(
            f"{API_BASE_URL}/devices/{GLOBAL_DEVICE_ID}/commands"
        ).json()
        for cmd in resp.get("data", []):
            self.handle_command(cmd)

    def handle_command(self, remote_command: Command):
        was_processed = True
        cmd_payload = remote_command.dict()
        container_id = cmd_payload.get("containerId")
        container = self.container_service.get_container_by_id(container_id)
        if not container:
            logger.warn(f"Container with id {container_id} not found")
            self.ack_command(cmd_payload.get("id"), False)
            return
        cmd = cmd_payload.get("command")
        if cmd == "restart":
            container.restart()
        elif cmd == "stop":
            container.stop()
        elif cmd == "ping":
            container.ping()
        else:
            was_processed = False
            logger.error("Unknown command: {}".format(cmd))
        self.ack_command(cmd_payload.get("id"), was_processed)

    def ack_command(self, command_id, was_processed):
        post_body = dict()
        post_body["commandId"] = command_id
        post_body["wasProcessed"] = was_processed
        requests.post(
            f"{API_BASE_URL}/devices/{GLOBAL_DEVICE_ID}/commands/ack", json=post_body
        )
