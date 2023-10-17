from typing import Literal, Optional

from pydantic import BaseModel

import docker
from inference.core.logger import logger
from inference.enterprise.device_manager.container_service import get_container_by_id
from inference.enterprise.device_manager.helpers import get_device_id


class Command(BaseModel):
    id: str
    containerId: str
    command: Literal["restart", "stop", "ping", "snapshot", "update_version"]
    deviceId: str
    requested_on: Optional[int] = None


def handle_command(cmd_payload: dict):
    was_processed = False
    if get_device_id() != cmd_payload.get("deviceId"):
        return was_processed, None
    container_id = cmd_payload.get("containerId")
    container = get_container_by_id(container_id)
    if not container:
        return was_processed, None
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
    if was_processed:
        from inference.enterprise.device_manager.metrics_service import (
            send_metrics,
        )  # isort: skip

        send_metrics()


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
