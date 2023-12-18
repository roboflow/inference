import subprocess
from typing import List

import typer
from docker.models.containers import Container

import docker

docker_client = docker.from_env()


def ask_user_to_kill_container(container: Container) -> bool:
    name = container.attrs.get("Name", "")
    env_vars = container.attrs.get("Config", {}).get("Env", {})
    port = 9001
    for var in env_vars:
        if var.startswith("PORT="):
            port = var.split("=")[1]
    should_delete = typer.confirm(
        f" An inference server is already running in container {name} on port {port}. Are you sure you want to delete it?"
    )
    return should_delete


def is_inference_server_container(container: Container) -> bool:
    image_tags = container.image.tags
    for t in image_tags:
        if t.startswith("roboflow/roboflow-inference-server"):
            return True
    return False


def terminate_running_containers(
    containers: List[Container], interactive_mode: bool = True
) -> bool:
    """
    Args:
        containers (List[Container]): List of containers to handle
        interactive_mode (bool): Flag to determine if user prompt should decide on container termination

    Returns: boolean value that informs if there are containers that have not received SIGKILL
        as a result of procedure.
    """
    running_inference_containers = [
        c for c in containers if is_container_running(container=c)
    ]
    containers_to_kill = running_inference_containers
    if interactive_mode:
        containers_to_kill = [
            c for c in running_inference_containers if ask_user_to_kill_container(c)
        ]
    kill_containers(containers=containers_to_kill)
    return len(containers_to_kill) < len(running_inference_containers)


def is_container_running(container: Container) -> str:
    return container.attrs.get("State", {}).get("Status", "").lower() == "running"


def kill_containers(containers: List[Container]) -> None:
    for container in containers:
        container.kill()


def find_running_inference_containers() -> List[Container]:
    containers = []
    for c in docker_client.containers.list():
        if is_inference_server_container(c):
            if c.attrs.get("State", {}).get("Status", "").lower() == "running":
                containers.append(c)
    return containers


def get_image() -> str:
    try:
        subprocess.check_output("nvidia-smi")
        print("GPU detected. Using a GPU image.")
        return "roboflow/roboflow-inference-server-gpu:latest"
    except:
        print("No GPU detected. Using a CPU image.")
        return "roboflow/roboflow-inference-server-cpu:latest"


def start_inference_container(
    api_key,
    image=None,
    port=9001,
    labels=None,
    project="roboflow-platform",
    metrics_enabled=True,
    device_id=None,
    num_workers=1,
):
    containers = find_running_inference_containers()
    if len(containers) > 0:
        still_has_containers = terminate_running_containers(containers)
        if still_has_containers:
            print("Please kill the existing containers and try again.")
            return

    if image is None:
        image = get_image()

    device_requests = None
    privileged = False
    if "gpu" in image:
        privileged = True
        device_requests = [
            docker.types.DeviceRequest(device_ids=["all"], capabilities=[["gpu"]])
        ]

    print(f"Starting inference server container...")
    docker_client.containers.run(
        image=image,
        privileged=privileged,
        detach=True,
        labels=labels,
        ports={"9001": port},
        # network="host",
        device_requests=device_requests,
        environment=[
            "HOST=0.0.0.0",
            f"PORT={port}",
            f"PROJECT={project}",
            f"METRICS_ENABLED={metrics_enabled}",
            f"DEVICE_ID={device_id}",
            f"API_KEY={api_key}",
            f"NUM_WORKERS={num_workers}",
        ],
    )


def stop_inference_containers() -> None:
    inference_containers = find_running_inference_containers()
    interactive_mode = len(inference_containers) > 1
    terminate_running_containers(
        containers=inference_containers, interactive_mode=interactive_mode
    )


def check_inference_server_status():
    containers = find_running_inference_containers()
    if len(containers) > 0:
        for c in containers:
            container_name = c.attrs.get("Name", "")
            created = c.attrs.get("Created", "")
            exposed_port = list(c.attrs.get("Config").get("ExposedPorts", {}).keys())[0]
            status = c.attrs.get("State", {}).get("Status", "unknown")
            image = c.attrs.get("Image", "")
            container_status_message = """
Container Name: {container_name}
Created: {created}
Exposed Port: {exposed_port}
Status: {status}
Image: {image}
            """
            print(
                container_status_message.format(
                    container_name=container_name,
                    created=created,
                    exposed_port=exposed_port,
                    status=status,
                    image=image,
                )
            )
            return
    print("No inference server container running.")
