import subprocess
from typing import Optional, Union, Dict, List

import typer

import docker

from inference_cli.lib.utils import read_env_file

docker_client = docker.from_env()


def ask_user_to_kill_container(c):
    name = c.attrs.get("Name", "")
    env_vars = c.attrs.get("Config", {}).get("Env", {})
    port = 9001
    for var in env_vars:
        if var.startswith("PORT="):
            port = var.split("=")[1]
    should_delete = typer.confirm(
        f" An inference server is already running in container {name} on port {port}. Are you sure you want to delete it?"
    )
    return should_delete


def is_inference_server_container(container):
    image_tags = container.image.tags
    for t in image_tags:
        if t.startswith("roboflow/roboflow-inference-server"):
            return True
    return False


def handle_existing_containers(containers):
    has_existing_containers = False
    for c in containers:
        if is_inference_server_container(c):
            has_existing_containers = True
            if c.attrs.get("State", {}).get("Status", "").lower() == "running":
                should_kill = ask_user_to_kill_container(c)
                if should_kill:
                    c.kill()
                    has_existing_containers = False
    return has_existing_containers


def find_existing_containers():
    containers = []
    for c in docker_client.containers.list():
        if is_inference_server_container(c):
            if c.attrs.get("State", {}).get("Status", "").lower() == "running":
                containers.append(c)
    return containers


def get_image():
    try:
        subprocess.check_output("nvidia-smi")
        print("GPU detected. Using a GPU image.")
        return "roboflow/roboflow-inference-server-gpu:latest"
    except:
        print("No GPU detected. Using a CPU image.")
        return "roboflow/roboflow-inference-server-cpu:latest"


def start_inference_container(
    image: Optional[str] = None,
    port: int = 9001,
    labels: Optional[Union[Dict[str, str], List[str]]] = None,
    project: str = "roboflow-platform",
    metrics_enabled: bool = True,
    device_id: Optional[str] = None,
    num_workers: int = 1,
    api_key: Optional[str] = None,
    env_file_path: Optional[str] = None,
) -> None:
    containers = find_existing_containers()
    if len(containers) > 0:
        still_has_containers = handle_existing_containers(containers)
        if still_has_containers:
            print("Please kill the existing containers and try again.")
            return

    if image is None:
        image = get_image()

    device_requests = None
    privileged = False
    if "gpu" in image:
        privileged = True
        device_requests = (
            [docker.types.DeviceRequest(device_ids=["all"], capabilities=[["gpu"]])],
        )
    environment = prepare_container_environment(
        port=port,
        project=project,
        metrics_enabled=metrics_enabled,
        device_id=device_id,
        num_workers=num_workers,
        api_key=api_key,
        env_file_path=env_file_path,
    )
    print(f"Starting inference server container...")
    docker_client.containers.run(
        image=image,
        privileged=privileged,
        detach=True,
        labels=labels,
        ports={"9001": port},
        device_requests=device_requests,
        environment=environment,
    )


def prepare_container_environment(
    port: int,
    project: str,
    metrics_enabled: bool,
    device_id: Optional[str],
    num_workers: int,
    api_key: Optional[str],
    env_file_path: Optional[str],
) -> List[str]:
    environment = {}
    if env_file_path is not None:
        environment = read_env_file(path=env_file_path)
    environment["HOST"] = "0.0.0.0"
    environment["PORT"] = str(port)
    environment["PROJECT"] = project
    environment["METRICS_ENABLED"] = str(metrics_enabled)
    if device_id is not None:
        environment["DEVICE_ID"] = device_id
    if api_key is not None:
        environment["API_KEY"] = api_key
    environment["NUM_WORKERS"] = str(num_workers)
    return [f"{key}={value}" for key, value in environment.items()]


def check_inference_server_status():
    containers = find_existing_containers()
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


if __name__ == "__main__":
    start_inference_container("my_api_key")
