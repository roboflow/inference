import os
import subprocess
from typing import Dict, List, Optional, Union

import typer
from docker.errors import ImageNotFound
from docker.models.containers import Container
from rich.progress import Progress, TaskID

import docker
from inference_cli.lib.exceptions import DockerConnectionErrorException
from inference_cli.lib.utils import read_env_file


def ensure_docker_is_running() -> None:
    try:
        _ = docker.from_env()
    except docker.errors.DockerException as e:
        raise DockerConnectionErrorException(
            "Error connecting to Docker daemon. Is docker installed and running? "
            "See https://www.docker.com/get-started/ for installation instructions."
        ) from e


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
    docker_client = docker.from_env()
    containers = []
    for c in docker_client.containers.list():
        if is_inference_server_container(c):
            if c.attrs.get("State", {}).get("Status", "").lower() == "running":
                containers.append(c)
    return containers


def get_image() -> str:
    jetpack_version = os.getenv("JETSON_JETPACK")
    if jetpack_version:
        return _get_jetpack_image(jetpack_version=jetpack_version)
    try:
        subprocess.check_output("nvidia-smi")
        print("GPU detected. Using a GPU image.")
        return "roboflow/roboflow-inference-server-gpu:latest"
    except:
        print("No GPU detected. Using a CPU image.")
        return "roboflow/roboflow-inference-server-cpu:latest"


def _get_jetpack_image(jetpack_version: str) -> str:
    if jetpack_version.startswith("4.5"):
        return "roboflow/roboflow-inference-server-jetson-4.5.0:latest"
    if jetpack_version.startswith("4.6"):
        return "roboflow/roboflow-inference-server-jetson-4.6.1:latest"
    if jetpack_version.startswith("5.1"):
        return "roboflow/roboflow-inference-server-jetson-5.1.1:latest"
    raise RuntimeError(f"Jetpack version: {jetpack_version} not supported")


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
    development: bool = False,
    use_local_images: bool = False,
) -> None:
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
    docker_run_kwargs = {}
    if "gpu" in image:
        privileged = True
        device_requests = [
            docker.types.DeviceRequest(device_ids=["all"], capabilities=[["gpu"]])
        ]
    if "jetson" in image:
        privileged = True
        docker_run_kwargs = {"runtime": "nvidia"}
    environment = prepare_container_environment(
        port=port,
        project=project,
        metrics_enabled=metrics_enabled,
        device_id=device_id,
        num_workers=num_workers,
        api_key=api_key,
        env_file_path=env_file_path,
        development=development,
    )
    pull_image(image, use_local_images=use_local_images)
    print(f"Starting inference server container...")
    ports = {"9001": port}
    if development:
        ports["9002"] = 9002
    docker_client = docker.from_env()
    docker_client.containers.run(
        image=image,
        privileged=privileged,
        detach=True,
        labels=labels,
        ports=ports,
        device_requests=device_requests,
        environment=environment,
        **docker_run_kwargs,
    )


def prepare_container_environment(
    port: int,
    project: str,
    metrics_enabled: bool,
    device_id: Optional[str],
    num_workers: int,
    api_key: Optional[str],
    env_file_path: Optional[str],
    development: bool = False,
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
        environment["ROBOFLOW_API_KEY"] = api_key
    environment["NUM_WORKERS"] = str(num_workers)
    if development:
        environment["NOTEBOOK_ENABLED"] = "True"
    return [f"{key}={value}" for key, value in environment.items()]


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


def pull_image(image: str, use_local_images: bool = False) -> None:
    docker_client = docker.from_env()
    progress_tasks = {}
    try:
        _ = docker_client.images.get(image)
        if use_local_images:
            print(f"Using locally cached image: {use_local_images}")
            return None
    except ImageNotFound:
        pass
    print(f"Pulling image: {image}")
    with Progress() as progress:
        logs_stream = docker_client.api.pull(image, stream=True, decode=True)
        for line in logs_stream:
            show_progress(
                log_line=line, progress=progress, progress_tasks=progress_tasks
            )
    print(f"Image {image} pulled.")


def show_progress(
    log_line: dict, progress: Progress, progress_tasks: Dict[str, TaskID]
) -> None:
    log_id, status = log_line.get("id"), log_line.get("status")
    if log_line["status"].lower() == "downloading":
        task_id = f"[red][Downloading {log_id}]"
    elif log_line["status"].lower() == "extracting":
        task_id = f"[green][Extracting {log_id}]"
    else:
        return None
    if task_id not in progress_tasks:
        progress_tasks[task_id] = progress.add_task(
            f"{task_id}", total=log_line.get("progressDetail", {}).get("total")
        )
    else:
        progress.update(
            progress_tasks[task_id],
            completed=log_line.get("progressDetail", {}).get("current"),
        )


if __name__ == "__main__":
    start_inference_container("my_api_key")
