import subprocess

import docker
import typer

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
    api_key,
    image=None,
    port=9001,
    labels=None,
    project="roboflow-platform",
    metrics_enabled=True,
    device_id=None,
    num_workers=1,
):
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


if __name__ == "__main__":
    start_inference_container("my_api_key")
