import time
from typing import Optional

from docker.models.containers import Container

import docker
from inference_cli.lib import container_adapter

docker_image = "us-central1-docker.pkg.dev/roboflow-proxy-425409/inference/tunnel"


def start_tunnel(api_key: str, inference_port: int = 9001) -> str:
    container = find_running_tunnel_container()
    if not container:
        container = start_tunnel_container(api_key, inference_port)

    return extract_tunnel_url(container)


def start_tunnel_container(api_key, inference_port: int = 9001) -> Container:
    container_adapter.pull_image(docker_image)

    # Any host value, it is mapped only inside the docker container
    upstream = "host.docker.internal"

    environment = {
        "TUNNEL_UPSTREAM": upstream,
        "TUNNEL_PORT": inference_port,
        "TUNNEL_API_KEY": api_key,
    }

    docker_client = docker.from_env()
    return docker_client.containers.run(
        docker_image,
        detach=True,
        environment=environment,
        extra_hosts={
            # Map the host gateway to a host inside the container to be able to access
            # inference from there. Later we can run both containers in the same
            # network.
            upstream: "host-gateway",
        },
    )


def find_running_tunnel_container() -> Optional[Container]:
    docker_client = docker.from_env()
    for container in docker_client.containers.list():
        if is_tunnel_container(container):
            if container.attrs.get("State", {}).get("Status", "").lower() == "running":
                return container


def is_tunnel_container(container: Container) -> bool:
    image_tags = container.image.tags
    return any("/inference/tunnel" in t for t in image_tags)


def extract_tunnel_url(container: Container) -> str:
    for n in range(5):
        tunnel_logs = container.logs(tail=10).decode()
        for logline in reversed(tunnel_logs.split("\n")):
            tunnel_url, *other_parts = logline.split(" is forwarding to ", 1)
            if other_parts:
                return tunnel_url
        if n == 0:
            print("Waiting for tunnel to start...")
        time.sleep(1)

    raise RuntimeError(f"Tunnel failed to start:\n{tunnel_logs}")


def stop_tunnel_container() -> None:
    tunnel_container = find_running_tunnel_container()
    if tunnel_container:
        container_adapter.terminate_running_containers(
            containers=[tunnel_container],
            interactive_mode=False,
        )
