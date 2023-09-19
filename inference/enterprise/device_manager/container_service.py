import docker
import requests
from dataclasses import dataclass


@dataclass
class InferServerContainer:
    status: str
    id: str
    port: int
    host: str

    def __init__(self, docker_container, details):
        self.container = docker_container
        self.status = details.get("status")
        self.id = details.get("uuid")
        self.port = details.get("port")
        self.host = details.get("host")

    def restart(self):
        self.container.restart()

    def stop(self):
        if self.status == "running":
            self.container.stop()

    def ping(self):
        info = requests.get(f"http://{self.host}:{self.port}/info").json()
        return info


class ContainerService:
    """
    ContainerService is a wrapper around the Docker SDK Container API

    It provides a way to discover inference server containers running on the host
    and perform actions on them.
    """

    def __init__(self):
        self.client = docker.from_env()
        self.inference_containers = []

    def is_inference_server_container(self, container):
        """
        Checks if a container is an inference server container

        Args:
            container (any): A container object from the Docker SDK

        Returns:
            boolean: True if the container is an inference server container, False otherwise
        """
        image_tags = container.get("image", {}).get("tags", [])
        for t in image_tags:
            if t.startswith("roboflow/roboflow-inference-server"):
                return True
        return False

    def discover_containers(self):
        """
        Discovers inference server containers running on the host
        and parses their information into a list of InferServerContainer objects
        """
        containers = self.client.containers.list()
        self.inference_containers = []
        for c in containers:
            if self.is_inference_server_container(c):
                details = self.parse_container_info(c)
                info = requests.get(
                    f"http://{details['host']}:{details['port']}/info"
                ).json()
                details.update(info)
                print(f"Found inference container: {details}")
                infer_container = InferServerContainer(c, details)
                self.inference_containers.append(infer_container)

    def parse_container_info(self, c):
        """
        Parses the container information into a dictionary

        Args:
            c (any): Docker SDK Container object

        Returns:
            dict: A dictionary containing the container information
        """
        env = c.attrs.get("Config", {}).get("Env", {})
        info = {"container_id": c.id, "port": 9001, "host": "0.0.0.0"}
        for var in env:
            if var.startswith("PORT="):
                info["port"] = var.split("=")[1]
            elif var.startswith("HOST="):
                info["host"] = var.split("=")[1]
            elif var.startswith("METRICS_INTERVAL="):
                if int(var.split("=")[1]) < self.pingback_interval:
                    self.pingback_interval = int(var.split("=")[1])
        status = c.attrs.get("State", {}).get("Status")
        if status:
            info["status"] = status
        container_name = c.attrs.get("Name")
        if container_name:
            info["container_name_on_host"] = container_name
        return info

    def get_container_by_id(self, id):
        """
        Gets an inference server container by its id

        Args:
            id (string): The id of the container

        Returns:
            container: The container object if found, None otherwise
        """
        self.discover_containers()
        for c in self.inference_containers:
            if c.id == id:
                return c
        return None

    def get_container_ids(self):
        """
        Gets the ids of the inference server containers

        Returns:
            list: A list of container ids
        """
        self.discover_containers()
        return [c.id for c in self.inference_containers]
