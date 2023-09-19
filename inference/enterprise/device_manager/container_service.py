import docker
import requests
from dataclasses import dataclass

@dataclass
class InferServerContainer():
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
    def __init__(self):
        self.client = docker.from_env()
        self.inference_containers = []
        self.discover_containers()

    def is_inference_server_container(self, container):
        image_tags = container.image.tags
        for t in image_tags:
            if t.startswith("roboflow/roboflow-inference-server"):
                return True

    def discover_containers(self):
        containers = self.client.containers.list()
        self.inference_containers = []
        for c in containers:
            if self.is_inference_server_container(c):
                details = self.parse_container_info(c)
                info = requests.get(f"http://{details['host']}:{details['port']}/info").json()
                details.update(info)
                print(f"Found inference container: {details}")
                infer_container = InferServerContainer(c, details)
                self.inference_containers.append(infer_container)


    def parse_container_info(self, c):
        env = c.attrs.get("Config", {}).get("Env", {})
        info = { "container_id": c.id, "port": 9001, "host": "0.0.0.0" }
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
        if len(self.inference_containers) == 0:
            self.discover_containers()
        for c in self.inference_containers:
            if c.id == id:
                return c
        return None

    def get_container_ids(self):
        if len(self.inference_containers) == 0:
            self.discover_containers()
        return [c.id for c in self.inference_containers]
