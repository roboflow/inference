import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from inference_cli.lib.exceptions import CLIError
from inference_cli.lib.logger import CLI_LOGGER

CDI_KIND_NVIDIA_GPU = "nvidia.com/gpu"


class PodmanContainer:
    """Drop-in stand-in for the docker SDK ``Container`` used by the shared
    terminate/status code paths: exposes ``id``, ``attrs`` (docker-inspect
    shaped), ``image.tags``, and ``kill()``."""

    def __init__(self, id: str, attrs: dict, image_tags: List[str]) -> None:
        self.id = id
        self.attrs = attrs
        self.image = _PodmanImage(image_tags)

    def kill(self) -> None:
        subprocess.run(
            ["podman", "kill", self.id],
            check=True,
            capture_output=True,
            text=True,
        )


class _PodmanImage:
    def __init__(self, tags: List[str]) -> None:
        self.tags = tags


def podman_is_installed() -> bool:
    return shutil.which("podman") is not None


def find_running_podman_inference_containers() -> List[PodmanContainer]:
    """Return running podman containers whose image tag marks them as an
    inference server, matching the same rule the docker path applies
    (image tag prefix ``roboflow/roboflow-inference-server``)."""
    output = subprocess.run(
        ["podman", "ps", "--all", "--format", "{{json .}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    containers = []
    for line in output.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("State", "")).lower() != "running":
            continue
        image_name = row.get("Image", "")
        if not _is_inference_server_image(image_name):
            continue
        containers.append(_as_container(row))
    return containers


def _is_inference_server_image(image_name: str) -> bool:
    # `podman ps` reports the short name (no registry prefix) for images pulled
    # from Docker Hub, while the docker SDK always reports the fully qualified
    # tag. Accept both spellings.
    for candidate in (image_name, f"docker.io/{image_name}"):
        bare = candidate.removeprefix("docker.io/")
        if bare.startswith("roboflow/roboflow-inference-server"):
            return True
    return False


def _as_container(row: dict) -> PodmanContainer:
    names = row.get("Names") or [row.get("Name", "")]
    config = row.get("Config") or {}
    exposed_ports = {
        "{}/{}".format(
            p.get("container_port") or p.get("containerPort"),
            p.get("protocol") or p.get("Type") or "tcp",
        )
        for p in row.get("Ports", [])
        if p.get("container_port") or p.get("containerPort")
    }
    attrs = {
        "Name": names[0] if names else "",
        "Created": row.get("Created", ""),
        "Image": row.get("Image", ""),
        "State": {"Status": row.get("State", "unknown")},
        "Config": {
            "Env": config.get("Env", []),
            "ExposedPorts": {port: {} for port in sorted(exposed_ports)},
        },
    }
    return PodmanContainer(
        id=row.get("Id", ""),
        attrs=attrs,
        image_tags=[row.get("Image", "")],
    )


def find_cdi_spec() -> Optional[Tuple[Path, str]]:
    """Locate an NVIDIA CDI spec in the directories podman scans, returning
    ``(path, --device value)``. Only ``nvidia*`` spec files qualify: any other
    vendor's spec in the same directory must not satisfy the GPU requirement."""
    for directory in _cdi_search_directories():
        if not directory.is_dir():
            continue
        for candidate in sorted(directory.iterdir()):
            if not candidate.is_file():
                continue
            if not candidate.name.startswith("nvidia"):
                continue
            if candidate.suffix not in (".yaml", ".yml", ".json"):
                continue
            kind = _cdi_kind(candidate)
            if kind is None:
                continue
            device = f"{kind}=all" if kind == CDI_KIND_NVIDIA_GPU else f"{kind}:all"
            return candidate, device
    return None


def _cdi_search_directories() -> List[Path]:
    data_home = os.getenv(
        "XDG_DATA_HOME", os.path.join(os.path.expanduser("~"), ".local", "share")
    )
    return [
        Path(data_home) / "containers" / "cdi",
        Path("/etc/containers/cdi"),
        Path("/run/cdi"),
        Path("/var/run/cdi"),
    ]


def _cdi_kind(path: Path) -> Optional[str]:
    """Read the CDI ``kind`` field without a YAML dependency: the key sits at
    the document root, and both JSON and YAML emit it on its own line."""
    try:
        with open(path, "r") as spec_file:
            content = spec_file.read()
    except OSError as error:
        CLI_LOGGER.warn(f"Could not read CDI spec {path}: {error}")
        return None
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            kind = parsed.get("kind")
            return kind if isinstance(kind, str) else None
    except json.JSONDecodeError:
        pass
    for line in content.splitlines():
        stripped = line.strip().lstrip("-").strip()
        if stripped.startswith('"kind"'):
            return stripped.split(":", 1)[1].strip().strip('",')
        if stripped.startswith("kind:"):
            return stripped.split(":", 1)[1].strip()
    return None


def pull_image_with_podman(image: str, use_local_images: bool = False) -> None:
    exists = subprocess.run(
        ["podman", "image", "exists", image],
        capture_output=True,
    )
    if exists.returncode == 0:
        if use_local_images:
            print(f"Using locally cached image: {image}")
            return None
        print(f"Image {image} is already present locally, pulling updates.")
    else:
        print(f"Pulling image: {image}")
    subprocess.run(["podman", "pull", image], check=True)
    print(f"Image {image} pulled.")


def build_podman_launch_command(
    image: str,
    development: bool,
    environment: List[str],
    extra_environment: List[str],
    bind_address: str,
    port: int,
    volumes: Dict[str, dict],
    labels: Optional[List[str]] = None,
    device_requests: Optional[List[str]] = None,
) -> Tuple[List[str], str]:
    """Translate the docker launch spec used by ``start_inference_container``
    into a ``podman run`` argv. Returns ``(argv, gpu_mode)`` where gpu_mode is
    ``"cdi"``, ``"none"``, or raises when a GPU image cannot be served."""
    command = ["podman", "run", "--detach"]
    command += ["--memory", "4g", "--memory-swap", "6g", "--cpu-shares", "1024"]
    command += ["--security-opt", "no-new-privileges"]
    command += ["--cap-drop", "ALL", "--cap-add", "NET_BIND_SERVICE"]
    command += ["--read-only", "--network", "bridge", "--ipc", "private"]
    for entry in environment + extra_environment:
        command += ["-e", entry]
    host_port_bindings = {port: port}
    if development:
        host_port_bindings[9002] = 9002
    for host_port, container_port in host_port_bindings.items():
        command += ["-p", f"{bind_address}:{host_port}:{container_port}"]
    for host_path, spec in volumes.items():
        command += ["-v", f"{host_path}:{spec['bind']}:{spec['mode']}"]
    for label in labels or []:
        command += ["--label", label]

    if device_requests:
        cdi_spec = find_cdi_spec()
        if cdi_spec is None:
            raise PodmanGPUNotConfiguredError(image=image)
        spec_path, device = cdi_spec
        CLI_LOGGER.info(f"Attaching GPUs via CDI spec {spec_path} ({device}).")
        command += ["--device", device]
        return command, "cdi"
    return command, "none"


class PodmanGPUNotConfiguredError(CLIError):
    def __init__(self, image: str) -> None:
        super().__init__(
            f"GPU container image ({image}) was requested with the podman "
            "runtime, but no NVIDIA CDI spec was found. Generate one and retry:\n"
            "  nvidia-ctk cdi generate "
            "--output=$XDG_DATA_HOME/containers/cdi/nvidia.yaml\n"
            "(docs: https://docs.podman.io/en/latest/markdown/podman.1.html"
            "#cdi-spec-dirs)"
        )


class PodmanJetsonUnsupportedError(CLIError):
    def __init__(self) -> None:
        super().__init__(
            "Jetson images require privileged mode and the nvidia container "
            "runtime, which inference-cli only drives through Docker. Run this "
            "image with Docker, or launch it manually with podman following "
            "the NVIDIA container runtime instructions."
        )


def launch_inference_container_with_podman(
    image: str,
    development: bool,
    environment: List[str],
    bind_address: str,
    port: int,
    volumes: Dict[str, dict],
    labels: Optional[Dict[str, str]] = None,
    require_gpu: bool = False,
    require_jetson: bool = False,
) -> None:
    if require_jetson:
        raise PodmanJetsonUnsupportedError()
    device_requests = ["nvidia.com/gpu=all"] if require_gpu else None
    command, _ = build_podman_launch_command(
        image=image,
        development=development,
        environment=environment,
        extra_environment=list(DEFAULT_CONTAINER_ENVIRONMENT),
        bind_address=bind_address,
        port=port,
        volumes=volumes,
        labels=[f"{k}={v}" for k, v in (labels or {}).items()],
        device_requests=device_requests,
    )
    subprocess.run(command, check=True)


DEFAULT_CONTAINER_ENVIRONMENT = [
    "MODEL_CACHE_DIR=/tmp/model-cache",
    "TRANSFORMERS_CACHE=/tmp/huggingface",
    "YOLO_CONFIG_DIR=/tmp/yolo",
    "MPLCONFIGDIR=/tmp/matplotlib",
    "HOME=/tmp/home",
]
