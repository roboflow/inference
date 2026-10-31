import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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

    def logs(self, tail: int = 10) -> bytes:
        result = subprocess.run(
            ["podman", "logs", "--tail", str(tail), self.id],
            check=True,
            capture_output=True,
        )
        return result.stdout + result.stderr


class _PodmanImage:
    def __init__(self, tags: List[str]) -> None:
        self.tags = tags


def podman_is_installed() -> bool:
    return shutil.which("podman") is not None


def find_running_podman_inference_containers() -> List[PodmanContainer]:
    """Return running podman containers whose image tag marks them as an
    inference server, matching the same rule the docker path applies
    (image tag prefix ``roboflow/roboflow-inference-server``)."""
    return find_running_podman_containers(
        predicate=lambda image: _is_inference_server_image(image)
    )


def find_running_podman_containers(
    predicate,
) -> List[PodmanContainer]:
    """Return running podman containers whose image satisfies ``predicate``."""
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
        if not predicate(image_name):
            continue
        containers.append(_as_container(row))
    return containers


def _is_inference_server_image(image_name: str) -> bool:
    # Depending on version and pull spelling, `podman ps` reports either the
    # short name or the fully qualified `docker.io/...` tag for Docker Hub
    # images (the docker SDK always reports the latter). Accept both spellings.
    for candidate in (image_name, f"docker.io/{image_name}"):
        bare = candidate.removeprefix("docker.io/")
        if bare.startswith("roboflow/roboflow-inference-server"):
            return True
    return False


def _inspect_env(container_id: str) -> List[str]:
    """Read the real ``Config.Env`` of a container via ``podman inspect``.
    ``podman ps --format {{json .}}`` has no Config key, so the docker-shaped
    attrs handed to the shared code paths would otherwise always report the
    default port in the "already running" prompt."""
    try:
        result = subprocess.run(
            ["podman", "inspect", "--format", "{{json .}}", container_id],
            check=True,
            capture_output=True,
            text=True,
        )
        config = json.loads(result.stdout).get("Config") or {}
    except (subprocess.CalledProcessError, OSError, json.JSONDecodeError):
        # Container vanished between ps and inspect, or inspect was refused.
        return []
    env = config.get("Env")
    if not isinstance(env, list):
        return []
    return [entry for entry in env if isinstance(entry, str)]


def _as_container(row: dict) -> PodmanContainer:
    names = row.get("Names") or [row.get("Name", "")]
    # `podman ps --format {{json .}}` emits "Ports": null for containers
    # without published ports (e.g. --network host).
    ports = row.get("Ports") or []
    exposed_ports = {
        "{}/{}".format(
            p.get("container_port") or p.get("containerPort"),
            p.get("protocol") or p.get("Type") or "tcp",
        )
        for p in ports
        if p.get("container_port") or p.get("containerPort")
    }
    env = _inspect_env(row.get("Id", ""))
    if not env:
        # Fallback (inspect failed / container gone): guess the served port
        # from the first published host port so the shared "already running"
        # prompt has something better than the default.
        for p in ports:
            host_port = p.get("host_port") or p.get("hostPort")
            if host_port:
                env = [f"PORT={host_port}"]
                break
    attrs = {
        "Name": names[0] if names else "",
        "Created": row.get("Created", ""),
        "Image": row.get("Image", ""),
        "State": {"Status": row.get("State", "unknown")},
        "Config": {
            "Env": env,
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
            # CDI device references are always vendor/class=device; a spec of
            # any other kind cannot serve NVIDIA GPUs.
            if kind != CDI_KIND_NVIDIA_GPU:
                continue
            return candidate, f"{CDI_KIND_NVIDIA_GPU}=all"
    return None


def _cdi_search_directories() -> List[Path]:
    # Podman's default cdi_spec_dirs (see containers.conf) — a per-user
    # directory is not scanned unless the user reconfigures podman.
    return [
        Path("/etc/cdi"),
        Path("/var/run/cdi"),
        Path("/run/cdi"),
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


def _resolve_pull_ref(image: str) -> str:
    """Fully-qualify a Docker Hub reference before pulling.

    Fedora ships podman with short-name resolution enforced: a name like
    ``roboflow/roboflow-inference-server-cpu`` without a registry prefix is
    ambiguous and `podman pull` refuses to resolve it without a TTY. The
    docker SDK resolves such names against Docker Hub implicitly, so match
    that behaviour here. Names that already carry a registry hint (a dot or
    a port in the first path segment, or the ``localhost`` host) pass through
    untouched.
    """
    first, _, _ = image.partition("/")
    if "/" in image and ("." in first or ":" in first or first == "localhost"):
        return image
    return f"docker.io/{image}"


def pull_image_with_podman(image: str, use_local_images: bool = False) -> None:
    image = _resolve_pull_ref(image)
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
    if volumes:
        # SELinux enforcing denies container writes to bind mounts unless the
        # mount is relabelled or labelling is turned off for the container;
        # docker-ce containers run unconfined, so the docker path never hits
        # this. Disabling labelling for the container matches that behaviour
        # without relabelling host paths.
        command += ["--security-opt", "label=disable"]
    cap_add = ["NET_BIND_SERVICE"]
    if device_requests:
        # Mirror the docker launch path, which adds SYS_ADMIN for GPU images.
        cap_add.append("SYS_ADMIN")
        # Device groups from the host must stay visible inside the container on
        # rootless podman, or CDI-provided device nodes can be inaccessible.
        command += ["--group-add", "keep-groups"]
    command += ["--cap-drop", "ALL"]
    for cap in cap_add:
        command += ["--cap-add", cap]
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
        _warn_if_selinux_devices_denied()
        CLI_LOGGER.info(f"Attaching GPUs via CDI spec {spec_path} ({device}).")
        command += ["--device", device]
        return command + [image], "cdi"
    return command + [image], "none"


def _warn_if_selinux_devices_denied() -> None:
    """SELinux enforcing with container_use_devices off can deny rootless
    CDI device injection at container creation. Warn instead of failing: the
    boolean is not the only path (typed policy may allow it), and podman
    surfaces a hard error at creation if the mount is truly denied."""
    getenforce = shutil.which("getenforce")
    if getenforce is None:
        return None
    try:
        enforcing = (
            subprocess.run(
                [getenforce], check=True, capture_output=True, text=True
            ).stdout.strip()
            == "Enforcing"
        )
        if not enforcing:
            return None
        getsebool = shutil.which("getsebool")
        if getsebool is None:
            return None
        bool_out = subprocess.run(
            [getsebool, "container_use_devices"],
            capture_output=True,
            text=True,
        ).stdout
    except (subprocess.CalledProcessError, OSError):
        return None
    if "off" in bool_out:
        CLI_LOGGER.warn(
            "SELinux is enforcing and the container_use_devices boolean is off. "
            "Rootless podman may refuse CDI GPU devices. If startup fails with a "
            "permission error, enable it (sudo setsebool -P container_use_devices 1) "
            "or see the podman NVIDIA GPU guide."
        )


class PodmanGPUNotConfiguredError(CLIError):
    def __init__(self, image: str) -> None:
        super().__init__(
            f"GPU container image ({image}) was requested with the podman "
            "runtime, but no NVIDIA CDI spec was found. Generate one and retry:\n"
            "  sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml\n"
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
    labels: Optional[Union[Dict[str, str], List[str]]] = None,
    require_gpu: bool = False,
    require_jetson: bool = False,
) -> None:
    if require_jetson:
        raise PodmanJetsonUnsupportedError()
    device_requests = ["nvidia.com/gpu=all"] if require_gpu else None
    if isinstance(labels, dict):
        labels = [f"{key}={value}" for key, value in labels.items()]
    command, _ = build_podman_launch_command(
        image=image,
        development=development,
        environment=environment,
        extra_environment=list(DEFAULT_CONTAINER_ENVIRONMENT),
        bind_address=bind_address,
        port=port,
        volumes=volumes,
        labels=labels,
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
