"""Exercise a frozen executable with isolated caches and no build DLL search path."""

import base64
import io
import json
import os
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import click
from PIL import Image


def _environment(directory: Path, *, port: int) -> dict[str, str]:
    names = ("SYSTEMROOT", "WINDIR", "COMSPEC", "PROGRAMDATA", "SYSTEMDRIVE")
    env = {name: os.environ[name] for name in names if name in os.environ}
    system_root = Path(os.environ.get("SYSTEMROOT", "C:/Windows"))
    for name in ("TEMP", "TMP", "USERPROFILE", "APPDATA", "LOCALAPPDATA"):
        env[name] = str(directory)
    env.update(
        PATH=str(system_root / "System32") + os.pathsep + str(system_root),
        HOST="127.0.0.1",
        PORT=str(port),
        OFFLINE_MODE="True",
        API_LOGGING_ENABLED="False",
        DISABLE_VERSION_CHECK="True",
        TORCHVISION_WARN_WHEN_EXTENSION_LOADING_FAILS="1",
        ENABLE_TENSOR_DATA_REPRESENTATION="True",
        # Windows KnownFolder lookup ignores spoofed APPDATA/USERPROFILE.
        # Explicit library paths prevent dist/installed tests sharing real caches.
        WIN_PD_OVERRIDE_LOCAL_APPDATA=str(directory / "local"),
        WIN_PD_OVERRIDE_APPDATA=str(directory / "roaming"),
        TORCH_HOME=str(directory / "torch"),
        HF_HOME=str(directory / "huggingface"),
        HF_HUB_CACHE=str(directory / "huggingface" / "hub"),
        HF_MODULES_CACHE=str(directory / "huggingface" / "modules"),
        TRANSFORMERS_CACHE=str(directory / "transformers"),
        TLD_EXTRACT_CACHE=str(directory / "tldextract"),
        MPLCONFIGDIR=str(directory / "matplotlib"),
        MATPLOTLIBCONFIGDIR=str(directory / "matplotlib"),
        YOLO_CONFIG_DIR=str(directory / "ultralytics"),
        MODEL_CACHE_DIR=str(directory / "models"),
    )

    return env


def _workflow_request(base_url: str) -> None:
    image = Image.new("RGB", (3, 2), color=(13, 27, 41))
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")
    payload = {
        "specification": {
            "version": "1.0",
            "inputs": [{"type": "WorkflowImage", "name": "image"}],
            "steps": [
                {
                    "type": "roboflow_core/image_preprocessing@v1",
                    "name": "resize",
                    "image": "$inputs.image",
                    "task_type": "resize",
                    "width": 6,
                    "height": 4,
                }
            ],
            "outputs": [
                {
                    "type": "JsonField",
                    "name": "image",
                    "selector": "$steps.resize.image",
                }
            ],
        },
        "inputs": {
            "image": {
                "type": "base64",
                "value": base64.b64encode(encoded.getvalue()).decode(),
            }
        },
    }
    request = urllib.request.Request(
        base_url + "/workflows/run",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    # Never consult user/system HTTP proxies, even for loopback requests.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=60) as response:
        output = json.load(response)["outputs"][0]["image"]
    decoded = Image.open(io.BytesIO(base64.b64decode(output["value"])))
    if decoded.size != (6, 4):
        raise RuntimeError(f"Wrong workflow output dimensions: {decoded.size}")


@click.command()
@click.option(
    "--flavor",
    type=click.Choice(["cpu", "gpu"]),
    required=True,
)
@click.option(
    "--exe",
    type=click.Path(exists=True, path_type=Path, dir_okay=False),
    required=True,
)
def main(*, flavor: str, exe: Path) -> None:
    """Run native checks, cold server startup and a model-free image workflow.

    Args:
        flavor: Requested CPU or GPU installer flavor.
        exe: Frozen production executable to exercise.
    """
    exe = exe.resolve()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="inference-frozen-smoke-") as directory:
        root = Path(directory)
        env = _environment(root, port=port)
        subprocess.run(
            [str(exe), "--packaging-smoke-test", "--flavor", flavor],
            cwd=root,
            env=env,
            timeout=180,
            check=True,
        )
        for stage in ("cold", "warm"):
            log = root / f"startup-{stage}.log"
            with log.open("w") as output:
                started = time.monotonic()
                process = subprocess.Popen(
                    [str(exe)],
                    cwd=root,
                    env=env,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                )
                try:
                    deadline = time.monotonic() + 300
                    base_url = f"http://127.0.0.1:{port}"
                    opener = urllib.request.build_opener(
                        urllib.request.ProxyHandler({})
                    )
                    while time.monotonic() < deadline:
                        if process.poll() is not None:
                            raise RuntimeError(
                                f"Frozen server exited: {process.returncode}"
                            )

                        try:
                            with opener.open(
                                base_url + "/readiness", timeout=2
                            ) as response:
                                if response.status == 200:
                                    break
                        except (urllib.error.URLError, TimeoutError):
                            pass

                        time.sleep(0.5)
                    else:
                        raise RuntimeError(
                            "Frozen server did not become ready in 300 seconds"
                        )

                    ready_seconds = time.monotonic() - started
                    request_started = time.monotonic()
                    _workflow_request(base_url)
                    first_request_seconds = time.monotonic() - request_started
                    request_started = time.monotonic()
                    _workflow_request(base_url)
                    print(
                        json.dumps(
                            {
                                "stage": stage,
                                "readiness_seconds": ready_seconds,
                                "first_request_seconds": first_request_seconds,
                                "warm_request_seconds": time.monotonic()
                                - request_started,
                            },
                            sort_keys=True,
                        )
                    )
                except Exception:
                    output.flush()
                    print(log.read_text(errors="replace")[-20000:])
                    raise
                finally:
                    if process.poll() is None:
                        subprocess.run(
                            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                            check=False,
                            capture_output=True,
                            timeout=30,
                        )
                        process.wait(timeout=30)


if __name__ == "__main__":
    main()
