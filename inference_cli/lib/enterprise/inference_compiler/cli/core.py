import logging
from enum import Enum
from typing import Annotated, Optional

import typer
from rich.console import Console

from inference_cli.lib.container_adapter import get_image, pull_image
from inference_cli.lib.env import ROBOFLOW_API_KEY
from inference_cli.lib.utils import read_env_file

logger = logging.getLogger("inference_cli.inference_compiler")

inference_compiler_app = typer.Typer(
    name="Inference compiler",
    help="Compile Roboflow models into optimized TensorRT engines for GPU-accelerated inference on NVIDIA GPUs and Jetson devices.",
)


class CompilationMode(str, Enum):
    AUTO = "auto"
    CONTAINER = "container"
    PYTHON = "python"


@inference_compiler_app.callback()
def compiler_callback():
    pass


@inference_compiler_app.command(
    name="compile-model",
    help="Compile an ONNX model from Roboflow into a TensorRT engine optimized for your GPU. "
    "The compiled engine is registered back to the Roboflow platform so it can be served to "
    "matching devices automatically. Compilation can run in-process (requires TensorRT and "
    "inference-models) or inside a Docker container.",
)
def compile_model(
    model_id: Annotated[
        str,
        typer.Option(
            "--model-id",
            "-m",
            help="Model ID in format project/version.",
        ),
    ],
    api_key: Annotated[
        Optional[str],
        typer.Option(
            "--api-key",
            "-a",
            help="Roboflow API key for your workspace. If not provided, the `ROBOFLOW_API_KEY` environment variable is used.",
        ),
    ] = None,
    debug_mode: Annotated[
        bool,
        typer.Option(
            "--debug-mode/--no-debug-mode",
            help="Display full stack traces on errors.",
        ),
    ] = False,
    trt_forward_compatible: Annotated[
        bool,
        typer.Option(
            "--trt-forward-compatible/--no-trt-forward-compatible",
            help="Enable TensorRT forward-compatibility mode, allowing engines to run on newer TRT versions.",
        ),
    ] = False,
    trt_same_cc_compatible: Annotated[
        bool,
        typer.Option(
            "--trt-same-cc-compatible/--no-trt-same-cc-compatible",
            help="Compile the engine to be portable across GPUs with the same CUDA compute capability.",
        ),
    ] = False,
    compilation_mode: Annotated[
        CompilationMode,
        typer.Option(
            "--compilation-mode",
            help="Selection of compilation mode. `container` runs the procedure inside an Inference server container, "
            "`python` runs in-process. `auto` (default) inspects environment dependencies to verify if "
            "the procedure can run in-process; if not, it offloads to the container.",
        ),
    ] = CompilationMode.AUTO,
    image: Annotated[
        Optional[str],
        typer.Option(
            "--image",
            help="Specify a Docker image to use for compilation (useful for custom builds of the Inference server).",
        ),
    ] = None,
    use_local_images: Annotated[
        bool,
        typer.Option(
            "--use-local-images/--not-use-local-images",
            help="Allow using local Docker images. If false, the image is always pulled from the registry.",
        ),
    ] = False,
    env_file_path: Annotated[
        Optional[str],
        typer.Option(
            "--env-file-path",
            help="Path to a key-value .env file to inject into the compilation container. "
            "For Python mode, export the variables to your environment instead.",
        ),
    ] = None,
) -> None:
    console = Console()
    console.print(
        "Inference Compiler is licensed under the Roboflow Enterprise License. "
        "By continuing, you acknowledge the terms of use: "
        "https://github.com/roboflow/inference/blob/main/inference/enterprise/LICENSE.txt",
    )
    if api_key is None:
        api_key = ROBOFLOW_API_KEY
    try:
        if len(model_id.split(" ")) > 1:
            raise ValueError(
                "Format of model_id is incorrect - expected string without whitespaces"
            )
        if compilation_to_run_in_container(
            compilation_mode=compilation_mode, console=console
        ):
            run_compilation_in_container(
                model_id=model_id,
                api_key=api_key,
                trt_forward_compatible=trt_forward_compatible,
                trt_same_cc_compatible=trt_same_cc_compatible,
                console=console,
                image=image,
                use_local_images=use_local_images,
                env_file_path=env_file_path,
            )
        else:
            run_compilation_in_python(
                model_id=model_id,
                api_key=api_key,
                trt_forward_compatible=trt_forward_compatible,
                trt_same_cc_compatible=trt_same_cc_compatible,
                console=console,
            )

    except KeyboardInterrupt:
        print("Command interrupted.")
        return
    except Exception as error:
        if debug_mode:
            raise error
        typer.echo(f"Command failed. Cause: {error}")
        raise typer.Exit(code=1)


def compilation_to_run_in_container(
    compilation_mode: CompilationMode,
    console: Console,
) -> bool:
    if compilation_mode == CompilationMode.CONTAINER:
        return True
    if compilation_mode == CompilationMode.PYTHON:
        return False
    try:
        import inference_models

    except Exception as error:
        logger.info(
            "Could not import inference-models, offloading to container: %s", error
        )
        console.print(
            "Compiler running in `auto` mode could not import `inference-models`, which is required "
            f"to compile in-process. Offloading to container. Error: {error}",
        )
        return True
    try:
        from inference_models.runtime_introspection.core import (
            x_ray_runtime_environment,
        )

        x_ray_result = x_ray_runtime_environment()

        assert x_ray_result.trt_version is not None, "TensorRT library not detected"
        assert (
            x_ray_result.trt_python_package_available
        ), "TensorRT Python package not detected"
    except Exception as error:
        logger.info("TensorRT not available, offloading to container: %s", error)
        console.print(
            "Compiler running in `auto` mode could not import `tensorrt`, which is required "
            f"to compile in-process. Offloading to container. Error: {error}",
        )
        return True
    return False


def run_compilation_in_container(
    model_id: str,
    api_key: Optional[str] = None,
    trt_forward_compatible: bool = False,
    trt_same_cc_compatible: bool = False,
    console: Optional[Console] = None,
    image: Optional[str] = None,
    use_local_images: bool = False,
    env_file_path: Optional[str] = None,
) -> None:
    import docker

    if image is None:
        image = get_image()
    if "-cpu" in image:
        raise ValueError(
            "Attempted to run compilation using an Inference server CPU image, which does not support TRT compilation. "
            "This may be caused by specifying an invalid Docker image with the `--image` parameter, or by "
            "automatic image selection when no GPU is detected."
        )
    is_gpu = "gpu" in image and "jetson" not in image
    is_jetson = "jetson" in image
    device_requests = None
    privileged = False
    docker_run_kwargs = {}
    if is_gpu:
        device_requests = [
            docker.types.DeviceRequest(device_ids=["all"], capabilities=[["gpu"]])
        ]
    if is_jetson:
        privileged = True
        docker_run_kwargs = {"runtime": "nvidia"}
    pull_image(image, use_local_images=use_local_images)
    logger.info(
        "Starting container compilation: image=%s, is_gpu=%s, is_jetson=%s",
        image,
        is_gpu,
        is_jetson,
    )
    console.print("Starting model compilation inside Docker container")
    command = build_container_command(
        model_id=model_id,
        api_key=api_key,
        trt_forward_compatible=trt_forward_compatible,
        trt_same_cc_compatible=trt_same_cc_compatible,
    )
    environment = []
    if env_file_path is not None:
        environment_values = read_env_file(path=env_file_path)
        environment = [f"{key}={value}" for key, value in environment_values.items()]
    docker_client = docker.from_env()
    container = docker_client.containers.run(
        image=image,
        command=["-c", command],
        entrypoint="bash",
        privileged=privileged,
        detach=True,
        device_requests=device_requests,
        security_opt=["no-new-privileges"] if not is_jetson else None,
        cap_drop=["ALL"] if not is_jetson else None,
        cap_add=(
            (["NET_BIND_SERVICE"] + (["SYS_ADMIN"] if is_gpu else []))
            if not is_jetson
            else None
        ),
        read_only=not is_jetson,
        volumes={"/tmp": {"bind": "/tmp", "mode": "rw"}},
        network_mode="bridge",
        ipc_mode="private" if not is_jetson else None,
        environment=environment,
        **docker_run_kwargs,
    )
    for line in container.logs(stream=True, follow=True):
        console.print(line.decode("utf-8"), end="")
    result = container.wait()
    exit_code = result["StatusCode"]
    console.print(f"\nTRT compilation container exited with code {exit_code}")
    if exit_code != 0:
        typer.Exit(code=result["StatusCode"])


def build_container_command(
    model_id: str,
    api_key: Optional[str] = None,
    trt_forward_compatible: bool = False,
    trt_same_cc_compatible: bool = False,
) -> str:
    command = (
        f'inference enterprise inference-compiler compile-model --model-id "{model_id}"'
    )
    if api_key:
        command += f" --api-key {api_key}"
    if trt_forward_compatible:
        command += (
            f' --trt-forward-compatible "{stringify_boolean(trt_forward_compatible)}"'
        )
    if trt_same_cc_compatible:
        command += (
            f' --trt-same-cc-compatible "{stringify_boolean(trt_same_cc_compatible)}"'
        )
    return command


def stringify_boolean(value: bool) -> str:
    if value:
        return "true"
    return "false"


def run_compilation_in_python(
    model_id: str,
    api_key: Optional[str] = None,
    trt_forward_compatible: bool = False,
    trt_same_cc_compatible: bool = False,
    console: Optional[Console] = None,
) -> None:
    logger.info("Running compilation in-process (Python mode)")
    from inference_cli.lib.enterprise.inference_compiler.core import compiler

    compiler.compile_model(
        model_id=model_id,
        api_key=api_key,
        trt_forward_compatible=trt_forward_compatible,
        trt_same_cc_compatible=trt_same_cc_compatible,
        console=console,
    )
