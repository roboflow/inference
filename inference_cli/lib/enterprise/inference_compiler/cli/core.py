from enum import Enum
from typing import Annotated, Optional

import typer
from rich.console import Console

from inference_cli.lib.container_adapter import get_image, pull_image
from inference_cli.lib.env import ROBOFLOW_API_KEY

inference_compiler_app = typer.Typer(name="Inference compiler")


class CompilationMode(str, Enum):
    AUTO = "auto"
    CONTAINER = "container"
    PYTHON = "python"


@inference_compiler_app.callback()
def compiler_callback():
    pass


@inference_compiler_app.command(name="compile-model")
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
            help="Roboflow API key for your workspace. If not given - env variable `ROBOFLOW_API_KEY` will be used",
        ),
    ] = None,
    debug_mode: Annotated[
        bool,
        typer.Option(
            "--debug-mode/--no-debug-mode",
            help="Flag enabling errors stack traces to be displayed (helpful for debugging)",
        ),
    ] = False,
    trt_forward_compatible: Annotated[
        bool,
        typer.Option(
            "--trt-forward-compatible/--no-trt-forward-compatible",
            help="Flag to decide if forward-compatibility mode in TRT compilation should be enabled",
        ),
    ] = False,
    trt_same_cc_compatible: Annotated[
        bool,
        typer.Option(
            "--trt-same-cc-compatible/--no-trt-same-cc-compatible",
            help="Flag to decide if engine should be compiled to be compatible with devices sharing the same CUDA CC "
            "to the one running compilation procedure",
        ),
    ] = False,
    compilation_mode: Annotated[
        CompilationMode,
        typer.Option(
            "--compilation-mode",
            help="Selection of compilation mode - `container` runs the procedure inside `inference` server, "
            "`python` runs in-process. `auto` (default) inspect environment dependencies to verify if "
            "the procedure can be run in-process, if not - offloading to the server.",
        ),
    ] = CompilationMode.AUTO,
    image: Annotated[
        Optional[str],
        typer.Option(
            "--image",
            help="Point specific docker image you would like to run with command (useful for development of custom "
            "builds of inference server)",
        ),
    ] = None,
    use_local_images: Annotated[
        bool,
        typer.Option(
            "--use-local-images/--not-use-local-images",
            help="Flag to allow using local images (if set False image is always attempted to be pulled)",
        ),
    ] = False,
) -> None:
    console = Console()
    console.print(
        "You are running component licensed under Roboflow Enterprise License - please acknowledge the "
        "terms of use: https://github.com/roboflow/inference/blob/main/inference/enterprise/LICENSE.txt",
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
        console.print(
            "Inference compiler running in `auto` mode could not import `inference-models`, which is required "
            f"to compile package in process - offloading to container. Error: {error}",
        )
        return True
    try:
        from inference_models.runtime_introspection.core import x_ray_runtime_environment

        x_ray_result = x_ray_runtime_environment()

        assert x_ray_result.trt_version is not None, "TensorRT library not detected"
        assert x_ray_result.trt_python_package_available, "TensorRT Python package not detected"
    except Exception as error:
        console.print(
            "Inference compiler running in `auto` mode could not import `tensorrt`, which is required "
            f"to compile package in process - offloading to container. Error: {error}",
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
) -> None:
    import docker

    if image is None:
        image = get_image()
    if "-cpu" in image:
        raise ValueError(
            "Attempted to run compilation using `inference-server` CPU image, which does not support TRT compilation. "
            "This error may be result of pointing invalid docker image with `--image` parameter or image "
            "auto-selection choice, due to lack of GPU detected."
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
    console.print("Starting model compilation inside docker container")
    command = build_container_command(
        model_id=model_id,
        api_key=api_key,
        trt_forward_compatible=trt_forward_compatible,
        trt_same_cc_compatible=trt_same_cc_compatible,
    )
    docker_client = docker.from_env()
    container = docker_client.containers.run(
        image=image,
        command=command.split(" "),
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
        **docker_run_kwargs,
    )
    for line in container.logs(stream=True, follow=True):
        console.print(line.decode("utf-8"), end="")
    # Once logs end, the container has stopped. Fetch exit code:
    result = container.wait()
    exit_code = result["StatusCode"]
    console.print(f"\nTRT compilation container exited with code {exit_code}")


def build_container_command(
    model_id: str,
    api_key: Optional[str] = None,
    trt_forward_compatible: bool = False,
    trt_same_cc_compatible: bool = False,
) -> str:
    command = (
        f"inference enterprise inference-compile compile-model --model-id {model_id}"
    )
    if api_key:
        command += f" --api-key {api_key}"
    command += f" --trt-forward-compatible {stringify_boolean(trt_forward_compatible)}"
    command += f" --trt-same-cc {stringify_boolean(trt_same_cc_compatible)}"
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
    from inference_cli.lib.enterprise.inference_compiler.core import compiler

    compiler.compile_model(
        model_id=model_id,
        api_key=api_key,
        trt_forward_compatible=trt_forward_compatible,
        trt_same_cc_compatible=trt_same_cc_compatible,
        console=console,
    )
