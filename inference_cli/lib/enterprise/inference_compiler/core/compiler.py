import logging
import tempfile
from functools import partial
from typing import Optional

from rich.console import Console

from inference_cli.lib.enterprise.inference_compiler.adapters.models_service import (
    ModelsServiceClient,
)
from inference_cli.lib.enterprise.inference_compiler.core.compilation_handlers.default import (
    compile_and_register_default_model,
)
from inference_cli.lib.enterprise.inference_compiler.core.entities import (
    CompilationConfig,
)
from inference_cli.lib.enterprise.inference_compiler.core.model_checks.default import (
    verify_auto_model,
)
from inference_cli.lib.enterprise.inference_compiler.errors import (
    ModelArchitectureNotSupportedError,
)
from inference_cli.lib.enterprise.inference_compiler.utils.logging import (
    print_model_info,
    print_to_console,
)
from inference_models.weights_providers.core import get_model_from_provider
from inference_models.weights_providers.entities import ModelMetadata

logger = logging.getLogger("inference_cli.inference_compiler")

REGISTERED_COMPILATION_HANDLERS = {
    "yolov8": partial(
        compile_and_register_default_model,
        compilation_config=CompilationConfig.for_yolo_models(
            verify_model=verify_auto_model,
        ),
    ),
    "yolov9": partial(
        compile_and_register_default_model,
        compilation_config=CompilationConfig.for_yolo_models(
            verify_model=verify_auto_model,
        ),
    ),
    "yolov5": partial(
        compile_and_register_default_model,
        compilation_config=CompilationConfig.for_yolo_models(
            verify_model=verify_auto_model,
        ),
    ),
    "yolov10": partial(
        compile_and_register_default_model,
        compilation_config=CompilationConfig.for_yolo_models(
            verify_model=verify_auto_model,
        ),
    ),
    "yolov11": partial(
        compile_and_register_default_model,
        compilation_config=CompilationConfig.for_yolo_models(
            verify_model=verify_auto_model,
        ),
    ),
    "yolov12": partial(
        compile_and_register_default_model,
        compilation_config=CompilationConfig.for_yolo_models(
            verify_model=verify_auto_model,
        ),
    ),
    "yolo26": partial(
        compile_and_register_default_model,
        compilation_config=CompilationConfig.for_yolo_models(
            verify_model=verify_auto_model,
        ),
    ),
    "rfdetr": partial(
        compile_and_register_default_model,
        compilation_config=CompilationConfig.for_rfdetr_models(
            verify_model=verify_auto_model,
        ),
    ),
    "yolact": partial(
        compile_and_register_default_model,
        compilation_config=CompilationConfig.for_yolo_models(
            verify_model=verify_auto_model,
        ),
    ),
    "resnet": partial(
        compile_and_register_default_model,
        compilation_config=CompilationConfig.for_resnet_models(
            verify_model=verify_auto_model,
        ),
    ),
    "deep-lab-v3-plus": partial(
        compile_and_register_default_model,
        compilation_config=CompilationConfig.for_deep_lab_models(
            verify_model=verify_auto_model,
        ),
    ),
    "vit": partial(
        compile_and_register_default_model,
        compilation_config=CompilationConfig.for_vit_models(
            verify_model=verify_auto_model,
        ),
    ),
    "yolonas": partial(
        compile_and_register_default_model,
        compilation_config=CompilationConfig.for_yolo_models(
            verify_model=verify_auto_model,
        ),
    ),
}


def compile_model(
    model_id: str,
    api_key: Optional[str] = None,
    trt_forward_compatible: bool = False,
    trt_same_cc_compatible: bool = False,
    console: Optional[Console] = None,
) -> None:
    print_to_console(
        message="Inference Compiler",
        justify="center",
        style="bold green4",
        console=console,
    )
    logger.info(
        "Starting compilation: model_id=%s, trt_forward_compatible=%s, trt_same_cc_compatible=%s",
        model_id,
        trt_forward_compatible,
        trt_same_cc_compatible,
    )
    models_service_client = ModelsServiceClient.init(api_key=api_key)
    print_to_console(message="Retrieving model metadata...", console=console)
    model_metadata = get_model_from_provider(
        model_id=model_id,
        provider="roboflow",
        api_key=api_key,
    )
    print_model_info(
        model_id=model_id,
        model_metadata=model_metadata,
        console=console,
    )
    compile_and_register(
        model_metadata=model_metadata,
        models_service_client=models_service_client,
        trt_forward_compatible=trt_forward_compatible,
        trt_same_cc_compatible=trt_same_cc_compatible,
        console=console,
    )


def compile_and_register(
    model_metadata: ModelMetadata,
    models_service_client: ModelsServiceClient,
    trt_forward_compatible: bool,
    trt_same_cc_compatible: bool,
    console: Optional[Console] = None,
) -> None:
    print_to_console(message="Model compilation in progress...", console=console)
    if model_metadata.model_architecture not in REGISTERED_COMPILATION_HANDLERS:
        raise ModelArchitectureNotSupportedError(
            f"Model architecture {model_metadata.model_architecture} not supported for compilation."
        )
    with tempfile.TemporaryDirectory() as compilation_directory:
        REGISTERED_COMPILATION_HANDLERS[model_metadata.model_architecture](
            model_metadata,
            models_service_client,
            compilation_directory,
            trt_forward_compatible,
            trt_same_cc_compatible,
            console,
        )
