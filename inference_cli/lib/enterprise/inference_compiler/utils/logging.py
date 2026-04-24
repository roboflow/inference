import logging
from typing import Optional, Union

from rich.console import Console, JustifyMethod
from rich.style import Style

from inference_models.models.auto_loaders.presentation_utils import (
    render_table_with_model_overview,
)
from inference_models.weights_providers.entities import ModelMetadata

logger = logging.getLogger("inference_cli.inference_compiler")


def print_to_console(
    message: str,
    console: Optional[Console] = None,
    style: Optional[Union[str, Style]] = None,
    justify: Optional[JustifyMethod] = None,
) -> None:
    if console is None:
        return None
    logger.info(message)
    console.print(message, style=style, justify=justify)


def print_model_info(
    model_id: str,
    model_metadata: ModelMetadata,
    console: Optional[Console] = None,
) -> None:
    logger.info(
        "Model info: id=%s, architecture=%s, variant=%s, task=%s, packages=%d",
        model_metadata.model_id,
        model_metadata.model_architecture,
        model_metadata.model_variant,
        model_metadata.task_type,
        len(model_metadata.model_packages),
    )
    if console is None:
        return None
    table = render_table_with_model_overview(
        model_id=model_metadata.model_id,
        requested_model_id=model_id,
        model_architecture=model_metadata.model_architecture,
        model_variant=model_metadata.model_variant,
        task_type=model_metadata.task_type,
        weights_provider="roboflow",
        registered_packages=len(model_metadata.model_packages),
        model_dependencies=(
            model_metadata.model_dependencies
            if hasattr(model_metadata, "model_dependencies")
            else None
        ),
    )
    console.print(table)
