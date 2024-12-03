import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, TextIO, Tuple

import cv2
from rich.progress import Progress, TaskID

from inference.core.cache import cache
from inference.core.env import API_KEY, MAX_ACTIVE_MODELS
from inference.core.exceptions import MissingApiKeyError
from inference.core.managers.active_learning import BackgroundTaskActiveLearningManager
from inference.core.managers.decorators.base import ModelManagerDecorator
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.core.roboflow_api import get_workflow_specification
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.profiling.core import (
    NullWorkflowsProfiler,
)
from inference.models.utils import ROBOFLOW_MODEL_TYPES
from inference_cli.lib.logger import CLI_LOGGER
from inference_cli.lib.workflows.common import (
    aggregate_batch_processing_results,
    denote_image_processed,
    dump_image_processing_results,
    get_all_images_in_directory,
    open_progress_log,
    report_failed_files,
)
from inference_cli.lib.workflows.entities import OutputFileType


def process_image_with_workflow_using_inference_package(
    image_path: str,
    output_directory: str,
    workflow_specification: Optional[dict] = None,
    workspace_name: Optional[str] = None,
    workflow_id: Optional[str] = None,
    workflow_parameters: Optional[Dict[str, Any]] = None,
    image_input_name: str = "image",
    api_key: Optional[str] = None,
    save_image_outputs: bool = True,
    force_reprocessing: bool = False,
) -> None:
    if api_key is None:
        api_key = API_KEY
    log_file, log_content = open_progress_log(output_directory=output_directory)
    try:
        image_name = os.path.basename(image_path)
        if image_name in log_content and not force_reprocessing:
            return None
        model_manager = _prepare_model_manager()
        workflow_specification = _get_workflow_specification(
            workflow_specification=workflow_specification,
            workspace_name=workspace_name,
            workflow_id=workflow_id,
            api_key=api_key,
        )
        result = _run_workflow_for_single_image_with_inference(
            model_manager=model_manager,
            image_path=image_path,
            workflow_specification=workflow_specification,
            workflow_id=workflow_id,
            image_input_name=image_input_name,
            workflow_parameters=workflow_parameters,
            api_key=api_key,
        )
        dump_image_processing_results(
            result=result,
            image_path=image_path,
            output_directory=output_directory,
            save_image_outputs=save_image_outputs,
        )
        denote_image_processed(log_file=log_file, image_path=image_path)
    except Exception as e:
        raise RuntimeError(f"Could not process image: {image_path}") from e
    finally:
        log_file.close()


def process_image_directory_with_workflow_using_inference_package(
    input_directory: str,
    output_directory: str,
    workflow_specification: Optional[dict] = None,
    workspace_name: Optional[str] = None,
    workflow_id: Optional[str] = None,
    workflow_parameters: Optional[Dict[str, Any]] = None,
    image_input_name: str = "image",
    api_key: Optional[str] = None,
    save_image_outputs: bool = True,
    force_reprocessing: bool = False,
    aggregate_structured_results: bool = True,
    aggregation_format: OutputFileType = OutputFileType.JSONL,
    debug_mode: bool = False,
) -> None:
    if api_key is None:
        api_key = API_KEY
    files_to_process = get_all_images_in_directory(input_directory=input_directory)
    log_file, log_content = open_progress_log(output_directory=output_directory)
    try:
        remaining_files = [
            f
            for f in files_to_process
            if os.path.basename(f) not in log_content or force_reprocessing
        ]
        print(f"Files to process: {len(remaining_files)}")
        failed_files = _process_images_within_directory(
            files_to_process=remaining_files,
            output_directory=output_directory,
            workflow_specification=workflow_specification,
            workspace_name=workspace_name,
            workflow_id=workflow_id,
            workflow_parameters=workflow_parameters,
            image_input_name=image_input_name,
            api_key=api_key,
            save_image_outputs=save_image_outputs,
            log_file=log_file,
            debug_mode=debug_mode,
        )
    finally:
        log_file.close()
    report_failed_files(failed_files=failed_files, output_directory=output_directory)
    if not aggregate_structured_results:
        return None
    aggregate_batch_processing_results(
        output_directory=output_directory,
        aggregation_format=aggregation_format,
    )


def _process_images_within_directory(
    files_to_process: List[str],
    output_directory: str,
    workflow_specification: Dict[str, Any],
    workspace_name: Optional[str],
    workflow_id: Optional[str],
    workflow_parameters: Optional[Dict[str, Any]],
    image_input_name: str,
    api_key: Optional[str],
    save_image_outputs: bool,
    log_file: TextIO,
    debug_mode: bool = False,
) -> List[Tuple[str, str]]:
    workflow_specification = _get_workflow_specification(
        workflow_specification=workflow_specification,
        workspace_name=workspace_name,
        workflow_id=workflow_id,
        api_key=api_key,
    )
    model_manager = _prepare_model_manager()
    progress_bar = Progress()
    processing_task = progress_bar.add_task(
        description="Processing images...",
        total=len(files_to_process),
    )
    failed_files = []
    on_success = partial(
        _on_success, progress_bar=progress_bar, task_id=processing_task
    )
    on_failure = partial(
        _on_failure,
        failed_files=failed_files,
        progress_bar=progress_bar,
        task_id=processing_task,
    )
    processing_fun = partial(
        _process_single_image_from_directory,
        model_manager=model_manager,
        workflow_specification=workflow_specification,
        workflow_id=workflow_id,
        image_input_name=image_input_name,
        workflow_parameters=workflow_parameters,
        api_key=api_key,
        output_directory=output_directory,
        save_image_outputs=save_image_outputs,
        log_file=log_file,
        on_success=on_success,
        on_failure=on_failure,
        debug_mode=debug_mode,
    )
    with progress_bar:
        for image_path in files_to_process:
            processing_fun(image_path)
    return failed_files


def _on_success(
    path: str,
    progress_bar: Progress,
    task_id: TaskID,
) -> None:
    progress_bar.update(task_id, advance=1)


def _on_failure(
    path: str,
    cause: str,
    failed_files: List[Tuple[str, str]],
    progress_bar: Progress,
    task_id: TaskID,
) -> None:
    failed_files.append((path, cause))
    progress_bar.update(task_id, advance=1)


def _process_single_image_from_directory(
    image_path: str,
    model_manager: ModelManagerDecorator,
    workflow_specification: Optional[Dict[str, Any]],
    workflow_id: Optional[str],
    image_input_name: str,
    workflow_parameters: Optional[Dict[str, Any]],
    api_key: Optional[str],
    output_directory: str,
    save_image_outputs: bool,
    log_file: TextIO,
    on_success: Callable[[str], None],
    on_failure: Callable[[str, str], None],
    log_file_lock: Optional[Lock] = None,
    debug_mode: bool = False,
) -> None:
    try:
        result = _run_workflow_for_single_image_with_inference(
            model_manager=model_manager,
            image_path=image_path,
            workflow_specification=workflow_specification,
            workflow_id=workflow_id,
            image_input_name=image_input_name,
            workflow_parameters=workflow_parameters,
            api_key=api_key,
        )
        dump_image_processing_results(
            result=result,
            image_path=image_path,
            output_directory=output_directory,
            save_image_outputs=save_image_outputs,
        )
        denote_image_processed(
            log_file=log_file, image_path=image_path, lock=log_file_lock
        )
        on_success(image_path)
    except Exception as error:
        error_summary = f"Error in processing {image_path}. Error type: {error.__class__.__name__} - {error}"
        if debug_mode:
            CLI_LOGGER.exception(error_summary)
        on_failure(image_path, error_summary)


def _get_workflow_specification(
    workflow_specification: Optional[dict] = None,
    workspace_name: Optional[str] = None,
    workflow_id: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    if workflow_specification is not None:
        return workflow_specification
    named_workflow_specified = (workspace_name is not None) and (
        workflow_id is not None
    )
    if not (named_workflow_specified != (workflow_specification is not None)):
        raise ValueError(
            "Parameters (`workspace_name`, `workflow_id`) can be used mutually exclusive with "
            "`workflow_specification`, but at least one must be set."
        )
    if api_key is None:
        raise MissingApiKeyError(
            "Roboflow API key needs to be provided either as parameter or via env variable "
            "ROBOFLOW_API_KEY. If you do not know how to get API key - visit "
            "https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key to learn how to "
            "retrieve one."
        )
    return get_workflow_specification(
        api_key=api_key,
        workspace_id=workspace_name,
        workflow_id=workflow_id,
        use_cache=False,
    )


def _prepare_model_manager() -> ModelManagerDecorator:
    model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
    model_manager = BackgroundTaskActiveLearningManager(
        model_registry=model_registry, cache=cache
    )
    return WithFixedSizeCache(
        model_manager,
        max_size=MAX_ACTIVE_MODELS,
    )


def _run_workflow_for_single_image_with_inference(
    model_manager: ModelManagerDecorator,
    image_path: str,
    workflow_specification: Dict[str, Any],
    workflow_id: Optional[str],
    image_input_name: str,
    workflow_parameters: Optional[Dict[str, Any]],
    api_key: Optional[str],
) -> Dict[str, Any]:
    profiler = NullWorkflowsProfiler.init()
    with ThreadPoolExecutor() as thread_pool_executor:
        workflow_init_parameters = {
            "workflows_core.model_manager": model_manager,
            "workflows_core.api_key": api_key,
            "workflows_core.thread_pool_executor": thread_pool_executor,
        }
        execution_engine = ExecutionEngine.init(
            workflow_definition=workflow_specification,
            init_parameters=workflow_init_parameters,
            workflow_id=workflow_id,
            profiler=profiler,
        )
        runtime_parameters = workflow_parameters or {}
        runtime_parameters[image_input_name] = cv2.imread(image_path)
        results = execution_engine.run(
            runtime_parameters=runtime_parameters,
            serialize_results=True,
        )
        return results[0]
