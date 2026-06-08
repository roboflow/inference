import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, TextIO, Tuple, Type

import cv2
from rich.progress import Progress, TaskID

from inference.core.cache import cache
from inference.core.env import API_KEY, MAX_ACTIVE_MODELS
from inference.core.managers.active_learning import BackgroundTaskActiveLearningManager
from inference.core.managers.decorators.base import ModelManagerDecorator
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.core.roboflow_api import get_workflow_specification
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.models.utils import ROBOFLOW_MODEL_TYPES
from inference_cli.lib.logger import CLI_LOGGER
from inference_cli.lib.utils import get_all_images_in_directory, read_json
from inference_cli.lib.workflows.common import (
    WorkflowsImagesProcessingIndex,
    aggregate_batch_processing_results,
    denote_image_processed,
    dump_image_processing_results,
    open_progress_log,
    replace_file_extension,
    report_failed_files,
)
from inference_cli.lib.workflows.entities import (
    ImagePath,
    ImageResultsIndexEntry,
    ImagesDirectoryProcessingDetails,
    OutputFileType,
)


def process_image_with_workflow_using_inference_package(
    image_path: str,
    output_directory: str,
    workflow_specification: Optional[dict] = None,
    workspace_name: Optional[str] = None,
    workflow_id: Optional[str] = None,
    workflow_version_id: Optional[str] = None,
    workflow_parameters: Optional[Dict[str, Any]] = None,
    image_input_name: str = "image",
    api_key: Optional[str] = None,
    save_image_outputs: bool = True,
    force_reprocessing: bool = False,
    max_concurrent_workflows_steps: int = 4,
) -> None:
    if api_key is None:
        api_key = API_KEY
    log_file, log_content = open_progress_log(output_directory=output_directory)
    with ThreadPoolExecutor(
        max_workers=max_concurrent_workflows_steps
    ) as thread_pool_executor:
        try:
            image_name = os.path.basename(image_path)
            if image_name in log_content and not force_reprocessing:
                return None
            model_manager = _prepare_model_manager()
            workflow_specification = _get_workflow_specification(
                workflow_specification=workflow_specification,
                workspace_name=workspace_name,
                workflow_id=workflow_id,
                workflow_version_id=workflow_version_id,
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
                thread_pool_executor=thread_pool_executor,
                max_concurrent_workflows_steps=max_concurrent_workflows_steps,
            )
            _ = dump_image_processing_results(
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
    workflow_version_id: Optional[str] = None,
    workflow_parameters: Optional[Dict[str, Any]] = None,
    image_input_name: str = "image",
    api_key: Optional[str] = None,
    save_image_outputs: bool = True,
    force_reprocessing: bool = False,
    aggregate_structured_results: bool = True,
    aggregation_format: OutputFileType = OutputFileType.JSONL,
    debug_mode: bool = False,
    max_failures: Optional[int] = None,
    max_concurrent_workflows_steps: int = 4,
    images_metadata_input_mapping: Optional[Dict[str, str]] = None,
    workflows_execution_engine_init_params: Optional[Dict[str, Any]] = None,
) -> ImagesDirectoryProcessingDetails:
    if api_key is None:
        api_key = API_KEY
    processing_index = WorkflowsImagesProcessingIndex.init()
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
            processing_index=processing_index,
            workflow_specification=workflow_specification,
            workspace_name=workspace_name,
            workflow_id=workflow_id,
            workflow_version_id=workflow_version_id,
            workflow_parameters=workflow_parameters,
            image_input_name=image_input_name,
            api_key=api_key,
            save_image_outputs=save_image_outputs,
            log_file=log_file,
            max_concurrent_workflows_steps=max_concurrent_workflows_steps,
            debug_mode=debug_mode,
            max_failures=max_failures,
            images_metadata_input_mapping=images_metadata_input_mapping,
            workflows_execution_engine_init_params=workflows_execution_engine_init_params,
        )
    finally:
        log_file.close()
    failures_report_path = report_failed_files(
        failed_files=failed_files, output_directory=output_directory
    )
    aggregated_results_path = None
    if aggregate_structured_results:
        aggregated_results_path = aggregate_batch_processing_results(
            output_directory=output_directory,
            aggregation_format=aggregation_format,
        )
    result_metadata_paths = processing_index.export_metadata()
    result_images_paths = processing_index.export_images()
    return ImagesDirectoryProcessingDetails(
        output_directory=output_directory,
        processed_images=len(remaining_files),
        failures=len(failed_files),
        result_metadata_paths=result_metadata_paths,
        result_images_paths=result_images_paths,
        aggregated_results_path=aggregated_results_path,
        failures_report_path=failures_report_path,
    )


def _process_images_within_directory(
    files_to_process: List[str],
    output_directory: str,
    processing_index: WorkflowsImagesProcessingIndex,
    workflow_specification: Dict[str, Any],
    workspace_name: Optional[str],
    workflow_id: Optional[str],
    workflow_version_id: Optional[str],
    workflow_parameters: Optional[Dict[str, Any]],
    image_input_name: str,
    api_key: Optional[str],
    save_image_outputs: bool,
    log_file: TextIO,
    max_concurrent_workflows_steps: int,
    debug_mode: bool = False,
    max_failures: Optional[int] = None,
    images_metadata_input_mapping: Optional[Dict[str, str]] = None,
    workflows_execution_engine_init_params: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, str]]:
    workflow_specification = _get_workflow_specification(
        workflow_specification=workflow_specification,
        workspace_name=workspace_name,
        workflow_id=workflow_id,
        workflow_version_id=workflow_version_id,
        api_key=api_key,
    )
    model_manager = _prepare_model_manager()
    progress_bar = Progress()
    processing_task = progress_bar.add_task(
        description="Processing images...",
        total=len(files_to_process),
    )
    if max_failures is None:
        max_failures = len(files_to_process) + 1
    failed_files = []
    on_success = partial(
        _on_success,
        progress_bar=progress_bar,
        task_id=processing_task,
        processing_index=processing_index,
    )
    on_failure = partial(
        _on_failure,
        failed_files=failed_files,
        progress_bar=progress_bar,
        task_id=processing_task,
    )
    with ThreadPoolExecutor(
        max_workers=max_concurrent_workflows_steps
    ) as thread_pool_executor:
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
            thread_pool_executor=thread_pool_executor,
            max_concurrent_workflows_steps=max_concurrent_workflows_steps,
            debug_mode=debug_mode,
            images_metadata_input_mapping=images_metadata_input_mapping,
            workflows_execution_engine_init_params=workflows_execution_engine_init_params,
        )
        failures = 0
        succeeded_files = set()
        with progress_bar:
            for image_path in files_to_process:
                success = processing_fun(image_path)
                if not success:
                    failures += 1
                else:
                    succeeded_files.add(image_path)
                if failures >= max_failures:
                    break
        failed_files_lookup = {f[0] for f in failed_files}
        aborted_files = [
            f
            for f in files_to_process
            if f not in succeeded_files and f not in failed_files_lookup
        ]
        for file in aborted_files:
            on_failure(
                file,
                "Aborted processing due to exceeding max failures of Workflows executions.",
            )
        return failed_files


def _on_success(
    path: ImagePath,
    index_entry: ImageResultsIndexEntry,
    progress_bar: Progress,
    task_id: TaskID,
    processing_index: WorkflowsImagesProcessingIndex,
) -> None:
    progress_bar.update(task_id, advance=1)
    processing_index.collect_entry(image_path=path, entry=index_entry)


def _on_failure(
    path: str,
    cause: str,
    failed_files: List[Tuple[str, str]],
    progress_bar: Progress,
    task_id: TaskID,
) -> None:
    failed_files.append((path, cause))
    progress_bar.update(task_id, advance=1)


def _resolve_metadata_driven_workflow_parameters(
    image_path: str,
    images_metadata_input_mapping: Optional[Dict[str, str]],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Resolve per-image metadata for parameter injection.

    When `images_metadata_input_mapping` is non-empty, the function looks
    up the sibling `.json` metadata file for `image_path`, validates that
    every value in the mapping is a key in the metadata, and projects the
    metadata into a `{workflow_input: value}` dict via the mapping.

    Returns (params, error_summary). On the success path the params dict
    is the projection (possibly empty when mapping is None/empty) and
    error_summary is None. On any failure (missing file, missing required
    keys) the params is None and error_summary explains why — the caller
    forwards it verbatim to `on_failure`.
    """
    if not images_metadata_input_mapping:
        return {}, None
    assumed_metadata_file_path = replace_file_extension(image_path, extension=".json")
    if not os.path.isfile(assumed_metadata_file_path):
        return None, (
            f"Cold not find required metadata file for image: {image_path}. "
            f"Since `images_metadata_input_mapping` was specified, presence of metadata file is enforced."
        )
    metadata = read_json(path=assumed_metadata_file_path)
    missing_keys = set(images_metadata_input_mapping.values()).difference(
        metadata.keys()
    )
    if len(missing_keys) > 0:
        return None, (
            f"Cold not find required metadata keys specified in `images_metadata_input_mapping` for image: "
            f"{image_path}. Missing keys: {missing_keys}."
        )
    return {
        key: metadata[value] for key, value in images_metadata_input_mapping.items()
    }, None


def _process_single_image_from_directory(
    image_path: ImagePath,
    model_manager: ModelManagerDecorator,
    workflow_specification: Dict[str, Any],
    workflow_id: Optional[str],
    image_input_name: str,
    workflow_parameters: Optional[Dict[str, Any]],
    api_key: Optional[str],
    output_directory: str,
    save_image_outputs: bool,
    log_file: TextIO,
    on_success: Callable[[ImagePath, ImageResultsIndexEntry], None],
    on_failure: Callable[[ImagePath, str], None],
    thread_pool_executor: ThreadPoolExecutor,
    max_concurrent_workflows_steps: int,
    log_file_lock: Optional[Lock] = None,
    debug_mode: bool = False,
    images_metadata_input_mapping: Optional[Dict[str, str]] = None,
    workflows_execution_engine_init_params: Optional[Dict[str, Any]] = None,
) -> bool:
    metadata_driven_workflow_parameters, metadata_error = (
        _resolve_metadata_driven_workflow_parameters(
            image_path=image_path,
            images_metadata_input_mapping=images_metadata_input_mapping,
        )
    )
    if metadata_error is not None:
        if debug_mode:
            CLI_LOGGER.exception(metadata_error)
        on_failure(image_path, metadata_error)
        return False
    try:
        workflow_parameters = workflow_parameters or {}
        metadata_driven_workflow_parameters.update(workflow_parameters)
        workflow_parameters = metadata_driven_workflow_parameters
        result = _run_workflow_for_single_image_with_inference(
            model_manager=model_manager,
            image_path=image_path,
            workflow_specification=workflow_specification,
            workflow_id=workflow_id,
            image_input_name=image_input_name,
            workflow_parameters=workflow_parameters,
            api_key=api_key,
            thread_pool_executor=thread_pool_executor,
            max_concurrent_workflows_steps=max_concurrent_workflows_steps,
            workflows_execution_engine_init_params=workflows_execution_engine_init_params,
        )
        index_entry = dump_image_processing_results(
            result=result,
            image_path=image_path,
            output_directory=output_directory,
            save_image_outputs=save_image_outputs,
        )
        denote_image_processed(
            log_file=log_file, image_path=image_path, lock=log_file_lock
        )
        on_success(image_path, index_entry)
        return True
    except Exception as error:
        error_summary = f"Error in processing {image_path}. Error type: {error.__class__.__name__} - {error}"
        if debug_mode:
            CLI_LOGGER.exception(error_summary)
        on_failure(image_path, error_summary)
        return False


def _get_workflow_specification(
    workflow_specification: Optional[dict] = None,
    workspace_name: Optional[str] = None,
    workflow_id: Optional[str] = None,
    workflow_version_id: Optional[str] = None,
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
    return get_workflow_specification(
        api_key=api_key,
        workspace_id=workspace_name,
        workflow_id=workflow_id,
        workflow_version_id=workflow_version_id,
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
    thread_pool_executor: ThreadPoolExecutor,
    max_concurrent_workflows_steps: int,
    workflows_execution_engine_init_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": api_key,
        "workflows_core.thread_pool_executor": thread_pool_executor,
    }
    if workflows_execution_engine_init_params:
        workflow_init_parameters.update(workflows_execution_engine_init_params)
    execution_engine = ExecutionEngine.init(
        workflow_definition=workflow_specification,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=max_concurrent_workflows_steps,
        workflow_id=workflow_id,
        executor=thread_pool_executor,
    )
    runtime_parameters = workflow_parameters or {}
    runtime_parameters[image_input_name] = cv2.imread(image_path)
    results = execution_engine.run(
        runtime_parameters=runtime_parameters,
        serialize_results=True,
    )
    return results[0]
