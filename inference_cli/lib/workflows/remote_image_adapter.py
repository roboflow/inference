import logging
import os
from functools import partial
from multiprocessing.pool import ThreadPool
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, TextIO, Tuple

import backoff
from rich.progress import Progress, TaskID

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
from inference_sdk import (
    InferenceConfiguration,
    InferenceHTTPClient,
    VisualisationResponseFormat,
)
from inference_sdk.http.errors import HTTPCallErrorError

HOSTED_API_URLS = {
    "https://detect.roboflow.com",
    "https://outline.roboflow.com",
    "https://classify.roboflow.com",
    "https://lambda-object-detection.staging.roboflow.com",
    "https://lambda-instance-segmentation.staging.roboflow.com",
    "https://lambda-classification.staging.roboflow.com",
}


def process_image_with_workflow_using_api(
    image_path: str,
    output_directory: str,
    workflow_specification: Optional[dict] = None,
    workspace_name: Optional[str] = None,
    workflow_id: Optional[str] = None,
    workflow_parameters: Optional[Dict[str, Any]] = None,
    image_input_name: str = "image",
    api_key: Optional[str] = None,
    api_url: str = "https://detect.roboflow.com",
    save_image_outputs: bool = True,
    force_reprocessing: bool = False,
) -> None:
    if api_key is None:
        api_key = _get_api_key_from_env()
    log_file, log_content = open_progress_log(output_directory=output_directory)
    try:
        image_name = os.path.basename(image_path)
        if image_name in log_content and not force_reprocessing:
            return None
        result = _run_workflow_for_single_image_through_api(
            image_path=image_path,
            workflow_specification=workflow_specification,
            workspace_name=workspace_name,
            workflow_id=workflow_id,
            image_input_name=image_input_name,
            workflow_parameters=workflow_parameters,
            api_key=api_key,
            api_url=api_url,
        )
        if result is None:
            return None
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


def process_image_directory_with_workflow_using_api(
    input_directory: str,
    output_directory: str,
    workflow_specification: Optional[dict] = None,
    workspace_name: Optional[str] = None,
    workflow_id: Optional[str] = None,
    workflow_parameters: Optional[Dict[str, Any]] = None,
    image_input_name: str = "image",
    api_key: Optional[str] = None,
    api_url: str = "https://detect.roboflow.com",
    save_image_outputs: bool = True,
    force_reprocessing: bool = False,
    aggregate_structured_results: bool = True,
    aggregation_format: OutputFileType = OutputFileType.JSONL,
    debug_mode: bool = False,
    processing_threads: Optional[int] = None,
) -> None:
    if api_key is None:
        api_key = _get_api_key_from_env()
    if processing_threads is None:
        if _is_roboflow_hosted_api(api_url=api_url):
            processing_threads = 32
        else:
            processing_threads = 1
    files_to_process = get_all_images_in_directory(input_directory=input_directory)
    log_file, log_content = open_progress_log(output_directory=output_directory)
    try:
        remaining_files = [
            f
            for f in files_to_process
            if os.path.basename(f) not in log_content or force_reprocessing
        ]
        print(f"Files to process: {len(remaining_files)}")
        failed_files = _process_images_within_directory_with_api(
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
            api_url=api_url,
            processing_threads=processing_threads,
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


def _process_images_within_directory_with_api(
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
    api_url: str,
    processing_threads: int,
    debug_mode: bool = False,
) -> List[Tuple[str, str]]:
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
    log_file_lock = Lock()
    processing_fun = partial(
        _process_single_image_from_directory,
        workflow_specification=workflow_specification,
        workspace_name=workspace_name,
        workflow_id=workflow_id,
        image_input_name=image_input_name,
        workflow_parameters=workflow_parameters,
        api_url=api_url,
        api_key=api_key,
        output_directory=output_directory,
        save_image_outputs=save_image_outputs,
        log_file=log_file,
        on_success=on_success,
        on_failure=on_failure,
        log_file_lock=log_file_lock,
        debug_mode=debug_mode,
    )
    with progress_bar:
        with ThreadPool(processes=processing_threads) as pool:
            _ = pool.map(
                processing_fun,
                files_to_process,
            )
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
    workflow_specification: Dict[str, Any],
    workspace_name: Optional[str],
    workflow_id: Optional[str],
    image_input_name: str,
    workflow_parameters: Optional[Dict[str, Any]],
    api_url: str,
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
        result = _run_workflow_for_single_image_through_api(
            image_path=image_path,
            workflow_specification=workflow_specification,
            workspace_name=workspace_name,
            workflow_id=workflow_id,
            image_input_name=image_input_name,
            workflow_parameters=workflow_parameters,
            api_key=api_key,
            api_url=api_url,
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


@backoff.on_exception(
    backoff.constant,
    exception=HTTPCallErrorError,
    max_tries=3,
    interval=1,
    backoff_log_level=logging.DEBUG,
    giveup_log_level=logging.DEBUG,
)
def _run_workflow_for_single_image_through_api(
    image_path: str,
    workflow_specification: Optional[dict],
    workspace_name: Optional[str],
    workflow_id: Optional[str],
    image_input_name: str,
    workflow_parameters: Optional[Dict[str, Any]],
    api_key: Optional[str],
    api_url: str,
) -> Dict[str, Any]:
    client = InferenceHTTPClient(
        api_url=api_url,
        api_key=api_key,
    ).configure(
        InferenceConfiguration(
            output_visualisation_format=VisualisationResponseFormat.NUMPY
        )
    )
    result = client.run_workflow(
        workspace_name=workspace_name,
        workflow_id=workflow_id,
        specification=workflow_specification,
        images={
            image_input_name: [image_path],
        },
        parameters=workflow_parameters,
    )[0]
    return result


def _is_roboflow_hosted_api(api_url: str) -> bool:
    return api_url in HOSTED_API_URLS


def _get_api_key_from_env() -> Optional[str]:
    return os.getenv("ROBOFLOW_API_KEY") or os.getenv("API_KEY")
