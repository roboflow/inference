import hashlib
import json
import random
import string
from collections import Counter
from datetime import datetime, timezone
from typing import Generator, List, Optional, Union

import backoff
import requests
from requests import Timeout
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn
from rich.table import Table
from rich.text import Text

from inference_cli.lib.env import API_BASE_URL
from inference_cli.lib.roboflow_cloud.batch_processing.entities import (
    AggregationFormat,
    ComputeConfigurationV2,
    GetJobMetadataResponse,
    JobLog,
    JobLogsResponse,
    JobMetadata,
    JobStageDetails,
    ListBatchJobsResponse,
    ListJobStagesResponse,
    ListJobStageTasksResponse,
    LogSeverity,
    MachineType,
    StagingBatchInputV1,
    TaskStatus,
    WorkflowProcessingJobV1,
    WorkflowsProcessingSpecificationV1,
)
from inference_cli.lib.roboflow_cloud.common import (
    get_workspace,
    handle_response_errors,
    prepare_status_type_emoji,
)
from inference_cli.lib.roboflow_cloud.config import REQUEST_TIMEOUT
from inference_cli.lib.roboflow_cloud.errors import RetryError, RFAPICallError
from inference_cli.lib.utils import dump_jsonl, read_json

WORKFLOWS_IMAGE_PROCESSING_JOB = "workflows-images-processing"
WORKFLOWS_VIDEO_PROCESSING_JOB = "workflows-videos-processing"


def display_batch_jobs(
    api_key: str,
    page_size: int = 10,
    max_pages: Optional[int] = None,
) -> None:
    workspace = get_workspace(api_key=api_key)
    batch_jobs = list_batch_jobs(
        workspace=workspace,
        page_size=page_size,
        api_key=api_key,
        max_pages=max_pages,
    )
    if len(batch_jobs) == 0:
        print("No jobs found")
        return None
    console = Console()
    table = Table(title="Batch Jobs Overview", show_lines=True)
    table.add_column(
        "ID", justify="center", style="cyan", no_wrap=True, vertical="middle"
    )
    table.add_column(
        "Name", justify="center", width=24, overflow="ellipsis", vertical="middle"
    )
    table.add_column(
        "Stage", justify="center", width=24, style="blue", vertical="middle"
    )
    table.add_column("Status", justify="center", vertical="middle")
    table.add_column("Notification", justify="left", vertical="middle")
    table.add_column("Errors", justify="center", vertical="middle")
    for batch_job in batch_jobs:
        error_status = "🚨" if batch_job.error else "🟢"
        terminal_status = "🏁" if batch_job.is_terminal else "🏃"
        stage_status = _prepare_stage_status(
            current_stage=batch_job.current_stage,
            planned_stages=batch_job.planned_stages,
        )
        last_notification = batch_job.last_notification
        if isinstance(last_notification, dict):
            last_notification = JSON.from_data(last_notification, indent=2)
        table.add_row(
            batch_job.job_id,
            batch_job.name,
            stage_status,
            terminal_status,
            last_notification or "🔄",
            error_status,
        )
    console.print(table)


def list_batch_jobs(
    workspace: str,
    api_key: str,
    page_size: Optional[int] = None,
    max_pages: Optional[int] = None,
) -> List[JobMetadata]:
    return list(
        iterate_batch_jobs(
            workspace=workspace,
            api_key=api_key,
            page_size=page_size,
            max_pages=max_pages,
        )
    )


def iterate_batch_jobs(
    workspace: str,
    api_key: str,
    page_size: Optional[int] = None,
    max_pages: Optional[int] = None,
) -> Generator[JobMetadata, None, None]:
    if max_pages is not None and max_pages <= 0:
        raise ValueError("Could not specify max_pages <= 0")
    next_page_token = None
    pages_fetched = 0
    while True:
        if max_pages is not None and pages_fetched >= max_pages:
            break
        listing_page = get_batch_jobs_listing_page(
            workspace=workspace,
            api_key=api_key,
            page_size=page_size,
            next_page_token=next_page_token,
        )
        yield from listing_page.jobs
        next_page_token = listing_page.next_page_token
        if next_page_token is None:
            break
        pages_fetched += 1


@backoff.on_exception(
    backoff.constant,
    exception=RetryError,
    max_tries=3,
    interval=1,
)
def get_batch_jobs_listing_page(
    workspace: str,
    api_key: str,
    page_size: Optional[int] = None,
    next_page_token: Optional[str] = None,
) -> ListBatchJobsResponse:
    params = {}
    if api_key is not None:
        params["api_key"] = api_key
    if page_size:
        params["pageSize"] = page_size
    if next_page_token:
        params["nextPageToken"] = next_page_token
    try:
        response = requests.get(
            f"{API_BASE_URL}/batch-processing/v1/external/{workspace}/jobs",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
    except (ConnectionError, Timeout):
        raise RetryError(
            f"Connectivity error. Try reaching Roboflow API in browser: {API_BASE_URL}"
        )
    handle_response_errors(response=response, operation_name="list jobs")
    try:
        return ListBatchJobsResponse.model_validate(response.json())
    except ValueError as error:
        raise RFAPICallError("Could not decode Roboflow API response.") from error


def display_batch_job_details(job_id: str, api_key: str) -> None:
    workspace = get_workspace(api_key=api_key)
    job_metadata = get_batch_job_metadata(
        workspace=workspace, job_id=job_id, api_key=api_key
    )
    console = Console()
    heading_text = Text(
        f"Batch Job Overview [id={job_id}]",
        style="bold grey89 on steel_blue",
        justify="center",
    )
    heading_panel = Panel(heading_text, expand=True, border_style="steel_blue")
    console.print(heading_panel)
    table = Table(show_lines=True, expand=True)
    table.add_column("Property", justify="left", style="cyan", no_wrap=True)
    table.add_column("Value", justify="full", overflow="fold")
    table.add_row("Name", job_metadata.name)
    last_notification = job_metadata.last_notification
    if isinstance(last_notification, dict):
        last_notification = JSON.from_data(last_notification, indent=2)
    table.add_row("Last Notification", last_notification or "🔄")
    error_status = "🚨" if job_metadata.error else "🟢"
    running_status = "🏁" if job_metadata.is_terminal else "🏃"
    table.add_row("Status", f"Errors: {error_status} Is Running: {running_status}")
    stage_status = _prepare_stage_status(
        current_stage=job_metadata.current_stage,
        planned_stages=job_metadata.planned_stages,
    )
    table.add_row("Progress", stage_status)
    table.add_row("Created At", job_metadata.created_at.strftime("%d %b %Y, %I:%M %p"))
    table.add_row(
        "Job Definition", JSON.from_data(job_metadata.job_definition, indent=2)
    )
    if job_metadata.restart_parameters_override:
        table.add_row(
            "Restart parameters override",
            JSON.from_data(job_metadata.restart_parameters_override),
        )
    console.print(table)
    job_stages = list_job_stages(workspace=workspace, job_id=job_id, api_key=api_key)
    job_stages = sorted(job_stages, key=lambda e: e.start_timestamp)
    for stage in job_stages:
        job_tasks = list_job_stage_tasks(
            workspace=workspace,
            job_id=job_id,
            stage_id=stage.processing_stage_id,
            api_key=api_key,
        )
        most_recent_task_update_time = stage.start_timestamp
        if job_tasks:
            most_recent_task_update_time = max([t.event_timestamp for t in job_tasks])
        single_task_progress = 1 / stage.tasks_number
        accumulated_progress = 0.0
        for task in job_tasks:
            if task.status_type != "info":
                accumulated_progress += single_task_progress
            else:
                accumulated_progress += (task.progress or 0.0) * single_task_progress
        succeeded_tasks = [t for t in job_tasks if "success" in t.status_type.lower()]
        failed_tasks = [t for t in job_tasks if "error" in t.status_type.lower()]
        errors_lookup = {
            hash_notification(t.notification): t.notification for t in failed_tasks
        }
        failed_tasks_statuses = Counter(
            [hash_notification(t.notification) for t in failed_tasks]
        )
        error_reports = [
            f"* {errors_lookup[e[0]]}" for e in failed_tasks_statuses.most_common()
        ]
        error_reports_str = "\n".join(error_reports)
        if not error_reports_str:
            error_reports_str = "All Good 😃"
        expected_tasks = stage.tasks_number
        registered_tasks = len([t for t in job_tasks if t.progress is not None])
        tasks_waiting_for_processing = expected_tasks - registered_tasks
        running_tasks = len(
            [t for t in job_tasks if t.status_type == "info" and t.progress is not None]
        )
        terminated_tasks = len([t for t in job_tasks if t.status_type != "info"])
        heading_text = Text(
            f"Stage: {stage.processing_stage_name} [{stage.processing_stage_id}]",
            style="bold grey89 on steel_blue",
            justify="center",
        )
        heading_panel = Panel(heading_text, expand=True, border_style="steel_blue")
        console.print(heading_panel)
        details_table = Table(show_lines=True, expand=True)
        output_batches_str = (
            "⚪️" if not stage.output_batches else ", ".join(stage.output_batches)
        )
        is_terminal_str = "🏁" if stage.status_type != "info" else "🏃"
        elapse_update = ""
        if stage.status_type == "info":
            most_recent_update = max(
                most_recent_task_update_time, stage.last_event_timestamp
            )
            time_from_start = round(
                (datetime.now(timezone.utc) - most_recent_update).total_seconds() / 60
            )
            elapse_update = f" (last update {max(time_from_start, 0)}m ago)"
        updates_string = f"Errors: {prepare_status_type_emoji(status_type=stage.status_type)} Is Running: {is_terminal_str}{elapse_update}"
        notification = stage.notification
        if isinstance(notification, dict):
            notification = JSON.from_data(stage.notification, indent=2)
        details_table.add_column("Property", justify="left", style="cyan", no_wrap=True)
        details_table.add_column("Value", justify="full", overflow="ellipsis")
        details_table.add_row(
            "ID",
            f"[bold green]StageID:[/bold green] {stage.processing_stage_id} "
            f"[bold green]Name:[/bold green] {stage.processing_stage_name}",
        )
        details_table.add_row("Output Batches", output_batches_str)
        elapse_update = ""
        if stage.status_type == "info":
            time_from_start = round(
                (datetime.now(timezone.utc) - stage.start_timestamp).total_seconds()
                / 60
            )
            elapse_update = f" ({max(time_from_start, 0)}m ago)"
        started_at_str = (
            f"{stage.start_timestamp.strftime('%d %b %Y, %I:%M %p')}{elapse_update}"
        )
        details_table.add_row("Started At", started_at_str)
        details_table.add_row("Status", updates_string)
        details_table.add_row("Notification", notification)
        details_table.add_row(
            "Downstream Tasks",
            f"⏳️: {tasks_waiting_for_processing}, 🏃: {running_tasks}, 🏁: {terminated_tasks} "
            f"(out of {expected_tasks})",
        )
        progress = Progress(BarColumn(), TaskProgressColumn())
        completed = round(accumulated_progress * 100, 2)
        _ = progress.add_task(
            description="Stage Progress",
            total=100.0,
            start=completed > 0,
            completed=completed,
        )

        details_table.add_row("Progress", progress.get_renderable())
        details_table.add_row(
            "Completed Tasks Status",
            f"✅: {len(succeeded_tasks)}, ❌: {len(failed_tasks)}",
        )
        details_table.add_row("Error Details", error_reports_str)
        console.print(details_table)


def hash_notification(notification: Union[dict, str]) -> str:
    if isinstance(notification, str):
        return hashlib.md5(notification.encode("utf-8")).hexdigest()
    return hashlib.md5(
        json.dumps(notification, sort_keys=True).encode("utf-8")
    ).hexdigest()


@backoff.on_exception(
    backoff.constant,
    exception=RetryError,
    max_tries=3,
    interval=1,
)
def get_batch_job_metadata(workspace: str, job_id: str, api_key: str) -> JobMetadata:
    params = {}
    if api_key is not None:
        params["api_key"] = api_key
    try:
        response = requests.get(
            f"{API_BASE_URL}/batch-processing/v1/external/{workspace}/jobs/{job_id}",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
    except (ConnectionError, Timeout):
        raise RetryError(
            f"Connectivity error. Try reaching Roboflow API in browser: {API_BASE_URL}"
        )
    handle_response_errors(response=response, operation_name="get job metadata")
    try:
        return GetJobMetadataResponse.model_validate(response.json()).job
    except ValueError as error:
        raise RFAPICallError("Could not decode Roboflow API response.") from error


def trigger_job_with_workflows_images_processing(
    batch_id: str,
    workflow_id: str,
    workflow_parameters_path: Optional[str],
    image_input_name: Optional[str],
    save_image_outputs: bool,
    image_outputs_to_save: Optional[List[str]],
    part_name: Optional[str],
    machine_type: Optional[MachineType],
    workers_per_machine: int,
    max_runtime_seconds: Optional[int],
    max_parallel_tasks: Optional[int],
    aggregation_format: Optional[AggregationFormat],
    job_id: Optional[str],
    notifications_url: Optional[str],
    api_key: str,
) -> str:
    workspace = get_workspace(api_key=api_key)
    compute_configuration = ComputeConfigurationV2(
        machine_type=machine_type,
        workers_per_machine=workers_per_machine,
    )
    input_configuration = StagingBatchInputV1(
        batch_id=batch_id,
        part_name=part_name,
    )
    workflow_parameters = None
    if workflow_parameters_path:
        workflow_parameters = read_json(path=workflow_parameters_path)
    processing_specification = WorkflowsProcessingSpecificationV1(
        workspace=workspace,
        workflow_id=workflow_id,
        workflow_parameters=workflow_parameters,
        image_input_name=image_input_name,
        persist_images_outputs=save_image_outputs,
        images_outputs_to_be_persisted=image_outputs_to_save,
        aggregation_format=aggregation_format,
    )
    if not job_id:
        job_id = f"job-{_generate_random_string(length=12)}"
    job_configuration = WorkflowProcessingJobV1(
        type="simple-image-processing-v1",
        job_input=input_configuration,
        compute_configuration=compute_configuration,
        processing_timeout_seconds=max_runtime_seconds,
        max_parallel_tasks=max_parallel_tasks,
        processing_specification=processing_specification,
        notifications_url=notifications_url,
    )
    create_batch_job(
        workspace=workspace,
        job_id=job_id,
        job_configuration=job_configuration,
        api_key=api_key,
    )
    return job_id


def trigger_job_with_workflows_videos_processing(
    batch_id: str,
    workflow_id: str,
    workflow_parameters_path: Optional[str],
    image_input_name: Optional[str],
    save_image_outputs: bool,
    image_outputs_to_save: Optional[List[str]],
    part_name: Optional[str],
    machine_type: Optional[MachineType],
    workers_per_machine: int,
    max_runtime_seconds: Optional[int],
    max_parallel_tasks: Optional[int],
    aggregation_format: Optional[AggregationFormat],
    max_video_fps: Optional[Union[float, int]],
    job_id: Optional[str],
    notifications_url: Optional[str],
    api_key: str,
) -> str:
    workspace = get_workspace(api_key=api_key)
    compute_configuration = ComputeConfigurationV2(
        machine_type=machine_type,
        workers_per_machine=workers_per_machine,
    )
    input_configuration = StagingBatchInputV1(
        batch_id=batch_id,
        part_name=part_name,
    )
    workflow_parameters = None
    if workflow_parameters_path:
        workflow_parameters = read_json(path=workflow_parameters_path)
    processing_specification = WorkflowsProcessingSpecificationV1(
        workspace=workspace,
        workflow_id=workflow_id,
        workflow_parameters=workflow_parameters,
        image_input_name=image_input_name,
        persist_images_outputs=save_image_outputs,
        images_outputs_to_be_persisted=image_outputs_to_save,
        aggregation_format=aggregation_format,
        max_video_fps=max_video_fps,
    )
    if not job_id:
        job_id = f"job-{_generate_random_string(length=12)}"
    job_configuration = WorkflowProcessingJobV1(
        type="simple-video-processing-v1",
        job_input=input_configuration,
        compute_configuration=compute_configuration,
        processing_timeout_seconds=max_runtime_seconds,
        max_parallel_tasks=max_parallel_tasks,
        processing_specification=processing_specification,
        notifications_url=notifications_url,
    )
    create_batch_job(
        workspace=workspace,
        job_id=job_id,
        job_configuration=job_configuration,
        api_key=api_key,
    )
    return job_id


@backoff.on_exception(
    backoff.constant,
    exception=RetryError,
    max_tries=3,
    interval=1,
)
def create_batch_job(
    workspace: str,
    job_id: str,
    job_configuration: WorkflowProcessingJobV1,
    api_key: str,
) -> None:
    params = {"api_key": api_key}
    try:
        response = requests.post(
            f"{API_BASE_URL}/batch-processing/v1/external/{workspace}/jobs/{job_id}",
            params=params,
            timeout=REQUEST_TIMEOUT,
            json=job_configuration.model_dump(by_alias=True, exclude_none=True),
        )
    except (ConnectionError, Timeout):
        raise RetryError(
            f"Connectivity error. Try reaching Roboflow API in browser: {API_BASE_URL}"
        )
    handle_response_errors(response=response, operation_name="create job")
    return None


@backoff.on_exception(
    backoff.constant,
    exception=RetryError,
    max_tries=3,
    interval=1,
)
def list_job_stages(
    workspace: str,
    job_id: str,
    api_key: str,
) -> List[JobStageDetails]:
    params = {}
    if api_key is not None:
        params["api_key"] = api_key
    try:
        response = requests.get(
            f"{API_BASE_URL}/batch-processing/v1/external/{workspace}/jobs/{job_id}/stages",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
    except (ConnectionError, Timeout):
        raise RetryError(
            f"Connectivity error. Try reaching Roboflow API in browser: {API_BASE_URL}"
        )
    handle_response_errors(response=response, operation_name="list job stages")
    try:
        return ListJobStagesResponse.model_validate(response.json()).stages
    except ValueError as error:
        raise RFAPICallError("Could not decode Roboflow API response.") from error


def list_job_stage_tasks(
    workspace: str,
    job_id: str,
    stage_id: str,
    api_key: str,
) -> List[TaskStatus]:
    return list(
        iterate_job_stage_tasks(
            workspace=workspace,
            job_id=job_id,
            stage_id=stage_id,
            api_key=api_key,
        )
    )


def iterate_job_stage_tasks(
    workspace: str,
    job_id: str,
    stage_id: str,
    api_key: str,
) -> Generator[TaskStatus, None, None]:
    next_page_token = None
    pages_fetched = 0
    while True:
        listing_page = get_job_stage_tasks_listing_page(
            workspace=workspace,
            job_id=job_id,
            stage_id=stage_id,
            api_key=api_key,
            next_page_token=next_page_token,
        )
        yield from listing_page.tasks
        next_page_token = listing_page.next_page_token
        if next_page_token is None:
            break
        pages_fetched += 1


@backoff.on_exception(
    backoff.constant,
    exception=RetryError,
    max_tries=3,
    interval=1,
)
def get_job_stage_tasks_listing_page(
    workspace: str,
    job_id: str,
    stage_id: str,
    api_key: str,
    page_size: Optional[int] = None,
    next_page_token: Optional[str] = None,
) -> ListJobStageTasksResponse:
    params = {"api_key": api_key}
    if page_size:
        params["pageSize"] = page_size
    if next_page_token:
        params["nextPageToken"] = next_page_token
    try:
        response = requests.get(
            f"{API_BASE_URL}/batch-processing/v1/external/{workspace}/jobs/{job_id}/stages/{stage_id}/tasks",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
    except (ConnectionError, Timeout):
        raise RetryError(
            f"Connectivity error. Try reaching Roboflow API in browser: {API_BASE_URL}"
        )
    handle_response_errors(response=response, operation_name="list job stage tasks")
    try:
        return ListJobStageTasksResponse.model_validate(response.json())
    except ValueError as error:
        raise RFAPICallError("Could not decode Roboflow API response.") from error


def abort_batch_job(job_id: str, api_key: str) -> None:
    workspace = get_workspace(api_key=api_key)
    response = send_abort_job_request(
        workspace=workspace,
        job_id=job_id,
        api_key=api_key,
    )
    console = Console()
    console.print(JSON.from_data(response, indent=2))


@backoff.on_exception(
    backoff.constant,
    exception=RetryError,
    max_tries=3,
    interval=1,
)
def send_abort_job_request(
    workspace: str,
    job_id: str,
    api_key: str,
) -> dict:
    params = {"api_key": api_key}
    try:
        response = requests.post(
            f"{API_BASE_URL}/batch-processing/v1/external/{workspace}/jobs/{job_id}/abort",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
    except (ConnectionError, Timeout):
        raise RetryError(
            f"Connectivity error. Try reaching Roboflow API in browser: {API_BASE_URL}"
        )
    handle_response_errors(response=response, operation_name="abort job")
    return response.json()


def restart_batch_job(
    job_id: str,
    api_key: str,
    machine_type: Optional[MachineType] = None,
    workers_per_machine: Optional[int] = None,
    max_runtime_seconds: Optional[float] = None,
    max_parallel_tasks: Optional[int] = None,
) -> None:
    workspace = get_workspace(api_key=api_key)
    response = send_restart_job_request(
        workspace=workspace,
        job_id=job_id,
        api_key=api_key,
        machine_type=machine_type,
        workers_per_machine=workers_per_machine,
        max_runtime_seconds=max_runtime_seconds,
        max_parallel_tasks=max_parallel_tasks,
    )
    console = Console()
    console.print(JSON.from_data(response, indent=2))


@backoff.on_exception(
    backoff.constant,
    exception=RetryError,
    max_tries=3,
    interval=1,
)
def send_restart_job_request(
    workspace: str,
    job_id: str,
    api_key: str,
    machine_type: Optional[MachineType] = None,
    workers_per_machine: Optional[int] = None,
    max_runtime_seconds: Optional[float] = None,
    max_parallel_tasks: Optional[int] = None,
) -> dict:
    payload = {
        "type": "parameters-override-v1",
    }
    if machine_type is not None or workers_per_machine is not None:
        compute_configuration = {"type": "compute-configuration-v2"}
        if machine_type:
            compute_configuration["machineType"] = machine_type.value
        if workers_per_machine:
            compute_configuration["workersPerMachine"] = workers_per_machine
        payload["computeConfiguration"] = compute_configuration
    if max_parallel_tasks is not None:
        payload["maxParallelTasks"] = max_parallel_tasks
    if max_runtime_seconds is not None:
        payload["processingTimeoutSeconds"] = max_runtime_seconds
    params = {"api_key": api_key}
    try:
        response = requests.post(
            f"{API_BASE_URL}/batch-processing/v1/external/{workspace}/jobs/{job_id}/restart",
            params=params,
            timeout=REQUEST_TIMEOUT,
            json=payload if len(payload) > 1 else None,
        )
    except (ConnectionError, Timeout):
        raise RetryError(
            f"Connectivity error. Try reaching Roboflow API in browser: {API_BASE_URL}"
        )
    handle_response_errors(response=response, operation_name="restart job")
    return response.json()


def fetch_job_logs(
    job_id: str,
    api_key: str,
    log_severity: Optional[LogSeverity] = None,
    output_file: Optional[str] = None,
) -> None:
    workspace = get_workspace(api_key=api_key)
    logs = []
    for log in iterate_over_job_logs(
        workspace=workspace,
        job_id=job_id,
        api_key=api_key,
        log_severity=log_severity,
    ):
        if output_file is None:
            print(log.model_dump())
        else:
            logs.append(log)
    if output_file:
        # for datetime serialization, we can afford overhead here most likely
        serialized_logs = [json.loads(log.model_dump_json()) for log in logs]
        dump_jsonl(path=output_file, content=serialized_logs)


def iterate_over_job_logs(
    workspace: str,
    job_id: str,
    api_key: str,
    log_severity: Optional[LogSeverity] = None,
) -> Generator[JobLog, None, None]:
    next_page_token = None
    while True:
        listing_page = get_one_page_of_logs(
            workspace=workspace,
            job_id=job_id,
            api_key=api_key,
            log_severity=log_severity,
            next_page_token=next_page_token,
        )
        for log in listing_page.logs:
            yield log
        if listing_page.next_page_token is None:
            break
        next_page_token = listing_page.next_page_token


def get_one_page_of_logs(
    workspace: str,
    job_id: str,
    api_key: str,
    next_page_token: Optional[str] = None,
    log_severity: Optional[LogSeverity] = None,
    page_size: Optional[int] = None,
) -> JobLogsResponse:
    params = {"api_key": api_key}
    if page_size:
        params["pageSize"] = page_size
    if log_severity:
        params["severity"] = log_severity.value
    if next_page_token:
        params["nextPageToken"] = next_page_token
    try:
        response = requests.get(
            f"{API_BASE_URL}/batch-processing/v1/external/{workspace}/jobs/{job_id}/logs",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
    except (ConnectionError, Timeout):
        raise RetryError(
            f"Connectivity error. Try reaching Roboflow API in browser: {API_BASE_URL}"
        )
    handle_response_errors(response=response, operation_name="get job logs")
    try:
        return JobLogsResponse.model_validate(response.json())
    except ValueError as error:
        raise RFAPICallError("Could not decode Roboflow API response.") from error


def _prepare_stage_status(
    current_stage: Optional[str], planned_stages: Optional[List[str]]
) -> str:
    if not current_stage:
        stage_status = "🔄"
    else:
        stage_status = current_stage
    if current_stage and planned_stages and current_stage in planned_stages:
        stage_names_str = []
        for stage in planned_stages:
            if stage != current_stage:
                stage_str = f"[medium_purple]{stage}[/medium_purple]"
            else:
                stage_str = f"[bold green]{stage}[/bold green]"
            stage_names_str.append(stage_str)
        stage_status = " - ".join(stage_names_str)
    return stage_status


def _generate_random_string(length: int = 6) -> str:
    letters = string.ascii_letters + string.digits
    return "".join(random.choice(letters) for _ in range(length)).lower()
