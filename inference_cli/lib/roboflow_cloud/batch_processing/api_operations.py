import json
import random
import string
from collections import Counter
from typing import List, Optional

import backoff
import requests
from requests import Timeout
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from inference_cli.lib.env import API_BASE_URL
from inference_cli.lib.roboflow_cloud.batch_processing.entities import (
    BatchJobMetadata,
    BatchJobMetadataResponse,
    BatchJobStateDescription,
    JobStageDetails,
    ListBatchJobsResponse,
    ListJobStagesResponse,
    ListJobStageTasksResponse,
    TaskStatus,
)
from inference_cli.lib.roboflow_cloud.common import (
    get_workspace,
    handle_response_errors,
    prepare_status_type_emoji,
)
from inference_cli.lib.roboflow_cloud.config import REQUEST_TIMEOUT
from inference_cli.lib.roboflow_cloud.data_staging.api_operations import (
    find_batch_by_id,
)
from inference_cli.lib.roboflow_cloud.errors import RetryError, RFAPICallError

WORKFLOWS_IMAGE_PROCESSING_JOB = "workflows-images-processing"


def display_batch_jobs(api_key: Optional[str]) -> None:
    workspace = get_workspace(api_key=api_key)
    batch_jobs = list_batch_jobs(workspace=workspace, api_key=api_key)
    if len(batch_jobs) == 0:
        print("No batches found")
        return None
    console = Console()
    table = Table(title="Batch Jobs Overview", show_lines=True)
    table.add_column("ID", justify="center", style="cyan", no_wrap=True)
    table.add_column("Name", justify="center", width=26, overflow="ellipsis")
    table.add_column("Stage", justify="center", width=26, style="blue")
    table.add_column("Errors", justify="center")
    table.add_column("Error Details", justify="center")
    for batch_job in batch_jobs:
        error_status = "🚨" if batch_job.error else "👍"
        stage_status = _prepare_stage_status(
            current_stage=batch_job.stage_name, planned_stages=batch_job.planned_stages
        )
        table.add_row(
            batch_job.job_id,
            batch_job.display_name,
            stage_status,
            error_status,
            batch_job.error_details if batch_job.error_details else "N/A",
        )
    console.print(table)


@backoff.on_exception(
    backoff.constant,
    exception=RetryError,
    max_tries=3,
    interval=1,
)
def list_batch_jobs(
    workspace: str, api_key: Optional[str]
) -> List[BatchJobStateDescription]:
    params = {}
    if api_key is not None:
        params["api_key"] = api_key
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
    handle_response_errors(response=response, operation_name="list batches")
    try:
        return ListBatchJobsResponse.model_validate(response.json()).batch_jobs
    except ValueError as error:
        raise RFAPICallError("Could not decode Roboflow API response.") from error


def display_batch_job_details(job_id: str, api_key: Optional[str]) -> None:
    workspace = get_workspace(api_key=api_key)
    job_metadata = get_batch_job_metadata(
        workspace=workspace, job_id=job_id, api_key=api_key
    )
    console = Console()
    heading_text = Text(
        f"Batch Job Overview [job={job_id}]",
        style="bold grey89 on steel_blue",
        justify="center",
    )
    heading_panel = Panel(heading_text, expand=True, border_style="steel_blue")
    console.print(heading_panel)
    table = Table(show_lines=True, expand=True)
    table.add_column("Property", justify="left", style="cyan", no_wrap=True)
    table.add_column("Value", justify="full", overflow="ellipsis")

    table.add_row("Name", job_metadata.display_name)
    table.add_row("Job Type", job_metadata.job_type)
    table.add_row("Job Parameters", json.dumps(job_metadata.job_parameters, indent=4))
    table.add_row("Input", json.dumps(job_metadata.input_definition, indent=4))
    stage_status = _prepare_stage_status(
        current_stage=job_metadata.stage_name,
        planned_stages=job_metadata.planned_stages,
    )
    table.add_row("Progress", stage_status)
    table.add_row("Last Update", job_metadata.event_timestamp.isoformat())
    console.print(table)

    job_stages = list_job_stages(workspace=workspace, job_id=job_id, api_key=api_key)
    for stage in job_stages:
        job_tasks = list_job_stage_tasks(
            workspace=workspace,
            job_id=job_id,
            stage_id=stage.processing_stage_id,
            api_key=api_key,
        )
        succeeded_tasks = [t for t in job_tasks if "success" in t.status_type.lower()]
        failed_tasks = [t for t in job_tasks if "error" in t.status_type.lower()]
        failed_tasks_statuses = Counter([t.status_name for t in failed_tasks])
        error_reports = [f"* {e[0]}" for e in failed_tasks_statuses.most_common()]
        error_reports_str = "\n".join(error_reports)
        if not error_reports_str:
            error_reports_str = "All Good 😃"
        expected_tasks = stage.tasks_number
        registered_tasks = len(job_tasks)
        tasks_waiting_for_processing = expected_tasks - registered_tasks
        running_tasks = len([t for t in job_tasks if t.is_terminal is False])
        terminated_tasks = len([t for t in job_tasks if t.is_terminal is True])
        heading_text = Text(
            f"Stage: {stage.processing_stage_name} [{stage.processing_stage_id}]",
            style="bold grey89 on steel_blue",
            justify="center",
        )
        heading_panel = Panel(heading_text, expand=True, border_style="steel_blue")
        console.print(heading_panel)
        details_table = Table(show_lines=True, expand=True)
        output_batches_str = (
            "❌" if not stage.output_batches else ", ".join(stage.output_batches)
        )
        status_update = f"{prepare_status_type_emoji(status_type=stage.status_type)} {stage.status_name}"
        is_terminal_str = "✋" if stage.is_terminal else "🏃"
        updates_string = (
            f"{is_terminal_str} - last update: {stage.last_event_timestamp.isoformat()}"
        )
        details_table.add_column("Property", justify="left", style="cyan", no_wrap=True)
        details_table.add_column("Value", justify="full", overflow="ellipsis")
        details_table.add_row("ID", stage.processing_stage_id)
        details_table.add_row("Name", stage.processing_stage_name)
        details_table.add_row("Output Batches", output_batches_str)
        details_table.add_row("Started At", stage.start_timestamp.isoformat())
        details_table.add_row("Status", status_update)
        details_table.add_row("Progress", updates_string)
        details_table.add_row(
            "Downstream Tasks",
            f"⏳️: {tasks_waiting_for_processing}, 🏃: {running_tasks}, 🏁: {terminated_tasks} "
            f"(out of {registered_tasks})",
        )
        details_table.add_row(
            "Completed Tasks Status",
            f"✅: {len(succeeded_tasks)}, ❌: {len(failed_tasks)}",
        )
        details_table.add_row("Error Details", error_reports_str)
        console.print(details_table)


@backoff.on_exception(
    backoff.constant,
    exception=RetryError,
    max_tries=3,
    interval=1,
)
def get_batch_job_metadata(
    workspace: str, job_id: str, api_key: Optional[str]
) -> BatchJobMetadata:
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
    handle_response_errors(response=response, operation_name="list batches")
    try:
        return BatchJobMetadataResponse.model_validate(response.json()).job_metadata
    except ValueError as error:
        raise RFAPICallError("Could not decode Roboflow API response.") from error


def trigger_job_with_workflows_images_processing(
    batch_id: str,
    job_id: Optional[str],
    api_key: Optional[str],
) -> str:
    workspace = get_workspace(api_key=api_key)
    selected_batch = find_batch_by_id(
        workspace=workspace, batch_id=batch_id, api_key=api_key
    )
    if not job_id:
        job_id = f"workflows-{_generate_random_string()}"
    input_definition = {
        "type": "images-batch",
        "batchId": selected_batch.batch_id,
    }
    create_batch_job(
        workspace=workspace,
        job_id=job_id,
        job_type=WORKFLOWS_IMAGE_PROCESSING_JOB,
        input_definition=input_definition,
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
    job_type: str,
    input_definition: dict,
    api_key: Optional[str],
) -> None:
    params = {}
    if api_key is not None:
        params["api_key"] = api_key
    try:
        response = requests.post(
            f"{API_BASE_URL}/batch-processing/v1/external/{workspace}/jobs/{job_id}",
            params=params,
            timeout=REQUEST_TIMEOUT,
            json={
                "jobType": job_type,
                "inputDefinition": input_definition,
            },
        )
    except (ConnectionError, Timeout):
        raise RetryError(
            f"Connectivity error. Try reaching Roboflow API in browser: {API_BASE_URL}"
        )
    handle_response_errors(response=response, operation_name="list batches")
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
    api_key: Optional[str],
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
        return ListJobStagesResponse.model_validate(response.json()).job_stages_metadata
    except ValueError as error:
        raise RFAPICallError("Could not decode Roboflow API response.") from error


@backoff.on_exception(
    backoff.constant,
    exception=RetryError,
    max_tries=3,
    interval=1,
)
def list_job_stage_tasks(
    workspace: str,
    job_id: str,
    stage_id: str,
    api_key: Optional[str],
) -> List[TaskStatus]:
    params = {}
    if api_key is not None:
        params["api_key"] = api_key
    try:
        response = requests.get(
            f"{API_BASE_URL}/batch-processing/v1/external/{workspace}/jobs/{job_id}/stages/{stage_id}",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
    except (ConnectionError, Timeout):
        raise RetryError(
            f"Connectivity error. Try reaching Roboflow API in browser: {API_BASE_URL}"
        )
    handle_response_errors(response=response, operation_name="list job stages")
    try:
        return ListJobStageTasksResponse.model_validate(response.json()).tasks_statuses
    except ValueError as error:
        raise RFAPICallError("Could not decode Roboflow API response.") from error


def _prepare_stage_status(
    current_stage: Optional[str], planned_stages: Optional[List[str]]
) -> str:
    if not current_stage:
        stage_status = "🕰"
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
