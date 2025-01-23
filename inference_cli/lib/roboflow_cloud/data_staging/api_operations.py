import math
import os
import tarfile
import tempfile
from datetime import datetime, timedelta
from functools import partial
from multiprocessing.pool import Pool, ThreadPool
from typing import Dict, List, Optional
from uuid import uuid4

import backoff
import requests
from requests import Timeout
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
import supervision as sv

from inference_cli.lib.env import API_BASE_URL
from inference_cli.lib.roboflow_cloud.batch_processing.entities import (
    ListMultipartBatchPartsResponse,
    MultipartBatchPartMetadata,
)
from inference_cli.lib.roboflow_cloud.common import (
    get_workspace,
    handle_response_errors,
)
from inference_cli.lib.roboflow_cloud.config import (
    MAX_SHARD_SIZE,
    MAX_SHARDS_UPLOAD_PROCESSES,
    MIN_IMAGES_TO_FORM_SHARD,
    REQUEST_TIMEOUT, SUGGESTED_MAX_VIDEOS_IN_BATCH,
)
from inference_cli.lib.roboflow_cloud.data_staging.entities import (
    BatchDetails,
    ListBatchesResponse,
    ShardDetails,
)
from inference_cli.lib.roboflow_cloud.errors import (
    RetryError,
    RFAPICallError,
    RoboflowCloudCommandError,
)
from inference_cli.lib.utils import create_batches, get_all_images_in_directory, get_all_videos_in_directory


def display_batches(api_key: Optional[str]) -> None:
    workspace = get_workspace(api_key=api_key)
    batches = list_batches(workspace=workspace, api_key=api_key)
    if len(batches) == 0:
        print("No batches found")
        return None
    console = Console()
    table = Table(title="Roboflow Data Staging Batches Overview", show_lines=True)
    table.add_column("ID", justify="center", style="cyan", no_wrap=True)
    table.add_column("Name", justify="center", width=32, overflow="ellipsis")
    table.add_column("Content", justify="center", style="blue")
    table.add_column("Expiry", justify="center")
    table.add_column("Batch Type", justify="center")
    for batch in batches:
        expiry_date_string = batch.expiry_date.strftime("%d/%m/%Y")
        if (batch.expiry_date - datetime.now().date()) <= timedelta(days=3):
            expiry_date_string = f"[bold red]{expiry_date_string}[/bold red]"
        else:
            expiry_date_string = f"[dark_cyan]{expiry_date_string}[/dark_cyan]"
        table.add_row(
            batch.batch_id,
            batch.display_name,
            batch.batch_content_type,
            expiry_date_string,
            batch.batch_type,
        )
    console.print(table)


@backoff.on_exception(
    backoff.constant,
    exception=RetryError,
    max_tries=3,
    interval=1,
)
def list_batches(workspace: str, api_key: Optional[str]) -> List[BatchDetails]:
    params = {}
    if api_key is not None:
        params["api_key"] = api_key
    try:
        response = requests.get(
            f"{API_BASE_URL}/data-staging/v1/external/{workspace}/batches",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
    except (ConnectionError, Timeout, requests.exceptions.ConnectionError):
        raise RetryError(
            f"Connectivity error. Try reaching Roboflow API in browser: {API_BASE_URL}"
        )
    handle_response_errors(response=response, operation_name="list batches")
    try:
        return ListBatchesResponse.model_validate(response.json()).batches
    except ValueError as error:
        raise RFAPICallError("Could not decode Roboflow API response.") from error


def create_images_batch_from_directory(
    directory: str,
    batch_id: str,
    api_key: Optional[str],
) -> None:
    workspace = get_workspace(api_key=api_key)
    images_paths = get_all_images_in_directory(input_directory=directory)
    if len(images_paths) > MIN_IMAGES_TO_FORM_SHARD:
        upload_images_to_sharded_batch(
            images_paths=images_paths,
            workspace=workspace,
            batch_id=batch_id,
            api_key=api_key,
        )
        return None
    upload_images_to_simple_batch(
        images_paths=images_paths,
        workspace=workspace,
        batch_id=batch_id,
        api_key=api_key,
    )


def create_videos_batch_from_directory(
    directory: str,
    batch_id: str,
    api_key: Optional[str],
) -> None:
    workspace = get_workspace(api_key=api_key)
    video_paths = get_all_videos_in_directory(input_directory=directory)
    video_paths = _filter_out_malformed_videos(video_paths=video_paths)
    if len(video_paths) > SUGGESTED_MAX_VIDEOS_IN_BATCH:
        print(
            f"You try to upload {len(video_paths)} to single batch. "
            f"Suggested max size is: {SUGGESTED_MAX_VIDEOS_IN_BATCH}."
        )
    for video in tqdm(video_paths, desc="Uploading video files..."):
        upload_video(
            video_path=video,
            workspace=workspace,
            batch_id=batch_id,
            api_key=api_key,
        )


def _filter_out_malformed_videos(video_paths: List[str]) -> List[str]:
    result = []
    for path in tqdm(video_paths, desc="Verifying videos..."):
        try:
            _ = sv.VideoInfo.from_video_path(path)
            result.append(path)
        except Exception:
            print(f"File: {path} is corrupted or is not a video file")
    return result


def upload_images_to_simple_batch(
    images_paths: List[str],
    workspace: str,
    batch_id: str,
    api_key: Optional[str],
) -> None:
    upload_image_closure = partial(
        upload_image,
        workspace=workspace,
        batch_id=batch_id,
        api_key=api_key,
    )
    with ThreadPool() as pool:
        for _ in tqdm(
            pool.imap(upload_image_closure, images_paths),
            desc="Uploading images...",
            total=len(images_paths),
        ):
            pass


def upload_images_to_sharded_batch(
    images_paths: List[str],
    workspace: str,
    batch_id: str,
    api_key: Optional[str],
) -> None:
    shards = list(create_batches(sequence=images_paths, batch_size=MAX_SHARD_SIZE))
    upload_images_shard_closure = partial(
        pack_and_upload_images_shard,
        workspace=workspace,
        batch_id=batch_id,
        api_key=api_key,
    )
    with Pool(processes=MAX_SHARDS_UPLOAD_PROCESSES) as pool:
        for _ in tqdm(
            pool.imap(upload_images_shard_closure, shards),
            desc=f"Uploading images shards (each {MAX_SHARD_SIZE} images)...",
            total=len(shards),
        ):
            pass


def pack_and_upload_images_shard(
    images_paths: List[str],
    workspace: str,
    batch_id: str,
    api_key: Optional[str],
) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        archive_path = os.path.join(tmp_dir, f"{uuid4()}.tar")
        create_images_shard_archive(
            archive_path=archive_path, images_paths=images_paths
        )
        upload_images_shard(
            archive_path=archive_path,
            workspace=workspace,
            batch_id=batch_id,
            api_key=api_key,
        )


def create_images_shard_archive(images_paths: List[str], archive_path: str) -> None:
    with tarfile.open(archive_path, "w") as tar:
        for image_path in images_paths:
            tar.add(image_path, arcname=os.path.basename(image_path))


@backoff.on_exception(
    backoff.constant,
    exception=RetryError,
    max_tries=3,
    interval=1,
)
def upload_image(
    image_path: str, workspace: str, batch_id: str, api_key: Optional[str]
) -> None:
    params = {}
    if api_key is not None:
        params["api_key"] = api_key
    image_file_name = os.path.basename(image_path)
    params["file_name"] = image_file_name
    try:
        with open(image_path, "rb") as f:
            image_content = f.read()
    except OSError as error:
        raise RoboflowCloudCommandError(
            f"Could not read file from {image_path}"
        ) from error
    try:
        response = requests.post(
            f"{API_BASE_URL}/data-staging/v1/external/{workspace}/batches/{batch_id}/upload/image",
            params=params,
            timeout=REQUEST_TIMEOUT,
            files={
                image_file_name: image_content,
            },
        )
    except (ConnectionError, Timeout, requests.exceptions.ConnectionError) as error:
        raise RetryError(
            f"Connectivity error. Try reaching Roboflow API in browser: {API_BASE_URL}"
        ) from error
    handle_response_errors(response=response, operation_name="list batches")


@backoff.on_exception(
    backoff.constant,
    exception=RetryError,
    max_tries=3,
    interval=1,
)
def upload_video(
    video_path: str,
    workspace: str,
    batch_id: str,
    api_key: Optional[str],
) -> None:
    params = {}
    if api_key is not None:
        params["api_key"] = api_key
    image_file_name = os.path.basename(video_path)
    params["file_name"] = image_file_name
    try:
        response = requests.post(
            f"{API_BASE_URL}/data-staging/v1/external/{workspace}/batches/{batch_id}/upload/video",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
    except (ConnectionError, Timeout, requests.exceptions.ConnectionError) as error:
        raise RetryError(
            f"Connectivity error. Try reaching Roboflow API in browser: {API_BASE_URL}"
        ) from error
    handle_response_errors(response=response, operation_name="upload video")
    try:
        response_data = response.json()
        upload_url, extension_headers = (
            response_data["uploadURL"],
            response_data["extensionHeaders"],
        )
    except (ValueError, KeyError) as error:
        raise RFAPICallError("Could not decode Roboflow API response.") from error
    upload_file_to_cloud(
        file_path=video_path,
        url=upload_url,
        headers=extension_headers,
    )


@backoff.on_exception(
    backoff.constant,
    exception=RetryError,
    max_tries=3,
    interval=1,
)
def upload_images_shard(
    archive_path: str, workspace: str, batch_id: str, api_key: Optional[str]
) -> None:
    # This flow is over-simplified - we should request shard at first and only then pack into archive
    params = {}
    if api_key is not None:
        params["api_key"] = api_key
    try:
        response = requests.post(
            f"{API_BASE_URL}/data-staging/v1/external/{workspace}/batches/{batch_id}/bulk-upload/image-files",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
    except (ConnectionError, Timeout, requests.exceptions.ConnectionError) as error:
        raise RetryError(
            f"Connectivity error. Try reaching Roboflow API in browser: {API_BASE_URL}"
        ) from error
    handle_response_errors(response=response, operation_name="register images shard")
    try:
        response_data = response.json()
        upload_url, extension_headers = (
            response_data["uploadURL"],
            response_data["extensionHeaders"],
        )
    except (ValueError, KeyError) as error:
        raise RFAPICallError("Could not decode Roboflow API response.") from error
    try:
        upload_file_to_cloud(
            file_path=archive_path,
            url=upload_url,
            headers=extension_headers,
        )
    except Exception as error:
        raise RoboflowCloudCommandError(
            f"Could not upload shard to Roboflow Data staging."
        ) from error


def upload_file_to_cloud(
    file_path: str,
    url: str,
    headers: Dict[str, str],
) -> None:
    with open(file_path, "rb") as f:
        response = requests.put(url, headers=headers, data=f.read())
        response.raise_for_status()


def display_batch_count(batch_id: str, api_key: Optional[str]) -> None:
    workspace = get_workspace(api_key=api_key)
    batch_details = find_batch_by_id(
        workspace=workspace, batch_id=batch_id, api_key=api_key
    )
    if batch_details.batch_type in {"simple-batch", "sharded-batch"}:
        count = get_batch_count(workspace=workspace, batch_id=batch_id, api_key=api_key)
        print(f"Elements in batch {batch_id}: {count}")
        return None
    elif batch_details.batch_type == "multipart-batch":
        parts = list_multipart_batch_parts(
            workspace=workspace, batch_id=batch_id, api_key=api_key
        )
        for part in parts:
            count = get_batch_count(
                workspace=workspace,
                batch_id=batch_id,
                api_key=api_key,
                part_name=part.part_name,
            )
            print(f"Elements in batch {batch_id} - part: {part.part_name}: {count}")
        return None
    raise RoboflowCloudCommandError(
        f"Batch content count not implemented for batch type: {batch_details.batch_type}"
    )


@backoff.on_exception(
    backoff.constant,
    exception=RetryError,
    max_tries=3,
    interval=1,
)
def get_batch_count(
    workspace: str,
    batch_id: str,
    api_key: Optional[str],
    part_name: Optional[str] = None,
) -> int:
    params = {}
    if api_key is not None:
        params["api_key"] = api_key
    if part_name is not None:
        params["partName"] = part_name
    try:
        response = requests.get(
            f"{API_BASE_URL}/data-staging/v1/external/{workspace}/batches/{batch_id}/count",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
    except (ConnectionError, Timeout, requests.exceptions.ConnectionError):
        raise RetryError(
            f"Connectivity error. Try reaching Roboflow API in browser: {API_BASE_URL}"
        )
    handle_response_errors(response=response, operation_name="list batches")
    try:
        return response.json()["count"]
    except (ValueError, KeyError) as error:
        raise RFAPICallError("Could not decode Roboflow API response.") from error


def display_batch_content(batch_id: str, api_key: Optional[str]) -> None:
    workspace = get_workspace(api_key=api_key)
    selected_batch = find_batch_by_id(
        workspace=workspace, batch_id=batch_id, api_key=api_key
    )
    if selected_batch.batch_type == "simple-batch":
        batch_content = list_batch_content(
            workspace=workspace,
            batch_id=batch_id,
            api_key=api_key,
        )
        display_list(batch_content)
        return None
    if selected_batch.batch_type == "sharded-batch":
        page = 0
        while True:
            batch_content = list_batch_content(
                workspace=workspace,
                batch_id=batch_id,
                api_key=api_key,
                page=page,
            )
            if not batch_content:
                break
            display_list(batch_content)
            page += 1
        return None
    if selected_batch.batch_type == "multipart-batch":
        parts = list_multipart_batch_parts(
            workspace=workspace, batch_id=batch_id, api_key=api_key
        )
        for part in parts:
            print(f"Part: {part.part_name} - content: {part.content_type}")
            if part.part_type == "simple-part":
                batch_content = list_batch_content(
                    workspace=workspace,
                    batch_id=batch_id,
                    api_key=api_key,
                    part_name=part.part_name,
                )
                display_list(batch_content)
            else:
                page = 0
                while True:
                    batch_content = list_batch_content(
                        workspace=workspace,
                        batch_id=batch_id,
                        api_key=api_key,
                        part_name=part.part_name,
                        page=page,
                    )
                    if not batch_content:
                        break
                    display_list(batch_content)
                    page += 1
        return None
    raise RoboflowCloudCommandError(
        f"Batch content display not implemented for batch type: {selected_batch.batch_type}"
    )


def find_batch_by_id(
    workspace: str, batch_id: str, api_key: Optional[str]
) -> BatchDetails:
    batches = list_batches(workspace=workspace, api_key=api_key)
    selected_batch_candidates = [b for b in batches if b.batch_id == batch_id]
    if not selected_batch_candidates:
        raise RoboflowCloudCommandError(f"Cannot find batch wih id: {batch_id}")
    return selected_batch_candidates[0]


def display_list(value: list) -> None:
    for element in value:
        print(element)


@backoff.on_exception(
    backoff.constant,
    exception=RetryError,
    max_tries=3,
    interval=1,
)
def list_batch_content(
    workspace: str,
    batch_id: str,
    api_key: Optional[str],
    page: Optional[int] = None,
    part_name: Optional[str] = None,
) -> List[str]:
    params = {}
    if api_key is not None:
        params["api_key"] = api_key
    if page is not None:
        params["page"] = page
    if part_name is not None:
        params["partName"] = part_name
    try:
        response = requests.get(
            f"{API_BASE_URL}/data-staging/v1/external/{workspace}/batches/{batch_id}/list",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
    except (ConnectionError, Timeout, requests.exceptions.ConnectionError):
        raise RetryError(
            f"Connectivity error. Try reaching Roboflow API in browser: {API_BASE_URL}"
        )
    handle_response_errors(response=response, operation_name="list batches")
    try:
        return response.json()["batchContent"]
    except (ValueError, KeyError) as error:
        raise RFAPICallError("Could not decode Roboflow API response.") from error


def display_batch_shards_statuses(
    batch_id: str,
    api_key: Optional[str],
    page: int = 0,
    page_size: int = 25,
) -> None:
    workspace = get_workspace(api_key=api_key)
    all_shards_ids = list_batch_shards(
        workspace=workspace, batch_id=batch_id, api_key=api_key
    )
    start = page * page_size
    end = start + page_size
    all_pages = math.ceil(len(all_shards_ids) / page_size)
    selected_shards_ids = all_shards_ids[start:end]
    get_shard_status_closure = partial(
        lambda s, w, b, a: get_shard_status(
            workspace=w, batch_id=b, shard_id=s, api_key=a
        ),
        w=workspace,
        b=batch_id,
        a=api_key,
    )
    with ThreadPool() as pool:
        shards_details = pool.map(get_shard_status_closure, selected_shards_ids)
    console = Console()
    table = Table(title=f"Shards statuses ({page + 1}/{all_pages})")
    table.add_column("ID", justify="center", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center")
    table.add_column("Type", justify="center")
    table.add_column("Is Terminal", justify="center")
    table.add_column("Timestamp", justify="center")
    table.add_column("Objects #", justify="center")
    for shard_details in shards_details:
        is_terminal = "yes" if shard_details.is_terminal else "no"
        status_type = shard_details.status_type
        if "error" in status_type.lower():
            status_type = f"[bold red]{status_type}[/bold red]"
        elif "info" in status_type.lower():
            status_type = f"[bold steel_blue]{status_type}[/bold steel_blue]"
        elif "success" in status_type.lower():
            status_type = f"[bold green]{status_type}[/bold green]"
        event_timestamp = shard_details.event_timestamp.isoformat()
        table.add_row(
            shard_details.shard_id,
            shard_details.status_name,
            status_type,
            is_terminal,
            event_timestamp,
            str(shard_details.shard_objects_count),
        )
    console.print(table)


@backoff.on_exception(
    backoff.constant,
    exception=RetryError,
    max_tries=3,
    interval=1,
)
def list_batch_shards(
    workspace: str,
    batch_id: str,
    api_key: Optional[str],
) -> List[str]:
    params = {}
    if api_key is not None:
        params["api_key"] = api_key
    try:
        response = requests.get(
            f"{API_BASE_URL}/data-staging/v1/external/{workspace}/batches/{batch_id}/shards",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
    except (ConnectionError, Timeout, requests.exceptions.ConnectionError):
        raise RetryError(
            f"Connectivity error. Try reaching Roboflow API in browser: {API_BASE_URL}"
        )
    handle_response_errors(response=response, operation_name="list batches")
    try:
        return response.json()["shardsIds"]
    except (ValueError, KeyError) as error:
        raise RFAPICallError("Could not decode Roboflow API response.") from error


@backoff.on_exception(
    backoff.constant,
    exception=RetryError,
    max_tries=3,
    interval=1,
)
def get_shard_status(
    workspace: str,
    batch_id: str,
    shard_id: str,
    api_key: Optional[str],
) -> ShardDetails:
    params = {}
    if api_key is not None:
        params["api_key"] = api_key
    try:
        response = requests.get(
            f"{API_BASE_URL}/data-staging/v1/external/{workspace}/batches/{batch_id}/shards/{shard_id}",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
    except (ConnectionError, Timeout, requests.exceptions.ConnectionError):
        raise RetryError(
            f"Connectivity error. Try reaching Roboflow API in browser: {API_BASE_URL}"
        )
    handle_response_errors(response=response, operation_name="list batches")
    try:
        return ShardDetails.model_validate(response.json()["shardStatus"])
    except (ValueError, KeyError) as error:
        raise RFAPICallError("Could not decode Roboflow API response.") from error


@backoff.on_exception(
    backoff.constant,
    exception=RetryError,
    max_tries=3,
    interval=1,
)
def list_multipart_batch_parts(
    workspace: str,
    batch_id: str,
    api_key: Optional[str],
) -> List[MultipartBatchPartMetadata]:
    params = {}
    if api_key is not None:
        params["api_key"] = api_key
    try:
        response = requests.get(
            f"{API_BASE_URL}/data-staging/v1/external/{workspace}/batches/{batch_id}/parts",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
    except (ConnectionError, Timeout, requests.exceptions.ConnectionError):
        raise RetryError(
            f"Connectivity error. Try reaching Roboflow API in browser: {API_BASE_URL}"
        )
    handle_response_errors(
        response=response, operation_name="list multipart batch parts"
    )
    try:
        return ListMultipartBatchPartsResponse.model_validate(
            response.json()
        ).batch_parts
    except (ValueError, KeyError) as error:
        raise RFAPICallError("Could not decode Roboflow API response.") from error
