import json
import os
import tarfile
import tempfile
from datetime import datetime, timedelta
from functools import partial
from multiprocessing.pool import Pool, ThreadPool
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse
from uuid import uuid4

import backoff
import requests
import supervision as sv
from requests import Timeout
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, TaskID
from rich.table import Table
from rich.text import Text
from tqdm import tqdm

from inference_cli.lib.env import API_BASE_URL
from inference_cli.lib.roboflow_cloud.common import (
    get_workspace,
    handle_response_errors,
)
from inference_cli.lib.roboflow_cloud.config import (
    MAX_DOWNLOAD_THREADS,
    MAX_SHARD_SIZE,
    MAX_SHARDS_UPLOAD_PROCESSES,
    MIN_IMAGES_TO_FORM_SHARD,
    REQUEST_TIMEOUT,
    SUGGESTED_MAX_VIDEOS_IN_BATCH,
)
from inference_cli.lib.roboflow_cloud.data_staging.entities import (
    BatchExportResponse,
    BatchMetadata,
    ListBatchesResponse,
    ListMultipartBatchPartsResponse,
    MultipartBatchPartMetadata,
    ShardDetails,
)
from inference_cli.lib.roboflow_cloud.errors import (
    RetryError,
    RFAPICallError,
    RoboflowCloudCommandError,
)
from inference_cli.lib.utils import (
    create_batches,
    get_all_images_in_directory,
    get_all_videos_in_directory,
)


def display_batches(
    api_key: Optional[str], pages: int, page_size: Optional[int]
) -> None:
    workspace = get_workspace(api_key=api_key)
    batches = list_batches(
        workspace=workspace, api_key=api_key, pages=pages, page_size=page_size
    )
    if len(batches) == 0:
        print("No batches found")
        return None
    console = Console()
    pages_status = f"(pages {pages} of ...)"
    table = Table(
        title=f"Roboflow Data Staging Batches {pages_status}", show_lines=True
    )
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


def list_batches(
    workspace: str,
    api_key: Optional[str],
    pages: Optional[int],
    page_size: Optional[int],
) -> List[BatchMetadata]:
    next_page_token = None
    pages_fetched = 0
    results = []
    while True:
        if pages is not None and pages_fetched >= pages:
            return results
        listing_page = get_workspace_batches_list_page(
            workspace=workspace,
            api_key=api_key,
            page_size=page_size,
            next_page_token=next_page_token,
        )
        results.extend(listing_page.batches)
        next_page_token = listing_page.next_page_token
        if next_page_token is None:
            break
        pages_fetched += 1
    return results


@backoff.on_exception(
    backoff.constant,
    exception=RetryError,
    max_tries=3,
    interval=1,
)
def get_workspace_batches_list_page(
    workspace: str,
    api_key: Optional[str],
    page_size: Optional[int],
    next_page_token: Optional[str] = None,
) -> ListBatchesResponse:
    params = {}
    if api_key is not None:
        params["api_key"] = api_key
    if page_size is not None:
        params["pageSize"] = page_size
    if next_page_token is not None:
        params["nextPageToken"] = next_page_token
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
        return ListBatchesResponse.model_validate(response.json())
    except ValueError as error:
        raise RFAPICallError("Could not decode Roboflow API response.") from error


def create_images_batch_from_directory(
    directory: str,
    batch_id: str,
    api_key: Optional[str],
    batch_name: Optional[str],
) -> None:
    workspace = get_workspace(api_key=api_key)
    images_paths = get_all_images_in_directory(input_directory=directory)
    if len(images_paths) > MIN_IMAGES_TO_FORM_SHARD:
        upload_images_to_sharded_batch(
            images_paths=images_paths,
            workspace=workspace,
            batch_id=batch_id,
            api_key=api_key,
            batch_name=batch_name,
        )
        return None
    upload_images_to_simple_batch(
        images_paths=images_paths,
        workspace=workspace,
        batch_id=batch_id,
        api_key=api_key,
        batch_name=batch_name,
    )


def create_videos_batch_from_directory(
    directory: str,
    batch_id: str,
    api_key: Optional[str],
    batch_name: Optional[str],
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
            batch_name=batch_name,
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
    batch_name: Optional[str],
) -> None:
    upload_image_closure = partial(
        upload_image,
        workspace=workspace,
        batch_id=batch_id,
        api_key=api_key,
        batch_name=batch_name,
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
    batch_name: Optional[str],
) -> None:
    shards = list(create_batches(sequence=images_paths, batch_size=MAX_SHARD_SIZE))
    upload_images_shard_closure = partial(
        pack_and_upload_images_shard,
        workspace=workspace,
        batch_id=batch_id,
        api_key=api_key,
        batch_name=batch_name,
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
    batch_name: Optional[str],
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
            batch_name=batch_name,
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
    image_path: str,
    workspace: str,
    batch_id: str,
    api_key: Optional[str],
    batch_name: Optional[str] = None,
) -> None:
    params = {}
    if api_key is not None:
        params["api_key"] = api_key
    if batch_name is not None:
        params["displayName"] = batch_name
    image_file_name = os.path.basename(image_path)
    params["fileName"] = image_file_name
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
    handle_response_errors(response=response, operation_name="upload image")


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
    batch_name: Optional[str] = None,
) -> None:
    params = {}
    if api_key is not None:
        params["api_key"] = api_key
    image_file_name = os.path.basename(video_path)
    params["fileName"] = image_file_name
    if batch_name is not None:
        params["displayName"] = batch_name
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
            response_data["signedURLDetails"]["uploadURL"],
            response_data["signedURLDetails"]["extensionHeaders"],
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
    archive_path: str,
    workspace: str,
    batch_id: str,
    api_key: Optional[str],
    batch_name: Optional[str],
) -> None:
    # This flow is over-simplified - we should request shard at first and only then pack into archive
    params = {}
    if api_key is not None:
        params["api_key"] = api_key
    if batch_name is not None:
        params["displayName"] = batch_name
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
            response_data["signedURLDetails"]["uploadURL"],
            response_data["signedURLDetails"]["extensionHeaders"],
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
    handle_response_errors(response=response, operation_name="get batch count")
    try:
        return response.json()["count"]
    except (ValueError, KeyError) as error:
        raise RFAPICallError("Could not decode Roboflow API response.") from error


CONTENT_TYPE_TO_ICON = {
    "images": " 🖼 ",
    "videos": " 🎬 ",
    "metadata": " 📋 ",
    "archives": " 🗃 ",
}


def display_batch_details(batch_id: str, api_key: Optional[str]) -> None:
    workspace = get_workspace(api_key=api_key)
    metadata = get_batch_metadata(
        workspace=workspace, batch_id=batch_id, api_key=api_key
    )
    console = Console()
    heading_text = Text(
        f"Staging Batch Overview [id={batch_id}]",
        style="bold grey89 on steel_blue",
        justify="center",
    )
    heading_panel = Panel(heading_text, expand=True, border_style="steel_blue")
    console.print(heading_panel)
    table = Table(show_lines=True, expand=True)
    table.add_column("Property", justify="left", style="cyan", no_wrap=True)
    table.add_column("Value", justify="full", overflow="ellipsis")
    identifier = f"{metadata.display_name} [id={metadata.batch_id}]"
    table.add_row("Name", identifier)
    content_type_icon = CONTENT_TYPE_TO_ICON.get(metadata.batch_content_type, "")
    table.add_row(
        "Type",
        f"[bold green]{metadata.batch_type}[/bold green] (of "
        f"[bold deep_sky_blue1]{metadata.batch_content_type}[/bold deep_sky_blue1]{content_type_icon})",
    )
    table.add_row("Created At", metadata.created_date.strftime("%d %b %Y"))
    is_short_to_expiry = (metadata.expiry_date - datetime.utcnow().date()) < timedelta(
        days=3
    )
    expiry_str = metadata.expiry_date.strftime("%d %b %Y")
    if is_short_to_expiry:
        expiry_str = f"[bold red]{expiry_str}[/bold red]"
    else:
        expiry_str = f"[bold green]{expiry_str}[/bold green]"
    table.add_row("Expiring At", expiry_str)
    if metadata.batch_type in {"sharded-batch", "simple-batch"}:
        batch_count = get_batch_count(
            workspace=workspace,
            batch_id=metadata.batch_id,
            api_key=api_key,
        )
        table.add_row("Files Count", str(batch_count))
    console.print(table)
    if metadata.batch_type == "multipart-batch":
        return display_multipart_batch_details(
            workspace=workspace, metadata=metadata, api_key=api_key, console=console
        )


def display_multipart_batch_details(
    workspace: str,
    metadata: BatchMetadata,
    api_key: str,
    console: Console,
) -> None:
    parts = list_multipart_batch_parts(
        workspace=workspace, batch_id=metadata.batch_id, api_key=api_key
    )
    parts_table_content = []
    for part in parts:
        part_count = get_batch_count(
            workspace=workspace,
            batch_id=metadata.batch_id,
            api_key=api_key,
            part_name=part.part_name,
        )
        icon = CONTENT_TYPE_TO_ICON.get(part.content_type, " ")
        content_type = (
            f"[bold deep_sky_blue1]{part.content_type}[/bold deep_sky_blue1]{icon}"
        )
        parts_table_content.append((part.part_name, content_type, str(part_count)))
    table = Table(title=f"Parts with content", show_lines=True)
    table.add_column("Name", justify="left", style="cyan", no_wrap=True)
    table.add_column("Content Type", justify="center", overflow="ellipsis")
    table.add_column("Files Count", justify="center", overflow="ellipsis")
    for row in parts_table_content:
        table.add_row(*row)
    console.print(table)


@backoff.on_exception(
    backoff.constant,
    exception=RetryError,
    max_tries=3,
    interval=1,
)
def get_batch_metadata(
    workspace: str,
    batch_id: str,
    api_key: Optional[str],
) -> BatchMetadata:
    params = {}
    if api_key is not None:
        params["api_key"] = api_key
    try:
        response = requests.get(
            f"{API_BASE_URL}/data-staging/v1/external/{workspace}/batches/{batch_id}",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
    except (ConnectionError, Timeout, requests.exceptions.ConnectionError):
        raise RetryError(
            f"Connectivity error. Try reaching Roboflow API in browser: {API_BASE_URL}"
        )
    handle_response_errors(response=response, operation_name="list batch content")
    try:
        return BatchMetadata.model_validate(response.json()["batch"])
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


def export_data(batch_id: str, api_key: Optional[str], target_directory: str) -> None:
    workspace = get_workspace(api_key=api_key)
    metadata = get_batch_metadata(
        workspace=workspace, batch_id=batch_id, api_key=api_key
    )
    if metadata.batch_type == "multipart-batch":
        return export_multipart_batch(
            workspace=workspace,
            batch_id=batch_id,
            api_key=api_key,
            target_directory=target_directory,
        )
    return export_simple_batch(
        workspace=workspace,
        batch_id=batch_id,
        api_key=api_key,
        target_directory=target_directory,
    )


def export_multipart_batch(
    workspace: str, batch_id: str, api_key: str, target_directory: str
) -> None:
    parts = list_multipart_batch_parts(
        workspace=workspace, batch_id=batch_id, api_key=api_key
    )
    os.makedirs(target_directory, exist_ok=True)
    next_page_token, covered_parts = get_export_progress(
        workspace=workspace, batch_id=batch_id, target_dir=target_directory
    )
    for part in parts:
        if part.part_name in covered_parts:
            continue
        export_part(
            workspace=workspace,
            batch_id=batch_id,
            api_key=api_key,
            target_directory=target_directory,
            part_name=part.part_name,
            covered_parts=covered_parts,
            next_page_token=next_page_token,
        )
        covered_parts.add(part.part_name)
        next_page_token = None


def export_simple_batch(
    workspace: str, batch_id: str, api_key: str, target_directory: str
) -> None:
    os.makedirs(target_directory, exist_ok=True)
    total_files = get_batch_count(
        workspace=workspace, batch_id=batch_id, api_key=api_key
    )
    next_page_token, _ = get_export_progress(
        workspace=workspace, batch_id=batch_id, target_dir=target_directory
    )
    with Progress() as progress_bar:
        task = progress_bar.add_task(
            f"Downloading {total_files} files...", total=total_files
        )
        while True:
            export_response = export_batch_data(
                workspace=workspace,
                batch_id=batch_id,
                api_key=api_key,
                next_page_token=next_page_token,
            )
            next_page_token = export_response.next_page_token
            pull_urls_to_directory(
                urls=export_response.urls,
                target_directory=target_directory,
                progress_bar=progress_bar,
                task=task,
            )
            if next_page_token is None:
                break
            denote_export_progress(
                workspace=workspace,
                batch_id=batch_id,
                target_dir=target_directory,
                next_page_token=next_page_token,
            )
    denote_export_progress(
        workspace=workspace,
        batch_id=batch_id,
        target_dir=target_directory,
        next_page_token="<END>",
    )
    return None


def export_part(
    workspace: str,
    batch_id: str,
    api_key: str,
    target_directory: str,
    part_name: str,
    covered_parts: Set[str],
    next_page_token: Optional[str] = None,
) -> None:
    part_target_directory = os.path.join(target_directory, part_name)
    os.makedirs(part_target_directory, exist_ok=True)
    total_files = get_batch_count(
        workspace=workspace,
        batch_id=batch_id,
        api_key=api_key,
        part_name=part_name,
    )
    with Progress() as progress_bar:
        task = progress_bar.add_task(
            f"Downloading part {part_name} - {total_files} files...", total=total_files
        )
        while True:
            export_response = export_batch_data(
                workspace=workspace,
                batch_id=batch_id,
                api_key=api_key,
                part_name=part_name,
                next_page_token=next_page_token,
            )
            next_page_token = export_response.next_page_token
            pull_urls_to_directory(
                urls=export_response.urls,
                target_directory=part_target_directory,
                progress_bar=progress_bar,
                task=task,
            )
            if next_page_token is None:
                break
            denote_export_progress(
                workspace=workspace,
                batch_id=batch_id,
                target_dir=target_directory,
                next_page_token=next_page_token,
                part_names=covered_parts,
            )
    denote_export_progress(
        workspace=workspace,
        batch_id=batch_id,
        target_dir=target_directory,
        next_page_token="<END>",
        part_names=covered_parts,
    )
    return None


def pull_urls_to_directory(
    urls: List[str],
    target_directory: str,
    progress_bar: Progress,
    task: TaskID,
) -> None:
    pull_url_to_directory_closure = partial(
        pull_url_to_directory,
        target_directory=target_directory,
    )
    with ThreadPool(processes=MAX_DOWNLOAD_THREADS) as pool:
        for _ in pool.imap(pull_url_to_directory_closure, urls):
            progress_bar.update(task, advance=1)
    return None


def pull_url_to_directory(url: str, target_directory: str) -> None:
    parsed_url = urlparse(url)
    file_name = os.path.basename(parsed_url.path)
    is_archive = file_name.endswith(".tar.gz") or file_name.endswith(".tar")
    if not is_archive:
        _ = pull_file_to_directory(
            url=url, target_directory=target_directory, file_name=file_name
        )
        return None
    with tempfile.TemporaryDirectory() as tmp_dir:
        download_and_unpack_archive(
            url=url,
            file_name=file_name,
            pull_dir=tmp_dir,
            target_dir=target_directory,
        )


def denote_export_progress(
    workspace: str,
    batch_id: str,
    target_dir: str,
    next_page_token: Optional[str],
    part_names: Set[str] = None,
) -> None:
    log_path = _get_export_progress_log_path(target_dir=target_dir)
    if part_names is None:
        part_names = set()
    log_content = {
        "workspace": workspace,
        "batch_id": batch_id,
        "next_page_token": next_page_token,
        "part_names": list(part_names),
    }
    with open(log_path, "w") as f:
        json.dump(log_content, f)


def get_export_progress(
    workspace: str,
    batch_id: str,
    target_dir: str,
) -> Tuple[Optional[str], Set[str]]:
    log_path = _get_export_progress_log_path(target_dir=target_dir)
    if not os.path.exists(log_path):
        return None, set()
    with open(log_path) as f:
        log_content = json.load(f)
    if any(
        k not in log_content
        for k in ["workspace", "batch_id", "next_page_token", "part_names"]
    ):
        raise RuntimeError(f"Log file saved under {log_path} is malformed")
    if log_content["workspace"] != workspace or log_content["batch_id"] != batch_id:
        raise RuntimeError(
            "Attempted to export batch to directory which was a host of another export"
        )
    if log_content["next_page_token"] == "<END>":
        raise RuntimeError("Nothing to download - content reported as fully fetched")
    return log_content["next_page_token"], set(log_content["part_names"])


def _get_export_progress_log_path(target_dir: str) -> str:
    return os.path.join(target_dir, ".export_progress.json")


@backoff.on_exception(
    backoff.constant,
    exception=Exception,
    max_tries=3,
    interval=1,
)
def pull_file_to_directory(url: str, target_directory: str, file_name: str) -> str:
    response = requests.get(url)
    response.raise_for_status()
    target_path = os.path.join(target_directory, file_name)
    with open(target_path, "wb") as f:
        f.write(response.content)
    return target_path


def download_and_unpack_archive(
    url: str, file_name: str, pull_dir: str, target_dir: str
) -> None:
    archive_path = pull_file_to_directory(
        url=url,
        file_name=file_name,
        target_directory=pull_dir,
    )
    is_compressed_archive = file_name.endswith(".tar.gz")
    mode = "r"
    if is_compressed_archive:
        mode = "r:gz"
    with tarfile.open(archive_path, mode) as tar_file:
        for member in tar_file.getmembers():
            tar_file.extract(member, target_dir)


@backoff.on_exception(
    backoff.constant,
    exception=RetryError,
    max_tries=3,
    interval=1,
)
def export_batch_data(
    workspace: str,
    batch_id: str,
    api_key: Optional[str],
    page_size: Optional[int] = None,
    next_page_token: Optional[str] = None,
    part_name: Optional[str] = None,
) -> BatchExportResponse:
    params = {}
    if api_key is not None:
        params["api_key"] = api_key
    if page_size is not None:
        params["pageSize"] = page_size
    if next_page_token is not None:
        params["nextPageToken"] = next_page_token
    if part_name is not None:
        params["partName"] = part_name
    try:
        response = requests.get(
            f"{API_BASE_URL}/data-staging/v1/external/{workspace}/batches/{batch_id}/export",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
    except (ConnectionError, Timeout, requests.exceptions.ConnectionError):
        raise RetryError(
            f"Connectivity error. Try reaching Roboflow API in browser: {API_BASE_URL}"
        )
    handle_response_errors(response=response, operation_name="list batches")
    try:
        return BatchExportResponse.model_validate(response.json())
    except ValueError as error:
        raise RFAPICallError("Could not decode Roboflow API response.") from error
