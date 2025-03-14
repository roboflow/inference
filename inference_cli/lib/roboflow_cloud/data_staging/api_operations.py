import json
import os
import tarfile
import tempfile
from datetime import datetime, timedelta, timezone
from functools import partial
from io import BytesIO
from multiprocessing.pool import Pool, ThreadPool
from threading import Lock
from typing import Dict, Generator, List, Optional, Set, TextIO, Tuple, Union
from urllib.parse import urlparse
from uuid import uuid4

import backoff
import requests
import supervision as sv
from pydantic import BaseModel, ValidationError
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
    read_jsonl_file,
)
from inference_cli.lib.roboflow_cloud.config import (
    MAX_DOWNLOAD_THREADS,
    MAX_IMAGE_REFERENCES_IN_INGEST_REQUEST,
    MAX_SHARD_SIZE,
    MAX_SHARDS_UPLOAD_PROCESSES,
    MIN_IMAGES_TO_FORM_SHARD,
    REQUEST_TIMEOUT,
    SUGGESTED_MAX_VIDEOS_IN_BATCH,
)
from inference_cli.lib.roboflow_cloud.data_staging.entities import (
    BatchExportResponse,
    BatchMetadata,
    DownloadLogEntry,
    FileMetadata,
    ImageReferencesIngestResponse,
    ListBatchesResponse,
    ListBatchResponse,
    ListMultipartBatchPartsResponse,
    MultipartBatchPartMetadata,
    PageOfBatchShardsStatuses,
    ShardDetails,
    VideoReferencesIngestResponse,
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


def display_batches(api_key: str, pages: int, page_size: Optional[int]) -> None:
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
        if (batch.expiry_date - datetime.now(timezone.utc)) <= timedelta(days=3):
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
    api_key: str,
    pages: Optional[int],
    page_size: Optional[int],
) -> List[BatchMetadata]:
    return list(
        iterate_batches(
            workspace=workspace,
            api_key=api_key,
            pages=pages,
            page_size=page_size,
        )
    )


def iterate_batches(
    workspace: str,
    api_key: str,
    pages: Optional[int],
    page_size: Optional[int],
) -> Generator[BatchMetadata, None, None]:
    next_page_token = None
    pages_fetched = 0
    while True:
        if pages is not None and pages_fetched >= pages:
            break
        listing_page = get_workspace_batches_list_page(
            workspace=workspace,
            api_key=api_key,
            page_size=page_size,
            next_page_token=next_page_token,
        )
        yield from listing_page.batches
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
def get_workspace_batches_list_page(
    workspace: str,
    api_key: str,
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


def get_batch_content(
    batch_id: str,
    api_key: str,
    part_names: Optional[Set[str]] = None,
    limit: Optional[int] = None,
    output_file: Optional[str] = None,
) -> None:
    workspace = get_workspace(api_key=api_key)
    metadata = get_batch_metadata(
        workspace=workspace, batch_id=batch_id, api_key=api_key
    )
    if metadata.batch_type == "multipart-batch":
        return get_content_of_multipart_batch(
            workspace=workspace,
            batch_id=batch_id,
            api_key=api_key,
            part_names=part_names,
            limit=limit,
            output_file=output_file,
        )
    if part_names is not None:
        raise ValueError(
            f"Requested listing of parts {part_names}, but batch is not multipart-batch."
        )
    return get_content_of_simple_or_sharded_batch(
        workspace=workspace,
        batch_id=batch_id,
        api_key=api_key,
        limit=limit,
        output_file=output_file,
    )


def get_content_of_simple_or_sharded_batch(
    workspace: str,
    batch_id: str,
    api_key: str,
    limit: Optional[int] = None,
    output_file: Optional[str] = None,
) -> None:
    if output_file:
        saver = MetadataSaver.init(file_path=output_file)
        on_new_metadata = saver.save_metadata
    else:
        on_new_metadata = lambda m: print(m.model_dump())
    for metadata in iterate_batch_content(
        workspace=workspace,
        batch_id=batch_id,
        api_key=api_key,
        limit=limit,
    ):
        on_new_metadata(metadata)


def get_content_of_multipart_batch(
    workspace: str,
    batch_id: str,
    api_key: str,
    part_names: Optional[Set[str]] = None,
    limit: Optional[int] = None,
    output_file: Optional[str] = None,
) -> None:
    parts = list_multipart_batch_parts(
        workspace=workspace, batch_id=batch_id, api_key=api_key
    )
    filtered_parts = (
        parts if part_names is None else [p for p in parts if p.part_name in part_names]
    )
    if output_file:
        saver = MetadataSaver.init(file_path=output_file)
        on_new_metadata = saver.save_metadata
    else:
        on_new_metadata = lambda m: print(m.model_dump())
    for part in filtered_parts:
        part_elements_listed = 0
        for metadata in iterate_batch_content(
            workspace=workspace,
            batch_id=batch_id,
            api_key=api_key,
            part_name=part.part_name,
            limit=limit,
        ):
            on_new_metadata(metadata)
            part_elements_listed += 1
        if limit is not None:
            limit = max(0, limit - part_elements_listed)


class MetadataSaver:

    @classmethod
    def init(cls, file_path: str) -> "MetadataSaver":
        abs_path = os.path.abspath(file_path)
        parent_dir = os.path.dirname(abs_path)
        os.makedirs(parent_dir, exist_ok=True)
        file_handler = open(abs_path, "w")
        return cls(file_handler=file_handler)

    def __init__(self, file_handler: TextIO):
        self._file_handler = file_handler

    def save_metadata(self, metadata: Union[dict, BaseModel]) -> None:
        if isinstance(metadata, dict):
            serialized_metadata = json.dumps(metadata)
        else:
            serialized_metadata = metadata.model_dump_json()
        self._file_handler.write(f"{serialized_metadata}\n")

    def __del__(self):
        self._file_handler.close()


def list_batch_content(
    workspace: str,
    batch_id: str,
    api_key: str,
    part_name: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[FileMetadata]:
    return list(
        iterate_batch_content(
            workspace=workspace,
            batch_id=batch_id,
            api_key=api_key,
            part_name=part_name,
            limit=limit,
        )
    )


def iterate_batch_content(
    workspace: str,
    batch_id: str,
    api_key: str,
    part_name: Optional[str] = None,
    limit: Optional[int] = None,
) -> Generator[FileMetadata, None, None]:
    if limit is not None and limit <= 0:
        return None
    items_listed = 0
    next_page_token = None
    while limit is None or items_listed < limit:
        listing_page = get_one_page_of_batch_content(
            workspace=workspace,
            batch_id=batch_id,
            api_key=api_key,
            next_page_token=next_page_token,
            part_name=part_name,
        )
        for element in listing_page.files_metadata:
            if limit is not None and items_listed >= limit:
                break
            items_listed += 1
            yield element
        next_page_token = listing_page.next_page_token
        if next_page_token is None:
            break


@backoff.on_exception(
    backoff.constant,
    exception=RetryError,
    max_tries=3,
    interval=1,
)
def get_one_page_of_batch_content(
    workspace: str,
    batch_id: str,
    api_key: str,
    page_size: Optional[int] = None,
    next_page_token: Optional[str] = None,
    part_name: Optional[str] = None,
) -> ListBatchResponse:
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
            f"{API_BASE_URL}/data-staging/v1/external/{workspace}/batches/{batch_id}/list",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
    except (ConnectionError, Timeout, requests.exceptions.ConnectionError):
        raise RetryError(
            f"Connectivity error. Try reaching Roboflow API in browser: {API_BASE_URL}"
        )
    handle_response_errors(response=response, operation_name="list batche")
    try:
        return ListBatchResponse.model_validate(response.json())
    except ValueError as error:
        raise RFAPICallError("Could not decode Roboflow API response.") from error


class DataImportLog:

    @classmethod
    def init(cls, sources_dir: str, batch_id: str) -> "DataImportLog":
        file_path = os.path.join(sources_dir, f".import_log-{batch_id}")
        if not os.path.exists(file_path):
            file_descriptor = open(file_path, "w+")
        else:
            file_descriptor = open(file_path, "r+")
        log_content = {
            line.strip()
            for line in file_descriptor.readlines()
            if len(line.strip()) > 0
        }
        return cls(log_file=file_descriptor, log_content=log_content)

    def __init__(self, log_file: TextIO, log_content: Set[str]):
        self._log_file = log_file
        self._log_content = log_content
        self._lock = Lock()

    def is_file_recorded(self, path: str) -> bool:
        return path in self._log_content

    def record_files(self, paths: List[str]) -> None:
        with self._lock:
            for path in paths:
                self._log_file.write(f"{path}\n")
                self._log_content.add(path)

    def record_file(self, path: str) -> None:
        with self._lock:
            self._log_file.write(f"{path}\n")
            self._log_content.add(path)


def create_images_batch_from_directory(
    directory: str,
    batch_id: str,
    api_key: str,
    batch_name: Optional[str] = None,
    ingest_id: Optional[str] = None,
    notifications_url: Optional[str] = None,
) -> None:
    workspace = get_workspace(api_key=api_key)
    images_paths = get_all_images_in_directory(input_directory=directory)
    upload_log = DataImportLog.init(sources_dir=directory, batch_id=batch_id)
    deduplicated_images = [
        path for path in images_paths if not upload_log.is_file_recorded(path=path)
    ]
    if len(images_paths) > MIN_IMAGES_TO_FORM_SHARD:
        upload_images_to_sharded_batch(
            images_paths=deduplicated_images,
            workspace=workspace,
            batch_id=batch_id,
            api_key=api_key,
            batch_name=batch_name,
            upload_log=upload_log,
            ingest_id=ingest_id,
            notifications_url=notifications_url,
        )
        return None
    if notifications_url:
        print(f"Ingesting images to simple-batch - notification URL will be ignored.")
    upload_images_to_simple_batch(
        images_paths=deduplicated_images,
        workspace=workspace,
        batch_id=batch_id,
        api_key=api_key,
        batch_name=batch_name,
        upload_log=upload_log,
    )


def create_images_batch_from_references_file(
    references: str,
    batch_id: str,
    api_key: str,
    ingest_id: Optional[str] = None,
    batch_name: Optional[str] = None,
    notifications_url: Optional[str] = None,
    notification_categories: Optional[List[str]] = None,
) -> None:
    workspace = get_workspace(api_key=api_key)
    if references.startswith("http://"):
        raise ValueError("Only HTTPS links are allowed to point references.")
    if not references.startswith("https://"):
        return create_images_batch_from_local_references_file(
            workspace=workspace,
            references=references,
            batch_id=batch_id,
            api_key=api_key,
            ingest_id=ingest_id,
            batch_name=batch_name,
            notifications_url=notifications_url,
            notification_categories=notification_categories,
        )
    response = trigger_images_references_ingest(
        workspace=workspace,
        batch_id=batch_id,
        references=references,
        api_key=api_key,
        ingest_id=ingest_id,
        batch_name=batch_name,
        notifications_url=notifications_url,
        notification_categories=notification_categories,
    )
    print(f"Your ingest ID: {response.ingest_id}")
    if notifications_url:
        print(f"Monitor updates that will be sent to: {notifications_url}")
        print(
            f"You can also use `inference rf-cloud data-staging list-ingest-details --batch-id {batch_id}` command "
            f"to check progress."
        )
    else:
        print(
            f"Use `inference rf-cloud data-staging list-ingest-details --batch-id {batch_id}` "
            "command to watch the ingest progress. If you want automated updates - use `--notifications-url` option "
            "of this command."
        )


def create_images_batch_from_local_references_file(
    workspace: str,
    references: str,
    batch_id: str,
    api_key: str,
    ingest_id: Optional[str] = None,
    batch_name: Optional[str] = None,
    notifications_url: Optional[str] = None,
    notification_categories: Optional[List[str]] = None,
) -> None:
    try:
        references = read_jsonl_file(path=references)
        if not references:
            raise ValueError("Empty reference file")
    except (OSError, ValueError) as error:
        raise ValueError(
            "Could not decode references file - provide JSONL document that contain "
            "list of JSON documents with keys 'name' and 'url` to describe all files "
            'you want to ingest. Document format: {"name": "<your-file-name>", "url": '
            '"https://<signed-url-from-cloud>"}'
        ) from error
    ingest_parts = list(
        create_batches(
            sequence=references, batch_size=MAX_IMAGE_REFERENCES_IN_INGEST_REQUEST
        )
    )
    if len(ingest_parts) > 1:
        print(
            f"Your ingest exceeds {MAX_IMAGE_REFERENCES_IN_INGEST_REQUEST} files - we split the ingest "
            f"into {len(ingest_parts)} chunks."
        )
    for part_id, part in enumerate(ingest_parts):
        if ingest_id is not None and len(ingest_parts) > 1:
            ingest_id = f"{ingest_id}-{part_id}"
        response = trigger_images_references_ingest(
            workspace=workspace,
            batch_id=batch_id,
            references=part,
            api_key=api_key,
            ingest_id=ingest_id,
            batch_name=batch_name,
            notifications_url=notifications_url,
            notification_categories=notification_categories,
        )
        print(f"Your ingest ID: {response.ingest_id}")
        print(
            f"System will create the following shards in batch {batch_id}: {response.shard_ids}"
        )
        if response.duplicated:
            print(
                "Your request was deduplicated - either it was retried by CLI on initial "
                "request failure or you attempt to use the same references file you used in the past."
            )
    if notifications_url:
        print(f"Monitor updates that will be sent to: {notifications_url}")
        print(
            f"You can also use `inference rf-cloud data-staging list-ingest-details --batch-id {batch_id}` command "
            f"to check progress."
        )
    else:
        print(
            f"Use `inference rf-cloud data-staging list-ingest-details --batch-id {batch_id}` "
            "command to watch the ingest progress. If you want automated updates - use `--notifications-url` option "
            "of this command."
        )


@backoff.on_exception(
    backoff.constant,
    exception=RetryError,
    max_tries=3,
    interval=1,
)
def trigger_images_references_ingest(
    workspace: str,
    batch_id: str,
    references: Union[str, List[dict]],
    api_key: str,
    ingest_id: Optional[str] = None,
    batch_name: Optional[str] = None,
    notifications_url: Optional[str] = None,
    notification_categories: Optional[List[str]] = None,
) -> ImageReferencesIngestResponse:
    params = {}
    if api_key is not None:
        params["api_key"] = api_key
    if batch_name is not None:
        params["displayName"] = batch_name
    payload = {}
    if isinstance(references, list):
        payload["imageReferences"] = references
    else:
        payload["imageReferencesURL"] = references
    if ingest_id:
        payload["ingestId"] = ingest_id
    if notifications_url:
        payload["notificationsURL"] = notifications_url
    if notification_categories:
        payload["notificationCategories"] = notification_categories
    try:
        response = requests.post(
            f"{API_BASE_URL}/data-staging/v1/external/{workspace}/batches/{batch_id}/bulk-upload/image-references",
            params=params,
            timeout=2 * REQUEST_TIMEOUT,
            json=payload,
        )
    except (ConnectionError, Timeout, requests.exceptions.ConnectionError) as error:
        raise RetryError(
            f"Connectivity error. Try reaching Roboflow API in browser: {API_BASE_URL}"
        ) from error
    handle_response_errors(response=response, operation_name="trigger images ingest")
    try:
        response_data = response.json()
        return ImageReferencesIngestResponse.model_validate(response_data)
    except (ValidationError, ValueError, KeyError) as error:
        raise RFAPICallError("Could not decode Roboflow API response.") from error


def create_videos_batch_from_directory(
    directory: str,
    batch_id: str,
    api_key: str,
    batch_name: Optional[str],
) -> None:
    workspace = get_workspace(api_key=api_key)
    video_paths = get_all_videos_in_directory(input_directory=directory)
    video_paths = _filter_out_malformed_videos(video_paths=video_paths)
    upload_log = DataImportLog.init(sources_dir=directory, batch_id=batch_id)
    if len(video_paths) > SUGGESTED_MAX_VIDEOS_IN_BATCH:
        print(
            f"You try to upload {len(video_paths)} to single batch. "
            f"Suggested max size is: {SUGGESTED_MAX_VIDEOS_IN_BATCH}."
        )
    video_paths = [
        path for path in video_paths if not upload_log.is_file_recorded(path=path)
    ]
    for video in tqdm(video_paths, desc="Uploading video files..."):
        upload_video(
            video_path=video,
            workspace=workspace,
            batch_id=batch_id,
            api_key=api_key,
            upload_log=upload_log,
            batch_name=batch_name,
        )


def create_videos_batch_from_references_file(
    references: str,
    batch_id: str,
    api_key: str,
    ingest_id: Optional[str] = None,
    batch_name: Optional[str] = None,
    notifications_url: Optional[str] = None,
    notification_categories: Optional[List[str]] = None,
) -> None:
    workspace = get_workspace(api_key=api_key)
    if references.startswith("http://"):
        raise ValueError("Only HTTPS links are allowed to point references.")
    if not references.startswith("https://"):
        try:
            references = read_jsonl_file(path=references)
            if not references:
                raise ValueError("Empty reference file")
        except (OSError, ValueError) as error:
            raise ValueError(
                "Could not decode references file - provide JSONL document that contain "
                "list of JSON documents with keys 'name' and 'url` to describe all files "
                'you want to ingest. Document format: {"name": "<your-file-name>", "url": '
                '"https://<signed-url-from-cloud>"}'
            ) from error
    response = trigger_videos_references_ingest(
        workspace=workspace,
        batch_id=batch_id,
        references=references,
        api_key=api_key,
        ingest_id=ingest_id,
        batch_name=batch_name,
        notifications_url=notifications_url,
        notification_categories=notification_categories,
    )
    print(f"Your ingest ID: {response.ingest_id}")
    if response.duplicated:
        print(
            "Your request was deduplicated - either it was retried by CLI on initial "
            "request failure or you attempt to use the same references file you used in the past."
        )
    if notifications_url:
        print(f"Monitor updates that will be sent to: {notifications_url}")
        print(
            f"You can also use `inference rf-cloud data-staging list-ingest-details --batch-id {batch_id}` command "
            f"to check progress."
        )
    else:
        print(
            f"Use `inference rf-cloud data-staging show-batch-details --batch-id {batch_id}` "
            "command to observe ingest progress. If you want automated updates - use `--notifications-url` option "
            "of this command."
        )


@backoff.on_exception(
    backoff.constant,
    exception=RetryError,
    max_tries=3,
    interval=1,
)
def trigger_videos_references_ingest(
    workspace: str,
    batch_id: str,
    references: Union[str, List[dict]],
    api_key: str,
    ingest_id: Optional[str] = None,
    batch_name: Optional[str] = None,
    notifications_url: Optional[str] = None,
    notification_categories: Optional[List[str]] = None,
) -> VideoReferencesIngestResponse:
    params = {}
    if api_key is not None:
        params["api_key"] = api_key
    if batch_name is not None:
        params["displayName"] = batch_name
    payload = {}
    if isinstance(references, list):
        payload["videoReferences"] = references
    else:
        payload["videoReferencesURL"] = references
    if ingest_id:
        payload["ingestId"] = ingest_id
    if notifications_url:
        payload["notificationsURL"] = notifications_url
    if notification_categories:
        payload["notificationCategories"] = notification_categories
    try:
        response = requests.post(
            f"{API_BASE_URL}/data-staging/v1/external/{workspace}/batches/{batch_id}/bulk-upload/video-references",
            params=params,
            timeout=2 * REQUEST_TIMEOUT,
            json=payload,
        )
    except (ConnectionError, Timeout, requests.exceptions.ConnectionError) as error:
        raise RetryError(
            f"Connectivity error. Try reaching Roboflow API in browser: {API_BASE_URL}"
        ) from error
    handle_response_errors(response=response, operation_name="trigger videos ingest")
    try:
        return VideoReferencesIngestResponse.model_validate(response.json())
    except (ValidationError, ValueError, KeyError) as error:
        raise RFAPICallError("Could not decode Roboflow API response.") from error


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
    api_key: str,
    batch_name: Optional[str],
    upload_log: DataImportLog,
) -> None:
    upload_image_closure = partial(
        upload_image,
        workspace=workspace,
        batch_id=batch_id,
        api_key=api_key,
        upload_log=upload_log,
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
    api_key: str,
    batch_name: Optional[str],
    upload_log: DataImportLog,
    ingest_id: Optional[str] = None,
    notifications_url: Optional[str] = None,
) -> None:
    shards = list(create_batches(sequence=images_paths, batch_size=MAX_SHARD_SIZE))
    if ingest_id is None:
        ingest_id = str(uuid4())
    upload_images_shard_closure = partial(
        pack_and_upload_images_shard,
        workspace=workspace,
        batch_id=batch_id,
        api_key=api_key,
        batch_name=batch_name,
        ingest_id=ingest_id,
        notifications_url=notifications_url,
    )
    with Pool(processes=MAX_SHARDS_UPLOAD_PROCESSES) as pool:
        for paths in tqdm(
            pool.imap(upload_images_shard_closure, shards),
            desc=f"Uploading images shards (each {MAX_SHARD_SIZE} images)...",
            total=len(shards),
        ):
            upload_log.record_files(paths=paths)


def pack_and_upload_images_shard(
    images_paths: List[str],
    workspace: str,
    batch_id: str,
    api_key: str,
    batch_name: Optional[str],
    ingest_id: str,
    notifications_url: Optional[str],
) -> List[str]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        archive_path = os.path.join(tmp_dir, f"{uuid4()}.tar")
        create_images_shard_archive(
            archive_path=archive_path,
            images_paths=images_paths,
            ingest_id=ingest_id,
            notifications_url=notifications_url,
        )
        upload_images_shard(
            archive_path=archive_path,
            workspace=workspace,
            batch_id=batch_id,
            api_key=api_key,
            batch_name=batch_name,
        )
        return images_paths


def create_images_shard_archive(
    images_paths: List[str],
    archive_path: str,
    ingest_id: str,
    notifications_url: Optional[str],
) -> None:
    with tarfile.open(archive_path, "w") as tar:
        for image_path in images_paths:
            tar.add(image_path, arcname=os.path.basename(image_path))
        if notifications_url:
            notification_config = {
                "ingestId": ingest_id,
                "notificationsURL": notifications_url,
            }
            json_data = json.dumps(notification_config).encode("utf-8")
            json_file = BytesIO(json_data)
            info = tarfile.TarInfo(name="notification_config.json")
            info.size = len(json_data)
            tar.addfile(info, json_file)


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
    api_key: str,
    upload_log: DataImportLog,
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
    upload_log.record_file(path=image_path)


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
    api_key: str,
    upload_log: DataImportLog,
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
    upload_log.record_file(path=video_path)


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
    api_key: str,
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
    api_key: str,
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
    "images": " 🖼",
    "videos": " 🎬",
    "metadata": " 📋",
    "archives": " 🗃",
    "mixed": " 🎁",
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
    is_short_to_expiry = (
        metadata.expiry_date - datetime.now(timezone.utc)
    ) < timedelta(days=3)
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
        icon = CONTENT_TYPE_TO_ICON.get(part.content_type, "❓")
        content_type = (
            f"[bold deep_sky_blue1]{part.content_type}[/bold deep_sky_blue1]{icon}"
        )
        if part.nestedContentType:
            nested_content_type_icon = CONTENT_TYPE_TO_ICON.get(
                part.nestedContentType, "❓"
            )
            content_type = f"{content_type}  (of [bold deep_sky_blue1]{part.nestedContentType}[/bold deep_sky_blue1]{nested_content_type_icon})"
        description = part.part_description or "Not Available"
        parts_table_content.append(
            (part.part_name, description, content_type, str(part_count))
        )
    table = Table(title=f"Parts with content", show_lines=True)
    table.add_column(
        "Name", justify="left", vertical="middle", style="cyan", no_wrap=True
    )
    table.add_column(
        "Description", vertical="middle", justify="left", overflow="ellipsis"
    )
    table.add_column(
        "Content Type", vertical="middle", justify="center", overflow="ellipsis"
    )
    table.add_column(
        "Files Count", vertical="middle", justify="center", overflow="ellipsis"
    )
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
    api_key: str,
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
    api_key: str,
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


class DataExportLog:

    @classmethod
    def init(cls, export_dir: str) -> "DataExportLog":
        file_path = os.path.join(export_dir, ".export_log.jsonl")
        if not os.path.exists(file_path):
            file_descriptor = open(file_path, "w+")
        else:
            file_descriptor = open(file_path, "r+")
        log_content = [
            DownloadLogEntry.model_validate(json.loads(line))
            for line in file_descriptor.readlines()
            if len(line.strip()) > 0
        ]
        return cls(log_file=file_descriptor, log_content=log_content)

    def __init__(
        self,
        log_file: TextIO,
        log_content: List[DownloadLogEntry],
    ):
        self._log_file = log_file
        self._metadata_hash2log_entry: Dict[str, DownloadLogEntry] = {
            _generate_file_metadata_hash(element.file_metadata): element
            for element in log_content
        }
        self._lock = Lock()

    def is_already_exported(
        self, file_metadata: FileMetadata, override_existing: bool
    ) -> bool:
        with self._lock:
            metadata_hash = _generate_file_metadata_hash(file_metadata)
            if metadata_hash not in self._metadata_hash2log_entry:
                return False
            if override_existing:
                return False
            entry = self._metadata_hash2log_entry[metadata_hash]
            return os.path.exists(entry.local_path)

    def denote_export(self, file_metadata: FileMetadata, local_path: str) -> None:
        with self._lock:
            entry = DownloadLogEntry(
                file_metadata=file_metadata,
                local_path=local_path,
            )
            self._metadata_hash2log_entry[
                _generate_file_metadata_hash(entry.file_metadata)
            ] = entry
            self._log_file.write(f"{json.dumps(entry.model_dump(by_alias=True))}\n")

    def __del__(self) -> None:
        self._log_file.close()


def _generate_file_metadata_hash(file_metadata: FileMetadata) -> str:
    return (
        f"{file_metadata.file_name}-{file_metadata.part_name}-{file_metadata.shard_id}"
    )


def export_data(
    batch_id: str,
    api_key: str,
    target_directory: str,
    part_names: Optional[Set[str]] = None,
    override_existing: bool = False,
) -> None:
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
            selected_part_names=part_names,
            override_existing=override_existing,
        )
    if part_names is not None:
        raise ValueError(
            f"Requested listing of parts {part_names}, but batch is not multipart-batch."
        )
    return export_simple_batch(
        workspace=workspace,
        batch_id=batch_id,
        api_key=api_key,
        target_directory=target_directory,
        override_existing=override_existing,
    )


def export_multipart_batch(
    workspace: str,
    batch_id: str,
    api_key: str,
    target_directory: str,
    selected_part_names: Optional[Set[str]] = None,
    override_existing: bool = False,
) -> None:
    parts = list_multipart_batch_parts(
        workspace=workspace, batch_id=batch_id, api_key=api_key
    )
    os.makedirs(target_directory, exist_ok=True)
    export_log = DataExportLog.init(export_dir=target_directory)
    for part in parts:
        if (
            selected_part_names is not None
            and part.part_name not in selected_part_names
        ):
            continue
        export_part(
            workspace=workspace,
            batch_id=batch_id,
            api_key=api_key,
            target_directory=target_directory,
            part_name=part.part_name,
            export_log=export_log,
            override_existing=override_existing,
        )


def export_simple_batch(
    workspace: str,
    batch_id: str,
    api_key: str,
    target_directory: str,
    override_existing: bool = False,
) -> None:
    os.makedirs(target_directory, exist_ok=True)
    total_files = get_batch_count(
        workspace=workspace, batch_id=batch_id, api_key=api_key
    )
    export_log = DataExportLog.init(export_dir=target_directory)
    with Progress() as progress_bar:
        task = progress_bar.add_task(
            f"Downloading {total_files} files...", total=total_files
        )
        next_page_token = None
        while True:
            listing_page = get_one_page_of_batch_content(
                workspace=workspace,
                batch_id=batch_id,
                api_key=api_key,
                next_page_token=next_page_token,
            )
            next_page_token = listing_page.next_page_token
            pull_batch_elements_to_directory(
                files_metadata=listing_page.files_metadata,
                target_directory=target_directory,
                progress_bar=progress_bar,
                task=task,
                export_log=export_log,
                override_existing=override_existing,
            )
            if next_page_token is None:
                break
    return None


def export_part(
    workspace: str,
    batch_id: str,
    api_key: str,
    target_directory: str,
    part_name: str,
    export_log: DataExportLog,
    override_existing: bool,
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
        next_page_token = None
        while True:
            listing_page = get_one_page_of_batch_content(
                workspace=workspace,
                batch_id=batch_id,
                api_key=api_key,
                next_page_token=next_page_token,
                part_name=part_name,
            )
            next_page_token = listing_page.next_page_token
            pull_batch_elements_to_directory(
                files_metadata=listing_page.files_metadata,
                target_directory=part_target_directory,
                progress_bar=progress_bar,
                task=task,
                export_log=export_log,
                override_existing=override_existing,
            )
            if next_page_token is None:
                break
    return None


def pull_batch_elements_to_directory(
    files_metadata: List[FileMetadata],
    target_directory: str,
    progress_bar: Progress,
    task: TaskID,
    export_log: DataExportLog,
    override_existing: bool,
) -> None:
    pull_batch_element_to_directory_closure = partial(
        pull_batch_element_to_directory,
        target_directory=target_directory,
        export_log=export_log,
        override_existing=override_existing,
    )
    with ThreadPool(processes=MAX_DOWNLOAD_THREADS) as pool:
        for _ in pool.imap(pull_batch_element_to_directory_closure, files_metadata):
            progress_bar.update(task, advance=1)
    return None


def pull_batch_element_to_directory(
    file_metadata: FileMetadata,
    target_directory: str,
    export_log: DataExportLog,
    override_existing: bool,
) -> None:
    if export_log.is_already_exported(
        file_metadata=file_metadata,
        override_existing=override_existing,
    ):
        return None
    parsed_url = urlparse(file_metadata.download_url)
    file_name = os.path.basename(parsed_url.path)
    is_archive = file_name.endswith(".tar.gz") or file_name.endswith(".tar")
    if not is_archive:
        result_path = pull_file_to_directory(
            url=file_metadata.download_url,
            target_directory=target_directory,
            file_name=file_name,
        )
        export_log.denote_export(file_metadata=file_metadata, local_path=result_path)
        return None
    with tempfile.TemporaryDirectory() as tmp_dir:
        download_and_unpack_archive(
            url=file_metadata.download_url,
            file_name=file_name,
            pull_dir=tmp_dir,
            target_dir=target_directory,
        )
        export_log.denote_export(
            file_metadata=file_metadata, local_path=target_directory
        )
        return None


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
    api_key: str,
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


def list_ingest_details(
    batch_id: str,
    api_key: str,
    output_file: Optional[str] = None,
) -> None:
    workspace = get_workspace(api_key=api_key)
    if output_file:
        saver = MetadataSaver.init(file_path=output_file)
        on_new_metadata = saver.save_metadata
    else:
        on_new_metadata = lambda m: print(m.model_dump())
    for shard_details in iterate_through_batch_shards_statuses(
        workspace=workspace,
        batch_id=batch_id,
        api_key=api_key,
    ):
        on_new_metadata(shard_details)


def list_batch_shards_statuses(
    workspace: str,
    batch_id: str,
    api_key: str,
    page_size: Optional[int] = None,
) -> List[ShardDetails]:
    return list(
        iterate_through_batch_shards_statuses(
            workspace=workspace,
            batch_id=batch_id,
            api_key=api_key,
            page_size=page_size,
        )
    )


def iterate_through_batch_shards_statuses(
    workspace: str,
    batch_id: str,
    api_key: str,
    page_size: Optional[int] = None,
) -> Generator[ShardDetails, None, None]:
    next_page_token = None
    while True:
        page = get_one_page_of_batch_shards_statuses(
            workspace=workspace,
            batch_id=batch_id,
            api_key=api_key,
            page_size=page_size,
            next_page_token=next_page_token,
        )
        yield from page.shards
        next_page_token = page.next_page_token
        if next_page_token is None:
            return None


def get_one_page_of_batch_shards_statuses(
    workspace: str,
    batch_id: str,
    api_key: str,
    page_size: Optional[int] = None,
    next_page_token: Optional[str] = None,
) -> PageOfBatchShardsStatuses:
    params = {"api_key": api_key}
    if page_size:
        params["pageSize"] = page_size
    if next_page_token:
        params["nextPageToken"] = next_page_token
    try:
        response = requests.get(
            f"{API_BASE_URL}/data-staging/v1/external/{workspace}/batches/{batch_id}/shards",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
    except (ConnectionError, Timeout):
        raise RetryError(
            f"Connectivity error. Try reaching Roboflow API in browser: {API_BASE_URL}"
        )
    handle_response_errors(response=response, operation_name="list batch shards")
    try:
        return PageOfBatchShardsStatuses.model_validate(response.json())
    except ValueError as error:
        raise RFAPICallError("Could not decode Roboflow API response.") from error
