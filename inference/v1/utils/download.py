import os
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_EXCEPTION
from threading import Lock
from typing import List, Tuple, Optional, Set, Callable
from uuid import uuid4

import backoff
import requests
from filelock import FileLock
from requests import Timeout, Response
from rich.progress import Progress, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn, TaskID

from inference.v1.configuration import IDEMPOTENT_API_REQUEST_CODES_TO_RETRY, API_CALLS_MAX_RETRIES, API_CALLS_TIMEOUT
from inference.v1.errors import RetryError
from inference.v1.logger import logger
from inference.v1.utils.file_system import ensure_parent_dir_exists, remove_file_if_exists, pre_allocate_file

FileName = str
DownloadUrl = str

DEFAULT_THREAD_CHUNK_SIZE = 1 * 1024 * 1024  # 32MB
DEFAULT_STREAM_DOWNLOAD_CHUNK = 128 * 1024  # 128kB


def download_files_to_directory(
    target_path: str,
    files_specs: List[Tuple[FileName, DownloadUrl]],
    verbose: bool = True,
    response_codes_to_retry: Optional[Set[int]] = None,
    request_timeout: Optional[int] = None,
    max_threads: Optional[int] = 16,
    thread_chunk_size: int = DEFAULT_THREAD_CHUNK_SIZE,
    file_lock_acquire_timeout: int = 10,
) -> None:
    files_specs = exclude_existing_files(target_dir=target_path, files_specs=files_specs)
    if not files_specs:
        return None
    if response_codes_to_retry is None:
        response_codes_to_retry = IDEMPOTENT_API_REQUEST_CODES_TO_RETRY
    if request_timeout is None:
        request_timeout = API_CALLS_TIMEOUT
    os.makedirs(target_path, exist_ok=True)
    progress = Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        disable=not verbose,
    )
    download_id = str(uuid4())
    with progress:
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = []
            for file_name, download_url in files_specs:
                future = executor.submit(
                    safe_download_file,
                    target_dir=target_path,
                    file_name=file_name,
                    download_url=download_url,
                    download_id=download_id,
                    executor=executor,
                    progress=progress,
                    response_codes_to_retry=response_codes_to_retry,
                    request_timeout=request_timeout,
                    thread_chunk_size=thread_chunk_size,
                    file_lock_acquire_timeout=file_lock_acquire_timeout,
                )
                futures.append(future)
            done_futures, pending_futures = wait(futures, return_when=FIRST_EXCEPTION)
            for pending_future in pending_futures:
                pending_future.cancel()
            _ = wait(pending_futures)
            for future in done_futures:
                future_exception = future.exception()
                if future_exception:
                    raise future_exception


def exclude_existing_files(
    target_dir: str,
    files_specs: List[Tuple[FileName, DownloadUrl]],
) -> List[Tuple[FileName, DownloadUrl]]:
    result = []
    for file_name, download_url in files_specs:
        target_path = os.path.join(target_dir, file_name)
        if not os.path.exists(target_path):
            result.append((file_name, download_url))
    return result


def safe_download_file(
    target_dir: str,
    file_name: str,
    download_url: str,
    download_id: str,
    executor: ThreadPoolExecutor,
    progress: Progress,
    response_codes_to_retry: Set[int],
    request_timeout: int,
    thread_chunk_size: int,
    file_lock_acquire_timeout: int,
) -> None:
    target_file_path = os.path.abspath(os.path.join(target_dir, file_name))
    ensure_parent_dir_exists(path=target_file_path)
    target_file_dir, target_file_name = os.path.split(target_file_path)
    lock_path = os.path.join(target_file_dir, f".{target_file_name}.lock")
    tmp_download_file = os.path.abspath(os.path.join(target_dir, f"{file_name}.{download_id}"))
    try:
        with FileLock(lock_path, timeout=file_lock_acquire_timeout):
            safe_execute_download(
                download_url=download_url,
                tmp_download_file=tmp_download_file,
                target_file_path=target_file_path,
                executor=executor,
                progress=progress,
                response_codes_to_retry=response_codes_to_retry,
                request_timeout=request_timeout,
                thread_chunk_size=thread_chunk_size,
                original_file_name=file_name,
            )
    except Exception as error:
        remove_file_if_exists(path=tmp_download_file)
        raise error


def safe_execute_download(
    download_url: str,
    tmp_download_file: str,
    target_file_path: str,
    executor: ThreadPoolExecutor,
    progress: Progress,
    response_codes_to_retry: Set[int],
    request_timeout: int,
    thread_chunk_size: int,
    original_file_name: str,
) -> None:
    expected_file_size = safe_check_range_download_option(url=download_url, timeout=request_timeout, response_codes_to_retry=response_codes_to_retry)
    progress_task = progress.add_task(original_file_name, total=expected_file_size, start=True, visible=True)
    progress_task_lock = Lock()

    def on_chunk_downloaded(bytes_num: int) -> None:
        with progress_task_lock:
            progress.advance(progress_task, bytes_num)

    if expected_file_size is None or expected_file_size < thread_chunk_size:
        stream_download(
            url=download_url,
            target_path=tmp_download_file,
            timeout=request_timeout,
            response_codes_to_retry=response_codes_to_retry,
            on_chunk_downloaded=on_chunk_downloaded,
        )
    else:
        threaded_download_file(
            url=download_url,
            target_path=tmp_download_file,
            file_size=expected_file_size,
            executor=executor,
            response_codes_to_retry=response_codes_to_retry,
            request_timeout=request_timeout,
            thread_chunk_size=thread_chunk_size,
            on_chunk_downloaded=on_chunk_downloaded,
        )
    os.rename(tmp_download_file, target_file_path)


def safe_check_range_download_option(url: str, timeout: int, response_codes_to_retry: Set[int]) -> Optional[int]:
    try:
        return check_range_download_option(url=url, timeout=timeout, response_codes_to_retry=response_codes_to_retry)
    except Exception:
        logger.warning(f"Cannot use range requests for {url}")
        return None


@backoff.on_exception(
    backoff.constant,
    exception=RetryError,
    max_tries=API_CALLS_MAX_RETRIES,
    interval=1,
)
def check_range_download_option(url: str, timeout: int, response_codes_to_retry: Set[int]) -> Optional[int]:
    try:
        response = requests.head(url, timeout=timeout)
    except (OSError, Timeout, requests.exceptions.ConnectionError):
        raise RetryError(f"Connectivity error for URL: {url}")
    if response.status_code in response_codes_to_retry:
        raise RetryError(f"Remote server returned response code {response.status_code} for URL {url}")
    response.raise_for_status()
    accept_ranges = response.headers.get("accept-ranges", "none")
    content_length = response.headers.get("content-length")
    if "bytes" not in accept_ranges.lower():
        return None
    if not content_length:
        return None
    return int(content_length)


def threaded_download_file(
    url: str,
    target_path: str,
    file_size: int,
    executor: ThreadPoolExecutor,
    response_codes_to_retry: Set[int],
    request_timeout: int,
    thread_chunk_size: int,
    on_chunk_downloaded: Optional[Callable[[int], None]] = None,
) -> None:
    chunks_boundaries = generate_chunks_boundaries(file_size=file_size, chunk_size=thread_chunk_size)
    pre_allocate_file(path=target_path, file_size=file_size)
    futures = []
    for start, end in chunks_boundaries:
        future = executor.submit(
            download_chunk,
            url=url,
            start=start,
            end=end,
            target_path=target_path,
            timeout=request_timeout,
            response_codes_to_retry=response_codes_to_retry,
            on_chunk_downloaded=on_chunk_downloaded,
        )
        futures.append(future)
    done_futures, pending_futures = wait(futures, return_when=FIRST_EXCEPTION)
    for pending_future in pending_futures:
        pending_future.cancel()
    _ = wait(pending_futures)
    for future in done_futures:
        future_exception = future.exception()
        if future_exception:
            raise future_exception


def generate_chunks_boundaries(file_size: int, chunk_size: int) -> List[Tuple[int, int]]:
    if file_size <= 0:
        return []
    ranges = []
    accumulated_size = 0
    while accumulated_size < file_size:
        ranges.append((accumulated_size, accumulated_size + chunk_size - 1))
        accumulated_size += chunk_size
    ranges[-1] = (ranges[-1][0], file_size - 1)
    return ranges


@backoff.on_exception(
    backoff.constant,
    exception=RetryError,
    max_tries=API_CALLS_MAX_RETRIES,
    interval=1,
)
def download_chunk(
    url: str,
    start: int,
    end: int,
    target_path: str,
    timeout: int,
    response_codes_to_retry: Set[int],
    file_chunk: int = DEFAULT_STREAM_DOWNLOAD_CHUNK,
    on_chunk_downloaded: Optional[Callable[[int], None]] = None,
) -> None:
    headers = {"Range": f"bytes={start}-{end}"}
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=timeout)
    except (ConnectionError, Timeout, requests.exceptions.ConnectionError):
        raise RetryError(f"Connectivity error")
    if response.status_code in response_codes_to_retry:
        raise RetryError(f"Image hosting returned {response.status_code}")
    response.raise_for_status()
    try:
        with open(target_path, "wb") as file:
            file.seek(start)
            for chunk in response.iter_content(file_chunk):
                file.write(chunk)
                on_chunk_downloaded(len(chunk))
    except requests.exceptions.ConnectionError:
        raise RetryError(f"Connectivity error")


@backoff.on_exception(
    backoff.constant,
    exception=RetryError,
    max_tries=API_CALLS_MAX_RETRIES,
    interval=1,
)
def stream_download(
    url: str,
    target_path: str,
    timeout: int,
    response_codes_to_retry: Set[int],
    file_chunk: int = DEFAULT_STREAM_DOWNLOAD_CHUNK,
    on_chunk_downloaded: Optional[Callable[[int], None]] = None,
) -> None:
    ensure_parent_dir_exists(path=target_path)
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            if response.status_code in response_codes_to_retry:
                raise RetryError(f"Image hosting returned {response.status_code}")
            response.raise_for_status()
            _handle_stream_download(
                response=response,
                target_path=target_path,
                file_chunk=file_chunk,
                on_chunk_downloaded=on_chunk_downloaded,
            )
    except (ConnectionError, Timeout, requests.exceptions.ConnectionError):
        raise RetryError(f"Connectivity error")


def _handle_stream_download(
    response: Response,
    target_path: str,
    file_chunk: int,
    on_chunk_downloaded: Optional[Callable[[int], None]] = None,
) -> None:
    with open(target_path, "wb") as file:
        for chunk in response.iter_content(file_chunk):
            file.write(chunk)
            if on_chunk_downloaded:
                on_chunk_downloaded(len(chunk))


