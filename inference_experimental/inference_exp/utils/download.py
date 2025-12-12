import hashlib
import math
import os
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from threading import Lock
from typing import Callable, Dict, List, Literal, Optional, Set, Tuple, Union
from uuid import uuid4

import backoff
import requests
from filelock import FileLock
from inference_exp.configuration import (
    API_CALLS_MAX_TRIES,
    API_CALLS_TIMEOUT,
    DISABLE_INTERACTIVE_PROGRESS_BARS,
    IDEMPOTENT_API_REQUEST_CODES_TO_RETRY,
)
from inference_exp.errors import (
    FileHashSumMissmatch,
    InvalidParameterError,
    RetryError,
    UntrustedFileError,
)
from inference_exp.logger import LOGGER
from inference_exp.utils.file_system import (
    ensure_parent_dir_exists,
    pre_allocate_file,
    remove_file_if_exists,
    stream_file_bytes,
)
from requests import Response, Timeout
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

FileHandle = str
DownloadUrl = str
MD5Hash = Optional[str]

MIN_SIZE_FOR_THREADED_DOWNLOAD = 32 * 1024 * 1024  # 32MB
MIN_THREAD_CHUNK_SIZE = 16 * 1024 * 1024  # 16MB
DEFAULT_STREAM_DOWNLOAD_CHUNK = 1 * 1024 * 1024  # 1MB


class HashNullObject:

    def update(self, *args, **kwargs) -> None:
        pass

    def hexdigest(self) -> None:
        return None


def download_files_to_directory(
    target_dir: str,
    files_specs: List[Tuple[FileHandle, DownloadUrl, MD5Hash]],
    verbose: bool = True,
    response_codes_to_retry: Optional[Set[int]] = None,
    request_timeout: Optional[int] = None,
    max_parallel_downloads: int = 8,
    max_threads_per_download: int = 8,
    file_lock_acquire_timeout: int = 10,
    verify_hash_while_download: bool = True,
    download_files_without_hash: bool = False,
    name_after: Literal["file_handle", "md5_hash"] = "file_handle",
    on_file_created: Optional[Callable[[str], None]] = None,
    on_file_renamed: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, str]:
    if name_after not in {"file_handle", "md5_hash"}:
        raise InvalidParameterError(
            message="Function download_files_to_directory(...) was called with "
            f"invalid value of parameter `name_after` - received value `{name_after}`. "
            f"This is a bug in `inference-exp` - submit new issue under "
            f"https://github.com/roboflow/inference/issues/",
            help_url="https://todo",
        )
    if DISABLE_INTERACTIVE_PROGRESS_BARS:
        verbose = False
    files_mapping = construct_files_path_mapping(
        target_dir=target_dir,
        files_specs=files_specs,
        name_after=name_after,
    )
    files_specs = exclude_existing_files(
        files_specs=files_specs,
        files_mapping=files_mapping,
    )
    if not files_specs:
        return files_mapping
    if response_codes_to_retry is None:
        response_codes_to_retry = IDEMPOTENT_API_REQUEST_CODES_TO_RETRY
    if request_timeout is None:
        request_timeout = API_CALLS_TIMEOUT
    if not download_files_without_hash:
        untrusted_files = [f[1] for f in files_specs if f[2] is None]
        if len(untrusted_files) > 0:
            raise UntrustedFileError(
                message=f"While downloading files detected {len(untrusted_files)} untrusted file(s): {untrusted_files} "
                f"without MD5 hash sum to verify the download content. The download method was used with "
                f"`download_files_without_hash=False` - which prevents from downloading such files. If you see "
                f"this error while using hosted Roboflow serving option - contact us to get support.",
                help_url="https://todo",
            )
    os.makedirs(target_dir, exist_ok=True)
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
        with ThreadPoolExecutor(max_workers=max_parallel_downloads) as executor:
            futures = []
            for file_handle, download_url, md5_hash in files_specs:
                future = executor.submit(
                    safe_download_file,
                    target_file_path=files_mapping[file_handle],
                    download_url=download_url,
                    md5_hash=md5_hash,
                    verify_hash_while_download=verify_hash_while_download,
                    download_id=download_id,
                    progress=progress,
                    response_codes_to_retry=response_codes_to_retry,
                    request_timeout=request_timeout,
                    max_threads_per_download=max_threads_per_download,
                    file_lock_acquire_timeout=file_lock_acquire_timeout,
                    on_file_created=on_file_created,
                    on_file_renamed=on_file_renamed,
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
        return files_mapping


def construct_files_path_mapping(
    target_dir: str,
    files_specs: List[Tuple[FileHandle, DownloadUrl, MD5Hash]],
    name_after: Literal["file_handle", "md5_hash"] = "file_handle",
) -> Dict[FileHandle, str]:
    result = {}
    for file_handle, download_url, content_hash in files_specs:
        if name_after == "md5_hash" and content_hash is None:
            raise UntrustedFileError(
                message="Attempted to download file without declared hash sum when "
                "`name_after='md5_hash'` - this problem is either misconfiguration "
                "of download procedure in `inference-exp` or bug in the codebase. "
                "If you see this error using hosted Roboflow solution - contact us to get "
                "help. Running locally, verify the download code and raise an issue if you see "
                "a bug: https://github.com/roboflow/inference/issues/",
                help_url="https://todo",
            )
        if name_after == "md5_hash":
            target_path = os.path.join(target_dir, content_hash)
        else:
            target_path = os.path.join(target_dir, file_handle)
        result[file_handle] = target_path
    return result


def exclude_existing_files(
    files_specs: List[Tuple[FileHandle, DownloadUrl, MD5Hash]],
    files_mapping: Dict[FileHandle, str],
) -> List[Tuple[FileHandle, DownloadUrl, MD5Hash]]:
    result = []
    for file_specs in files_specs:
        target_path = files_mapping[file_specs[0]]
        if not os.path.isfile(target_path):
            result.append(file_specs)
    return result


def safe_download_file(
    target_file_path: str,
    download_url: str,
    download_id: str,
    md5_hash: MD5Hash,
    verify_hash_while_download: bool,
    progress: Progress,
    response_codes_to_retry: Set[int],
    request_timeout: int,
    max_threads_per_download: int,
    file_lock_acquire_timeout: int,
    on_file_created: Optional[Callable[[str], None]] = None,
    on_file_renamed: Optional[Callable[[str, str], None]] = None,
) -> None:
    ensure_parent_dir_exists(path=target_file_path)
    target_file_dir, target_file_name = os.path.split(target_file_path)
    lock_path = os.path.join(target_file_dir, f".{target_file_name}.lock")
    tmp_download_file = os.path.abspath(
        os.path.join(target_file_dir, f"{target_file_name}.{download_id}")
    )
    try:
        with FileLock(lock_path, timeout=file_lock_acquire_timeout):
            safe_execute_download(
                download_url=download_url,
                tmp_download_file=tmp_download_file,
                target_file_path=target_file_path,
                md5_hash=md5_hash,
                verify_hash_while_download=verify_hash_while_download,
                progress=progress,
                response_codes_to_retry=response_codes_to_retry,
                request_timeout=request_timeout,
                max_threads_per_download=max_threads_per_download,
                original_file_name=target_file_name,
                on_file_created=on_file_created,
                on_file_renamed=on_file_renamed,
            )
    finally:
        remove_file_if_exists(path=tmp_download_file)


def safe_execute_download(
    download_url: str,
    tmp_download_file: str,
    target_file_path: str,
    md5_hash: MD5Hash,
    verify_hash_while_download: bool,
    progress: Progress,
    response_codes_to_retry: Set[int],
    request_timeout: int,
    max_threads_per_download: int,
    original_file_name: str,
    on_file_created: Optional[Callable[[str], None]] = None,
    on_file_renamed: Optional[Callable[[str, str], None]] = None,
) -> None:
    expected_file_size = safe_check_range_download_option(
        url=download_url,
        timeout=request_timeout,
        response_codes_to_retry=response_codes_to_retry,
    )
    download_task = progress.add_task(
        description=f"{original_file_name}: Download",
        total=expected_file_size,
        start=True,
        visible=True,
    )
    hash_calculation_task = (
        []
    )  # yeah, this is a dirty trick to add task in closure in runtime

    progress_task_lock = Lock()

    def on_chunk_downloaded(bytes_num: int) -> None:
        with progress_task_lock:
            progress.advance(download_task, bytes_num)

    def on_hash_calculation_started() -> None:
        if len(hash_calculation_task) > 0:
            return None
        progress.remove_task(download_task)
        new_hash_calculation_task = progress.add_task(
            description=f"{original_file_name}: Verify hash",
            total=expected_file_size,
            start=True,
            visible=True,
        )
        hash_calculation_task.append(new_hash_calculation_task)

    def on_hash_chunk_calculated(bytes_num: int) -> None:
        if len(hash_calculation_task) != 1:
            return None
        progress.advance(hash_calculation_task[0], bytes_num)

    if (
        expected_file_size is None
        or expected_file_size < MIN_SIZE_FOR_THREADED_DOWNLOAD
        or max_threads_per_download <= 1
    ):
        stream_download(
            url=download_url,
            target_path=tmp_download_file,
            timeout=request_timeout,
            md5_hash=md5_hash,
            verify_hash_while_download=verify_hash_while_download,
            response_codes_to_retry=response_codes_to_retry,
            on_chunk_downloaded=on_chunk_downloaded,
            on_file_created=on_file_created,
        )
    else:
        threaded_download_file(
            url=download_url,
            target_path=tmp_download_file,
            file_size=expected_file_size,
            response_codes_to_retry=response_codes_to_retry,
            request_timeout=request_timeout,
            md5_hash=md5_hash,
            verify_hash_while_download=verify_hash_while_download,
            max_threads_per_download=max_threads_per_download,
            on_chunk_downloaded=on_chunk_downloaded,
            on_file_created=on_file_created,
            on_hash_calculation_started=on_hash_calculation_started,
            on_hash_chunk_calculated=on_hash_chunk_calculated,
        )
    os.rename(tmp_download_file, target_file_path)
    if on_file_renamed:
        on_file_renamed(tmp_download_file, target_file_path)


def safe_check_range_download_option(
    url: str, timeout: int, response_codes_to_retry: Set[int]
) -> Optional[int]:
    try:
        return check_range_download_option(
            url=url, timeout=timeout, response_codes_to_retry=response_codes_to_retry
        )
    except Exception:
        LOGGER.warning(f"Cannot use range requests for {url}")
        return None


@backoff.on_exception(
    backoff.constant,
    exception=RetryError,
    max_tries=API_CALLS_MAX_TRIES,
    interval=1,
)
def check_range_download_option(
    url: str, timeout: int, response_codes_to_retry: Set[int]
) -> Optional[int]:
    try:
        response = requests.head(url, timeout=timeout)
    except (OSError, Timeout, requests.exceptions.ConnectionError):
        raise RetryError(
            message=f"Connectivity error for URL: {url}", help_url="https://todo"
        )
    if response.status_code in response_codes_to_retry:
        raise RetryError(
            message=f"Remote server returned response code {response.status_code} for URL {url}",
            help_url="https://todo",
        )
    response.raise_for_status()
    accept_ranges = response.headers.get("accept-ranges", "none")
    content_length = response.headers.get("content-length")
    if "bytes" not in accept_ranges.lower():
        return None
    if not content_length:
        return None
    return int(content_length)


@backoff.on_exception(
    backoff.constant,
    exception=RetryError,
    max_tries=API_CALLS_MAX_TRIES,
    interval=1,
)
def get_content_length(
    url: str,
    timeout: Optional[int] = None,
    response_codes_to_retry: Optional[Set[int]] = None,
) -> Optional[int]:
    if response_codes_to_retry is None:
        response_codes_to_retry = IDEMPOTENT_API_REQUEST_CODES_TO_RETRY
    if timeout is None:
        timeout = API_CALLS_TIMEOUT
    try:
        response = requests.head(url, timeout=timeout)
    except (OSError, Timeout, requests.exceptions.ConnectionError):
        raise RetryError(
            message=f"Connectivity error for URL: {url}", help_url="https://todo"
        )
    if response.status_code in response_codes_to_retry:
        raise RetryError(
            message=f"Remote server returned response code {response.status_code} for URL {url}",
            help_url="https://todo",
        )
    response.raise_for_status()
    content_length = response.headers.get("content-length")
    if content_length is None:
        return None
    return int(content_length)


def threaded_download_file(
    url: str,
    target_path: str,
    file_size: int,
    response_codes_to_retry: Set[int],
    request_timeout: int,
    max_threads_per_download: int,
    md5_hash: MD5Hash,
    verify_hash_while_download: bool,
    on_chunk_downloaded: Optional[Callable[[int], None]] = None,
    on_file_created: Optional[Callable[[str], None]] = None,
    on_hash_calculation_started: Optional[Callable[[], None]] = None,
    on_hash_chunk_calculated: Optional[Callable[[int], None]] = None,
) -> None:
    chunks_boundaries = generate_chunks_boundaries(
        file_size=file_size,
        max_threads=max_threads_per_download,
        min_chunk_size=MIN_THREAD_CHUNK_SIZE,
    )
    pre_allocate_file(
        path=target_path, file_size=file_size, on_file_created=on_file_created
    )
    futures = []
    max_workers = min(len(chunks_boundaries), max_threads_per_download)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
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
    if not verify_hash_while_download:
        return None
    if on_hash_calculation_started:
        on_hash_calculation_started()
    verify_hash_sum_of_local_file(
        url=url,
        file_path=target_path,
        expected_md5_hash=md5_hash,
        on_hash_chunk_calculated=on_hash_chunk_calculated,
    )


def verify_hash_sum_of_local_file(
    url: str,
    file_path: str,
    expected_md5_hash: MD5Hash,
    on_hash_chunk_calculated: Optional[Callable[[int], None]] = None,
) -> None:
    computed_hash = hashlib.md5()
    for file_chunk in stream_file_bytes(
        path=file_path, chunk_size=MIN_THREAD_CHUNK_SIZE
    ):
        computed_hash.update(file_chunk)
        if on_hash_chunk_calculated:
            on_hash_chunk_calculated(len(file_chunk))
    if computed_hash.hexdigest() != expected_md5_hash:
        raise FileHashSumMissmatch(
            f"Could not confirm the validity of file content for url: {url}. "
            f"Expected MD5: {expected_md5_hash}, calculated hash: {computed_hash.hexdigest()}",
            help_url="https://todo",
        )


def generate_chunks_boundaries(
    file_size: int,
    max_threads: int,
    min_chunk_size: int,
) -> List[Tuple[int, int]]:
    if file_size <= 0:
        return []
    chunk_size = math.ceil(file_size / max_threads)
    if chunk_size < min_chunk_size:
        chunk_size = min_chunk_size
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
    max_tries=API_CALLS_MAX_TRIES,
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
        with requests.get(
            url, headers=headers, stream=True, timeout=timeout
        ) as response:
            if response.status_code in response_codes_to_retry:
                raise RetryError(
                    message=f"File hosting returned {response.status_code}",
                    help_url="https://todo",
                )
            response.raise_for_status()
            if response.status_code != 206:
                raise RetryError(
                    message=f"Server does not support range requests (returned {response.status_code} instead of 206)",
                    help_url="https://todo",
                )
            _handle_stream_download(
                response=response,
                target_path=target_path,
                file_chunk=file_chunk,
                on_chunk_downloaded=on_chunk_downloaded,
                file_open_mode="r+b",
                offset=start,
            )
    except (ConnectionError, Timeout, requests.exceptions.ConnectionError):
        raise RetryError(
            message=f"Connectivity error",
            help_url="https://todo",
        )


@backoff.on_exception(
    backoff.constant,
    exception=RetryError,
    max_tries=API_CALLS_MAX_TRIES,
    interval=1,
)
def stream_download(
    url: str,
    target_path: str,
    timeout: int,
    response_codes_to_retry: Set[int],
    md5_hash: MD5Hash,
    verify_hash_while_download: bool,
    file_chunk: int = DEFAULT_STREAM_DOWNLOAD_CHUNK,
    on_chunk_downloaded: Optional[Callable[[int], None]] = None,
    on_file_created: Optional[Callable[[str], None]] = None,
) -> None:
    ensure_parent_dir_exists(path=target_path)
    computed_hash = (
        HashNullObject()
        if md5_hash is None or verify_hash_while_download is None
        else hashlib.md5()
    )
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            if response.status_code in response_codes_to_retry:
                raise RetryError(
                    message=f"File hosting returned {response.status_code}",
                    help_url="https://todo",
                )
            response.raise_for_status()
            _handle_stream_download(
                response=response,
                target_path=target_path,
                file_chunk=file_chunk,
                on_chunk_downloaded=on_chunk_downloaded,
                content_storage=computed_hash,
                on_file_created=on_file_created,
            )
    except (ConnectionError, Timeout, requests.exceptions.ConnectionError):
        raise RetryError(
            message=f"Connectivity error",
            help_url="https://todo",
        )
    if not verify_hash_while_download:
        return None
    if computed_hash.hexdigest() != md5_hash:
        raise FileHashSumMissmatch(
            f"Could not confirm the validity of file content for url: {url}. Expected MD5: {md5_hash}, "
            f"calculated hash: {computed_hash.hexdigest()}",
            help_url="https://todo",
        )
    return None


def _handle_stream_download(
    response: Response,
    target_path: str,
    file_chunk: int,
    on_chunk_downloaded: Optional[Callable[[int], None]] = None,
    file_open_mode: str = "wb",
    offset: Optional[int] = None,
    content_storage: Optional[Union[hashlib.md5, HashNullObject]] = None,
    on_file_created: Optional[Callable[[str], None]] = None,
) -> None:
    with open(target_path, file_open_mode) as file:
        if on_file_created:
            on_file_created(target_path)
        if offset:
            file.seek(offset)
        for chunk in response.iter_content(file_chunk):
            file.write(chunk)
            if content_storage is not None:
                content_storage.update(chunk)
            if on_chunk_downloaded:
                on_chunk_downloaded(len(chunk))