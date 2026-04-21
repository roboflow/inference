import hashlib
import math
import os
import time
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from threading import Lock
from typing import Callable, Dict, List, Literal, Optional, Set, Tuple, Union
from uuid import uuid4

import backoff
import requests
from filelock import FileLock
from requests import Response, Timeout
from requests.exceptions import ChunkedEncodingError
from urllib3.exceptions import ProtocolError
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from inference_models.configuration import (
    API_CALLS_MAX_TRIES,
    API_CALLS_TIMEOUT,
    CHUNK_DOWNLOAD_CONNECT_TIMEOUT,
    CHUNK_DOWNLOAD_READ_TIMEOUT,
    CHUNK_DOWNLOAD_MAX_ATTEMPTS,
    DISABLE_INTERACTIVE_PROGRESS_BARS,
    FILE_LOCK_ACQUIRE_TIMEOUT,
    IDEMPOTENT_API_REQUEST_CODES_TO_RETRY,
)
from inference_models.errors import (
    FileHashSumMissmatch,
    InvalidParameterError,
    RangeRequestNotSupportedError,
    RetryError,
    UntrustedFileError,
)
from inference_models.logger import LOGGER
from inference_models.utils.file_system import (
    ensure_parent_dir_exists,
    pre_allocate_file,
    remove_file_if_exists,
    stream_file_bytes,
)

FileHandle = str
DownloadUrl = str
MD5Hash = Optional[str]

MIN_SIZE_FOR_THREADED_DOWNLOAD = 32 * 1024 * 1024  # 32MB
MIN_THREAD_CHUNK_SIZE = 16 * 1024 * 1024  # 16MB
DEFAULT_STREAM_DOWNLOAD_CHUNK = 1 * 1024 * 1024  # 1MB

_CONNECTIVITY_ERRORS = (
    ChunkedEncodingError,
    ProtocolError,
    ConnectionError,
    Timeout,
    requests.exceptions.ConnectionError,
)


class PartialDownloadError(Exception):
    """Raised when the remote end closes mid-body after some bytes were written."""

    __slots__ = ("bytes_written",)

    def __init__(self, bytes_written: int) -> None:
        self.bytes_written = bytes_written


def _chunk_download_backoff_sleep(attempt_index: int) -> None:
    time.sleep(2 ** min(attempt_index, 5))


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
    file_lock_acquire_timeout: int = FILE_LOCK_ACQUIRE_TIMEOUT,
    verify_hash_while_download: bool = True,
    download_files_without_hash: bool = False,
    name_after: Literal["file_handle", "md5_hash"] = "file_handle",
    on_file_created: Optional[Callable[[str], None]] = None,
    on_file_renamed: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, str]:
    """Download multiple files to a directory with parallel downloads and hash verification.

    Downloads files from URLs to a target directory with support for parallel downloads,
    automatic retries, hash verification, and progress tracking. Skips files that already
    exist in the target directory.

    Args:
        target_dir: Absolute path to the directory where files should be downloaded.
            Will be created if it doesn't exist.

        files_specs: List of tuples, each containing:
            - file_handle (str): Logical name for the file (used as filename by default)
            - download_url (str): URL to download the file from
            - md5_hash (Optional[str]): Expected MD5 hash for verification (None if unknown)

        verbose: Show progress bars during download. Default: True.

        response_codes_to_retry: HTTP status codes that should trigger a retry.
            Default: Uses library defaults (typically 429, 500, 502, 503, 504).

        request_timeout: Timeout in seconds for HTTP requests. Default: Uses library default.

        max_parallel_downloads: Maximum number of files to download simultaneously.
            Default: 8.

        max_threads_per_download: Maximum number of threads to use for downloading a
            single large file. Default: 8.

        file_lock_acquire_timeout: Timeout in seconds for acquiring file locks during
            concurrent downloads. Default: 10.

        verify_hash_while_download: Verify MD5 hash during download. Default: True.

        download_files_without_hash: Allow downloading files without MD5 hashes.
            **Security risk**. Default: False.

        name_after: How to name downloaded files. Options:
            - "file_handle": Use the file_handle from files_specs
            - "md5_hash": Use the MD5 hash as filename
            Default: "file_handle".

        on_file_created: Optional callback called when a file is created.
            Receives the file path as argument.

        on_file_renamed: Optional callback called when a file is renamed.
            Receives old and new paths as arguments.

    Returns:
        Dictionary mapping file handles to their absolute paths in the target directory.

    Raises:
        UntrustedFileError: If `download_files_without_hash=False` and files without
            hashes are encountered.
        FileHashSumMissmatch: If downloaded file's hash doesn't match expected hash.
        RetryError: If download fails after all retry attempts.
        InvalidParameterError: If `name_after` has an invalid value.

    Examples:
        Download model files:

        >>> from inference_models.developer_tools import download_files_to_directory
        >>>
        >>> files_to_download = [
        ...     ("model.onnx", "https://example.com/model.onnx", "abc123..."),
        ...     ("config.json", "https://example.com/config.json", "def456..."),
        ... ]
        >>>
        >>> file_paths = download_files_to_directory(
        ...     target_dir="/path/to/cache",
        ...     files_specs=files_to_download,
        ...     verbose=True
        ... )
        >>>
        >>> print(file_paths["model.onnx"])  # /path/to/cache/model.onnx
        >>> print(file_paths["config.json"])  # /path/to/cache/config.json

        Download without hash verification (not recommended):

        >>> files_to_download = [
        ...     ("weights.pt", "https://example.com/weights.pt", None),
        ... ]
        >>>
        >>> file_paths = download_files_to_directory(
        ...     target_dir="/path/to/cache",
        ...     files_specs=files_to_download,
        ...     download_files_without_hash=True,  # Allow files without hashes
        ...     verify_hash_while_download=False
        ... )

    Note:
        - Files are downloaded in parallel for better performance
        - Large files (>32MB) are downloaded using multiple threads
        - Existing files are skipped automatically
        - Progress bars are disabled if DISABLE_INTERACTIVE_PROGRESS_BARS env var is set
    """
    if name_after not in {"file_handle", "md5_hash"}:
        raise InvalidParameterError(
            message="Function download_files_to_directory(...) was called with "
            f"invalid value of parameter `name_after` - received value `{name_after}`. "
            f"This is a bug in `inference-models` - submit new issue under "
            f"https://github.com/roboflow/inference/issues/",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#invalidparametererror",
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
                help_url="https://inference-models.roboflow.com/errors/file-download/#untrustedfileerror",
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
                "of download procedure in `inference-models` or bug in the codebase. "
                "If you see this error using hosted Roboflow solution - contact us to get "
                "help. Running locally, verify the download code and raise an issue if you see "
                "a bug: https://github.com/roboflow/inference/issues/",
                help_url="https://inference-models.roboflow.com/errors/file-download/#untrustedfileerror",
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
            if os.path.isfile(target_file_path):
                LOGGER.debug(
                    f"File {target_file_path} already exists after acquiring lock, "
                    f"skipping download."
                )
                return
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
    os.replace(tmp_download_file, target_file_path)
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
            message=f"Connectivity error for URL: {url}",
            help_url="https://inference-models.roboflow.com/errors/file-download/#retryerror",
        )
    if response.status_code in response_codes_to_retry:
        raise RetryError(
            message=f"Remote server returned response code {response.status_code} for URL {url}",
            help_url="https://inference-models.roboflow.com/errors/file-download/#retryerror",
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
            message=f"Connectivity error for URL: {url}",
            help_url="https://inference-models.roboflow.com/errors/file-download/#retryerror",
        )
    if response.status_code in response_codes_to_retry:
        raise RetryError(
            message=f"Remote server returned response code {response.status_code} for URL {url}",
            help_url="https://inference-models.roboflow.com/errors/file-download/#retryerror",
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
                connect_timeout=CHUNK_DOWNLOAD_CONNECT_TIMEOUT,
                read_timeout=request_timeout,  # TODO: this needs to be updated further upstream to allow for providing read and connect timeouts
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
            help_url="https://inference-models.roboflow.com/errors/file-download/#filehashsummissmatch",
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


def download_chunk(
    url: str,
    start: int,
    end: int,
    target_path: str,
    response_codes_to_retry: Set[int],
    connect_timeout: int = CHUNK_DOWNLOAD_CONNECT_TIMEOUT,
    read_timeout: int = CHUNK_DOWNLOAD_READ_TIMEOUT,
    max_attempts: int = CHUNK_DOWNLOAD_MAX_ATTEMPTS,
    file_chunk: int = DEFAULT_STREAM_DOWNLOAD_CHUNK,
    on_chunk_downloaded: Optional[Callable[[int], None]] = None,
) -> None:    
    current_start = start

    for attempt in range(max_attempts):
        if current_start > end:
            return None

        headers = {"Range": f"bytes={current_start}-{end}"}
        try:
            with requests.get(
                url, headers=headers, stream=True, timeout=(connect_timeout, read_timeout)
            ) as response:
                if response.status_code in response_codes_to_retry:
                    LOGGER.warning(
                        f"Download chunk got status {response.status_code} for "
                        f"bytes={current_start}-{end}, retrying…"
                    )
                    _chunk_download_backoff_sleep(attempt)
                    continue

                response.raise_for_status()

                if response.status_code != 206:
                    raise RangeRequestNotSupportedError(
                        message=(
                            "Server does not support range requests "
                            f"(returned {response.status_code} instead of 206)"
                        ),
                        help_url="https://inference-models.roboflow.com/errors/file-download/#rangerequestnotsupportederror",
                    )

                segment_len = end - current_start + 1
                try:
                    written = _handle_range_request_download(
                        response=response,
                        target_path=target_path,
                        file_chunk=file_chunk,
                        offset=current_start,
                        on_chunk_downloaded=on_chunk_downloaded,
                    )
                except PartialDownloadError as error:
                    current_start += error.bytes_written
                    LOGGER.warning(
                        f"Download chunk interrupted after {error.bytes_written} bytes "
                        f"(bytes={current_start - error.bytes_written}-{end}), resuming…"
                    )
                    _chunk_download_backoff_sleep(attempt)
                    continue

                # Sanity check: the server should have returned the correct content length
                # Doesn't run in the unlikely scenario the content-length header is missing or malformed
                content_length = response.headers.get("Content-Length")
                if content_length and content_length.isdigit() and written < int(content_length):
                    current_start += written
                    _chunk_download_backoff_sleep(attempt)
                    continue

                # Sanity check: the server should have returned the length we asked for
                if written < segment_len:
                    current_start += written
                    _chunk_download_backoff_sleep(attempt)
                    continue

                return None

        except _CONNECTIVITY_ERRORS as error:
            LOGGER.warning(
                f"Download chunk failed ({type(error).__name__}: {error}), retrying…"
            )
            _chunk_download_backoff_sleep(attempt)
            continue

    raise RetryError(
        message="Connectivity error. Max retries reached.",
        help_url="https://inference-models.roboflow.com/errors/file-download/#retryerror",
    )


@backoff.on_exception(
    backoff.constant,
    exception=RetryError,
    max_tries=API_CALLS_MAX_TRIES,
    interval=10,
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
                    help_url="https://inference-models.roboflow.com/errors/file-download/#retryerror",
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
    except (
        ConnectionError,
        Timeout,
        requests.exceptions.ConnectionError,
        ChunkedEncodingError,
    ) as error:
        LOGGER.warning(
            f"Download failed ({type(error).__name__}: {error}), retrying in 10s..."
        )
        raise RetryError(
            message=f"Connectivity error",
            help_url="https://inference-models.roboflow.com/errors/file-download/#retryerror",
        ) from error
    if not verify_hash_while_download:
        return None
    if computed_hash.hexdigest() != md5_hash:
        raise FileHashSumMissmatch(
            f"Could not confirm the validity of file content for url: {url}. Expected MD5: {md5_hash}, "
            f"calculated hash: {computed_hash.hexdigest()}",
            help_url="https://inference-models.roboflow.com/errors/file-download/#filehashsummissmatch",
        )
    return None


def _handle_stream_download(
    response: Response,
    target_path: str,
    file_chunk: int,
    on_chunk_downloaded: Optional[Callable[[int], None]] = None,
    file_open_mode: str = "wb",
    content_storage: Optional[Union[hashlib.md5, HashNullObject]] = None,
    on_file_created: Optional[Callable[[str], None]] = None,
) -> None:
    with open(target_path, file_open_mode) as file:
        if on_file_created:
            on_file_created(target_path)
        for chunk in response.iter_content(file_chunk):
            if not chunk:
                continue
            file.write(chunk)
            if content_storage is not None:
                content_storage.update(chunk)
            if on_chunk_downloaded:
                on_chunk_downloaded(len(chunk))


def _handle_range_request_download(
    response: Response,
    target_path: str,
    file_chunk: int,
    offset: int,
    on_chunk_downloaded: Optional[Callable[[int], None]] = None,
) -> int:
    bytes_written = 0

    try:
        with open(target_path, "r+b") as file:
            file.seek(offset)

            for chunk in response.iter_content(file_chunk):
                if not chunk:
                    continue

                file.write(chunk)

                if on_chunk_downloaded:
                    on_chunk_downloaded(len(chunk))

                bytes_written += len(chunk)

    except _CONNECTIVITY_ERRORS as error:
        if bytes_written > 0:
            raise PartialDownloadError(bytes_written) from error
        raise

    return bytes_written
