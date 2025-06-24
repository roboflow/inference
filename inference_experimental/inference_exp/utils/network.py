from typing import Optional

import backoff
import requests
from inference_exp.constants import DOWNLOAD_CHUNK_SIZE, HTTP_CODES_TO_RETRY
from inference_exp.errors import RetryError
from inference_exp.utils.file_system import ensure_parent_dir_exists
from requests import Response, Timeout
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


@backoff.on_exception(
    backoff.constant,
    exception=RetryError,
    max_tries=3,
    interval=1,
)
def download_file(
    url: str,
    target_path: str,
    description: Optional[str] = None,
    verbose: bool = False,
) -> None:
    ensure_parent_dir_exists(path=target_path)
    try:
        with requests.get(url, stream=True) as response:
            if response.status_code in HTTP_CODES_TO_RETRY:
                raise RetryError(message=f"Image hosting returned {response.status_code}", help_url="https://todo",)
            response.raise_for_status()
            _handle_stream_download(
                response=response,
                target_path=target_path,
                description=description,
                verbose=verbose,
            )
    except (ConnectionError, Timeout, requests.exceptions.ConnectionError):
        raise RetryError(message=f"Connectivity error", help_url="https://todo",)


def _handle_stream_download(
    response: Response,
    target_path: str,
    description: Optional[str] = None,
    verbose: bool = False,
) -> None:
    announced_content_length = int(response.headers.get("content-length", "0"))
    progress = Progress(
        TextColumn("[bold green]Downloading"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    )
    description = description or "download"
    with open(target_path, "wb") as file, progress as p_bar:
        task = p_bar.add_task(
            description, total=announced_content_length, visible=verbose
        )
        for chunk in response.iter_content(DOWNLOAD_CHUNK_SIZE):
            file.write(chunk)
            p_bar.update(task, advance=len(chunk))
