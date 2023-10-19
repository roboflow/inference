import os
from typing import List

from botocore.client import BaseClient


def download_s3_files_to_directory(
    bucket: str,
    keys: List[str],
    target_dir: str,
    s3_client: BaseClient,
) -> None:
    os.makedirs(target_dir, exist_ok=True)
    for key in keys:
        target_path = os.path.join(target_dir, key)
        s3_client.download_file(
            bucket,
            key,
            target_path,
        )
