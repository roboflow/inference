import argparse
import json
import os
from typing import Tuple, Union, List

from google.cloud import storage
from google.cloud.storage import Bucket, Blob
from slack_sdk import WebClient

GCSUri = str
BucketName = str
BucketKey = str
GCSUriChunks = Tuple[BucketName, BucketKey]
LocalPath = str

GOOGLE_STORAGE_URI_PREFIX = "gs://"

STORAGE_CLIENT = storage.Client()

SLACK_TOKEN = os.environ["SLACK_TOKEN"]
SLACK_CHANNEL = os.environ["SLACK_CHANNEL"]
ENV_STATE_DOCUMENT_URI = "gs://roboflow-misc/google-colab-ci/google-colab-env.json"


def main(current_env_state_path: str) -> None:
    current_env_state = read_json_from_local_storage(path=current_env_state_path)
    current_env_latest_digest = get_latest_docker_image_digest(docker_repository_state=current_env_state)
    if file_exists_in_gcs(uri=ENV_STATE_DOCUMENT_URI):
        previous_state = read_json_file_from_gcs(uri=ENV_STATE_DOCUMENT_URI)
        previous_latest_digest = get_latest_docker_image_digest(docker_repository_state=previous_state)
        new_latest_digest = current_env_latest_digest != previous_latest_digest
    else:
        new_latest_digest = True
    if new_latest_digest:
        notify_about_new_google_colab_env()
        upload_json_to_gcs(uri=ENV_STATE_DOCUMENT_URI, content=current_env_state)


def file_exists_in_gcs(uri: GCSUri, storage_client: storage.Client = STORAGE_CLIENT) -> bool:
    blob = get_blob(uri=uri, storage_client=storage_client)
    return blob.exists()


def read_json_from_local_storage(path: str) -> Union[dict, list]:
    with open(path, "r") as f:
        return json.load(f)


def read_json_file_from_gcs(
    uri: GCSUri,
    encoding: str = "utf-8",
    storage_client: storage.Client = STORAGE_CLIENT,
) -> Union[dict, list]:
    data_bytes = read_file_from_gcs(uri=uri, storage_client=storage_client)
    decoded_content = data_bytes.decode(encoding=encoding)
    return json.loads(decoded_content)


def read_file_from_gcs(
    uri: GCSUri,
    storage_client: storage.Client = STORAGE_CLIENT,
) -> bytes:
    blob = get_blob(uri=uri, storage_client=storage_client)
    return blob.download_as_string()


def upload_json_to_gcs(
    uri: GCSUri,
    content: Union[dict, list],
    storage_client: storage.Client = STORAGE_CLIENT,
) -> None:
    content_serialized = json.dumps(content)
    target_blob = get_blob(uri=uri, storage_client=storage_client)
    target_blob.upload_from_string(content_serialized)


def get_blob(
    uri: GCSUri,
    storage_client: storage.Client = STORAGE_CLIENT,
) -> Blob:
    bucket_name, key = get_bucket_and_key(uri=uri)
    bucket: Bucket = storage_client.bucket(bucket_name=bucket_name)
    return bucket.blob(blob_name=key)


def get_bucket_and_key(uri: Union[GCSUri, GCSUriChunks]) -> GCSUriChunks:
    if isinstance(uri, tuple):
        if len(uri) != 2:
            raise ValueError(f"Not a GCS reference: {uri}")
        return uri
    if uri.startswith(GOOGLE_STORAGE_URI_PREFIX):
        uri = uri[len(GOOGLE_STORAGE_URI_PREFIX) :]
    uri_chunks = uri.split("/")
    if len(uri_chunks) <= 1:
        raise ValueError(f"Not a GCS reference: {uri}")
    bucket = uri_chunks[0]
    key = "/".join(uri_chunks[1:])
    return bucket, key


def get_latest_docker_image_digest(docker_repository_state: List[dict]) -> str:
    latest_tag_entry = [entry for entry in docker_repository_state if entry["tag"].endswith("/latest")]
    if len(latest_tag_entry) != 1:
        raise ValueError(f"Found {len(latest_tag_entry)} entries matching latest tag")
    return latest_tag_entry[0]["version"].split("/")[-1]


def notify_about_new_google_colab_env() -> None:
    slack_client = WebClient(token=SLACK_TOKEN)
    blocks = [
        {
            "type": "section",
            "block_id": "section-1",
            "text": {
                "type": "mrkdwn",
                "text": "*:rotating_light: Google Colab environment has changed*"
            }
        },
        {
            "type": "section",
            "block_id": "section-2",
            "text": {
                "type": "mrkdwn",
                "text": "Google Colab is likely to be running with a new environment. "
                        "*Check dependencies of Roboflow packages*."
            }
        }
    ]
    slack_client.chat_postMessage(
        channel=SLACK_CHANNEL,
        blocks=blocks,
        text="Google Colab is likely to be running with a new environment. "
             "Check dependencies of Roboflow packages."
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--current_env_state", type=str, required=True)
    args = parser.parse_args()
    main(current_env_state_path=args.current_env_state)
