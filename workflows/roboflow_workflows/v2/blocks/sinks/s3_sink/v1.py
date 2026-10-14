"""S3 sink block: upload text content to an AWS S3 bucket.

Port of V1 ``roboflow_core/s3_sink@v1``::

    disabled?       ─▶ error_status=False, V1's disabled message
    credentials     ─▶ revealed only here ─▶ s3_client_factory ─▶ client
    separate_files  each call ─▶ put_object(<prefix>/<name>_<time>.<type>)
    append_log      each call ─▶ entry added to the buffer of the current key
                                 ─▶ put_object(whole buffer, same key)
                    a new key after max_entries_per_file entries, or when
                    bucket_name, s3_prefix, file_name_prefix or file_type change

As in V1, ``append_log`` uploads the whole buffer on every call. A successful
upload makes the object complete through that entry. A failed upload keeps
its entry in the buffer; the next call uploads it again, as in V1. A CSV
entry after the first loses its header line. JSON content is rewritten on
one line into a ``.jsonl`` key.

A throttling or connection error is retried three times (1 s, 2 s, 4 s);
an auth or bucket error is not. Like V1, the block has no ``fire_and_forget``:
each call waits for its upload, retry sleeps included (up to 7 s).

Credentials must come from the workflow, as in V1: a string literal, an
input, or a ``secret`` such as ``roboflow_core/environment_secrets_store@v1`` outputs.
They are revealed only to build the client; messages are scrubbed of them.
A replayed (unavailable) secret fails the step and uploads nothing.

The ``s3_client_factory`` resource builds the client; the default is
``create_s3_client`` (``boto3.client("s3", ...)``). Tests pass a factory that
returns a ``botocore.stub.Stubber``-wrapped client.

Keys, CSV header removal, JSON lines, retries and messages match V1 on equal
inputs. Differences from V1:

- A change of bucket, prefix, name prefix or file type starts a new object.
  V1 kept its buffer and key, so the next upload copied the earlier rows
  into the other bucket.
- ``max_entries_per_file`` accepts an integer selector (V1 declared a string).
- Failure messages are scrubbed of credentials; a client factory error fails
  the step without its text.
"""

import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
from pydantic import Field, StrictInt, StrictStr
from roboflow_workflows.v2.blocks.formatters._shared.values import WallClock
from roboflow_workflows.v2.blocks.sinks.local_file_sink.v1 import FILE_TIMESTAMP_FORMAT, FileType, with_line_end, without_header
from roboflow_workflows.v2.blocks.sinks.runtime import (
    ERROR_STATUS_OUTPUT,
    MESSAGE_OUTPUT,
    disabled_sink_message,
)
from roboflow_workflows.v2.data.kinds import INTEGER_KIND, STRING_KIND
from roboflow_workflows.v2.data.secrets import (
    SECRET_KIND,
    Revealed,
    reveal_secrets,
    to_secret,
)
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Ref


LOGGER = logging.getLogger(__name__)

BLOCK_TYPE = "roboflow_core/s3_sink@v1"

SAVED_MESSAGE = "Data saved successfully to s3://{}/{}"

MISSING_CREDENTIALS_MESSAGE = (
    "S3 sink requires `aws_access_key_id` and `aws_secret_access_key` to be "
    "provided by the workflow. The server's ambient AWS credentials are not used."
)

CONTENT_TYPES = {
    "csv": "text/csv",
    "json": "application/json",
    "jsonl": "application/x-ndjson",
    "txt": "text/plain",
}

NON_RETRYABLE_CLIENT_ERROR_CODES = frozenset(
    {
        "InvalidAccessKeyId",
        "AccessDenied",
        "NoSuchBucket",
        "InvalidBucketName",
        "SignatureDoesNotMatch",
        "AuthFailure",
    }
)

MAX_UPLOAD_RETRIES = 3

UPLOAD_RETRY_INITIAL_DELAY = 1.0

AppendTarget = Tuple[str, str, str, str]

S3ClientFactory = Callable[..., Any]

def create_s3_client(
    *, aws_access_key_id: str, aws_secret_access_key: str, aws_region: Optional[str]
) -> Any:
    """Build a boto3 S3 client from explicit credentials, as V1 did.

    Args:
        aws_access_key_id: Access key ID.
        aws_secret_access_key: Secret access key.
        aws_region: Region name; ``None`` uses boto3's default region.

    Returns:
        A boto3 S3 client.
    """
    import boto3

    options = {
        "aws_access_key_id": aws_access_key_id,
        "aws_secret_access_key": aws_secret_access_key,
    }
    if aws_region:
        options["region_name"] = aws_region
    client = boto3.client("s3", **options)

    return client

class S3SinkBlock(Block):
    """Upload string content (CSV, JSON or text) to an S3 bucket.

    Example: ``content="$steps.csv_formatter.csv_content"``,
    ``output_mode="separate_files"``, ``bucket_name="my-results"`` and
    ``s3_prefix="logs"`` upload one object per call to
    ``s3://my-results/logs/workflow_output_<time>.csv``.
    """

    type = BLOCK_TYPE
    metadata = {
        "name": "S3 Sink",
        "port_of": "roboflow_core/s3_sink@v1",
        "section": "data_storage",
    }
    outputs = {"error_status": ERROR_STATUS_OUTPUT, "message": MESSAGE_OUTPUT}

    class Params(BlockParams):
        content: Ref(STRING_KIND) = Field(
            description=(
                "Text to upload, matching file_type. In append_log mode CSV "
                "content must include its header line."
            ),
            examples=["$steps.csv_formatter.csv_content"],
        )
        file_type: FileType = Field(
            default="csv",
            description="File format: csv, json (jsonl in append_log mode) or txt.",
            examples=["csv"],
        )
        output_mode: Literal["append_log", "separate_files"] = Field(
            description=(
                "append_log: one object holds many entries, re-uploaded whole "
                "on each call; separate_files: one new object per call."
            ),
            examples=["append_log"],
        )
        bucket_name: StrictStr | Ref(STRING_KIND) = Field(
            description="Name of the target bucket.",
            examples=["my-inference-results"],
        )
        s3_prefix: StrictStr | Ref(STRING_KIND) = Field(
            default="",
            description=(
                "Key prefix (folder path); trailing slashes are removed. Empty "
                "puts objects at the bucket root."
            ),
            examples=["logs/detections"],
        )
        file_name_prefix: StrictStr | Ref(STRING_KIND) = Field(
            default="workflow_output",
            description=(
                "Object name prefix; a timestamp and the extension follow it."
            ),
            examples=["my_output"],
        )
        max_entries_per_file: StrictInt | Ref(INTEGER_KIND) = Field(
            default=1024,
            ge=1,
            description="Entries per object in append_log mode before a new key.",
            examples=[1024],
        )
        aws_access_key_id: Optional[StrictStr] | Ref(SECRET_KIND, STRING_KIND) = Field(
            default=None,
            description=(
                "AWS access key ID; required together with aws_secret_access_key. "
                "Prefer a secret, e.g. from an environment secrets store step."
            ),
            examples=["$steps.secrets.aws_access_key_id"],
        )
        aws_secret_access_key: Optional[StrictStr] | Ref(SECRET_KIND, STRING_KIND) = (
            Field(
                default=None,
                description=(
                    "AWS secret access key; required together with "
                    "aws_access_key_id. Prefer a secret."
                ),
                examples=["$steps.secrets.aws_secret_access_key"],
            )
        )
        aws_region: Optional[StrictStr] | Ref(STRING_KIND) = Field(
            default=None,
            description="Bucket region; None uses boto3's default region.",
            examples=["us-east-1"],
        )

    def __init__(
        self,
        *,
        s3_client_factory: S3ClientFactory = create_s3_client,
        disable_sinks: bool = False,
        wall_clock: WallClock = datetime.now,
        retry_sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client_factory = s3_client_factory
        self._disable_sinks = disable_sinks
        self._wall_clock = wall_clock
        self._retry_sleep = retry_sleep
        # One lock for the buffer and its upload: an older buffer must never
        # overwrite a newer one under the same key.
        self._lock = threading.Lock()
        self._buffer: List[str] = []
        self._append_key: Optional[str] = None
        self._append_target: Optional[AppendTarget] = None

    def run(
        self,
        *,
        content: str,
        file_type: str,
        output_mode: str,
        bucket_name: str,
        s3_prefix: str,
        file_name_prefix: str,
        max_entries_per_file: int,
        aws_access_key_id: Any,
        aws_secret_access_key: Any,
        aws_region: Optional[str],
    ) -> Dict[str, Any]:
        """Upload one entry unless disabled.

        Args:
            content: Text to upload.
            file_type: ``csv``, ``json`` or ``txt``.
            output_mode: ``append_log`` or ``separate_files``.
            bucket_name: Target bucket.
            s3_prefix: Key prefix.
            file_name_prefix: Object name prefix.
            max_entries_per_file: Entries per append_log object.
            aws_access_key_id: Access key ID, a str or a ``Secret``.
            aws_secret_access_key: Secret access key, a str or a ``Secret``.
            aws_region: Region, or ``None``.

        Returns:
            ``error_status`` and ``message``.

        Raises:
            ValueError: When a credential is missing or empty.
            SecretUnavailableError: When a credential is a replayed secret.
            RuntimeError: When the client factory fails; its text is dropped.
        """
        if self._disable_sinks:
            message = disabled_sink_message(by_policy=True)
            return {"error_status": False, "message": message}

        credentials = (
            to_secret(aws_access_key_id, name="aws_access_key_id"),
            to_secret(aws_secret_access_key, name="aws_secret_access_key"),
        )
        if None in credentials:
            raise ValueError(MISSING_CREDENTIALS_MESSAGE)
        revealed = reveal_secrets(credentials)
        key_id, secret_key = revealed.value
        if not key_id or not secret_key:
            raise ValueError(MISSING_CREDENTIALS_MESSAGE)
        try:
            client = self._client_factory(
                aws_access_key_id=key_id,
                aws_secret_access_key=secret_key,
                aws_region=aws_region,
            )
        except Exception as error:
            # Library text may echo a credential; keep only the type.
            raise RuntimeError(
                f"{BLOCK_TYPE}: could not create the S3 client "
                f"({type(error).__name__})"
            ) from None

        if output_mode == "separate_files":
            key = self._new_key(s3_prefix, file_name_prefix, extension=file_type)
            outcome = self._upload(
                client,
                revealed,
                bucket=bucket_name,
                key=key,
                content=content,
                content_type=CONTENT_TYPES[file_type],
            )
            return outcome

        target = (bucket_name, s3_prefix, file_name_prefix, file_type)
        outcome = self._append(
            client,
            revealed,
            content,
            target=target,
            max_entries_per_file=max_entries_per_file,
        )

        return outcome

    def _append(
        self,
        client: Any,
        revealed: Revealed,
        content: str,
        *,
        target: AppendTarget,
        max_entries_per_file: int,
    ) -> Dict[str, Any]:
        bucket, prefix, file_name_prefix, file_type = target
        extension = "jsonl" if file_type == "json" else file_type
        if file_type == "json":
            try:
                content = json.dumps(json.loads(content))
            except ValueError:
                return {"error_status": True, "message": "Invalid JSON content"}

        with self._lock:
            starts_object = (
                self._append_key is None
                or self._append_target != target
                or len(self._buffer) >= max_entries_per_file
            )
            if starts_object:
                self._append_key = self._new_key(
                    prefix, file_name_prefix, extension=extension
                )
                self._append_target = target
                self._buffer = []
            elif file_type == "csv":
                content = without_header(content)
            self._buffer.append(with_line_end(content))
            outcome = self._upload(
                client,
                revealed,
                bucket=bucket,
                key=self._append_key,
                content="".join(self._buffer),
                content_type=CONTENT_TYPES[extension],
            )

        return outcome

    def _new_key(self, s3_prefix: str, file_name_prefix: str, *, extension: str) -> str:
        local_now = self._wall_clock(timezone.utc).astimezone()
        name = (
            f"{file_name_prefix.rstrip('/')}_"
            f"{local_now.strftime(FILE_TIMESTAMP_FORMAT)}.{extension}"
        )
        if not s3_prefix:
            return name

        key = f"{s3_prefix.rstrip('/')}/{name}"

        return key

    def _upload(
        self,
        client: Any,
        revealed: Revealed,
        *,
        bucket: str,
        key: str,
        content: str,
        content_type: str,
    ) -> Dict[str, Any]:
        from botocore.exceptions import BotoCoreError, ClientError

        body = content.encode("utf-8")
        last_error: Optional[Exception] = None
        for attempt in range(1 + MAX_UPLOAD_RETRIES):
            if attempt > 0:
                self._retry_sleep(UPLOAD_RETRY_INITIAL_DELAY * 2 ** (attempt - 1))
            try:
                client.put_object(
                    Bucket=bucket, Key=key, Body=body, ContentType=content_type
                )
            except ClientError as error:
                code = error.response.get("Error", {}).get("Code")
                if code in NON_RETRYABLE_CLIENT_ERROR_CODES:
                    return failed(revealed, error)
                last_error = error
            except BotoCoreError as error:
                last_error = error
            except Exception as error:
                return failed(revealed, error)
            else:
                message = SAVED_MESSAGE.format(bucket, key)
                return {"error_status": False, "message": message}

            LOGGER.warning(
                "S3 upload attempt %d/%d to s3://%s/%s failed: %s",
                attempt + 1,
                1 + MAX_UPLOAD_RETRIES,
                bucket,
                key,
                revealed.scrub(str(last_error)),
            )

        outcome = failed(revealed, last_error)

        return outcome

def failed(revealed: Revealed, error: Exception) -> Dict[str, Any]:
    """Return a failed outcome with V1's ``str(error)``, scrubbed.

    Args:
        revealed: Revealed credentials of this call.
        error: The upload error.

    Returns:
        ``error_status=True`` and the scrubbed message.
    """
    outcome = {"error_status": True, "message": revealed.scrub(str(error))}

    return outcome
