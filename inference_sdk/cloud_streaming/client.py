import contextlib
import os
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from inference_sdk.cloud_streaming.api import CloudStreamingAPI, CloudStreamingError
from inference_sdk.cloud_streaming.sources import LocalSource, Publisher
from inference_sdk.cloud_streaming.stream import CloudStream

# The relay admits a temporary source's publisher within two minutes of allocation.
PUBLISH_TIMEOUT_S = 60.0

ProcessorSize = Literal["gpu-s-v1", "gpu-m-v1", "gpu-l-v1", "cpu-standard-v1"]


class CloudStreamingClient:
    """Run Workflows on live video with Roboflow Cloud Streaming.

    Args:
        api_key: A workspace API key; defaults to the ROBOFLOW_API_KEY environment variable.
        workspace: The workspace's URL slug; looked up from the key when omitted.
        api_url: The platform API; defaults to RF_API_BASE_URL, ROBOFLOW_REGION and
            ROBOFLOW_ENVIRONMENT (https://api.roboflow.com).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        api_url: Optional[str] = None,
    ):
        api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            raise ValueError("api_key is required (or set ROBOFLOW_API_KEY)")
        self._api = CloudStreamingAPI(api_key, workspace=workspace, api_url=api_url)

    @property
    def workspace(self) -> str:
        return self._api.workspace

    def list_sources(self) -> List[Dict[str, Any]]:
        """The workspace's Cloud Streaming sources (cameras, RTSP and RTMP sources, files)."""
        return self._api.list_sources()

    def start_stream(
        self,
        source: Union[str, LocalSource],
        workflow: Union[str, Dict[str, Any]],
        *,
        parameters: Optional[Dict[str, Any]] = None,
        image_output: Optional[str] = None,
        max_fps: Optional[float] = None,
        processor_size: Optional[ProcessorSize] = None,
        purpose: Literal["preview", "deployment"] = "preview",
    ) -> CloudStream:
        """Run a Workflow on a source and return the stream.

        Args:
            source: A registered source id (see `list_sources()`), or local video to publish:
                `WebcamSource`, `VideoFileSource`, `RTSPSource` or `ManualSource`.
            workflow: A saved Workflow's id, or a Workflow specification.
            parameters: Values for the Workflow's non-image inputs.
            image_output: The image output that `video()` shows by default.
            max_fps: Process at most this many frames per second.
            processor_size: The GPU or CPU size to run on.
            purpose: `preview` (the default) runs while this stream keeps it alive, for at most an
                hour. `deployment`, on a registered source only, keeps running after this process
                exits, until stopped. Both bill for processing.
        """
        job: Dict[str, Any] = {"purpose": purpose}
        if isinstance(workflow, str):
            job["workflowUrl"] = workflow
        else:
            job["workflowSpecification"] = workflow
        for field, value in (
            ("workflowParameters", parameters),
            ("imageOutput", image_output),
            ("maxFps", max_fps),
            ("processorSize", processor_size),
        ):
            if value is not None:
                job[field] = value
        if isinstance(source, str):
            created = self._api.start_job(
                source, job, idempotency_key=str(uuid.uuid4())
            )
            return CloudStream(self._api, created, owns_job=True)
        if purpose != "preview":
            raise ValueError(
                "local video runs as a preview; deploy a registered source instead"
            )
        return self._start_local(source, job)

    def attach_stream(self, job_id: str) -> CloudStream:
        """Follow a running job, such as a deployment, without taking ownership of it."""
        return CloudStream(self._api, self._api.get_job(job_id), owns_job=False)

    def _start_local(self, source: LocalSource, job: Dict[str, Any]) -> CloudStream:
        session = self._api.create_session(
            source.name, idempotency_key=str(uuid.uuid4())
        )
        publisher = None
        try:
            if not session.get("rtspsUrl"):
                raise CloudStreamingError(
                    "this region's Cloud Streaming cells do not accept published video yet"
                )
            publisher = Publisher(source, session["rtspsUrl"])
            publisher.start()
            publisher.wait_until_publishing(PUBLISH_TIMEOUT_S)
            created = self._api.start_job(
                session["sourceId"], job, idempotency_key=str(uuid.uuid4())
            )
        except BaseException:
            if publisher:
                publisher.stop()
            with contextlib.suppress(CloudStreamingError):
                self._api.stop_session(session["sourceId"])
            raise
        return CloudStream(
            self._api,
            created,
            owns_job=True,
            session_id=session["sourceId"],
            publisher=publisher,
        )
