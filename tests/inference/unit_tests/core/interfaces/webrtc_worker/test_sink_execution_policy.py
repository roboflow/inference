from unittest.mock import MagicMock

from inference.core.interfaces.stream.inference_pipeline import InferencePipeline
from inference.core.interfaces.stream_manager.manager_app.entities import (
    WorkflowConfiguration,
)
from inference.core.interfaces.webrtc_worker.webrtc import VideoFrameProcessor


def test_video_frame_processor_passes_sink_execution_policy(monkeypatch) -> None:
    pipeline_init = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(InferencePipeline, "init_with_workflow", pipeline_init)
    workflow_configuration = WorkflowConfiguration(
        type="WorkflowConfiguration",
        workflow_specification={"version": "1.0", "outputs": []},
        disable_sinks=True,
    )

    VideoFrameProcessor(
        asyncio_loop=MagicMock(),
        workflow_configuration=workflow_configuration,
        api_key="api-key",
        has_video_track=False,
    )

    assert pipeline_init.call_args.kwargs["disable_sinks"] is True
