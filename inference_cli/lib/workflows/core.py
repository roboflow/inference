import os
from typing import Any, Dict, Literal, Optional, Union

from inference_cli.lib.utils import ensure_inference_is_installed, read_json
from inference_cli.lib.workflows.entities import OutputFileType


def run_video_processing_with_workflows(
    input_video_path: str,
    output_directory: str,
    output_file_type: OutputFileType,
    workflow_specification: Optional[dict] = None,
    workspace_name: Optional[str] = None,
    workflow_id: Optional[str] = None,
    workflow_parameters: Optional[Dict[str, Any]] = None,
    max_fps: Optional[float] = None,
    save_image_outputs_as_video: bool = True,
    api_key: Optional[str] = None,
) -> None:
    # enabling new behaviour ensuring frame rate will be subsample (needed until
    # this becomes default)
    os.environ["ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING"] = "True"

    ensure_inference_is_installed()

    from inference_cli.lib.workflows.video_adapter import process_video_with_workflow

    process_video_with_workflow(
        input_video_path=input_video_path,
        output_directory=output_directory,
        output_file_type=output_file_type,
        workflow_specification=workflow_specification,
        workspace_name=workspace_name,
        workflow_id=workflow_id,
        workflow_parameters=workflow_parameters,
        max_fps=max_fps,
        save_image_outputs_as_video=save_image_outputs_as_video,
        api_key=api_key,
    )
