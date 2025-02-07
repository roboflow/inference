import os
from typing import Any, Dict, Optional

from inference_cli.lib.utils import ensure_inference_is_installed
from inference_cli.lib.workflows.entities import OutputFileType, ProcessingTarget
from inference_cli.lib.workflows.remote_image_adapter import (
    process_image_directory_with_workflow_using_api,
    process_image_with_workflow_using_api,
)


def run_video_processing_with_workflows(
    input_video_path: str,
    output_directory: str,
    output_file_type: OutputFileType,
    workflow_specification: Optional[dict] = None,
    workspace_name: Optional[str] = None,
    workflow_id: Optional[str] = None,
    workflow_parameters: Optional[Dict[str, Any]] = None,
    image_input_name: str = "image",
    max_fps: Optional[float] = None,
    save_image_outputs_as_video: bool = True,
    api_key: Optional[str] = None,
) -> None:
    # enabling new behaviour ensuring frame rate will be subsample (needed until
    # this becomes default)
    os.environ["ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING"] = "True"

    ensure_inference_is_installed()

    from inference_cli.lib.workflows.video_adapter import process_video_with_workflow

    _ = process_video_with_workflow(
        input_video_path=input_video_path,
        output_directory=output_directory,
        output_file_type=output_file_type,
        workflow_specification=workflow_specification,
        workspace_name=workspace_name,
        workflow_id=workflow_id,
        workflow_parameters=workflow_parameters,
        image_input_name=image_input_name,
        max_fps=max_fps,
        save_image_outputs_as_video=save_image_outputs_as_video,
        api_key=api_key,
    )


def process_image_with_workflow(
    image_path: str,
    output_directory: str,
    processing_target: ProcessingTarget,
    workflow_specification: Optional[dict] = None,
    workspace_name: Optional[str] = None,
    workflow_id: Optional[str] = None,
    workflow_parameters: Optional[Dict[str, Any]] = None,
    image_input_name: str = "image",
    api_key: Optional[str] = None,
    api_url: str = "https://detect.roboflow.com",
    save_image_outputs: bool = True,
    force_reprocessing: bool = False,
) -> None:
    if processing_target is ProcessingTarget.INFERENCE_PACKAGE:

        ensure_inference_is_installed()

        from inference_cli.lib.workflows.local_image_adapter import (
            process_image_with_workflow_using_inference_package,
        )

        process_image_with_workflow_using_inference_package(
            image_path=image_path,
            output_directory=output_directory,
            workflow_specification=workflow_specification,
            workspace_name=workspace_name,
            workflow_id=workflow_id,
            workflow_parameters=workflow_parameters,
            image_input_name=image_input_name,
            api_key=api_key,
            save_image_outputs=save_image_outputs,
            force_reprocessing=force_reprocessing,
        )
        return None
    process_image_with_workflow_using_api(
        image_path=image_path,
        output_directory=output_directory,
        workflow_specification=workflow_specification,
        workspace_name=workspace_name,
        workflow_id=workflow_id,
        workflow_parameters=workflow_parameters,
        image_input_name=image_input_name,
        api_key=api_key,
        api_url=api_url,
        save_image_outputs=save_image_outputs,
        force_reprocessing=force_reprocessing,
    )
    return None


def process_images_directory_with_workflow(
    input_directory: str,
    output_directory: str,
    processing_target: ProcessingTarget,
    workflow_specification: Optional[dict] = None,
    workspace_name: Optional[str] = None,
    workflow_id: Optional[str] = None,
    workflow_parameters: Optional[Dict[str, Any]] = None,
    image_input_name: str = "image",
    api_key: Optional[str] = None,
    save_image_outputs: bool = True,
    force_reprocessing: bool = False,
    aggregate_structured_results: bool = True,
    aggregation_format: OutputFileType = OutputFileType.JSONL,
    debug_mode: bool = False,
    api_url: str = "https://detect.roboflow.com",
    processing_threads: Optional[int] = None,
    max_failures: Optional[int] = None,
) -> None:
    if processing_target is ProcessingTarget.INFERENCE_PACKAGE:

        ensure_inference_is_installed()

        from inference_cli.lib.workflows.local_image_adapter import (
            process_image_directory_with_workflow_using_inference_package,
        )

        _ = process_image_directory_with_workflow_using_inference_package(
            input_directory=input_directory,
            output_directory=output_directory,
            workflow_specification=workflow_specification,
            workspace_name=workspace_name,
            workflow_id=workflow_id,
            workflow_parameters=workflow_parameters,
            image_input_name=image_input_name,
            api_key=api_key,
            save_image_outputs=save_image_outputs,
            force_reprocessing=force_reprocessing,
            aggregate_structured_results=aggregate_structured_results,
            aggregation_format=aggregation_format,
            debug_mode=debug_mode,
            max_failures=max_failures,
        )
        return None
    _ = process_image_directory_with_workflow_using_api(
        input_directory=input_directory,
        output_directory=output_directory,
        workflow_specification=workflow_specification,
        workspace_name=workspace_name,
        workflow_id=workflow_id,
        workflow_parameters=workflow_parameters,
        image_input_name=image_input_name,
        api_key=api_key,
        api_url=api_url,
        save_image_outputs=save_image_outputs,
        force_reprocessing=force_reprocessing,
        aggregate_structured_results=aggregate_structured_results,
        aggregation_format=aggregation_format,
        debug_mode=debug_mode,
        processing_threads=processing_threads,
        max_failures=max_failures,
    )
    return None
