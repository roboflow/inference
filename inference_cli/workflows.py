from typing import Any, Dict, List, Optional

import typer
from typing_extensions import Annotated

from inference_cli.lib.utils import ensure_target_directory_is_empty, read_json
from inference_cli.lib.workflows.core import (
    process_image_with_workflow,
    process_images_directory_with_workflow,
    run_video_processing_with_workflows,
)
from inference_cli.lib.workflows.entities import OutputFileType, ProcessingTarget

workflows_app = typer.Typer(help="Commands for interacting with Roboflow Workflows")


@workflows_app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    help="Process video file with your Workflow locally (inference Python package required)",
)
def process_video(
    context: typer.Context,
    video_path: Annotated[
        str,
        typer.Option(
            "--video_path",
            "-v",
            help="Path to video to be processed",
        ),
    ],
    output_directory: Annotated[
        str,
        typer.Option(
            "--output_dir",
            "-o",
            help="Path to output directory",
        ),
    ],
    output_file_type: Annotated[
        OutputFileType,
        typer.Option(
            "--output_file_type",
            "-ft",
            help="Type of the output file",
            case_sensitive=False,
        ),
    ] = OutputFileType.CSV,
    workflow_specification_path: Annotated[
        Optional[str],
        typer.Option(
            "--workflow_spec",
            "-ws",
            help="Path to JSON file with Workflow definition "
            "(mutually exclusive with `workspace_name` and `workflow_id`)",
        ),
    ] = None,
    workspace_name: Annotated[
        Optional[str],
        typer.Option(
            "--workspace_name",
            "-wn",
            help="Name of Roboflow workspace the that Workflow belongs to "
            "(mutually exclusive with `workflow_specification_path`)",
        ),
    ] = None,
    workflow_id: Annotated[
        Optional[str],
        typer.Option(
            "--workflow_id",
            "-wid",
            help="Identifier of a Workflow on Roboflow platform "
            "(mutually exclusive with `workflow_specification_path`)",
        ),
    ] = None,
    workflow_parameters_path: Annotated[
        Optional[str],
        typer.Option(
            "--workflow_params",
            help="Path to JSON document with Workflow parameters - helpful when Workflow is parametrized and "
            "passing the parameters in CLI is not handy / impossible due to typing conversion issues.",
        ),
    ] = None,
    image_input_name: Annotated[
        str,
        typer.Option(
            "--image_input_name",
            help="Name of the Workflow input that defines placeholder for image to be processed",
        ),
    ] = "image",
    max_fps: Annotated[
        Optional[float],
        typer.Option(
            "--max_fps",
            help="Use the parameter to limit video FPS (additional frames will be skipped in processing).",
        ),
    ] = None,
    image_outputs_as_video: Annotated[
        bool,
        typer.Option(
            "--save_out_video/--no_save_out_video",
            help="Flag deciding if image outputs of the workflow should be saved as video file",
        ),
    ] = True,
    api_key: Annotated[
        Optional[str],
        typer.Option(
            "--api-key",
            "-a",
            help="Roboflow API key for your workspace. If not given - env variable `ROBOFLOW_API_KEY` will be used",
        ),
    ] = None,
    allow_override: Annotated[
        bool,
        typer.Option(
            "--allow_override/--no_override",
            help="Flag to decide if content of output directory can be overridden.",
        ),
    ] = False,
    debug_mode: Annotated[
        bool,
        typer.Option(
            "--debug_mode/--no_debug_mode",
            help="Flag enabling errors stack traces to be displayed (helpful for debugging)",
        ),
    ] = False,
):
    try:
        ensure_target_directory_is_empty(
            output_directory=output_directory,
            allow_override=allow_override,
        )
        workflow_parameters = prepare_workflow_parameters(
            context=context,
            workflows_parameters_path=workflow_parameters_path,
        )
        workflow_specification = None
        if workflow_specification_path is not None:
            workflow_specification = read_json(path=workflow_specification_path)
        run_video_processing_with_workflows(
            input_video_path=video_path,
            output_directory=output_directory,
            output_file_type=output_file_type,
            workflow_specification=workflow_specification,
            workspace_name=workspace_name,
            workflow_id=workflow_id,
            workflow_parameters=workflow_parameters,
            image_input_name=image_input_name,
            max_fps=max_fps,
            save_image_outputs_as_video=image_outputs_as_video,
            api_key=api_key,
        )
    except KeyboardInterrupt:
        print("Command interrupted - results may not be fully consistent.")
        return
    except Exception as error:
        if debug_mode:
            raise error
        typer.echo(f"Command failed. Cause: {error}")
        raise typer.Exit(code=1)


@workflows_app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    help="Process single image with Workflows (inference Package may be needed dependent on mode)",
)
def process_image(
    context: typer.Context,
    image_path: Annotated[
        str,
        typer.Option(
            "--image_path",
            "-i",
            help="Path to image to be processed",
        ),
    ],
    output_directory: Annotated[
        str,
        typer.Option(
            "--output_dir",
            "-o",
            help="Path to output directory",
        ),
    ],
    processing_target: Annotated[
        ProcessingTarget,
        typer.Option(
            "--processing_target",
            "-pt",
            help="Defines where the actual processing will be done, either in inference Python package "
            "running locally, or behind the API (which ensures greater throughput).",
            case_sensitive=False,
        ),
    ] = ProcessingTarget.API,
    workflow_specification_path: Annotated[
        Optional[str],
        typer.Option(
            "--workflow_spec",
            "-ws",
            help="Path to JSON file with Workflow definition "
            "(mutually exclusive with `workspace_name` and `workflow_id`)",
        ),
    ] = None,
    workspace_name: Annotated[
        Optional[str],
        typer.Option(
            "--workspace_name",
            "-wn",
            help="Name of Roboflow workspace the that Workflow belongs to "
            "(mutually exclusive with `workflow_specification_path`)",
        ),
    ] = None,
    workflow_id: Annotated[
        Optional[str],
        typer.Option(
            "--workflow_id",
            "-wid",
            help="Identifier of a Workflow on Roboflow platform "
            "(mutually exclusive with `workflow_specification_path`)",
        ),
    ] = None,
    workflow_parameters_path: Annotated[
        Optional[str],
        typer.Option(
            "--workflow_params",
            help="Path to JSON document with Workflow parameters - helpful when Workflow is parametrized and "
            "passing the parameters in CLI is not handy / impossible due to typing conversion issues.",
        ),
    ] = None,
    image_input_name: Annotated[
        str,
        typer.Option(
            "--image_input_name",
            help="Name of the Workflow input that defines placeholder for image to be processed",
        ),
    ] = "image",
    api_key: Annotated[
        Optional[str],
        typer.Option(
            "--api-key",
            "-a",
            help="Roboflow API key for your workspace. If not given - env variable `ROBOFLOW_API_KEY` will be used",
        ),
    ] = None,
    api_url: Annotated[
        str,
        typer.Option(
            "--api_url",
            help="URL of the API that will be used for processing, when API processing target pointed.",
        ),
    ] = "https://detect.roboflow.com",
    allow_override: Annotated[
        bool,
        typer.Option(
            "--allow_override/--no_override",
            help="Flag to decide if content of output directory can be overridden.",
        ),
    ] = False,
    save_image_outputs: Annotated[
        bool,
        typer.Option(
            "--save_image_outputs/--no_save_image_outputs",
            help="Flag controlling persistence of Workflow outputs that are images",
        ),
    ] = True,
    force_reprocessing: Annotated[
        bool,
        typer.Option(
            "--force_reprocessing/--no_reprocessing",
            help="Flag to enforce re-processing of specific images. Images are identified by file name.",
        ),
    ] = False,
    debug_mode: Annotated[
        bool,
        typer.Option(
            "--debug_mode/--no_debug_mode",
            help="Flag enabling errors stack traces to be displayed (helpful for debugging)",
        ),
    ] = False,
) -> None:
    try:
        ensure_target_directory_is_empty(
            output_directory=output_directory,
            allow_override=allow_override,
            only_files=False,
        )
        workflow_parameters = prepare_workflow_parameters(
            context=context,
            workflows_parameters_path=workflow_parameters_path,
        )
        workflow_specification = None
        if workflow_specification_path is not None:
            workflow_specification = read_json(path=workflow_specification_path)
        process_image_with_workflow(
            image_path=image_path,
            output_directory=output_directory,
            processing_target=processing_target,
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
    except KeyboardInterrupt:
        print("Command interrupted - results may not be fully consistent.")
        return
    except Exception as error:
        if debug_mode:
            raise error
        typer.echo(f"Command failed. Cause: {error}")
        raise typer.Exit(code=1)


@workflows_app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    help="Process whole images directory with Workflows (inference Package may be needed dependent on mode)",
)
def process_images_directory(
    context: typer.Context,
    input_directory: Annotated[
        str,
        typer.Option(
            "--input_directory",
            "-i",
            help="Path to directory with images",
        ),
    ],
    output_directory: Annotated[
        str,
        typer.Option(
            "--output_dir",
            "-o",
            help="Path to output directory",
        ),
    ],
    processing_target: Annotated[
        ProcessingTarget,
        typer.Option(
            "--processing_target",
            "-pt",
            help="Defines where the actual processing will be done, either in inference Python package "
            "running locally, or behind the API (which ensures greater throughput).",
            case_sensitive=False,
        ),
    ] = ProcessingTarget.API,
    workflow_specification_path: Annotated[
        Optional[str],
        typer.Option(
            "--workflow_spec",
            "-ws",
            help="Path to JSON file with Workflow definition "
            "(mutually exclusive with `workspace_name` and `workflow_id`)",
        ),
    ] = None,
    workspace_name: Annotated[
        Optional[str],
        typer.Option(
            "--workspace_name",
            "-wn",
            help="Name of Roboflow workspace the that Workflow belongs to "
            "(mutually exclusive with `workflow_specification_path`)",
        ),
    ] = None,
    workflow_id: Annotated[
        Optional[str],
        typer.Option(
            "--workflow_id",
            "-wid",
            help="Identifier of a Workflow on Roboflow platform "
            "(mutually exclusive with `workflow_specification_path`)",
        ),
    ] = None,
    workflow_parameters_path: Annotated[
        Optional[str],
        typer.Option(
            "--workflow_params",
            help="Path to JSON document with Workflow parameters - helpful when Workflow is parametrized and "
            "passing the parameters in CLI is not handy / impossible due to typing conversion issues.",
        ),
    ] = None,
    image_input_name: Annotated[
        str,
        typer.Option(
            "--image_input_name",
            help="Name of the Workflow input that defines placeholder for image to be processed",
        ),
    ] = "image",
    api_key: Annotated[
        Optional[str],
        typer.Option(
            "--api-key",
            "-a",
            help="Roboflow API key for your workspace. If not given - env variable `ROBOFLOW_API_KEY` will be used",
        ),
    ] = None,
    api_url: Annotated[
        str,
        typer.Option(
            "--api_url",
            help="URL of the API that will be used for processing, when API processing target pointed.",
        ),
    ] = "https://detect.roboflow.com",
    allow_override: Annotated[
        bool,
        typer.Option(
            "--allow_override/--no_override",
            help="Flag to decide if content of output directory can be overridden.",
        ),
    ] = False,
    save_image_outputs: Annotated[
        bool,
        typer.Option(
            "--save_image_outputs/--no_save_image_outputs",
            help="Flag controlling persistence of Workflow outputs that are images",
        ),
    ] = True,
    force_reprocessing: Annotated[
        bool,
        typer.Option(
            "--force_reprocessing/--no_reprocessing",
            help="Flag to enforce re-processing of specific images. Images are identified by file name.",
        ),
    ] = False,
    aggregate_structured_results: Annotated[
        bool,
        typer.Option(
            "--aggregate/--no_aggregate",
            help="Flag to decide if processing results for a directory should be aggregated to a single file "
            "at the end of processing.",
        ),
    ] = True,
    aggregation_format: Annotated[
        OutputFileType,
        typer.Option(
            "--aggregation_format",
            "-af",
            help="Defines the format of aggregated results - either CSV of JSONL",
            case_sensitive=False,
        ),
    ] = OutputFileType.CSV,
    processing_threads: Annotated[
        Optional[int],
        typer.Option(
            "--threads",
            help="Defines number of threads that will be used to send requests when processing target is API. "
            "Default for Roboflow Hosted API is 32, and for on-prem deployments: 1.",
        ),
    ] = None,
    debug_mode: Annotated[
        bool,
        typer.Option(
            "--debug_mode/--no_debug_mode",
            help="Flag enabling errors stack traces to be displayed (helpful for debugging)",
        ),
    ] = False,
    max_failures: Annotated[
        Optional[int],
        typer.Option(
            "--max-failures",
            help="Maximum number of Workflow executions for directory images which will be tolerated before give up. "
            "If not set - unlimited.",
        ),
    ] = None,
) -> None:
    try:
        ensure_target_directory_is_empty(
            output_directory=output_directory,
            allow_override=allow_override,
            only_files=False,
        )
        workflow_parameters = prepare_workflow_parameters(
            context=context,
            workflows_parameters_path=workflow_parameters_path,
        )
        workflow_specification = None
        if workflow_specification_path is not None:
            workflow_specification = read_json(path=workflow_specification_path)
        process_images_directory_with_workflow(
            input_directory=input_directory,
            output_directory=output_directory,
            processing_target=processing_target,
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
            processing_threads=processing_threads,
            debug_mode=debug_mode,
            max_failures=max_failures,
        )
    except KeyboardInterrupt:
        print("Command interrupted - results may not be fully consistent.")
        return
    except Exception as error:
        if debug_mode:
            raise error
        typer.echo(f"Command failed. Cause: {error}")
        raise typer.Exit(code=1)


def prepare_workflow_parameters(
    context: typer.Context,
    workflows_parameters_path: Optional[str],
) -> Optional[dict]:
    workflow_parameters = _parse_extra_args(context=context)
    if workflows_parameters_path is None:
        return workflow_parameters
    workflows_parameters_from_file = read_json(path=workflows_parameters_path)
    if workflow_parameters is None:
        return workflows_parameters_from_file
    # explicit params in CLI override file params
    workflows_parameters_from_file.update(workflow_parameters)
    return workflows_parameters_from_file


def _parse_extra_args(context: typer.Context) -> Optional[Dict[str, Any]]:
    indices_of_named_parameters = _get_indices_of_named_parameters(
        arguments=context.args
    )
    if not indices_of_named_parameters:
        return None
    params_values_spans = _calculate_spans_between_indices(
        indices=indices_of_named_parameters,
        list_length=len(context.args),
    )
    result = {}
    for param_index, values_span in zip(
        indices_of_named_parameters, params_values_spans
    ):
        if values_span < 1:
            continue
        name = context.args[param_index].lstrip("-")
        values = [
            _parse_value(value=v)
            for v in context.args[param_index + 1 : param_index + values_span + 1]
        ]
        if len(values) == 1:
            values = values[0]
        result[name] = values
    if not result:
        return None
    return result


def _get_indices_of_named_parameters(arguments: List[str]) -> List[int]:
    result = []
    for index, arg in enumerate(arguments):
        if arg.startswith("-"):
            result.append(index)
    return result


def _calculate_spans_between_indices(indices: List[int], list_length: int) -> List[int]:
    if not indices:
        return []
    diffs = [x - y - 1 for x, y in zip(indices[1:], indices)]
    diffs.append(list_length - indices[-1] - 1)
    return diffs


def _parse_value(value: str) -> Any:
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.lower() in {"y", "yes", "true"}:
        return True
    if value.lower() in {"n", "no", "false"}:
        return False
    return value
