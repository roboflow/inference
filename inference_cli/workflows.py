from typing import Any, Dict, List, Optional

import typer
from typing_extensions import Annotated

from inference_cli.lib.utils import read_json
from inference_cli.lib.workflows.core import run_video_processing_with_workflows
from inference_cli.lib.workflows.entities import OutputFileType

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
            "--workflow_params_file",
            help="Path to JSON document with Workflow parameters - helpful when Workflow is parametrized and "
            "passing the parameters in CLI is not handy / impossible due to typing conversion issues.",
        ),
    ] = None,
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
    debug_mode: Annotated[
        bool,
        typer.Option(
            "--debug_mode/--no_debug_mode",
            help="Flag enabling errors stack traces to be displayed (helpful for debugging)",
        ),
    ] = False,
):
    try:
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
        return float(value)
    except ValueError:
        pass
    try:
        return int(value)
    except ValueError:
        pass
    if value.lower() in {"y", "yes", "true"}:
        return True
    if value.lower() in {"n", "no", "false"}:
        return False
    return value
