from typing import Optional

import typer
from typing_extensions import Annotated

from inference_cli.lib.env import ROBOFLOW_API_KEY
from inference_cli.lib.roboflow_cloud.data_staging.api_operations import (
    create_images_batch_from_directory,
    display_batch_content,
    display_batch_count,
    display_batch_shards_statuses,
    display_batches, create_videos_batch_from_directory,
)

data_staging_app = typer.Typer(
    help="Commands for interacting with Roboflow Data Staging"
)


@data_staging_app.command(help="List staging batches in your workspace.")
def list_staging_batches(
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
            "--debug-mode/--no-debug-mode",
            help="Flag enabling errors stack traces to be displayed (helpful for debugging)",
        ),
    ] = False,
) -> None:
    if api_key is None:
        api_key = ROBOFLOW_API_KEY
    try:
        display_batches(api_key=api_key)
    except KeyboardInterrupt:
        print("Command interrupted.")
        return
    except Exception as error:
        if debug_mode:
            raise error
        typer.echo(f"Command failed. Cause: {error}")
        raise typer.Exit(code=1)


@data_staging_app.command(help="Create new batch with your images.")
def create_batch_of_images(
    batch_id: Annotated[
        str,
        typer.Option(
            "--batch-id",
            "-b",
            help="Identifier of new batch (must be lower-cased letters with '-' and '_' allowed",
        ),
    ],
    images_dir: Annotated[
        str,
        typer.Option(
            "--images-dir",
            "-i",
            help="Path to your images directory to upload",
        ),
    ],
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
            "--debug-mode/--no-debug-mode",
            help="Flag enabling errors stack traces to be displayed (helpful for debugging)",
        ),
    ] = False,
) -> None:
    if api_key is None:
        api_key = ROBOFLOW_API_KEY
    try:
        create_images_batch_from_directory(
            directory=images_dir, batch_id=batch_id, api_key=api_key
        )
    except KeyboardInterrupt:
        print("Command interrupted.")
        return
    except Exception as error:
        if debug_mode:
            raise error
        typer.echo(f"Command failed. Cause: {error}")
        raise typer.Exit(code=1)


@data_staging_app.command(help="Create new batch with your videos.")
def create_batch_of_videos(
    batch_id: Annotated[
        str,
        typer.Option(
            "--batch-id",
            "-b",
            help="Identifier of new batch (must be lower-cased letters with '-' and '_' allowed",
        ),
    ],
    videos_dir: Annotated[
        str,
        typer.Option(
            "--videos-dir",
            "-v",
            help="Path to your videos directory to upload",
        ),
    ],
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
            "--debug-mode/--no-debug-mode",
            help="Flag enabling errors stack traces to be displayed (helpful for debugging)",
        ),
    ] = False,
) -> None:
    if api_key is None:
        api_key = ROBOFLOW_API_KEY
    try:
        create_videos_batch_from_directory(
            directory=videos_dir, batch_id=batch_id, api_key=api_key
        )
    except KeyboardInterrupt:
        print("Command interrupted.")
        return
    except Exception as error:
        if debug_mode:
            raise error
        typer.echo(f"Command failed. Cause: {error}")
        raise typer.Exit(code=1)


@data_staging_app.command(help="Display batch size")
def display_batch_size(
    batch_id: Annotated[
        str,
        typer.Option(
            "--batch-id",
            "-b",
            help="Identifier of new batch (must be lower-cased letters with '-' and '_' allowed",
        ),
    ],
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
            "--debug-mode/--no-debug-mode",
            help="Flag enabling errors stack traces to be displayed (helpful for debugging)",
        ),
    ] = False,
) -> None:
    if api_key is None:
        api_key = ROBOFLOW_API_KEY
    try:
        display_batch_count(batch_id=batch_id, api_key=api_key)
    except KeyboardInterrupt:
        print("Command interrupted.")
        return
    except Exception as error:
        if debug_mode:
            raise error
        typer.echo(f"Command failed. Cause: {error}")
        raise typer.Exit(code=1)


@data_staging_app.command(help="List batch content")
def list_batch_content(
    batch_id: Annotated[
        str,
        typer.Option(
            "--batch-id",
            "-b",
            help="Identifier of new batch (must be lower-cased letters with '-' and '_' allowed",
        ),
    ],
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
            "--debug-mode/--no-debug-mode",
            help="Flag enabling errors stack traces to be displayed (helpful for debugging)",
        ),
    ] = False,
) -> None:
    if api_key is None:
        api_key = ROBOFLOW_API_KEY
    try:
        display_batch_content(batch_id=batch_id, api_key=api_key)
    except KeyboardInterrupt:
        print("Command interrupted.")
        return
    except Exception as error:
        if debug_mode:
            raise error
        typer.echo(f"Command failed. Cause: {error}")
        raise typer.Exit(code=1)


@data_staging_app.command(help="Display shards statuses")
def display_shards_details(
    batch_id: Annotated[
        str,
        typer.Option(
            "--batch-id",
            "-b",
            help="Identifier of new batch (must be lower-cased letters with '-' and '_' allowed",
        ),
    ],
    page: Annotated[
        int,
        typer.Option(
            "--page",
            "-p",
            help="Pagination page",
        ),
    ] = 0,
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
            "--debug-mode/--no-debug-mode",
            help="Flag enabling errors stack traces to be displayed (helpful for debugging)",
        ),
    ] = False,
) -> None:
    if api_key is None:
        api_key = ROBOFLOW_API_KEY
    try:
        display_batch_shards_statuses(batch_id=batch_id, api_key=api_key, page=page)
    except KeyboardInterrupt:
        print("Command interrupted.")
        return
    except Exception as error:
        if debug_mode:
            raise error
        typer.echo(f"Command failed. Cause: {error}")
        raise typer.Exit(code=1)
