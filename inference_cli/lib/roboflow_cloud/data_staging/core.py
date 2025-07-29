from typing import List, Optional
import time

import typer
from typing_extensions import Annotated
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from inference_cli.lib.env import ROBOFLOW_API_KEY
from inference_cli.lib.roboflow_cloud.common import ensure_api_key_is_set, get_workspace
from inference_cli.lib.roboflow_cloud.data_staging import api_operations
from inference_cli.lib.roboflow_cloud.data_staging.entities import DataSource

data_staging_app = typer.Typer(
    help="Commands for interacting with Roboflow Data Staging. THIS IS ALPHA PREVIEW OF THE FEATURE."
)


@data_staging_app.command(help="List staging batches in your workspace.")
def list_batches(
    api_key: Annotated[
        Optional[str],
        typer.Option(
            "--api-key",
            "-a",
            help="Roboflow API key for your workspace. If not given - env variable `ROBOFLOW_API_KEY` will be used",
        ),
    ] = None,
    pages: Annotated[
        int,
        typer.Option("--pages", "-p", help="Number of pages to pull"),
    ] = 1,
    page_size: Annotated[
        Optional[int],
        typer.Option("--page-size", help="Size of pagination page"),
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
        ensure_api_key_is_set(api_key=api_key)
        api_operations.display_batches(
            api_key=api_key, pages=pages, page_size=page_size
        )
    except KeyboardInterrupt:
        print("Command interrupted.")
        raise typer.Exit(code=2)
    except Exception as error:
        if debug_mode:
            raise error
        typer.echo(f"Command failed. Cause: {error}")
        raise typer.Exit(code=1)


@data_staging_app.command(help="List content of individual batch.")
def list_batch_content(
    batch_id: Annotated[
        str,
        typer.Option(
            "--batch-id",
            "-b",
            help="Identifier of new batch (must be lower-cased letters with '-' and '_' allowed",
        ),
    ],
    part_names: Annotated[
        Optional[List[str]],
        typer.Option(
            "--part-name",
            "-pn",
            help="Name of the part to be listed (if not given - all parts are presented). Invalid if batch is "
            "not multipart",
        ),
    ] = None,
    limit: Annotated[
        Optional[int],
        typer.Option(
            "--limit",
            "-l",
            help="Number of entries to display / dump into th output files. If not given - whole "
            "content will be presented.",
        ),
    ] = None,
    output_file: Annotated[
        Optional[str],
        typer.Option(
            "--output-file",
            "-o",
            help="Path to the output file - if not provided, command will dump result to the console. "
            "File type is JSONL.",
        ),
    ] = None,
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
    if part_names:
        part_names = set(part_names)
    else:
        # this is needed - just Typer things - part_names is empty list instead of node, and we need default
        part_names = None
    try:
        ensure_api_key_is_set(api_key=api_key)
        api_operations.get_batch_content(
            batch_id=batch_id,
            api_key=api_key,
            part_names=part_names,
            limit=limit,
            output_file=output_file,
        )
    except KeyboardInterrupt:
        print("Command interrupted.")
        raise typer.Exit(code=2)
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
    source: Annotated[
        DataSource,
        typer.Option(
            "--data-source",
            "-ds",
            help="Source of the data - either local directory or JSON file with references.",
        ),
    ] = DataSource.LOCAL_DIRECTORY,
    images_dir: Annotated[
        Optional[str],
        typer.Option(
            "--images-dir",
            "-i",
            help="Path to your images directory to upload (required if data source is 'local-directory')",
        ),
    ] = None,
    references: Annotated[
        Optional[str],
        typer.Option(
            "--references",
            "-r",
            help="Path to JSON file with URLs of files to be ingested (required if data source is 'references-file')",
        ),
    ] = None,
    ingest_id: Annotated[
        Optional[str],
        typer.Option(
            "--ingest-id",
            "-i",
            help="Identifier assigned for references ingest (if value not provided - system will auto-assign) - "
            "only relevant when `notifications-url` specified",
        ),
    ] = None,
    notifications_url: Annotated[
        Optional[str],
        typer.Option(
            "--notifications-url",
            help="Webhook URL where system should send notifications about ingest status",
        ),
    ] = None,
    notification_categories: Annotated[
        Optional[List[str]],
        typer.Option(
            "--notification-category",
            help="Selecting specific notification categories (ingest-status / files-status) "
            "in combination with `--notifications-url` you may filter which notifications are "
            "going to be sent to your system. Please note that filtering of notifications do only "
            "work with ingests of files references via signed URLs and will not be applicable "
            "for ingests from local storage.",
        ),
    ] = None,
    batch_name: Annotated[
        Optional[str],
        typer.Option(
            "--batch-name",
            "-bn",
            help="Display name of your batch",
        ),
    ] = None,
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
    if notification_categories:
        notification_categories = list(set(notification_categories))
    else:
        # this is needed - notification_categories is empty list instead of node, and we need default
        notification_categories = None
    try:
        ensure_api_key_is_set(api_key=api_key)
        if source is DataSource.LOCAL_DIRECTORY:
            if images_dir is None:
                raise ValueError(
                    "`images-dir` not provided when `local-directory` specified as a data source"
                )
            if notifications_url and notification_categories:
                print(
                    "--notification-category option is not supported for ingests from local storage"
                )
            api_operations.create_images_batch_from_directory(
                directory=images_dir,
                batch_id=batch_id,
                api_key=api_key,
                batch_name=batch_name,
                ingest_id=ingest_id,
                notifications_url=notifications_url,
            )
        else:
            if references is None:
                raise ValueError(
                    "`references` path not provided when `references-file` specified as a data source"
                )
            api_operations.create_images_batch_from_references_file(
                references=references,
                batch_id=batch_id,
                api_key=api_key,
                ingest_id=ingest_id,
                batch_name=batch_name,
                notifications_url=notifications_url,
                notification_categories=notification_categories,
            )
    except KeyboardInterrupt:
        print("Command interrupted.")
        raise typer.Exit(code=2)
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
    source: Annotated[
        DataSource,
        typer.Option(
            "--data-source",
            "-ds",
            help="Source of the data - either local directory or JSON file with references.",
        ),
    ] = DataSource.LOCAL_DIRECTORY,
    videos_dir: Annotated[
        Optional[str],
        typer.Option(
            "--videos-dir",
            "-v",
            help="Path to your videos directory to upload",
        ),
    ] = None,
    references: Annotated[
        Optional[str],
        typer.Option(
            "--references",
            "-r",
            help="Path to JSON file with URLs of files to be ingested (required if data source is 'references-file')",
        ),
    ] = None,
    ingest_id: Annotated[
        Optional[str],
        typer.Option(
            "--ingest-id",
            "-i",
            help="Identifier assigned for references ingest (if value not provided - system will auto-assign) - "
            "only relevant if data source is 'references-file'",
        ),
    ] = None,
    notifications_url: Annotated[
        Optional[str],
        typer.Option(
            "--notifications-url",
            help="Webhook URL where system should send notifications about ingest status - only relevant "
            "if data source is 'references-file'",
        ),
    ] = None,
    notification_categories: Annotated[
        Optional[List[str]],
        typer.Option(
            "--notification-category",
            help="Selecting specific notification categories (ingest-status / files-status) "
            "in combination with `--notifications-url` you may filter which notifications are "
            "going to be sent to your system. Please note that filtering of notifications do only "
            "work with ingests of files references via signed URLs and will not be applicable "
            "for ingests from local storage.",
        ),
    ] = None,
    api_key: Annotated[
        Optional[str],
        typer.Option(
            "--api-key",
            "-a",
            help="Roboflow API key for your workspace. If not given - env variable `ROBOFLOW_API_KEY` will be used",
        ),
    ] = None,
    batch_name: Annotated[
        Optional[str],
        typer.Option(
            "--batch-name",
            "-bn",
            help="Display name of your batch",
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
    if notification_categories:
        notification_categories = list(set(notification_categories))
    else:
        # this is needed - notification_categories is empty list instead of node, and we need default
        notification_categories = None
    try:
        ensure_api_key_is_set(api_key=api_key)
        if source is DataSource.LOCAL_DIRECTORY:
            if videos_dir is None:
                raise ValueError(
                    "`videos-dir` not provided when `local-directory` specified as a data source"
                )
            if notifications_url:
                print(
                    "`--notifications-url` option not supported for ingests of videos from local storage"
                )
            api_operations.create_videos_batch_from_directory(
                directory=videos_dir,
                batch_id=batch_id,
                api_key=api_key,
                batch_name=batch_name,
            )
        else:
            if references is None:
                raise ValueError(
                    "`references` path not provided when `references-file` specified as a data source"
                )
            api_operations.create_videos_batch_from_references_file(
                references=references,
                batch_id=batch_id,
                api_key=api_key,
                ingest_id=ingest_id,
                batch_name=batch_name,
                notifications_url=notifications_url,
                notification_categories=notification_categories,
            )
    except KeyboardInterrupt:
        print("Command interrupted.")
        return
    except Exception as error:
        if debug_mode:
            raise error
        typer.echo(f"Command failed. Cause: {error}")
        raise typer.Exit(code=1)


@data_staging_app.command(help="Show batch details")
def show_batch_details(
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
        ensure_api_key_is_set(api_key=api_key)
        api_operations.display_batch_details(batch_id=batch_id, api_key=api_key)
    except KeyboardInterrupt:
        print("Command interrupted.")
        raise typer.Exit(code=1)
    except Exception as error:
        if debug_mode:
            raise error
        typer.echo(f"Command failed. Cause: {error}")
        raise typer.Exit(code=1)


@data_staging_app.command(help="Export batch")
def export_batch(
    batch_id: Annotated[
        str,
        typer.Option(
            "--batch-id",
            "-b",
            help="Identifier of new batch (must be lower-cased letters with '-' and '_' allowed",
        ),
    ],
    target_dir: Annotated[
        str,
        typer.Option(
            "--target-dir",
            "-t",
            help="Path to export directory",
        ),
    ],
    part_names: Annotated[
        Optional[List[str]],
        typer.Option(
            "--part-name",
            "-pn",
            help="Name of the part to be exported (if not given - all parts are presented). Invalid if batch is "
            "not multipart",
        ),
    ] = None,
    api_key: Annotated[
        Optional[str],
        typer.Option(
            "--api-key",
            "-a",
            help="Roboflow API key for your workspace. If not given - env variable `ROBOFLOW_API_KEY` will be used",
        ),
    ] = None,
    override_existing: Annotated[
        bool,
        typer.Option(
            "--override-existing/--no-override-existing",
            help="Flag to enforce export even if partial content is already exported",
        ),
    ] = False,
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
    if part_names:
        part_names = set(part_names)
    else:
        # this is needed - just Typer things - part_names is empty list instead of node, and we need default
        part_names = None
    try:
        ensure_api_key_is_set(api_key=api_key)
        api_operations.export_data(
            batch_id=batch_id,
            api_key=api_key,
            target_directory=target_dir,
            part_names=part_names,
            override_existing=override_existing,
        )
    except KeyboardInterrupt:
        print("Command interrupted.")
        raise typer.Exit(code=2)
    except Exception as error:
        if debug_mode:
            raise error
        typer.echo(f"Command failed. Cause: {error}")
        raise typer.Exit(code=1)


@data_staging_app.command(help="List details of your data ingest.")
def list_ingest_details(
    batch_id: Annotated[
        str,
        typer.Option(
            "--batch-id",
            "-b",
            help="Identifier of new batch (must be lower-cased letters with '-' and '_' allowed",
        ),
    ],
    output_file: Annotated[
        Optional[str],
        typer.Option(
            "--output-file",
            "-o",
            help="Path to the output file - if not provided, command will dump result to the console. "
            "File type is JSONL.",
        ),
    ] = None,
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
        ensure_api_key_is_set(api_key=api_key)
        api_operations.list_ingest_details(
            batch_id=batch_id,
            api_key=api_key,
            output_file=output_file,
        )
    except KeyboardInterrupt:
        print("Command interrupted.")
        raise typer.Exit(code=2)
    except Exception as error:
        if debug_mode:
            raise error
        typer.echo(f"Command failed. Cause: {error}")
        raise typer.Exit(code=1)


@data_staging_app.command(help="Create batch of images and monitor until completion.")
def run_image_batch(
    batch_id: Annotated[
        str,
        typer.Option(
            "--batch-id",
            "-b",
            help="Identifier of new batch (must be lower-cased letters with '-' and '_' allowed",
        ),
    ],
    source: Annotated[
        DataSource,
        typer.Option(
            "--data-source",
            "-ds",
            help="Source of the data - either local directory or JSON file with references.",
        ),
    ] = DataSource.LOCAL_DIRECTORY,
    images_dir: Annotated[
        Optional[str],
        typer.Option(
            "--images-dir",
            "-i",
            help="Path to your images directory to upload (required if data source is 'local-directory')",
        ),
    ] = None,
    references: Annotated[
        Optional[str],
        typer.Option(
            "--references",
            "-r",
            help="Path to JSON file with URLs of files to be ingested (required if data source is 'references-file')",
        ),
    ] = None,
    ingest_id: Annotated[
        Optional[str],
        typer.Option(
            "--ingest-id",
            help="Identifier assigned for references ingest (if value not provided - system will auto-assign) - "
            "only relevant when `notifications-url` specified",
        ),
    ] = None,
    notifications_url: Annotated[
        Optional[str],
        typer.Option(
            "--notifications-url",
            help="Webhook URL where system should send notifications about ingest status",
        ),
    ] = None,
    notification_categories: Annotated[
        Optional[List[str]],
        typer.Option(
            "--notification-category",
            help="Selecting specific notification categories (ingest-status / files-status) "
            "in combination with `--notifications-url` you may filter which notifications are "
            "going to be sent to your system. Please note that filtering of notifications do only "
            "work with ingests of files references via signed URLs and will not be applicable "
            "for ingests from local storage.",
        ),
    ] = None,
    batch_name: Annotated[
        Optional[str],
        typer.Option(
            "--batch-name",
            "-bn",
            help="Display name of your batch",
        ),
    ] = None,
    output_file: Annotated[
        Optional[str],
        typer.Option(
            "--output-file",
            "-o",
            help="Path to the output file for final ingest details - if not provided, details will be shown in console. "
            "File type is JSONL.",
        ),
    ] = None,
    polling_interval: Annotated[
        int,
        typer.Option(
            "--polling-interval",
            "-pi",
            help="Polling interval in seconds for checking batch status",
        ),
    ] = 3,
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
    if notification_categories:
        notification_categories = list(set(notification_categories))
    else:
        # this is needed - notification_categories is empty list instead of node, and we need default
        notification_categories = None

    console = Console()

    try:
        ensure_api_key_is_set(api_key=api_key)
        workspace = get_workspace(api_key=api_key)

        # Step 1: Create the batch
        console.print(f"[bold cyan]Creating batch '{batch_id}'...[/bold cyan]")

        if source is DataSource.LOCAL_DIRECTORY:
            if images_dir is None:
                raise ValueError(
                    "`images-dir` not provided when `local-directory` specified as a data source"
                )
            if notifications_url and notification_categories:
                console.print(
                    "[yellow]--notification-category option is not supported for ingests from local storage[/yellow]"
                )
            api_operations.create_images_batch_from_directory(
                directory=images_dir,
                batch_id=batch_id,
                api_key=api_key,
                batch_name=batch_name,
                ingest_id=ingest_id,
                notifications_url=notifications_url,
            )
        else:
            if references is None:
                raise ValueError(
                    "`references` path not provided when `references-file` specified as a data source"
                )
            api_operations.create_images_batch_from_references_file(
                references=references,
                batch_id=batch_id,
                api_key=api_key,
                ingest_id=ingest_id,
                batch_name=batch_name,
                notifications_url=notifications_url,
                notification_categories=notification_categories,
            )

        console.print(f"[bold green]✓[/bold green] Batch '{batch_id}' created successfully!")

        # Step 2: Poll for completion
        console.print(f"[bold cyan]Monitoring batch progress...[/bold cyan]")
        console.print(f"[dim]Polling every {polling_interval} seconds. Press Ctrl+C to stop monitoring.[/dim]")

        completed_shards = set()
        total_shards = 0
        previous_status_summary = ""

        while True:
            try:
                # Get batch metadata for basic info
                batch_metadata = api_operations.get_batch_metadata(
                    workspace=workspace,
                    batch_id=batch_id,
                    api_key=api_key,
                )

                # Get shard statuses
                shard_statuses = {}
                all_shards_complete = True
                current_completed_shards = set()

                for shard_details in api_operations.iterate_through_batch_shards_statuses(
                    workspace=workspace,
                    batch_id=batch_id,
                    api_key=api_key,
                ):
                    shard_id = shard_details.shard_id
                    status = shard_details.status_name

                    if status not in shard_statuses:
                        shard_statuses[status] = 0
                    shard_statuses[status] += 1

                    # Check if this shard is complete (you may need to adjust these status names based on your API)
                    if status.lower() in ['completed', 'success', 'done', 'finished']:
                        current_completed_shards.add(shard_id)
                    else:
                        all_shards_complete = False

                total_shards = sum(shard_statuses.values())
                completed_count = len(current_completed_shards)

                # Create status summary
                status_parts = []
                for status, count in sorted(shard_statuses.items()):
                    status_parts.append(f"{status}: {count}")

                current_status_summary = f"Progress: {completed_count}/{total_shards} shards complete"
                if status_parts:
                    current_status_summary += f" | Status breakdown: {', '.join(status_parts)}"

                # Only print if status changed to avoid spam
                if current_status_summary != previous_status_summary:
                    timestamp = time.strftime("%H:%M:%S")
                    console.print(f"[dim]{timestamp}[/dim] {current_status_summary}")
                    previous_status_summary = current_status_summary

                # Check for newly completed shards
                newly_completed = current_completed_shards - completed_shards
                if newly_completed:
                    for shard_id in newly_completed:
                        console.print(f"[green]✓ Shard {shard_id} completed[/green]")
                    completed_shards = current_completed_shards

                # Check if all shards are complete
                if all_shards_complete and total_shards > 0:
                    console.print(f"[bold green]🎉 All shards completed! Batch '{batch_id}' is ready.[/bold green]")
                    break

                # If no shards yet, batch might still be initializing
                if total_shards == 0:
                    console.print("[dim]Waiting for batch initialization...[/dim]")

            except Exception as e:
                if debug_mode:
                    raise e
                console.print(f"[yellow]Warning: Error checking batch status: {e}[/yellow]")

            time.sleep(polling_interval)

        # Step 3: Show final batch details and ingest details
        console.print(f"\n[bold cyan]Final batch details:[/bold cyan]")
        api_operations.display_batch_details(batch_id=batch_id, api_key=api_key)

        console.print(f"\n[bold cyan]Ingest details:[/bold cyan]")
        api_operations.list_ingest_details(
            batch_id=batch_id,
            api_key=api_key,
            output_file=output_file,
        )

        console.print(f"\n[bold green]✅ Batch '{batch_id}' processing completed successfully![/bold green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring interrupted by user. Batch processing may continue in the background.[/yellow]")
        console.print(f"[dim]You can check batch status later using: inference data-staging show-batch-details --batch-id {batch_id}[/dim]")
        raise typer.Exit(code=2)
    except Exception as error:
        if debug_mode:
            raise error
        typer.echo(f"Command failed. Cause: {error}")
        raise typer.Exit(code=1)
