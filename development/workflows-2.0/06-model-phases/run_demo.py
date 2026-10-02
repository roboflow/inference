"""Run the model phase examples with the trained ResNet-18 weights."""

import json
import sys
from pathlib import Path
from typing import Optional

import click
from assets import AssetError, describe_assets, locate_weights
from cases import CASES
from gallery import Gallery
from host import DemoContext
from resnet18 import load_state_dict


@click.command()
@click.option(
    "--case",
    "case_name",
    type=click.Choice(("all",) + tuple(CASES)),
    default="classify",
    show_default=True,
    help="Example to run, or all of them.",
)
@click.option(
    "--weights-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Directory with resnet18-f37072fd.pth; the torch hub checkpoint directory by default.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Directory for evidence.json and the gallery (gallery/index.html).",
)
@click.option(
    "--list",
    "list_cases",
    is_flag=True,
    default=False,
    help="Print the example names with a one-line description and exit.",
)
def main(
    case_name: str,
    weights_dir: Optional[Path],
    output_dir: Optional[Path],
    list_cases: bool,
) -> None:
    """Run examples locally; nothing is downloaded.

    Args:
        case_name: One example name, or ``all``.
        weights_dir: Where the pinned weights are.
        output_dir: Optional directory for evidence and the gallery.
        list_cases: Only list the examples.
    """
    if list_cases:
        width = max(len(name) for name in CASES)
        for name, case in CASES.items():
            click.echo(f"{name.ljust(width)}  {case.description}")
        return

    try:
        weights = locate_weights(weights_dir)
    except AssetError as error:
        click.echo(str(error), err=True)
        sys.exit(2)

    output_dir = None if output_dir is None else output_dir.resolve()
    gallery = Gallery(
        None if output_dir is None else output_dir / "gallery",
        title="Model phases: flip-averaged ResNet-18",
    )
    context = DemoContext(state_dict=load_state_dict(weights), gallery=gallery)
    results = {"assets": describe_assets(weights)}
    failed = []
    for name in CASES if case_name == "all" else (case_name,):
        click.echo(f"\n{name}: {CASES[name].description}")
        try:
            results[name] = CASES[name].run(context)
        except Exception as error:
            failed.append(name)
            results[name] = {"failed": f"{type(error).__name__}: {error}"}
            click.echo(f"FAIL {name}: {type(error).__name__}: {error}")
            continue
        status = "SKIP" if "skipped" in results[name] else "PASS"
        click.echo(f"{status} {name}")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / "evidence.json"
        destination.write_text(json.dumps(results, indent=2, default=str) + "\n")
        click.echo(f"\nEvidence: {destination}")
        page = gallery.write_index()
        if page is not None:
            click.echo(f"Gallery: {page}")
    if failed:
        click.echo(f"\n{len(failed)} case(s) failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
