"""Run the temporal operator examples: alignment, windows and temporal blocks."""

import json
import sys
from pathlib import Path
from typing import Optional

import click
from cases import CASES


@click.command()
@click.option(
    "--case",
    "case_name",
    type=click.Choice(("all",) + tuple(CASES)),
    default="rig",
    show_default=True,
    help="Example to run, or all of them.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Directory for evidence.json, compiled plans, results and the gallery.",
)
@click.option(
    "--list",
    "list_cases",
    is_flag=True,
    default=False,
    help="Print the example names with a one-line description and exit.",
)
def main(case_name: str, output_dir: Optional[Path], list_cases: bool) -> None:
    """Run deterministic local examples; no model, camera or network is needed.

    Args:
        case_name: One example name, or ``all``.
        output_dir: Optional directory for machine-readable evidence.
        list_cases: Only list the examples.
    """
    if list_cases:
        width = max(len(name) for name in CASES)
        for name, case in CASES.items():
            click.echo(f"{name.ljust(width)}  {case.description}")
        return

    output_dir = None if output_dir is None else output_dir.resolve()
    results = {}
    failed = []
    for name in CASES if case_name == "all" else (case_name,):
        click.echo(f"\n{name}: {CASES[name].description}")
        case_dir = None if output_dir is None else output_dir / name
        try:
            results[name] = CASES[name].run(case_dir)
        except Exception as error:
            failed.append(name)
            results[name] = {"failed": f"{type(error).__name__}: {error}"}
            click.echo(f"FAIL {name}: {type(error).__name__}: {error}")
            continue
        click.echo(f"PASS {name}")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / "evidence.json"
        destination.write_text(json.dumps(results, indent=2, default=str) + "\n")
        click.echo(f"\nEvidence: {destination}")
        gallery = output_dir / "rig" / "gallery" / "index.html"
        if gallery.exists():
            click.echo(f"Gallery: {gallery}")
    if failed:
        click.echo(f"\n{len(failed)} case(s) failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
