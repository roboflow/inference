"""Command line entry point of the passive V2 foundation demo.

Run from the repository root with the checkout's sources on the path::

    PYTHONPATH=workflows python development/workflows-2.0/01-passive-foundation/run_demo.py \\
        --scenario nested --output-dir /tmp/workflows-2.0-demo

Every scenario compiles an ordinary JSON definition with ``compile_workflow``
and executes it with ``plan.run``; the demo only prepares inputs and records
what the engine returned. The exit code is 1 when any expectation failed.
"""

import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenarios import SCENARIO_NAMES, run_scenario  # noqa: E402


@click.command()
@click.option(
    "--scenario",
    type=click.Choice(
        [*SCENARIO_NAMES, "all"],
    ),
    default="all",
    show_default=True,
    help="Named scenario to run, or 'all'.",
)
@click.option(
    "--output-dir",
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
    required=True,
    help="Directory receiving PNG artifacts and JSON reports.",
)
def main(scenario: str, output_dir: Path) -> None:
    """Run the passive V2 image demo scenarios and write their artifacts."""
    names = list(SCENARIO_NAMES) if scenario == "all" else [scenario]
    output_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for name in names:
        click.echo(f"== scenario: {name}")
        report = run_scenario(name, output_dir=output_dir)
        for check in report.checks:
            marker = "PASS" if check["passed"] else "FAIL"
            click.echo(f"  [{marker}] {check['check']}")
            if not check["passed"]:
                click.echo(f"         detail: {check['detail']}")
        for note in report.notes:
            click.echo(f"  note: {note}")
        click.echo(f"  report: {output_dir / name / 'report.json'}")
        if not report.ok:
            failed.append(name)

    if failed:
        click.echo(f"FAILED scenarios: {failed}")
        sys.exit(1)

    click.echo(f"All scenarios passed: {names}")


if __name__ == "__main__":
    main()
