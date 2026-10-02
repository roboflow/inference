"""Sequential parity demo: the real V1 engine next to the V2 engine.

Run from anywhere with the repository's Python environment::

    python development/workflows-2.0/02-sequential-parity/run_demo.py \\
        --output-dir /tmp/workflows-2.0-parity

``parity`` runs the 45 reference cases on V1 (in a child process, see
``reference/``) and on V2 (compile_workflow -> create_session -> run -> rows)
and compares them. ``capabilities`` runs V2-only examples of features beyond
the reference cases. The exit code is 1 when any comparison or check fails.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import click

DEMO_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = next(
    parent
    for parent in DEMO_DIR.parents
    if (parent / "workflows" / "roboflow_workflows").is_dir()
)
sys.path[0:0] = [str(REPOSITORY_ROOT / "workflows"), str(DEMO_DIR)]

import capabilities  # noqa: E402
from comparison import compare_case  # noqa: E402
from reference import REFERENCE_CASES, collect_v1_observations  # noqa: E402
from v2_runner import observe_v2_case  # noqa: E402

SCENARIOS = ("parity", "capabilities")
VERDICT_MARKERS = {
    "parity": "PARITY",
    "difference_confirmed": "DIFFERENCE",
    "unexpected_difference": "FAIL",
    "v2_error": "FAIL",
}


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=repr) + "\n")


def run_parity(case_ids: List[str], *, output_dir: Path) -> Dict[str, Any]:
    """Compare the selected reference cases on V1 and V2.

    Args:
        case_ids: Reference case identifiers to run.
        output_dir: Directory receiving one folder per case.

    Returns:
        Summary with the verdict of every case and the V1 pin status.
    """
    v1_report = collect_v1_observations(case_ids)
    v1_cases = {case["case_id"]: case for case in v1_report["cases"]}
    verdicts: Dict[str, str] = {}
    input_preparation_runs: List[str] = []
    for case in REFERENCE_CASES:
        if case.case_id not in case_ids:
            continue
        v1 = v1_cases[case.case_id]
        try:
            v2 = observe_v2_case(case)
            comparison = compare_case(case, v1, v2)
        except Exception as error:
            v2 = {"error": f"{type(error).__name__}: {error}"}
            comparison = {"verdict": "v2_error", "failed_checks": [v2["error"]]}

        case_dir = output_dir / "parity" / case.case_id
        _write_json(case_dir / "v1.json", v1)
        _write_json(case_dir / "v2.json", v2)
        _write_json(case_dir / "v2_definition.json", v2.get("definition"))
        _write_json(case_dir / "comparison.json", comparison)
        verdicts[case.case_id] = comparison["verdict"]
        input_preparation_runs.extend(
            f"{case.case_id}@{run['run']}"
            for run in comparison.get("runs", [])
            for check in run["checks"]
            if check.get("label") == "D012-INPUT-PREPARATION"
        )

        labels = sorted(
            {run["label"] for run in comparison.get("runs", []) if run["label"]}
        )
        marker = VERDICT_MARKERS[comparison["verdict"]]
        click.echo(f"  [{marker}] {case.case_id} {' '.join(labels)}".rstrip())
        for failed in comparison["failed_checks"]:
            click.echo(f"         failed: {failed}")

    counts: Dict[str, int] = {}
    for verdict in verdicts.values():
        counts[verdict] = counts.get(verdict, 0) + 1
    summary = {
        "v1_pin_failures": v1_report["summary"]["pin_failures"],
        "v1_network_attempts": v1_report["network_attempts"],
        "verdict_counts": counts,
        "input_preparation_difference_runs": input_preparation_runs,
        "verdicts": verdicts,
    }
    _write_json(output_dir / "parity" / "summary.json", summary)

    return summary


@click.command()
@click.option(
    "--scenario",
    type=click.Choice(
        [*SCENARIOS, "all"],
    ),
    default="all",
    show_default=True,
    help="Scenario group to run.",
)
@click.option(
    "--case",
    "selected",
    multiple=True,
    help="Run only these parity case or capability names; repeatable.",
)
@click.option(
    "--output-dir",
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
    default=Path("/tmp/workflows-2.0-parity"),
    show_default=True,
    help="Directory receiving JSON artifacts per case.",
)
@click.option(
    "--list",
    "list_only",
    is_flag=True,
    default=False,
    help="List parity cases and capability examples, then exit.",
)
def main(scenario: str, selected: tuple, output_dir: Path, list_only: bool) -> None:
    """Run the sequential parity demo and print one line per case."""
    parity_ids = [case.case_id for case in REFERENCE_CASES]
    capability_names = list(capabilities.EXAMPLES)
    if list_only:
        for case in REFERENCE_CASES:
            label = case.comparison.label or ""
            click.echo(f"parity        {case.case_id} {label}".rstrip())
        for name in capability_names:
            click.echo(f"capabilities  {name}")
        return

    unknown = set(selected) - set(parity_ids) - set(capability_names)
    if unknown:
        raise click.BadParameter(f"unknown names {sorted(unknown)}; see --list")

    failed = False
    if scenario in ("parity", "all"):
        case_ids = [name for name in parity_ids if not selected or name in selected]
        if case_ids:
            click.echo(f"== parity: {len(case_ids)} reference cases, V1 vs V2")
            summary = run_parity(case_ids, output_dir=output_dir)
            click.echo(f"  verdicts: {summary['verdict_counts']}")
            click.echo(
                "  D012-INPUT-PREPARATION: V1 rewrote the caller's inputs, V2 did not, in "
                f"{len(summary['input_preparation_difference_runs'])} runs"
            )
            failed = failed or bool(
                summary["v1_pin_failures"]
                or summary["v1_network_attempts"]
                or set(summary["verdict_counts"]) - {"parity", "difference_confirmed"}
            )
    if scenario in ("capabilities", "all"):
        names = [name for name in capability_names if not selected or name in selected]
        if names:
            click.echo(f"== capabilities: {len(names)} V2 examples")
            failed = (
                not capabilities.run_examples(names, output_dir=output_dir) or failed
            )

    click.echo(f"artifacts: {output_dir}")
    if failed:
        click.echo("FAILED")
        sys.exit(1)

    click.echo("All selected scenarios passed")


if __name__ == "__main__":
    main()
