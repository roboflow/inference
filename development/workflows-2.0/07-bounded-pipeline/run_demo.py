"""Run the bounded pipeline examples; write evidence.json and index.html."""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import click
from bounded_authoring import authoring
from bounded_report import write_report
from bounded_scheduling import active_timeline, lifecycle, overload, passive_timeline


@dataclass(frozen=True)
class Example:
    """One runnable example.

    Attributes:
        description: One line shown by ``--list``.
        needs_model: Whether it loads the trained ResNet-18 weights.
    """

    description: str
    needs_model: bool


EXAMPLES: Dict[str, Example] = {
    "timeline": Example(
        "SYNTHETIC passive phase timeline: run 1 in phase first while run 0 is in second",
        needs_model=False,
    ),
    "active-timeline": Example(
        "SYNTHETIC active frames: phase overlap, per-source order, independent sources",
        needs_model=False,
    ),
    "overload": Example(
        "SYNTHETIC held consumer: block (lossless read-ahead) versus latest (drops, age)",
        needs_model=False,
    ),
    "lifecycle": Example(
        "SYNTHETIC stop, cancel and failure: counters balance, no thread survives",
        needs_model=False,
    ),
    "authoring": Example(
        "SYNTHETIC block authoring: scratch on self across phases (wrong) and two fixes",
        needs_model=False,
    ),
    "model": Example(
        "Trained ResNet-18 (M3 classifier): serial versus pipelined, CPU and MPS when present",
        needs_model=True,
    ),
}


@click.command()
@click.option(
    "--case",
    "case_name",
    type=click.Choice(("all",) + tuple(EXAMPLES)),
    default="all",
    show_default=True,
    help="Example to run, or all of them.",
)
@click.option(
    "--weights",
    type=click.Path(path_type=Path, dir_okay=False, exists=True),
    default=None,
    help="resnet18-f37072fd.pth; verified, never downloaded.",
)
@click.option(
    "--weights-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Directory holding the weights; torch's checkpoint directory by default.",
)
@click.option(
    "--repeats",
    type=click.IntRange(
        min=1,
    ),
    default=4,
    show_default=True,
    help="Times each pinned image (or frame triple) is classified.",
)
@click.option(
    "--max-in-flight",
    type=click.IntRange(
        min=1,
    ),
    default=3,
    show_default=True,
    help="Pipeline workers of the model examples.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Directory for evidence.json and index.html.",
)
@click.option(
    "--list",
    "list_examples",
    is_flag=True,
    default=False,
    help="Print the example names and exit.",
)
def main(
    case_name: str,
    weights: Optional[Path],
    weights_dir: Optional[Path],
    repeats: int,
    max_in_flight: int,
    output_dir: Optional[Path],
    list_examples: bool,
) -> None:
    """Run examples locally; nothing is downloaded.

    Args:
        case_name: One example name, or ``all``.
        weights: Explicit weights file.
        weights_dir: Directory searched when ``weights`` is not given.
        repeats: Repetitions of the model inputs.
        max_in_flight: Pipeline workers of the model examples.
        output_dir: Optional directory for evidence and the report.
        list_examples: Only list the examples.
    """
    if list_examples:
        width = max(len(name) for name in EXAMPLES)
        for name, example in EXAMPLES.items():
            click.echo(f"{name.ljust(width)}  {example.description}")
        return

    selected = list(EXAMPLES) if case_name == "all" else [case_name]
    runners: Dict[str, Callable[[], Dict[str, Any]]] = {
        "timeline": passive_timeline,
        "active-timeline": active_timeline,
        "overload": overload,
        "lifecycle": lifecycle,
        "authoring": authoring,
    }
    if any(EXAMPLES[name].needs_model for name in selected):
        runners["model"] = _model_runner(
            weights=weights,
            weights_dir=weights_dir,
            repeats=repeats,
            max_in_flight=max_in_flight,
        )

    evidence: Dict[str, Any] = {}
    failed = []
    for name in selected:
        click.echo(f"\n{name}: {EXAMPLES[name].description}")
        try:
            evidence[name] = runners[name]()
        except Exception as error:
            failed.append(name)
            evidence[name] = {"failed": f"{type(error).__name__}: {error}"}
            click.echo(f"FAIL {name}: {type(error).__name__}: {error}")
            continue
        click.echo(f"PASS {name}")

    if output_dir is not None:
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / "evidence.json"
        destination.write_text(json.dumps(evidence, indent=2, default=str) + "\n")
        page = write_report(evidence, destination=output_dir)
        click.echo(f"\nEvidence: {destination}\nReport: {page}")
    if failed:
        click.echo(f"\n{len(failed)} example(s) failed: {', '.join(failed)}")
        sys.exit(1)


def _model_runner(
    *,
    weights: Optional[Path],
    weights_dir: Optional[Path],
    repeats: int,
    max_in_flight: int,
) -> Callable[[], Dict[str, Any]]:
    """Verify and load the weights once; return the model example."""
    import bounded_model

    try:
        path, state_dict = bounded_model.load_weights(
            weights=weights, weights_dir=weights_dir
        )
    except bounded_model.AssetError as error:
        click.echo(str(error), err=True)
        sys.exit(2)

    def run() -> Dict[str, Any]:
        evidence = bounded_model.model_example(
            state_dict, repeats=repeats, max_in_flight=max_in_flight
        )
        evidence["weights"] = str(path)

        return evidence

    return run


if __name__ == "__main__":
    main()
