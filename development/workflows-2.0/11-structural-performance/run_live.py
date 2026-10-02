"""Live 16-source run of 10 ``run_batched.py`` with a chosen drawing.

    run_live.py --drawing D  <every 10 run_batched option>
      └─ 10 run_batched.main, unchanged, except that its backend comes from
         drawing_backend.build_for_model(..., drawing=D)

Sources, admission, batch collection, warmup, drain, metrics and outputs
(``summary.json``, ``frames.csv``, ``batches.csv``) are 10's. The drawing is in
``summary.json`` under ``backend.drawing``; each ``frames.csv`` result also
carries ``prep_ms`` for the shared drawings.

How the backend is swapped: 10 ``run_batched`` calls
``batched_backend.build_for_model`` through its module global
``batched_backend``. For the duration of the call only, that global points at
a namespace whose ``build_for_model`` is ``drawing_backend.build_for_model``
with ``drawing`` bound. No other module and no class is changed.

    python run_live.py --drawing shared-prep --mode v2_pipeline --batch-size 8 \\
        --pipeline-depth 2 --sources 16 --output-dir /tmp/m46/live/shared_b8_d2
"""

import functools
import types
from contextlib import contextmanager
from typing import Iterator, List

import click
import structural_imports

# isort: split

import batched_backend
import drawing_backend


@click.command(
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
)
@click.option(
    "--drawing",
    type=click.Choice(
        drawing_backend.DRAWINGS,
    ),
    required=True,
    help="per-painter is 10 unchanged; the others share one prep transfer per batch.",
)
@click.pass_context
def main(context: click.Context, drawing: str) -> None:
    """Run 10 run_batched.py with DRAWING; every other option is passed to it."""
    arguments: List[str] = list(context.args)
    mode = _option_value(arguments, name="--mode")
    if drawing != "per-painter" and mode == "v1_tensor":
        raise click.UsageError(f"--drawing {drawing} needs a V2 --mode")

    run_batched = structural_imports.load_10_module("run_batched")
    with _drawing_backends(run_batched, drawing=drawing):
        run_batched.main.main(args=arguments, prog_name="run_live.py")


@contextmanager
def _drawing_backends(module: types.ModuleType, *, drawing: str) -> Iterator[None]:
    # Scoped to one run: restore 10's own module global afterwards.
    original = module.batched_backend
    module.batched_backend = types.SimpleNamespace(
        MODES=batched_backend.MODES,
        build_for_model=functools.partial(
            drawing_backend.build_for_model, drawing=drawing
        ),
    )
    try:
        yield
    finally:
        module.batched_backend = original


def _option_value(arguments: List[str], *, name: str) -> str:
    for position, argument in enumerate(arguments):
        if argument == name and position + 1 < len(arguments):
            return arguments[position + 1]
        if argument.startswith(f"{name}="):
            return argument.split("=", 1)[1]

    return ""


if __name__ == "__main__":
    main()
