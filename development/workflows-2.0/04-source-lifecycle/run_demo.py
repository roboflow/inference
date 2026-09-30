"""Observe independent finite sources, lifecycle boundaries and timestamped ingress."""

import json
from pathlib import Path

import click
from lifecycle_probe import DemoControl, ProbedFrames, ProbedTemperature
from passive_example import run_passive
from processing import CelsiusToFahrenheit, IsBright, RecordTemperature
from roboflow_workflows.execution_engine.v2 import ActiveRunError, GroupResult
from roboflow_workflows.execution_engine.v2.blocks import create_catalogue
from roboflow_workflows.execution_engine.v2.blocks.image_data import ImageData
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.errors import LineageError
from source_plugins import CsvTemperature, TensorFrames

DEMO_DIR = Path(__file__).resolve().parent
CASES = (
    "csv",
    "csv-stop",
    "independent",
    "lifecycle",
    "stop",
    "failure",
    "passive",
    "invalid-join",
)


def _catalogue(*, probe=False):
    catalogue = Catalogue.merge(
        create_catalogue(),
        Catalogue(
            [CelsiusToFahrenheit, IsBright, RecordTemperature],
            sources=(
                [ProbedTemperature, ProbedFrames]
                if probe
                else [CsvTemperature, TensorFrames]
            ),
        ),
    )

    return catalogue


def _definition():
    definition = json.loads(
        (DEMO_DIR / "workflows" / "independent_sources.json").read_text()
    )

    return definition


def _timestamp(timestamp):
    if timestamp is None:
        return None

    record = {
        "ticks": timestamp.ticks,
        "time_base": str(timestamp.time_base),
        "clock": timestamp.clock_id,
    }

    return record


def _value(value):
    if not isinstance(value, ImageData):
        return value

    record = {
        "image_id": value.image_id,
        "tensor_shape": list(value.tensor_image.shape),
        "dtype": str(value.tensor_image.dtype),
        "device": str(value.device),
        "root": value.root.to_dict(),
    }

    return record


def _observe(result: GroupResult) -> dict:
    fields = {}
    for name, selection in result.selections.items():
        (entry,) = selection.values()
        status = result.statuses[entry]
        record = {"status": status}
        if status == "complete":
            metadata = result.outputs.metadata[entry]
            sample = metadata.sample_at(())
            timing = metadata.temporal_at(())
            record.update(
                value=_value(result.outputs.data[entry]),
                source=sample.source_id,
                pts=_timestamp(timing.media_coverage),
                observed=_timestamp(timing.observed_coverage),
                axis_ids=list(result.outputs.layout[entry].axis_ids),
            )
        fields[name] = record

    record = {
        "group": result.group,
        "source": result.source,
        "pulse": result.pulse.sequence,
        "lineage": result.pulse.lineage_id,
        "run_id": result.run_id,
        "fields": fields,
    }
    first_present = next(
        (field for field in fields.values() if field["status"] == "complete"), None
    )
    pts = first_present["pts"] if first_present else None
    observed = first_present["observed"] if first_present else None
    statuses = ",".join(f"{name}:{field['status']}" for name, field in fields.items())
    click.echo(
        f"group={result.group} source={result.source} pulse={result.pulse.sequence} "
        f"PTS={pts} observed={observed} status={statuses}"
    )

    return record


def _check_independent(records, *, audit):
    frames = [record for record in records if record["group"] == "frames"]
    temperatures = [record for record in records if record["group"] == "temperatures"]
    twins = [record for record in records if record["group"] == "temperature_audit"]
    assert [record["pulse"] for record in frames] == list(range(5))
    assert [record["pulse"] for record in temperatures] == list(range(3))
    assert [record["fields"] for record in temperatures] == [
        record["fields"] for record in twins
    ]
    assert [record["fields"]["count"]["value"] for record in temperatures] == [1, 2, 3]
    assert audit == [32.0, 68.0, 212.0]

    for index, record in enumerate(frames):
        fields = record["fields"]
        bright = index % 2 == 0
        assert fields["original"]["value"]["tensor_shape"] == [3, 12, 16]
        for name in ("resized", "forwarded"):
            assert fields[name]["status"] == ("complete" if bright else "filtered")
        if bright:
            resized = fields["resized"]["value"]
            assert resized["tensor_shape"] == [3, 6, 8]
            assert resized["root"]["frame_id"] == f"frame-{index}"
            assert resized["root"]["scale_xy"] == [2.0, 2.0]
            assert fields["resized"]["pts"] == fields["original"]["pts"]
            assert fields["forwarded"]["pts"] == fields["original"]["pts"]

    for record in records:
        for field in record["fields"].values():
            if field["status"] != "complete":
                continue
            assert field["source"] == record["source"]
            assert field["axis_ids"] == []
            assert field["pts"]["clock"] != field["observed"]["clock"]


def _run_active(case):
    control = DemoControl()
    audit = []
    records = []
    probe = case in ("lifecycle", "stop", "failure")
    plan = compile_workflow(_definition(), catalogue=_catalogue(probe=probe))
    assert control.lifecycle == []  # Compilation does not acquire sources.
    resources = {"audit": audit}
    if probe:
        resources["control"] = control
    session = plan.create_session(resources)
    assert control.lifecycle == []

    def receive(result: GroupResult) -> None:
        records.append(_observe(result))
        control.lifecycle.append(f"{result.group}:{result.pulse.sequence}")
        if result.group == "frames" and result.pulse.sequence == 0:
            if case == "stop":
                # Bound 2 includes this executing pulse. Reaching a third read
                # proves the second was admitted and is waiting for this handler.
                assert control.third_image_read.wait(timeout=5)
                assert active.counters["camera"].admitted == 2
                active.stop()
                active.stop()
            control.first_frame.set()
            if case == "failure":
                raise RuntimeError("intentional frames handler failure")

    active = session.start(
        {"csv_path": str(DEMO_DIR / "fixtures" / "temperatures.csv")},
        handlers={
            name: receive for name in ("frames", "temperatures", "temperature_audit")
        },
        admission_bound=2,
    )
    error_record = None
    try:
        if probe:
            assert control.temperature_opened.wait(timeout=5)
        control.begin.set()
        try:
            assert active.wait(timeout=10), "Sources did not finish within 10 seconds"
        except ActiveRunError as error:
            if case != "failure":
                raise
            error_record = {
                "type": type(error).__name__,
                "message": str(error),
                "stage": error.stage,
                "source": error.source,
                "group": error.group,
            }
            assert error.stage == "handler" and error.group == "frames"
            assert "frames" in str(error)
            assert "intentional frames handler failure" in str(error)
        else:
            assert case != "failure", "Handler failure was swallowed"
    finally:
        active.stop()
        if not active.done:
            active.wait(timeout=10)

    assert active.done
    if probe:
        assert control.lifecycle.count("camera:close") == 1
        assert control.lifecycle.count("temperature:close") == 1
    if case == "lifecycle":
        assert control.lifecycle.index("frames:0") < control.lifecycle.index(
            "temperatures:0"
        )
        assert control.lifecycle.index("temperature:eof") < control.lifecycle.index(
            "frames:3"
        )
    if case in ("independent", "lifecycle"):
        _check_independent(records, audit=audit)
    elif case == "stop":
        assert records[0]["group"] == "frames"
        assert len([record for record in records if record["group"] == "frames"]) == 2
        assert all(record["source"] == "camera" for record in records)

    counters = {
        source: {
            "admitted": counts.admitted,
            "processed": counts.processed,
            "delivered": counts.delivered,
            "cancelled": counts.cancelled,
        }
        for source, counts in active.counters.items()
    }
    if case != "failure":
        assert all(
            value["admitted"] == value["processed"] for value in counters.values()
        )
        assert all(value["cancelled"] == 0 for value in counters.values())
        for source, counts in counters.items():
            assert counts["delivered"] == counts["processed"] * (
                1 if source == "camera" else 2
            )
            assert counts["delivered"] == sum(
                record["source"] == source for record in records
            )
    evidence = {
        "records": records,
        "audit": audit,
        "lifecycle": control.lifecycle,
        "counters": counters,
        "state": active.state,
        "error": error_record,
    }

    return evidence


def _run_csv(*, stop_on_output=False):
    definition = json.loads(
        (DEMO_DIR / "workflows" / "csv_temperature.json").read_text()
    )
    plan = compile_workflow(definition, catalogue=_catalogue())
    records = []
    session = plan.create_session()

    def receive(result: GroupResult) -> None:
        records.append(_observe(result))
        if stop_on_output:
            # The session already exists even if this runs before start returns.
            session.stop()

    active = session.start(
        {"csv_path": str(DEMO_DIR / "fixtures" / "temperatures.csv")},
        handlers={"temperatures": receive},
    )
    try:
        assert active.wait(timeout=10), "CSV source did not reach EOF"
    finally:
        active.stop()
        if not active.done:
            active.wait(timeout=10)
    expected = [32.0, 68.0, 212.0]
    assert records
    if stop_on_output:
        # Acquisition may admit ahead of the first callback. All admitted work
        # must drain, so the callback count is intentionally not fixed.
        expected = expected[: len(records)]
    assert [record["fields"]["fahrenheit"]["value"] for record in records] == expected
    counts = active.counters["temperature"]
    assert counts.admitted == counts.processed == counts.delivered == len(records)
    assert counts.cancelled == 0
    assert counts.closed
    evidence = {
        "records": records,
        "state": active.state,
        "stop_requested": stop_on_output,
        "counters": {
            "admitted": counts.admitted,
            "processed": counts.processed,
            "delivered": counts.delivered,
            "cancelled": counts.cancelled,
            "closed": counts.closed,
        },
    }

    return evidence


def _run_invalid_join():
    definition = _definition()
    definition["outputs"][1]["outputs"].append(
        {
            "type": "JsonField",
            "name": "unrelated_image",
            "selector": "$sources.camera.image",
        }
    )
    try:
        compile_workflow(definition, catalogue=_catalogue())
    except LineageError as error:
        message = str(error)
        assert "camera" in message and "temperature" in message
        click.echo(f"Expected independent-source join rejection: {message}")
        evidence = {"rejected": True, "type": type(error).__name__, "message": message}
        return evidence

    raise AssertionError("Independent source domains were silently joined")


@click.command()
@click.option(
    "--case",
    "case_name",
    type=click.Choice(("all",) + CASES),
    default="all",
    show_default=True,
    help="Choose a runnable lifecycle or metadata example.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Optionally save machine-readable evidence outside the checkout.",
)
def main(case_name: str, output_dir: Path | None) -> None:
    """Run local deterministic examples with no model, device or network setup.

    Args:
        case_name: One example name, or all examples.
        output_dir: Optional directory for evidence.json.
    """
    results = {}
    for case in CASES if case_name == "all" else (case_name,):
        click.echo(f"\n{case}")
        if case in ("csv", "csv-stop"):
            result = _run_csv(stop_on_output=case == "csv-stop")
        elif case == "passive":
            result = run_passive()
            click.echo(json.dumps(result))
        elif case == "invalid-join":
            result = _run_invalid_join()
        else:
            result = _run_active(case)
        results[case] = result
        click.echo(f"PASS {case}")

    if output_dir is not None:
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / "evidence.json"
        destination.write_text(json.dumps(results, indent=2) + "\n")
        click.echo(f"\nEvidence: {destination}")


if __name__ == "__main__":
    main()
