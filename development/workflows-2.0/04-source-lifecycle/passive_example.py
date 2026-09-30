"""The same ordinary transformation with timestamped passive ingress."""

from processing import CelsiusToFahrenheit
from roboflow_workflows.execution_engine.v2.blocks import create_catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.data import (
    EntryMetadata,
    InputValue,
    SampleContext,
)
from source_fixtures import make_timing


def run_passive() -> dict:
    """Check timestamp preservation, unwrapped inputs and persistent block state.

    Returns:
        JSON-friendly observations for the CLI and scenario test.
    """
    definition = {
        "version": "2.0",
        "inputs": [{"name": "celsius", "kind": ["float"], "axes": []}],
        "steps": [
            {
                "type": CelsiusToFahrenheit.type,
                "name": "convert",
                "celsius": "$inputs.celsius",
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "fahrenheit",
                "selector": "$steps.convert.fahrenheit",
            },
            {"type": "JsonField", "name": "count", "selector": "$steps.convert.count"},
        ],
    }
    timing = make_timing(pts_ms=110, observed_ms=10110, media_clock="thermometer-media")
    metadata = EntryMetadata(
        sample={(): SampleContext(source_id="thermometer", source_type="csv")},
        temporal={(): timing},
    )
    catalogue = create_catalogue().with_blocks([CelsiusToFahrenheit])
    session = compile_workflow(definition, catalogue=catalogue).create_session()
    result = session.run({"celsius": InputValue(data=20.0, metadata=metadata)})
    assert result.rows() == [{"fahrenheit": 68.0, "count": 1}]
    (entry,) = result.selections["fahrenheit"].values()
    observed = result.outputs.metadata[entry]
    assert observed.sample_at(()).source_id == "thermometer"
    assert observed.temporal_at(()) == timing
    assert result.outputs.layout[entry].axes == ()

    # The caller may still supply an ordinary scalar on the same session.
    second = session.run({"celsius": 0.0})
    assert second.rows() == [{"fahrenheit": 32.0, "count": 2}]
    direct = CelsiusToFahrenheit().run(celsius=20.0)
    assert direct["fahrenheit"] == result.rows()[0]["fahrenheit"]
    evidence = {
        "source": observed.sample_at(()).source_id,
        "pts_ticks": timing.media_coverage.ticks,
        "pts_time_base": str(timing.media_coverage.time_base),
        "media_clock": timing.media_coverage.clock_id,
        "observed_ticks": timing.observed_coverage.ticks,
        "observation_clock": timing.observed_coverage.clock_id,
        "axis_ids": [],
        "timestamped_rows": result.rows(),
        "unwrapped_rows": second.rows(),
        "direct_block_matches": True,
    }

    return evidence
