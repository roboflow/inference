"""Active example: timed frames, nested gated crops and a window of predictions."""

from typing import Any, Dict, List, Optional

from assets import load_image
from host import (
    DemoContext,
    compile_for,
    load_definition,
    output_metadata,
    output_value,
    run_active,
)
from observations import compare_outputs, expect, leaves, media_pts_ms, phase_events
from resnet18 import top_classes

FRAMES = ["beagle", "dogs", "car"]
INTERVAL_MS = 40


def _value_or_none(result: Any, name: str) -> Optional[Any]:
    # A gated output is delivered as a filtered entry, without a value.
    (entry,) = result.selections[name].values()
    if result.statuses[entry] != "complete":
        return None

    value = output_value(result, name)

    return value


def _run(mode: str, context: DemoContext) -> Dict[str, List[Any]]:
    delivered = run_active(
        compile_for(load_definition("active.json"), execution=mode),
        context.state_dict,
        groups=["frames", "recent"],
    )
    by_group = {
        group: [result for result in delivered if result.group == group]
        for group in ("frames", "recent")
    }

    return by_group


def active(context: DemoContext) -> Dict[str, Any]:
    """Frames keep their PTS and ancestry through crops, the child and a window."""
    runs = {mode: _run(mode, context) for mode in ("run", "phases")}
    frames = runs["phases"]["frames"]
    expect("one frames result per image", len(frames), len(FRAMES))
    expect("one window of three frames", len(runs["phases"]["recent"]), 1)

    observed = []
    for position, result in enumerate(frames):
        pts = media_pts_ms(output_metadata(result, "predictions"), ())
        prediction = output_value(result, "predictions")
        band = _value_or_none(result, "band_predictions")
        frame_id = f"{FRAMES[position]}@{pts}ms"
        expect("frame PTS", pts, position * INTERVAL_MS)
        expect(
            "prediction root is the frame",
            prediction.images_metadata[0]["root_parent_id"],
            frame_id,
        )
        band_leaves = [] if band is None else leaves(band)
        for index, crop_prediction in band_leaves:
            expect(
                "band crop root",
                crop_prediction.images_metadata[0]["root_parent_id"],
                frame_id,
            )
            expect(
                "band crop PTS",
                media_pts_ms(output_metadata(result, "band_predictions"), index),
                pts,
            )
        observed.append(
            {
                "frame": frame_id,
                "pts_ms": pts,
                "top_class": output_value(result, "top_class"),
                "band": [top_classes(p, 1)[0]["class_name"] for _, p in band_leaves],
                "phase_events": phase_events(result.trace),
            }
        )
    expect(
        "labels per frame",
        [item["top_class"] for item in observed],
        ["basset", "Norfolk terrier", "car wheel"],
    )
    expect(
        "the dark car band is gated before the classifier",
        [len(item["band"]) for item in observed],
        [1, 1, 0],
    )

    context.gallery.section("active: timed frames, gated lower band, window of labels")
    for name, item in zip(FRAMES, observed):
        band = item["band"][0] if item["band"] else "gated (dark), never classified"
        context.gallery.add(
            load_image(name),
            name=f"active-{name}",
            lines=[
                f"{item['frame']} PTS {item['pts_ms']} ms",
                item["top_class"],
                f"band: {band}",
            ],
        )

    (window,) = runs["phases"]["recent"]
    labels = leaves(output_value(window, "labels"))
    window_metadata = output_metadata(window, "labels")
    expect(
        "window members keep their own PTS",
        [(label, media_pts_ms(window_metadata, index)) for index, label in labels],
        [("basset", 0), ("Norfolk terrier", 40), ("car wheel", 80)],
    )

    comparisons = [
        compare_outputs(
            output_value(run_frame, "predictions"),
            output_value(phase_frame, "predictions"),
        )
        for run_frame, phase_frame in zip(runs["run"]["frames"], frames)
    ]
    window_comparison = compare_outputs(
        output_value(runs["run"]["recent"][0], "predictions"),
        output_value(window, "predictions"),
    )

    evidence = {
        "frames": observed,
        "window": [{"index": list(index), "label": label} for index, label in labels],
        "run_vs_phases": {"frames": comparisons, "window": window_comparison},
    }

    return evidence
