"""Real model: the M3 flip-averaged ResNet-18, serial versus pipelined.

The classifier, its implementations, the pinned assets and the comparison
helpers are imported from ``../06-model-phases``; nothing is copied. Each
image gets a unique ``image_id``, and every pipelined prediction is matched
to its input through the prediction's own ``images_metadata``, not through
submission order.

Timings are single wall-clock observations on this host. They are recorded
for inspection, not as a performance claim, and say nothing about CUDA.
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

MODEL_PHASES_DIR = Path(__file__).resolve().parents[1] / "06-model-phases"
if str(MODEL_PHASES_DIR) not in sys.path:
    sys.path.append(str(MODEL_PHASES_DIR))

import torch  # noqa: E402
from assets import (  # noqa: E402
    RESNET18_WEIGHTS,
    AssetError,
    load_image,
    locate_weights,
    verify,
)
from host import (  # noqa: E402
    compile_for,
    load_definition,
    output_metadata,
    output_value,
)
from native_comparison import compare_predictions  # noqa: E402
from observations import compare_outputs, expect, leaves, media_pts_ms  # noqa: E402
from resnet18 import load_state_dict, top_classes  # noqa: E402
from roboflow_workflows.execution_engine.v2.blocks.image_data import (  # noqa: E402
    ImageData,
)
from roboflow_workflows.execution_engine.v2.pipelining import (  # noqa: E402
    PipelineOptions,
)
from roboflow_workflows.execution_engine.v2.targets import Target  # noqa: E402

IMAGE_NAMES = ["beagle", "dogs", "car"]
EXPECTED_TOP = {"beagle": "basset", "dogs": "Norfolk terrier", "car": "car wheel"}
WAIT_SECONDS = 300

StateDict = Mapping[str, torch.Tensor]

__all__ = [
    "AssetError",
    "active_model",
    "load_weights",
    "model_example",
    "passive_model",
]


def load_weights(
    *, weights: Optional[Path], weights_dir: Optional[Path]
) -> Tuple[Path, StateDict]:
    """Verify the pinned ResNet-18 weights and load them; never downloads.

    Args:
        weights: Explicit weights file, checked against its pinned identity.
        weights_dir: Directory searched when ``weights`` is None; torch's
            checkpoint directory when both are None.

    Returns:
        The verified path and the loaded state dict.

    Raises:
        AssetError: When the file is missing or differs from its identity.
    """
    path = (
        verify(weights, RESNET18_WEIGHTS)
        if weights is not None
        else locate_weights(weights_dir)
    )
    state_dict = load_state_dict(path)

    return path, state_dict


def model_example(
    state_dict: StateDict, *, repeats: int, max_in_flight: int
) -> Dict[str, Any]:
    """Passive comparison on CPU (and MPS when present), active comparison on CPU.

    Args:
        state_dict: Trained ResNet-18 weights.
        repeats: Repetitions of the pinned images (or frame triples).
        max_in_flight: Pipeline workers.

    Returns:
        Evidence of every comparison; MPS is recorded as skipped when absent.
    """
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    passive: Dict[str, Any] = {
        device: passive_model(
            state_dict, device=device, repeats=repeats, max_in_flight=max_in_flight
        )
        for device in devices
    }
    if "mps" not in devices:
        passive["mps"] = {"skipped": "MPS is not available on this host"}
    evidence = {
        "passive": passive,
        "active_cpu": active_model(
            state_dict, repeats=repeats, max_in_flight=max_in_flight
        ),
    }

    return evidence


def _images(repeats: int) -> List[ImageData]:
    """Every pinned image ``repeats`` times, each with its own ``image_id``."""
    originals = {name: load_image(name) for name in IMAGE_NAMES}
    images = [
        ImageData.from_tensor(originals[name].tensor_image, image_id=f"{name}#{repeat}")
        for repeat in range(repeats)
        for name in IMAGE_NAMES
    ]

    return images


def passive_model(
    state_dict: StateDict, *, device: str, repeats: int, max_in_flight: int
) -> Dict[str, Any]:
    """Classify the same images with ``session.run`` and with ``session.pipeline``.

    Args:
        state_dict: Trained ResNet-18 weights.
        device: ``cpu`` or ``mps``; selects the implementation by target.
        repeats: How often each pinned image is submitted.
        max_in_flight: Pipeline workers.

    Returns:
        Per-image comparison, association and the raw timing observation.
    """
    target = Target.cpu() if device == "cpu" else Target(frozenset({"mps"}))
    plan = compile_for(
        load_definition("classify_one.json"), target=target, execution="phases"
    )
    expect("selected implementation", plan.steps[0].selected.name, device)
    session = plan.create_session({"resnet18_state_dict": state_dict})
    images = _images(repeats)
    # One untimed run first, so lazy initialization is not in either timing.
    session.run({"image": images[0]})

    started = time.perf_counter()
    serial = [session.run({"image": image}) for image in images]
    serial_seconds = time.perf_counter() - started

    started = time.perf_counter()
    with session.pipeline(options=PipelineOptions(max_in_flight=max_in_flight)) as pipe:
        futures = [pipe.submit({"image": image}) for image in images]
        pipelined = [future.result(timeout=WAIT_SECONDS) for future in futures]
    pipelined_seconds = time.perf_counter() - started

    by_image = {}
    for image, reference in zip(images, serial):
        by_image[image.image_id] = output_value(reference, "predictions")
    records = []
    for result in pipelined:
        prediction = output_value(result, "predictions")
        image_id = prediction.images_metadata[0]["parent_id"]
        comparison = compare_predictions(by_image.pop(image_id), prediction)
        expect(
            f"{image_id}: pipelined equals serial (bitwise)", comparison["equal"], True
        )
        top = top_classes(prediction, 1)[0]["class_name"]
        expect(f"{image_id}: top class", top, EXPECTED_TOP[image_id.split("#")[0]])
        records.append({"image_id": image_id, "top_class": top, **comparison})
    expect("every image classified exactly once", by_image, {})

    snapshot = pipe.counters.snapshot()
    expect(
        "submissions in flight never exceed the bound",
        snapshot["peak"]["executing"] <= max_in_flight,
        True,
    )
    evidence = {
        "device": device,
        "implementation": plan.steps[0].selected.name,
        "images": len(images),
        "max_in_flight": max_in_flight,
        "comparisons": records,
        "pipeline_counters": snapshot,
        "observed_seconds": {
            "serial": round(serial_seconds, 4),
            "pipelined": round(pipelined_seconds, 4),
            "note": "single wall-clock observation on this host; not a performance claim",
        },
        "torch_threads": torch.get_num_threads(),
    }

    return evidence


def active_model(
    state_dict: StateDict, *, repeats: int, max_in_flight: int
) -> Dict[str, Any]:
    """Run M3's active workflow (frames, gated child crops, window) both ways on CPU.

    Args:
        state_dict: Trained ResNet-18 weights.
        repeats: How often the three frames repeat in the source.
        max_in_flight: Pipeline workers.

    Returns:
        Per-frame and per-window comparison and the raw timing observation.
    """
    definition = load_definition("active.json")
    definition["sources"][0]["images"] = IMAGE_NAMES * repeats
    plan = compile_for(definition, execution="phases")
    runs = {}
    seconds = {}
    counters: Optional[Dict[str, Any]] = None
    for mode, pipeline in (
        ("serial", None),
        ("pipelined", PipelineOptions(max_in_flight=max_in_flight)),
    ):
        delivered: Dict[str, List[Any]] = {"frames": [], "recent": []}
        session = plan.create_session({"resnet18_state_dict": state_dict})
        started = time.perf_counter()
        run = session.start(
            {},
            handlers={name: delivered[name].append for name in delivered},
            admission_bound=max_in_flight,
            pipeline=pipeline,
        )
        expect(f"{mode}: run completes", run.wait(timeout=WAIT_SECONDS), True)
        seconds[mode] = round(time.perf_counter() - started, 4)
        runs[mode] = delivered
        if pipeline is not None:
            counters = run.pipeline_counters.snapshot()

    frames = []
    for position, (reference, result) in enumerate(
        zip(runs["serial"]["frames"], runs["pipelined"]["frames"])
    ):
        pts = media_pts_ms(output_metadata(result, "predictions"), ())
        prediction = output_value(result, "predictions")
        frame_id = f"{IMAGE_NAMES[position % 3]}@{pts}ms"
        expect("frames arrive in PTS order", pts, position * 40)
        expect(
            "prediction names its original frame",
            prediction.images_metadata[0]["root_parent_id"],
            frame_id,
        )
        compare_outputs(output_value(reference, "predictions"), prediction)
        frames.append(
            {
                "frame": frame_id,
                "top_class": output_value(result, "top_class"),
                "band_crops": _band_count(result),
            }
        )
    expect(
        "same number of frames",
        len(runs["pipelined"]["frames"]),
        len(runs["serial"]["frames"]),
    )
    windows = [
        [label for _, label in leaves(output_value(result, "labels"))]
        for result in runs["pipelined"]["recent"]
    ]
    expect(
        "windows equal serial",
        windows,
        [
            [label for _, label in leaves(output_value(result, "labels"))]
            for result in runs["serial"]["recent"]
        ],
    )
    evidence = {
        "frames": frames,
        "windows": windows,
        "pipeline_counters": counters,
        "observed_seconds": {
            **seconds,
            "note": "single wall-clock observation on this host; not a performance claim",
        },
    }

    return evidence


def _band_count(result: Any) -> int:
    (entry,) = result.selections["band_predictions"].values()
    if result.statuses[entry] != "complete":
        return 0

    count = len(leaves(output_value(result, "band_predictions")))

    return count
