"""Real-model examples: the main workflow, direct calls and the MPS backend."""

from typing import Any, Dict

import torch
from assets import load_image
from classifier import CpuResNet18, FlipAveragedClassifier, MpsResNet18
from host import DemoContext, compile_for, load_definition, output_value, run_passive
from native_comparison import compare_predictions
from observations import (
    check_own_crops,
    compare_outputs,
    describe_prediction,
    expect,
    leaves,
    phase_events,
)
from resnet18 import top_classes
from roboflow_workflows.execution_engine.v2.declaration import spec_of
from roboflow_workflows.execution_engine.v2.phases import run_phases
from roboflow_workflows.execution_engine.v2.targets import Target

CLASSIFIER_PHASES = [
    "tensor",
    "logits",
    "flipped",
    "flipped_logits",
    "probabilities",
    "result",
]
REGIONS = {
    "dogs": [[215, 65, 375, 385], [310, 215, 585, 410]],
    "beagle": [[0, 0, 720, 400], [0, 700, 720, 1280]],
}
"""Per image: each dog of ``dogs``; the beagle's head and the backpack below it."""
CROSS_DEVICE_TOLERANCE = 1e-3


def main_inputs() -> Dict[str, Any]:
    """Freshly decoded inputs of ``classify.json``.

    Returns:
        Two images and their crop rectangles.
    """
    inputs = {
        "images": [load_image(name) for name in REGIONS],
        "regions": list(REGIONS.values()),
    }

    return inputs


def classify(context: DemoContext) -> Dict[str, Any]:
    """Main workflow in run and phase mode: crops, nested gate, same predictions."""
    definition = load_definition("classify.json")
    results = {
        mode: run_passive(
            compile_for(definition, execution=mode), context.state_dict, main_inputs()
        )
        for mode in ("run", "phases")
    }
    run, phased = results["run"], results["phases"]

    images = compare_outputs(
        output_value(run, "predictions"), output_value(phased, "predictions")
    )
    crops = compare_outputs(
        output_value(run, "crop_predictions"),
        output_value(phased, "crop_predictions"),
        ignore_metadata=("parent_id",),
    )
    for result in (run, phased):
        check_own_crops(
            output_value(result, "crop_predictions"), output_value(result, "crops")
        )
    predictions = dict(leaves(output_value(phased, "predictions")))
    crop_predictions = dict(leaves(output_value(phased, "crop_predictions")))
    expect(
        "beagle photo: basset then beagle",
        [item["class_name"] for item in top_classes(predictions[(1,)], 2)],
        ["basset", "beagle"],
    )
    expect(
        "crop labels; the dark backpack crop [1, 1] is gated",
        {
            index: top_classes(prediction, 1)[0]["class_name"]
            for index, prediction in crop_predictions.items()
        },
        {(0, 0): "Saint Bernard", (0, 1): "toy terrier", (1, 0): "beagle"},
    )
    small_dog = crop_predictions[(0, 1)].images_metadata[0]
    expect("small dog crop root", small_dog["root_parent_id"], "dogs")
    expect("small dog crop offset", small_dog["root_parent_coordinates"], [310, 215])
    expect("small dog crop size", small_dog["image_dimensions"], [195, 275])

    events = phase_events(phased.trace)
    expect("run mode records no phase events", phase_events(run.trace), {})
    expect(
        "every classification ran the six phases once, gated crop none",
        events,
        {
            key: CLASSIFIER_PHASES
            for key in (
                "classify@[0]",
                "classify@[1]",
                "refine/classify@[0, 0]",
                "refine/classify@[0, 1]",
                "refine/classify@[1, 0]",
            )
        },
    )

    _add_figures(context, phased)

    evidence = {
        "images": {
            str(list(index)): describe_prediction(p, 5)
            for index, p in predictions.items()
        },
        "crops": {
            str(list(index)): describe_prediction(p)
            for index, p in crop_predictions.items()
        },
        "run_vs_phases": {"images": images, "crops": crops},
        "phase_events": events,
    }

    return evidence


def _add_figures(context: DemoContext, result: Any) -> None:
    context.gallery.section("classify: whole images (flip-averaged ResNet-18)")
    images = dict(leaves(output_value(result, "predictions")))
    for (position,), prediction in images.items():
        name = list(REGIONS)[position]
        context.gallery.add(
            load_image(name),
            name=f"classify-{name}",
            lines=[f"{name}.jpg"] + _top_lines(prediction, 5),
        )

    context.gallery.section("classify: crops refined in a nested workflow")
    crop_predictions = dict(leaves(output_value(result, "crop_predictions")))
    for index, crop in leaves(output_value(result, "crops")):
        label = "-".join(str(part) for part in index)
        prediction = crop_predictions.get(index)
        lines = [f"crop {list(index)} of {crop.root.frame_id}"]
        if prediction is None:
            lines.append("gated: mean brightness below 120, never classified")
        else:
            lines.extend(_top_lines(prediction, 3))
        context.gallery.add(crop, name=f"crop-{label}", lines=lines)


def _top_lines(prediction: Any, k: int) -> list:
    lines = [
        f"{item['class_name']}: {item['confidence']:.3f}"
        for item in top_classes(prediction, k)
    ]

    return lines


def direct(context: DemoContext) -> Dict[str, Any]:
    """Call the implementation without the engine: run() and run_phases agree."""
    implementation = CpuResNet18(resnet18_state_dict=context.state_dict)
    graph = next(
        choice.phases
        for choice in spec_of(FlipAveragedClassifier).implementations
        if choice.name == CpuResNet18.name
    )
    image = load_image("beagle")
    pixels_before = image.tensor_image.clone()

    explicit = implementation.run(image=image)["predictions"]
    order = []
    phased = run_phases(implementation, graph, {"image": image}, on_phase=order.append)
    engine = output_value(
        run_passive(
            compile_for(load_definition("classify_one.json"), execution="phases"),
            context.state_dict,
            {"image": image},
        ),
        "predictions",
    )

    expect("phase order", order, CLASSIFIER_PHASES)
    explicit_vs_phases = compare_predictions(explicit, phased["predictions"])
    explicit_vs_engine = compare_predictions(explicit, engine)
    expect("run() equals run_phases bitwise", explicit_vs_phases["equal"], True)
    expect("run() equals the engine's phase mode", explicit_vs_engine["equal"], True)
    expect(
        "input pixels unchanged", torch.equal(image.tensor_image, pixels_before), True
    )

    evidence = {
        "graph": graph.describe(),
        "phase_order": order,
        "explicit": describe_prediction(explicit, 5),
        "explicit_vs_run_phases": explicit_vs_phases,
        "explicit_vs_engine_phases": explicit_vs_engine,
    }

    return evidence


def mps(context: DemoContext) -> Dict[str, Any]:
    """The MPS implementation on this host: phases equal run(); CPU within 1e-3."""
    if not torch.backends.mps.is_available():
        return {"skipped": "torch.backends.mps.is_available() is False on this host"}

    definition = load_definition("classify_one.json")
    target = Target(frozenset({"mps"}))
    plans = {
        "cpu": compile_for(definition, execution="phases"),
        "mps-run": compile_for(definition, target=target, execution="run"),
        "mps-phases": compile_for(definition, target=target, execution="phases"),
    }
    expect(
        "selected implementations",
        {name: plan.steps[0].selected.name for name, plan in plans.items()},
        {"cpu": "cpu", "mps-run": "mps", "mps-phases": "mps"},
    )

    implementation = MpsResNet18(resnet18_state_dict=context.state_dict)
    beagle = load_image("beagle")
    devices = {
        "network": next(implementation.network.parameters()).device.type,
        "tensor phase": implementation.tensor(beagle).device.type,
        "result predictions": implementation.run(image=beagle)[
            "predictions"
        ].confidence.device.type,
    }
    expect(
        "intermediates on MPS, outputs on the image's device",
        devices,
        {"network": "mps", "tensor phase": "mps", "result predictions": "cpu"},
    )

    images = {}
    for name in ("beagle", "dogs", "car"):
        predictions = {
            label: output_value(
                run_passive(plan, context.state_dict, {"image": load_image(name)}),
                "predictions",
            )
            for label, plan in plans.items()
        }
        same_device = compare_predictions(
            predictions["mps-run"], predictions["mps-phases"]
        )
        cross_device = compare_predictions(
            predictions["cpu"],
            predictions["mps-phases"],
            tolerance=CROSS_DEVICE_TOLERANCE,
        )
        expect(
            f"{name}: MPS run() equals MPS phases bitwise", same_device["equal"], True
        )
        expect(
            f"{name}: CPU and MPS agree within tolerance", cross_device["equal"], True
        )
        images[name] = {
            "mps": describe_prediction(predictions["mps-phases"]),
            "mps_run_vs_phases": same_device,
            "cpu_vs_mps": cross_device,
        }

    evidence = {
        "device": "mps (Apple Metal), executed on this host",
        "devices": devices,
        "readiness": "the result phase copies probabilities to the CPU; the copy waits for MPS work",
        "images": images,
    }

    return evidence
