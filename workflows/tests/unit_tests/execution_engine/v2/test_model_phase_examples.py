"""Model phase examples (development/workflows-2.0/06-model-phases).

Two kinds of evidence, kept apart:

- Unit tests use an *untrained*, seeded ResNet-18 state dict. Its predictions
  are meaningless; it only exercises declaration, selection, phase execution,
  metadata and mutation plumbing, and never needs a download.
- ``test_real_model_examples`` runs the demo with the pinned trained weights.
  It is skipped unless they are present locally (``RESNET18_WEIGHTS_DIR`` or
  the torch hub checkpoint directory).
"""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPOSITORY_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "development" / "workflows-2.0").is_dir()
)
DEMO_DIR = REPOSITORY_ROOT / "development/workflows-2.0/06-model-phases"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import assets  # noqa: E402
import classifier  # noqa: E402
import host  # noqa: E402
from observations import (  # noqa: E402
    check_own_crops,
    compare_outputs,
    leaves,
    media_pts_ms,
    phase_events,
)
from roboflow_workflows.execution_engine.v2.blocks.image_data import (  # noqa: E402
    ImageData,
)
from roboflow_workflows.execution_engine.v2.declaration import spec_of  # noqa: E402
from roboflow_workflows.execution_engine.v2.errors import (  # noqa: E402
    MutationConflictError,
    SelectorError,
    StepExecutionError,
)
from roboflow_workflows.execution_engine.v2.introspection import (  # noqa: E402
    describe_catalogue,
    describe_workflow,
    discover_workload,
)
from roboflow_workflows.execution_engine.v2.phases import (  # noqa: E402
    PhaseFailure,
    run_phases,
)
from roboflow_workflows.execution_engine.v2.targets import (  # noqa: E402
    Target,
    UnsupportedTargetError,
)
from torchvision.models import resnet18  # noqa: E402

PHASES = ["tensor", "logits", "flipped", "flipped_logits", "probabilities", "result"]


def test_demo_modules_come_from_the_example_directory():
    assert Path(classifier.__file__).parent == DEMO_DIR
    assert Path(host.__file__).parent == DEMO_DIR


@pytest.fixture(scope="module")
def untrained_state_dict():
    torch.manual_seed(0)
    state_dict = resnet18(weights=None).state_dict()

    return state_dict


def _two_tone_image(image_id: str) -> ImageData:
    # Bright left half, dark right half: the nested brightness gate admits a
    # crop of the left half only.
    pixels = torch.empty((3, 120, 160), dtype=torch.uint8)
    pixels[:, :, :80] = 200
    pixels[:, :, 80:] = 40
    pixels[0, ::7, ::5] = 90

    image = ImageData.from_tensor(pixels, image_id=image_id)

    return image


def _main_inputs():
    inputs = {
        "images": [_two_tone_image("first"), _two_tone_image("second")],
        "regions": [[[0, 0, 80, 120], [80, 0, 160, 120]], [[10, 20, 70, 100]]],
    }

    return inputs


def test_classifier_declares_three_choices_one_diamond_and_no_fused_phases():
    implementations = {
        choice.name: choice
        for choice in spec_of(classifier.FlipAveragedClassifier).implementations
    }

    assert list(implementations) == ["mps", "cpu-batched-views", "cpu"]
    assert implementations["cpu-batched-views"].phases is None
    graph = implementations["cpu"].phases.describe()
    upstream = {phase["name"]: phase["upstream"] for phase in graph["phases"]}
    assert graph["result"] == "result"
    assert upstream["logits"] == ["tensor"]
    assert upstream["flipped"] == ["tensor"]
    assert upstream["probabilities"] == ["logits", "flipped_logits"]
    assert implementations["mps"].phases == implementations["cpu"].phases


@pytest.mark.parametrize(
    "capabilities, implementation, execution",
    [
        ({"cpu"}, "cpu", "phases"),
        ({"cpu", "batched_views"}, "cpu-batched-views", "run"),
        ({"mps"}, "mps", "phases"),
        ({"mps", "cpu", "batched_views"}, "mps", "phases"),
    ],
)
def test_selection_follows_declared_order_and_falls_back_to_run(
    capabilities, implementation, execution
):
    plan = host.compile_for(
        host.load_definition("classify.json"),
        target=Target(frozenset(capabilities)),
        execution="phases",
    )

    selected = {
        step.path: (step.selected.name, step.execution)
        for step in plan.steps
        if step.block_type == classifier.FlipAveragedClassifier.type
    }
    assert selected == {
        ("classify",): (implementation, execution),
        ("refine", "classify"): (implementation, execution),
    }


def test_unsupported_target_lists_what_each_implementation_misses():
    with pytest.raises(UnsupportedTargetError) as raised:
        host.compile_for(
            host.load_definition("classify_one.json"),
            target=Target(frozenset({"cuda"})),
        )

    assert raised.value.considered == (
        ("mps", ("mps",)),
        ("cpu-batched-views", ("batched_views", "cpu")),
        ("cpu", ("cpu",)),
    )


def test_compilation_and_inspection_never_read_weights(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("weights were loaded during compilation or inspection")

    monkeypatch.setattr(torch, "load", refuse)

    plan = host.compile_for(host.load_definition("active.json"), execution="phases")
    description = describe_workflow(plan)
    workload = discover_workload(plan).step("$steps.classify").describe()
    catalogue = describe_catalogue(host.create_demo_catalogue())

    assert description["block_execution"] == "phases"
    assert workload["resources"]["items"][0]["identifier"] == (
        "torchvision/resnet18/IMAGENET1K_V1"
    )
    assert "model_demo/flip_averaged_classifier@v1" in json.dumps(catalogue)


def test_run_and_phase_modes_agree_bitwise_with_untrained_weights(
    untrained_state_dict,
):
    definition = host.load_definition("classify.json")
    run, phased = (
        host.run_passive(
            host.compile_for(definition, execution=mode),
            untrained_state_dict,
            _main_inputs(),
        )
        for mode in ("run", "phases")
    )

    compare_outputs(
        host.output_value(run, "predictions"),
        host.output_value(phased, "predictions"),
    )
    compare_outputs(
        host.output_value(run, "crop_predictions"),
        host.output_value(phased, "crop_predictions"),
        ignore_metadata=("parent_id",),
    )
    crop_predictions = dict(leaves(host.output_value(phased, "crop_predictions")))
    assert sorted(crop_predictions) == [(0, 0), (1, 0)]
    assert (
        check_own_crops(
            host.output_value(phased, "crop_predictions"),
            host.output_value(phased, "crops"),
        )
        == 2
    )
    assert phase_events(run.trace) == {}
    assert phase_events(phased.trace) == {
        key: PHASES
        for key in (
            "classify@[0]",
            "classify@[1]",
            "refine/classify@[0, 0]",
            "refine/classify@[1, 0]",
        )
    }


def test_crop_prediction_metadata_maps_to_the_root_image(untrained_state_dict):
    result = host.run_passive(
        host.compile_for(host.load_definition("classify.json"), execution="phases"),
        untrained_state_dict,
        _main_inputs(),
    )

    crop_predictions = dict(leaves(host.output_value(result, "crop_predictions")))
    (metadata,) = crop_predictions[(1, 0)].images_metadata
    assert metadata["root_parent_id"] == "second"
    assert metadata["root_parent_coordinates"] == [10, 20]
    assert metadata["image_dimensions"] == [80, 60]
    (whole,) = dict(leaves(host.output_value(result, "predictions")))[
        (0,)
    ].images_metadata
    assert whole["parent_id"] == "first"


def test_direct_run_equals_run_phases_and_leaves_input_untouched(
    untrained_state_dict,
):
    implementation = classifier.CpuResNet18(resnet18_state_dict=untrained_state_dict)
    graph = spec_of(classifier.FlipAveragedClassifier).implementations[-1].phases
    image = _two_tone_image("direct")
    pixels = image.tensor_image.clone()
    order = []

    explicit = implementation.run(image=image)["predictions"]
    phased = run_phases(implementation, graph, {"image": image}, on_phase=order.append)[
        "predictions"
    ]

    assert order == PHASES
    assert torch.equal(explicit.confidence, phased.confidence)
    assert explicit.images_metadata == phased.images_metadata
    assert torch.equal(image.tensor_image, pixels)


def test_batched_views_match_within_tolerance(untrained_state_dict):
    image = _two_tone_image("batched")
    separate = classifier.CpuResNet18(resnet18_state_dict=untrained_state_dict)
    batched = classifier.CpuBatchedViews(resnet18_state_dict=untrained_state_dict)

    reference = separate.run(image=image)["predictions"]
    candidate = batched.run(image=image)["predictions"]

    assert torch.equal(reference.class_id, candidate.class_id)
    assert torch.allclose(reference.confidence, candidate.confidence, atol=1e-5)


def test_unordered_mutation_warns_and_strict_mode_rejects():
    definition = host.load_definition("mutation_unordered.json")

    plan = host.compile_for(definition)

    assert len(plan.warnings) == 1
    assert "$steps.redact" in plan.warnings[0]
    with pytest.raises(MutationConflictError):
        host.compile_for(definition, mutation_conflicts="error")


def test_ordered_redaction_mutates_the_caller_tensor_inside_a_phase(
    untrained_state_dict,
):
    image = ImageData.from_tensor(
        torch.full((3, 800, 720), 50, dtype=torch.uint8), image_id="caller"
    )
    plan = host.compile_for(
        host.load_definition("mutation_ordered.json"), execution="phases"
    )

    result = host.run_passive(plan, untrained_state_dict, {"image": image})

    assert plan.warnings == ()
    assert bool((image.tensor_image[:, :640, :] == 128).all())
    assert bool((image.tensor_image[:, 640:, :] == 50).all())
    redacted = host.output_value(result, "redacted")
    assert redacted.tensor_image.data_ptr() == image.tensor_image.data_ptr()
    assert phase_events(result.trace)["redact@[]"] == ["bounds", "result"]


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_failing_phase_is_named_in_both_modes(untrained_state_dict, mode):
    definition = copy.deepcopy(host.load_definition("mutation_ordered.json"))
    definition["steps"][0]["region"] = [2000, 2000, 2100, 2100]
    plan = host.compile_for(definition, execution=mode)

    with pytest.raises(StepExecutionError) as raised:
        host.run_passive(
            plan, untrained_state_dict, {"image": _two_tone_image("small")}
        )

    assert raised.value.phase == "bounds"
    assert raised.value.step_path == ("redact",)
    assert "lies outside the 160x120 image" in str(raised.value)


def test_direct_call_failure_is_a_phase_failure():
    from redaction import RedactRegion

    with pytest.raises(PhaseFailure) as raised:
        RedactRegion().run(image=_two_tone_image("small"), region=[500, 500, 600, 600])

    assert raised.value.phase == "bounds"


def test_selecting_a_phase_result_as_output_is_rejected():
    with pytest.raises(SelectorError, match="has no output 'probabilities'"):
        host.compile_for(host.load_definition("invalid/private_phase.json"))


def test_active_frames_keep_pts_and_ancestry_with_untrained_weights(
    untrained_state_dict,
):
    delivered = {
        mode: host.run_active(
            host.compile_for(host.load_definition("active.json"), execution=mode),
            untrained_state_dict,
            groups=["frames", "recent"],
        )
        for mode in ("run", "phases")
    }

    frames = [result for result in delivered["phases"] if result.group == "frames"]
    assert [
        host.output_value(result, "predictions").images_metadata[0]["root_parent_id"]
        for result in frames
    ] == ["beagle@0ms", "dogs@40ms", "car@80ms"]
    assert [
        media_pts_ms(host.output_metadata(result, "predictions"), ())
        for result in frames
    ] == [0, 40, 80]
    (window,) = [result for result in delivered["phases"] if result.group == "recent"]
    labels = leaves(host.output_value(window, "labels"))
    window_metadata = host.output_metadata(window, "labels")
    assert [media_pts_ms(window_metadata, index) for index, _ in labels] == [0, 40, 80]
    run_frames = [result for result in delivered["run"] if result.group == "frames"]
    for run_frame, phase_frame in zip(run_frames, frames):
        compare_outputs(
            host.output_value(run_frame, "predictions"),
            host.output_value(phase_frame, "predictions"),
        )


def test_weights_are_located_and_verified_without_download(tmp_path):
    with pytest.raises(assets.AssetError, match="prepare_assets.py"):
        assets.locate_weights(tmp_path)

    impostor = tmp_path / assets.RESNET18_WEIGHTS.name
    impostor.write_bytes(b"0" * assets.RESNET18_WEIGHTS.size)
    with pytest.raises(assets.AssetError, match="SHA-256"):
        assets.locate_weights(tmp_path)


def _local_weights_dir():
    configured = os.environ.get("RESNET18_WEIGHTS_DIR")
    directory = Path(configured) if configured else assets.default_weights_dir()
    try:
        assets.locate_weights(directory)
    except assets.AssetError:
        return None

    return directory


@pytest.mark.slow
def test_real_model_examples(tmp_path):
    weights_dir = _local_weights_dir()
    if weights_dir is None:
        pytest.skip("pinned ResNet-18 weights are not present locally")

    environment = dict(os.environ)
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        MPLCONFIGDIR=str(tmp_path / "matplotlib"),
        PYTHONPATH=os.pathsep.join(
            str(REPOSITORY_ROOT / part)
            for part in (".", "workflows", "inference_models", "stream_vision")
        ),
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(DEMO_DIR / "run_demo.py"),
            "--case",
            "all",
            "--weights-dir",
            str(weights_dir),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    evidence = json.loads((tmp_path / "evidence.json").read_text())
    assert evidence["assets"]["weights"]["sha256"] == assets.RESNET18_WEIGHTS.sha256
    beagle = evidence["classify"]["images"]["[1]"]["top"]
    assert [item["class_name"] for item in beagle[:2]] == ["basset", "beagle"]
    assert evidence["classify"]["crops"]["[1, 0]"]["top"][0]["class_name"] == "beagle"
    assert all(
        record["rule"] == "bitwise" and record["equal"]
        for record in evidence["classify"]["run_vs_phases"]["crops"]
    )
    assert [frame["top_class"] for frame in evidence["active"]["frames"]] == [
        "basset",
        "Norfolk terrier",
        "car wheel",
    ]
    assert evidence["mutation"]["top_class_after"] != "basset"
    assert (tmp_path / "gallery" / "index.html").is_file()
