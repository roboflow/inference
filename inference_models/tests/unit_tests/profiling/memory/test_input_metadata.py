from __future__ import annotations

from profiling.memory.metadata import build_input_metadata_with_registry


def test_non_image_inputs_marked_profiled_from_infer_kwargs() -> None:
    metadata = build_input_metadata_with_registry(
        module_name="inference_models.models.paligemma.paligemma_hf",
        class_name="PaliGemmaHF",
        architecture="paligemma",
        task_type="vlm",
        backend="torch",
        batch_size=1,
        height=640,
        width=640,
        infer_kwargs={
            "prompt": "Describe this image.",
            "max_new_tokens": 128,
        },
    )

    assert "prompt" in metadata.inputs
    assert metadata.inputs["prompt"]["prompt"].value == "Describe this image."
    prompt_declared = next(
        item for item in metadata.declared_inputs if item.name == "prompt"
    )
    assert prompt_declared.profiled is True

    max_tokens_declared = next(
        item
        for item in metadata.declared_inputs
        if item.name == "max_new_tokens"
    )
    assert max_tokens_declared.profiled is True
    assert max_tokens_declared.axes["max_new_tokens"].value == 128


def test_open_vocabulary_classes_profiled_from_infer_kwargs() -> None:
    metadata = build_input_metadata_with_registry(
        module_name="inference_models.models.grounding_dino.grounding_dino_torch",
        class_name="GroundingDinoForObjectDetectionTorch",
        architecture="grounding-dino",
        task_type="open-vocabulary-object-detection",
        backend="torch",
        batch_size=1,
        height=640,
        width=640,
        infer_kwargs={"classes": ["cat", "dog"]},
    )

    classes_declared = next(
        item for item in metadata.declared_inputs if item.name == "classes"
    )
    assert classes_declared.profiled is True
    assert classes_declared.axes["num_classes"].value == 2
    assert "classes" in metadata.inputs
