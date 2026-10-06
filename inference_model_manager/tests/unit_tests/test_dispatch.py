"""Unit tests for dispatch module — action resolution and listing."""

from inference_model_manager.dispatch import list_actions_by_mro_names
from inference_model_manager.registry_defaults import lazy_register_by_names


def test_list_actions_by_mro_names_object_detection():
    """ObjectDetectionModel MRO names should return 'infer' action."""
    lazy_register_by_names(["ObjectDetectionModel"])
    actions = list_actions_by_mro_names(["ObjectDetectionModel"])
    assert "infer" in actions
    assert actions["infer"]["default"] is True
    assert "images" in actions["infer"]["params"]


def test_list_actions_by_mro_names_unknown_class():
    """Unknown class names should return empty dict."""
    actions = list_actions_by_mro_names(["CompletelyUnknownModel"])
    assert actions == {}


def test_list_actions_by_mro_names_walks_mro():
    """Should match on any ancestor in the MRO list."""
    lazy_register_by_names(["ObjectDetectionModel"])
    actions = list_actions_by_mro_names(
        [
            "YOLOv8ForObjectDetectionTorchScript",
            "ObjectDetectionModel",
            "object",
        ]
    )
    assert "infer" in actions


def test_invoke_action_applies_param_aliases():
    from inference_model_manager.dispatch import invoke_action
    from inference_model_manager.registry_defaults import registry

    class AliasedModel:
        def ask(self, question):
            return f"answer:{question}"

    registry.register(
        AliasedModel,
        "query",
        method="ask",
        default=True,
        params={"prompt": {"type": "str", "required": True}},
        validator=lambda kw: kw,
        serializer=lambda out, m: {"text": out},
        response_type="roboflow-text-v1",
        param_aliases={"prompt": "question"},
    )
    result = invoke_action(AliasedModel(), action="query", prompt="hi")
    assert result == "answer:hi"


def _register_capturing_model(declare_flags=True):
    from inference_model_manager.registry_defaults import (
        PRE_PROCESSING_OVERRIDE_FIELDS,
        registry,
    )

    params = {"images": {"type": "image", "required": True}}
    if declare_flags:
        params.update(
            {
                flag: {"type": "bool", "required": False}
                for flag in PRE_PROCESSING_OVERRIDE_FIELDS
            }
        )

    class CapturingModel:
        def __init__(self):
            self.received = None

        def infer(self, images, **kwargs):
            self.received = kwargs
            return []

    registry.register(
        CapturingModel,
        "infer",
        default=True,
        params=params,
        validator=lambda kw: kw,
        serializer=lambda out, m: {"predictions": out},
        response_type="roboflow-object-detection-compact-v1",
    )
    return CapturingModel()


def test_invoke_action_builds_pre_processing_overrides_from_legacy_flags():
    from inference_model_manager.dispatch import invoke_action
    from inference_models.models.auto_loaders.entities import PreProcessingOverrides

    model = _register_capturing_model()
    invoke_action(
        model,
        action="infer",
        images=["img"],
        confidence=0.5,
        disable_preproc_contrast=True,
        disable_preproc_static_crop=True,
    )

    assert model.received == {
        "confidence": 0.5,
        "pre_processing_overrides": PreProcessingOverrides(
            disable_contrast_enhancement=True,
            disable_grayscale=False,
            disable_static_crop=True,
        ),
    }


def test_invoke_action_keeps_explicit_overrides_when_flag_is_false():
    from inference_model_manager.dispatch import invoke_action
    from inference_models.models.auto_loaders.entities import PreProcessingOverrides

    model = _register_capturing_model()
    invoke_action(
        model,
        action="infer",
        images=["img"],
        pre_processing_overrides=PreProcessingOverrides(disable_static_crop=True),
        disable_preproc_contrast=False,
    )

    assert model.received == {
        "pre_processing_overrides": PreProcessingOverrides(
            disable_contrast_enhancement=False,
            disable_grayscale=False,
            disable_static_crop=True,
        ),
    }


def test_invoke_action_merges_true_flag_into_explicit_overrides():
    from inference_model_manager.dispatch import invoke_action
    from inference_models.models.auto_loaders.entities import PreProcessingOverrides

    model = _register_capturing_model()
    invoke_action(
        model,
        action="infer",
        images=["img"],
        pre_processing_overrides=PreProcessingOverrides(disable_static_crop=True),
        disable_preproc_contrast=True,
    )

    assert model.received == {
        "pre_processing_overrides": PreProcessingOverrides(
            disable_contrast_enhancement=True,
            disable_grayscale=False,
            disable_static_crop=True,
        ),
    }


def test_invoke_action_drops_legacy_flags_for_undeclared_entry():
    from inference_model_manager.dispatch import invoke_action

    model = _register_capturing_model(declare_flags=False)
    invoke_action(
        model,
        action="infer",
        images=["img"],
        confidence=0.5,
        disable_preproc_contrast=True,
        disable_preproc_static_crop=True,
    )

    assert model.received == {"confidence": 0.5}


def test_invoke_action_without_legacy_flags_passes_no_overrides():
    from inference_model_manager.dispatch import invoke_action

    model = _register_capturing_model()
    invoke_action(model, action="infer", images=["img"], confidence=0.5)

    assert model.received == {"confidence": 0.5}


def _register_sam_segment_model(embeddings_type):
    from inference_model_manager.registry_defaults import registry

    class SegmentingModel:
        def __init__(self):
            self.received = None

        def segment_images(self, **kwargs):
            self.received = kwargs
            return []

    registry.register(
        SegmentingModel,
        "segment",
        method="segment_images",
        default=True,
        params={
            "images": {"type": "image", "required": False},
            "embeddings": {"type": embeddings_type, "required": False},
        },
        validator=lambda kw: kw,
        serializer=lambda out, m: {"predictions": out},
        response_type="roboflow-sam-segmentation-compact-v1",
    )
    return SegmentingModel()


def _wire_embeddings(image_hash=None, image_size_hw=None):
    import numpy as np

    return {
        "embeddings": np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2),
        "image_hash": image_hash,
        "image_size_hw": image_size_hw,
    }


def test_invoke_action_builds_sam_image_embeddings_from_wire_dict():
    import numpy as np
    import torch

    from inference_model_manager.dispatch import invoke_action
    from inference_model_manager.registry_defaults import SAM_IMAGE_EMBEDDINGS_TYPE
    from inference_models.models.sam.entities import SAMImageEmbeddings

    model = _register_sam_segment_model(SAM_IMAGE_EMBEDDINGS_TYPE)
    wire = _wire_embeddings(image_hash="tenant:img-1", image_size_hw=[480, 640])
    invoke_action(model, action="segment", embeddings=wire, multi_mask_output=False)

    entity = model.received["embeddings"]
    assert isinstance(entity, SAMImageEmbeddings)
    assert entity.image_hash == "tenant:img-1"
    assert entity.image_size_hw == (480, 640)
    assert isinstance(entity.embeddings, torch.Tensor)
    np.testing.assert_array_equal(entity.embeddings.numpy(), wire["embeddings"])
    assert model.received["multi_mask_output"] is False
    assert "images" not in model.received


def test_invoke_action_hashes_and_sizes_sam_embeddings_from_the_image():
    import numpy as np

    from inference_model_manager.dispatch import invoke_action
    from inference_model_manager.registry_defaults import SAM_IMAGE_EMBEDDINGS_TYPE
    from inference_models.models.sam.sam_torch import compute_image_hash

    model = _register_sam_segment_model(SAM_IMAGE_EMBEDDINGS_TYPE)
    image = np.random.default_rng(0).integers(0, 255, (6, 8, 3), dtype=np.uint8)
    invoke_action(model, action="segment", images=image, embeddings=_wire_embeddings())

    entity = model.received["embeddings"]
    assert entity.image_hash == compute_image_hash(image=image)
    assert entity.image_size_hw == (6, 8)
    assert model.received["images"] is image


def test_invoke_action_converts_a_batch_of_sam_embeddings():
    import numpy as np

    from inference_model_manager.dispatch import invoke_action
    from inference_model_manager.registry_defaults import SAM_IMAGE_EMBEDDINGS_TYPE
    from inference_models.models.sam.sam_torch import compute_image_hash

    model = _register_sam_segment_model(SAM_IMAGE_EMBEDDINGS_TYPE)
    images = [
        np.zeros((2, 3, 3), dtype=np.uint8),
        np.full((4, 5, 3), 7, dtype=np.uint8),
    ]
    invoke_action(
        model,
        action="segment",
        images=images,
        embeddings=[_wire_embeddings(image_hash="h-0"), _wire_embeddings()],
    )

    first, second = model.received["embeddings"]
    assert first.image_hash == "h-0" and first.image_size_hw == (2, 3)
    assert second.image_hash == compute_image_hash(image=images[1])
    assert second.image_size_hw == (4, 5)


def _malformed_sam_wires():
    import numpy as np

    good = _wire_embeddings(image_hash="h", image_size_hw=[2, 2])
    return [
        "not-a-dict",
        {key: good[key] for key in ("embeddings", "image_hash")},
        {**good, "embeddings": np.array([["a", "b"]])},
        {**good, "embeddings": np.array([object()], dtype=object)},
        {**good, "embeddings": [1.0, 2.0]},
        {**good, "image_hash": 5},
        {**good, "image_size_hw": [2]},
        {**good, "image_size_hw": [2, 2, 2]},
        {**good, "image_size_hw": [0, 2]},
        {**good, "image_size_hw": [2, -3]},
        {**good, "image_size_hw": [2.5, 2]},
        {**good, "image_size_hw": 4},
    ]


def test_malformed_sam_wire_embeddings_raise_value_error():
    import pytest

    from inference_model_manager.dispatch import _sam_image_embeddings_from_wire

    for wire in _malformed_sam_wires():
        with pytest.raises(ValueError):
            _sam_image_embeddings_from_wire(wire, None)


def test_a_well_formed_sam_wire_still_converts():
    from inference_model_manager.dispatch import _sam_image_embeddings_from_wire

    wire = _wire_embeddings(image_hash="h", image_size_hw=(2, 3))
    entity = _sam_image_embeddings_from_wire(wire, None)
    assert entity.image_hash == "h"
    assert entity.image_size_hw == (2, 3)


def test_invoke_action_requires_an_image_hash_when_no_image_is_sent():
    import pytest

    from inference_model_manager.dispatch import invoke_action
    from inference_model_manager.registry_defaults import SAM_IMAGE_EMBEDDINGS_TYPE

    model = _register_sam_segment_model(SAM_IMAGE_EMBEDDINGS_TYPE)
    with pytest.raises(
        ValueError, match="image_id is required when image not provided"
    ):
        invoke_action(
            model, action="segment", embeddings=_wire_embeddings(image_size_hw=[2, 2])
        )


def test_invoke_action_requires_an_image_size_when_no_image_is_sent():
    import pytest

    from inference_model_manager.dispatch import invoke_action
    from inference_model_manager.registry_defaults import SAM_IMAGE_EMBEDDINGS_TYPE

    model = _register_sam_segment_model(SAM_IMAGE_EMBEDDINGS_TYPE)
    with pytest.raises(
        ValueError,
        match="orig_im_size is required when image not provided and embeddings are "
        "injected by client.",
    ):
        invoke_action(model, action="segment", embeddings=_wire_embeddings("h-1"))


def test_invoke_action_leaves_tensor_typed_embeddings_untouched():
    from inference_model_manager.dispatch import invoke_action

    model = _register_sam_segment_model("tensor")
    wire = _wire_embeddings(image_hash="h-1", image_size_hw=[2, 2])
    invoke_action(model, action="segment", embeddings=wire)

    assert model.received["embeddings"] is wire


def test_invoke_action_passes_prebuilt_sam_embeddings_through():
    import torch

    from inference_model_manager.dispatch import invoke_action
    from inference_model_manager.registry_defaults import SAM_IMAGE_EMBEDDINGS_TYPE
    from inference_models.models.sam.entities import SAMImageEmbeddings

    model = _register_sam_segment_model(SAM_IMAGE_EMBEDDINGS_TYPE)
    entity = SAMImageEmbeddings(
        image_hash="h-1", image_size_hw=(2, 2), embeddings=torch.zeros(1, 2, 2, 2)
    )
    invoke_action(model, action="segment", embeddings=entity)

    assert model.received["embeddings"] is entity


def test_sam_segment_declares_the_sam_image_embeddings_type_only_for_sam1():
    from inference_model_manager.registry_defaults import (
        _ACTION_CONFIGS,
        SAM_IMAGE_EMBEDDINGS_TYPE,
        _unpack_config,
    )

    sam = {c[0]: _unpack_config(c) for c in _ACTION_CONFIGS["SAMTorch"]}
    sam2 = {c[0]: _unpack_config(c) for c in _ACTION_CONFIGS["SAM2Torch"]}
    sam3 = {c[0]: _unpack_config(c) for c in _ACTION_CONFIGS["SAM3Torch"]}
    assert sam["segment"][3]["embeddings"]["type"] == SAM_IMAGE_EMBEDDINGS_TYPE
    assert sam2["segment"][3]["embeddings"]["type"] == "tensor"
    assert sam3["segment_with_visual_prompts"][3]["embeddings"]["type"] == "tensor"


def test_unpack_config_handles_7_and_8_tuples():
    from inference_model_manager.registry_defaults import _unpack_config

    seven = ("t", "m", True, {}, "v", "s", "r")
    assert _unpack_config(seven)[7] == {}
    eight = ("t", "m", True, {}, "v", "s", "r", {"a": "b"})
    assert _unpack_config(eight)[7] == {"a": "b"}
