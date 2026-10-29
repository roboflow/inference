import json
from types import SimpleNamespace

import pytest
import torch
from transformers import (
    CLIPTokenizer,
    Owlv2Config,
    Owlv2ForObjectDetection,
    Owlv2ImageProcessor,
    Owlv2Processor,
)

from inference_models.entities import ImageDimensions
from inference_models.errors import CorruptedModelPackageError, ModelInputError
from inference_models.models.owlv2.owlv2_hf import OWLv2HF
from inference_models.weights_providers.entities import RecommendedParameters


@pytest.fixture
def package(tmp_path):
    config = Owlv2Config(
        text_config={
            "vocab_size": 4,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "max_position_embeddings": 16,
            "bos_token_id": 0,
            "eos_token_id": 1,
            "pad_token_id": 1,
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "image_size": 16,
            "patch_size": 8,
        },
        projection_dim=32,
    )
    model = Owlv2ForObjectDetection(config)
    model.save_pretrained(tmp_path)
    tokenizer = CLIPTokenizer(
        vocab={"<|startoftext|>": 0, "<|endoftext|>": 1, "a</w>": 2, "b</w>": 3},
        merges=[],
        model_max_length=16,
    )
    processor = Owlv2Processor(
        image_processor=Owlv2ImageProcessor(size={"height": 16, "width": 16}),
        tokenizer=tokenizer,
    )
    processor.save_pretrained(tmp_path)
    (tmp_path / "class_names.txt").write_text("label-a\nlabel-b\n")
    (tmp_path / "owlv2_prompts.json").write_text(json.dumps(["a", "b"]))
    return tmp_path


def test_all_images_receive_all_prompts_and_preserve_color(package):
    model = OWLv2HF.from_pretrained(str(package), device=torch.device("cpu"))
    rgb = torch.zeros((3, 16, 16), dtype=torch.uint8)
    rgb[0] = 255
    bgr = rgb.permute(1, 2, 0).numpy()[:, :, ::-1].copy()
    pixels, metadata = model.pre_process([bgr, bgr], input_color_format="bgr")
    tensor_pixels, _ = model.pre_process(rgb, input_color_format="rgb")
    torch.testing.assert_close(pixels[0], tensor_pixels[0])
    assert metadata[0] == ImageDimensions(height=16, width=16)
    outputs = model.forward(pixels, classes=["a", "b"])
    assert outputs.logits.shape == (2, 4, 2)
    torch.testing.assert_close(outputs.logits[0], outputs.logits[1])
    single_outputs = model.forward(tensor_pixels, classes=["a", "b"])
    torch.testing.assert_close(outputs.logits[0], single_outputs.logits[0])
    implicit = model.infer(rgb, confidence=0.0)
    explicit = model.infer(rgb, classes=["a", "b"], confidence=0.0)
    torch.testing.assert_close(implicit[0].xyxy, explicit[0].xyxy)
    torch.testing.assert_close(implicit[0].confidence, explicit[0].confidence)
    assert implicit[0].image_metadata["class_names"] == ["label-a", "label-b"]
    assert explicit[0].image_metadata["class_names"] == ["a", "b"]


def test_per_class_thresholds_precede_suppression_and_limit(package):
    outputs = SimpleNamespace(
        logits=torch.logit(torch.tensor([[[0.9, 0.1], [0.1, 0.8]]])),
        pred_boxes=torch.tensor([[[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]]),
    )
    model = OWLv2HF.from_pretrained(
        str(package),
        device=torch.device("cpu"),
        recommended_parameters=RecommendedParameters(
            confidence=0.5, per_class_confidence={"label-a": 0.95, "label-b": 0.5}
        ),
    )
    metadata = [ImageDimensions(height=16, width=16)]
    detections = model.post_process(
        outputs,
        metadata,
        confidence="best",
        class_names=["label-a", "label-b"],
        class_agnostic_nms=True,
        max_detections=1,
    )
    assert detections[0].class_id.tolist() == [1]
    assert detections[0].confidence.tolist() == pytest.approx([0.8])


@pytest.mark.parametrize("confidence", [0.5, "best"])
def test_confidence_filter_excludes_exact_threshold_with_real_processor(
    package, confidence
):
    model = OWLv2HF.from_pretrained(
        str(package),
        device=torch.device("cpu"),
        recommended_parameters=RecommendedParameters(
            confidence=0.5, per_class_confidence={"label-a": 0.5, "label-b": 0.25}
        ),
    )
    outputs = SimpleNamespace(
        logits=torch.tensor([[[0.0, -1.0], [0.5, -1.0], [-1.0, -0.5]]]),
        pred_boxes=torch.tensor(
            [[[0.2, 0.2, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]]]
        ),
    )
    detections = model.post_process(
        outputs,
        [ImageDimensions(height=100, width=100)],
        confidence=confidence,
        class_names=["label-a", "label-b"],
    )[0]
    if confidence == "best":
        assert detections.class_id.tolist() == [0, 1]
    else:
        assert detections.class_id.tolist() == [0]
    assert detections.confidence[0].item() == pytest.approx(
        torch.sigmoid(torch.tensor(0.5)).item()
    )


def test_package_without_saved_prompts_keeps_explicit_queries(package):
    (package / "class_names.txt").unlink()
    (package / "owlv2_prompts.json").unlink()
    model = OWLv2HF.from_pretrained(str(package), device=torch.device("cpu"))
    image = torch.zeros((3, 16, 16), dtype=torch.uint8)
    predictions = model(image, ["a", "b"], confidence=0.0)
    assert len(predictions) == 1
    assert predictions[0].image_metadata is None
    with pytest.raises(ModelInputError, match="Provide classes"):
        model.infer(image)


@pytest.mark.parametrize(
    "filename,contents",
    [
        ("class_names.txt", "duplicate\nduplicate\n"),
        ("owlv2_prompts.json", '["one prompt"]'),
        ("owlv2_prompts.json", None),
        ("class_names.txt", None),
    ],
)
def test_invalid_saved_metadata_does_not_fall_back_to_open_vocabulary(
    package, filename, contents
):
    if contents is None:
        (package / filename).unlink()
    else:
        (package / filename).write_text(contents)
    with pytest.raises(CorruptedModelPackageError):
        OWLv2HF.from_pretrained(str(package), device=torch.device("cpu"))
