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
from inference_models.models.owlv2.owlv2_custom_hf import (
    OWLv2CustomHF,
    OWLv2PreprocessingMetadata,
)
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
    model = OWLv2CustomHF.from_pretrained(str(package), device=torch.device("cpu"))
    rgb = torch.zeros((3, 16, 16), dtype=torch.uint8)
    rgb[0] = 255
    bgr = rgb.permute(1, 2, 0).numpy()[:, :, ::-1].copy()
    pixels, metadata = model.pre_process([bgr, bgr], input_color_format="bgr")
    tensor_pixels, _ = model.pre_process(rgb, input_color_format="rgb")
    torch.testing.assert_close(pixels[0], tensor_pixels[0])
    assert metadata[0].original_size == ImageDimensions(height=16, width=16)
    outputs = model.forward(pixels)
    assert outputs.logits.shape == (2, 4, 2)
    torch.testing.assert_close(outputs.logits[0], outputs.logits[1])
    single_outputs = model.forward(tensor_pixels)
    torch.testing.assert_close(outputs.logits[0], single_outputs.logits[0])
    assert model.class_names == ["label-a", "label-b"]


def test_per_class_thresholds_precede_suppression_and_limit():
    processor = SimpleNamespace(
        post_process_grounded_object_detection=lambda **kwargs: [
            {
                "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]]),
                "scores": torch.tensor([0.9, 0.8]),
                "labels": torch.tensor([0, 1]),
            }
        ]
    )
    wrapped = SimpleNamespace(
        _processor=processor,
        _model=SimpleNamespace(
            config=SimpleNamespace(vision_config=SimpleNamespace(image_size=16))
        ),
    )
    model = OWLv2CustomHF(
        wrapped,
        ["a", "b"],
        ["a", "b"],
        RecommendedParameters(
            confidence=0.5, per_class_confidence={"a": 0.95, "b": 0.5}
        ),
    )
    metadata = [OWLv2PreprocessingMetadata(ImageDimensions(height=16, width=16))]
    detections = model.post_process(
        None, metadata, confidence="best", class_agnostic_nms=True, max_detections=1
    )
    assert detections[0].class_id.tolist() == [1]
    assert detections[0].confidence.tolist() == pytest.approx([0.8])
