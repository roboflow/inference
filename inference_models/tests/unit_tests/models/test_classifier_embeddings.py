import pytest
import timm
import torch
from transformers import ViTConfig, ViTModel

from inference_models.models.dinov3.dinov3_classification_torch import (
    DinoV3ForClassificationTorch,
    DinoV3ForMultiLabelClassificationTorch,
    DinoV3Model,
)
from inference_models.models.resnet.resnet_classification_torch import (
    ResNetClassifier,
    ResNetForClassificationTorch,
    ResNetForMultiLabelClassificationTorch,
    ResNetMultiLabelClassifier,
)
from inference_models.models.vit.vit_classification_huggingface import (
    VITClassifier,
    VITForClassificationHF,
    VITForMultiLabelClassificationHF,
    VITMultiLabelClassifier,
)


def check_features(model, head, model_class, inputs):
    expected = []
    expected_logits = []
    handle = head.register_forward_pre_hook(
        lambda module, args: expected.append(args[0].clone())
    )
    logits_handle = head.register_forward_hook(
        lambda module, args, output: expected_logits.append(output.clone())
    )
    classifier = model_class(
        model=model.eval(),
        inference_config=None,
        class_names=["a", "b", "c"],
        device=torch.device("cpu"),
    )
    scores = classifier.forward(inputs)
    handle.remove()
    logits_handle.remove()
    embeddings = classifier.forward_embedding(inputs)
    torch.testing.assert_close(embeddings, expected[0], rtol=0, atol=0)
    logits = classifier.forward_embedding(inputs, output_type="logits")
    torch.testing.assert_close(logits, expected_logits[0], rtol=0, atol=0)
    assert logits.shape == (2, 3)
    assert (
        classifier.get_embedding_info("logits")["feature_definition"]
        != classifier.embedding_info["feature_definition"]
    )
    classifier.prepare_image_embeddings(output_type="logits")
    torch.testing.assert_close(
        classifier.forward_embedding(inputs), logits, rtol=0, atol=0
    )
    classifier.prepare_image_embeddings(output_type="feature_vector")
    torch.testing.assert_close(classifier.forward(inputs), scores, rtol=0, atol=0)
    assert embeddings.shape[0] == 2
    assert classifier.embedding_info["normalization"] == "none"
    assert not torch.allclose(embeddings.norm(dim=1), torch.ones(2))


@pytest.mark.parametrize("fused", [False, True])
def test_resnet_single_and_multi_label_features_are_exact_head_inputs(fused):
    for wrapper, model_class, flag in [
        (ResNetClassifier, ResNetForClassificationTorch, {"softmax_fused": fused}),
        (
            ResNetMultiLabelClassifier,
            ResNetForMultiLabelClassificationTorch,
            {"sigmoid_fused": fused},
        ),
    ]:
        backbone = timm.create_model("resnet18", pretrained=False, num_classes=3)
        model = wrapper(backbone=backbone, **flag)
        check_features(model, backbone.fc, model_class, torch.randn(2, 3, 32, 32))


@pytest.mark.parametrize("fused", [False, True])
def test_vit_single_and_multi_label_features_include_trained_layer_norm(fused):
    for wrapper, model_class, flag in [
        (VITClassifier, VITForClassificationHF, {"softmax_fused": fused}),
        (
            VITMultiLabelClassifier,
            VITForMultiLabelClassificationHF,
            {"sigmoid_fused": fused},
        ),
    ]:
        backbone = ViTModel(
            ViTConfig(
                image_size=16,
                patch_size=8,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
            )
        )
        head = torch.nn.Linear(32, 3)
        model = wrapper(backbone=backbone, classifier=head, **flag)
        check_features(model, head, model_class, torch.randn(2, 3, 16, 16))


def test_dinov3_single_and_multi_label_features_are_probe_inputs():
    model = DinoV3Model(num_classes=3)
    inputs = torch.randn(2, 3, 224, 224)
    for model_class in [
        DinoV3ForClassificationTorch,
        DinoV3ForMultiLabelClassificationTorch,
    ]:
        check_features(model, model.linear_layer, model_class, inputs)
