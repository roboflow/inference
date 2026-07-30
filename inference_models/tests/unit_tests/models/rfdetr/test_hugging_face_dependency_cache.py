from unittest.mock import MagicMock, patch

from inference_models.models.rfdetr import rfdetr_backbone_pytorch


def test_non_windowed_backbone_uses_explicit_hugging_face_cache() -> None:
    expected_cache_path = "/mounted/inference-cache/hf_home/hub"
    encoder = MagicMock()

    with (
        patch.object(
            rfdetr_backbone_pytorch,
            "HF_HUB_CACHE",
            expected_cache_path,
        ),
        patch.object(
            rfdetr_backbone_pytorch.AutoBackbone,
            "from_pretrained",
            return_value=encoder,
        ) as model_loader,
    ):
        model = rfdetr_backbone_pytorch.DinoV2(
            use_windowed_attn=False,
            gradient_checkpointing=False,
        )

    assert model.encoder is encoder
    assert model_loader.call_args.kwargs["cache_dir"] == expected_cache_path
    assert (
        model_loader.call_args.kwargs["local_files_only"]
        is rfdetr_backbone_pytorch.OFFLINE_MODE
    )


def test_windowed_backbone_uses_explicit_hugging_face_cache() -> None:
    expected_cache_path = "/mounted/inference-cache/hf_home/hub"
    encoder = MagicMock()

    with (
        patch.object(
            rfdetr_backbone_pytorch,
            "HF_HUB_CACHE",
            expected_cache_path,
        ),
        patch.object(
            rfdetr_backbone_pytorch.WindowedDinov2WithRegistersBackbone,
            "from_pretrained",
            return_value=encoder,
        ) as model_loader,
        patch.object(
            rfdetr_backbone_pytorch,
            "WindowedDinov2WithRegistersConfig",
            return_value=MagicMock(),
        ),
    ):
        model = rfdetr_backbone_pytorch.DinoV2()

    assert model.encoder is encoder
    assert model_loader.call_args.kwargs["cache_dir"] == expected_cache_path
    assert (
        model_loader.call_args.kwargs["local_files_only"]
        is rfdetr_backbone_pytorch.OFFLINE_MODE
    )
