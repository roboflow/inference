from unittest.mock import MagicMock


def test_trocr_uses_explicit_hugging_face_cache_path(monkeypatch) -> None:
    from inference.models.trocr import trocr

    expected_cache_path = "/mounted/inference-cache/hf_home/hub"
    monkeypatch.setattr(trocr, "HF_HUB_CACHE", expected_cache_path)

    model = MagicMock()
    model.eval.return_value = model
    model.to.return_value = model
    model_loader = MagicMock(return_value=model)
    processor_loader = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(
        trocr.VisionEncoderDecoderModel,
        "from_pretrained",
        model_loader,
    )
    monkeypatch.setattr(
        trocr.TrOCRProcessor,
        "from_pretrained",
        processor_loader,
    )

    trocr.TrOCR(model_id="trocr/trocr-small-printed")

    assert model_loader.call_args.kwargs["cache_dir"] == expected_cache_path
    assert processor_loader.call_args.kwargs["cache_dir"] == expected_cache_path


def test_owlv2_singleton_uses_explicit_hugging_face_cache_path(
    monkeypatch,
) -> None:
    from inference.models.owlv2 import owlv2

    expected_cache_path = "/mounted/inference-cache/hf_home/hub"
    monkeypatch.setattr(owlv2, "HF_HUB_CACHE", expected_cache_path)
    # The singleton is asked about the cache path, not compilation: torch>=2.14
    # validates what torch.compile is handed and rejects the MagicMock model.
    monkeypatch.setattr(owlv2, "OWLV2_COMPILE_MODEL", False)
    owlv2.Owlv2Singleton._instances.clear()

    model = MagicMock()
    model.eval.return_value = model
    model.to.return_value = model
    model_loader = MagicMock(return_value=model)
    monkeypatch.setattr(
        owlv2.Owlv2ForObjectDetection,
        "from_pretrained",
        model_loader,
    )

    singleton = owlv2.Owlv2Singleton("google/owlv2-base-patch16-ensemble")

    assert singleton.model is model
    assert model_loader.call_args.kwargs["cache_dir"] == expected_cache_path
    owlv2.Owlv2Singleton._instances.clear()
