import ast
import os
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np


def _literal_artifact_list(
    source_path: Path,
    class_name: str,
) -> list[str]:
    syntax_tree = ast.parse(source_path.read_text())
    class_definition = next(
        node
        for node in syntax_tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in class_definition.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_infer_bucket_file_list"
    )
    return_statement = next(
        node for node in ast.walk(method) if isinstance(node, ast.Return)
    )
    return ast.literal_eval(return_statement.value)


def test_easyocr_stages_runtime_aliases_outside_model_cache(
    tmp_path,
    monkeypatch,
) -> None:
    from inference.models.easy_ocr import easy_ocr

    model_id = "easy_ocr/english_g2"
    cache_dir = tmp_path / "easy_ocr" / "english_g2"
    cache_dir.mkdir(parents=True)
    recognition_weights = cache_dir / "weights.pt"
    detector_weights = cache_dir / "craft_mlt_25k.pth"
    recognition_weights.write_bytes(b"recognition")
    detector_weights.write_bytes(b"detector")
    original_entries = {path.name: path.read_bytes() for path in cache_dir.iterdir()}

    reader = MagicMock()
    reader.readtext.return_value = []

    def reader_factory(*args, **kwargs):
        runtime_dir = Path(kwargs["model_storage_directory"])
        assert runtime_dir == Path(kwargs["user_network_directory"])
        assert os.path.commonpath([str(runtime_dir), str(cache_dir)]) != str(cache_dir)
        assert (runtime_dir / "english_g2.pth").read_bytes() == b"recognition"
        assert (runtime_dir / "craft_mlt_25k.pth").read_bytes() == b"detector"
        return reader

    monkeypatch.setattr(easy_ocr.easyocr, "Reader", reader_factory)
    model = object.__new__(easy_ocr.EasyOCR)
    model.endpoint = model_id
    model.cache_dir = str(cache_dir)
    model.recognizer = "english_g2"

    assert model.predict(np.zeros((4, 4, 3), dtype=np.uint8)) == []
    assert {
        path.name: path.read_bytes() for path in cache_dir.iterdir()
    } == original_entries


def test_doctr_loads_source_weights_without_copying_into_shared_cache(
    tmp_path,
    monkeypatch,
) -> None:
    from inference.models.doctr import doctr_model

    detector_version = "db_resnet50_v2"
    recognizer_version = "crnn_vgg16_bn_v2"
    detector_weights = tmp_path / "doctr_det" / detector_version / "model.pt"
    recognizer_weights = tmp_path / "doctr_rec" / recognizer_version / "model.pt"
    detector_weights.parent.mkdir(parents=True)
    recognizer_weights.parent.mkdir(parents=True)
    detector_weights.write_bytes(b"detector")
    recognizer_weights.write_bytes(b"recognizer")

    detector_metadata = MagicMock(version_id=detector_version)
    detector_metadata.cache_file.return_value = str(detector_weights)
    recognizer_metadata = MagicMock(version_id=recognizer_version)
    recognizer_metadata.cache_file.return_value = str(recognizer_weights)
    detector_model = MagicMock()
    recognizer_model = MagicMock()
    torch_load = MagicMock(side_effect=[{"detector": 1}, {"recognizer": 2}])

    monkeypatch.setattr(
        doctr_model,
        "DocTRDet",
        MagicMock(return_value=detector_metadata),
    )
    monkeypatch.setattr(
        doctr_model,
        "DocTRRec",
        MagicMock(return_value=recognizer_metadata),
    )
    monkeypatch.setattr(
        doctr_model,
        "db_resnet50",
        MagicMock(return_value=detector_model),
    )
    monkeypatch.setattr(
        doctr_model,
        "crnn_vgg16_bn",
        MagicMock(return_value=recognizer_model),
    )
    monkeypatch.setattr(doctr_model.torch, "load", torch_load)
    monkeypatch.setattr(doctr_model, "ocr_predictor", MagicMock())

    doctr_model.DocTR()

    assert torch_load.call_args_list == [
        call(str(detector_weights), map_location=doctr_model.DEVICE, weights_only=True),
        call(
            str(recognizer_weights),
            map_location=doctr_model.DEVICE,
            weights_only=True,
        ),
    ]
    detector_metadata.cache_file.assert_called_once_with("model.pt")
    recognizer_metadata.cache_file.assert_called_once_with("model.pt")
    assert not (tmp_path / "doctr" / "models").exists()


def test_grounding_dino_resolves_bert_to_explicit_local_snapshot(
    tmp_path,
    monkeypatch,
) -> None:
    from inference.models.grounding_dino import grounding_dino

    model_cache_dir = tmp_path / "grounding_dino"
    model_cache_dir.mkdir()
    text_encoder_dir = tmp_path / "hf_home" / "hub" / "bert-snapshot"
    loaded_model = MagicMock()
    loaded_model.to.return_value = loaded_model
    snapshot_download = MagicMock(return_value=str(text_encoder_dir))
    load_model = MagicMock(return_value=loaded_model)

    def initialise_cached_model(instance, *args, **kwargs):
        instance.cache_dir = str(model_cache_dir)

    monkeypatch.setattr(
        grounding_dino.RoboflowCoreModel,
        "__init__",
        initialise_cached_model,
    )
    monkeypatch.setattr(
        grounding_dino,
        "HF_HUB_CACHE",
        str(tmp_path / "hf_home" / "hub"),
    )
    monkeypatch.setattr(grounding_dino, "OFFLINE_MODE", True)
    monkeypatch.setattr(
        grounding_dino,
        "snapshot_download",
        snapshot_download,
    )
    monkeypatch.setattr(grounding_dino, "load_model", load_model)
    monkeypatch.setattr(
        grounding_dino.torch.cuda,
        "is_available",
        MagicMock(return_value=False),
    )

    model = grounding_dino.GroundingDINO()

    snapshot_download.assert_called_once_with(
        repo_id="google-bert/bert-base-uncased",
        cache_dir=str(tmp_path / "hf_home" / "hub"),
        local_files_only=True,
        allow_patterns=grounding_dino.BERT_SNAPSHOT_ALLOW_PATTERNS,
    )
    assert load_model.call_args.kwargs["text_encoder_type"] == str(text_encoder_dir)
    assert load_model.call_args.kwargs["device"] == "cpu"
    assert load_model.call_args.kwargs["model_checkpoint_path"] == str(
        model_cache_dir / "groundingdino_swint_ogc.pth"
    )
    assert model.model.model is loaded_model
    assert model.model.device == "cpu"


def test_grounding_dino_uses_legacy_bert_cache_as_offline_fallback(
    tmp_path,
    monkeypatch,
) -> None:
    from huggingface_hub.errors import LocalEntryNotFoundError

    from inference.models.grounding_dino import grounding_dino

    legacy_snapshot = tmp_path / "legacy-bert-snapshot"
    snapshot_download = MagicMock(
        side_effect=[
            LocalEntryNotFoundError("canonical snapshot is not cached"),
            str(legacy_snapshot),
        ]
    )
    monkeypatch.setattr(
        grounding_dino,
        "HF_HUB_CACHE",
        str(tmp_path / "hf_home" / "hub"),
    )
    monkeypatch.setattr(grounding_dino, "OFFLINE_MODE", True)
    monkeypatch.setattr(
        grounding_dino,
        "snapshot_download",
        snapshot_download,
    )

    assert grounding_dino._download_bert_snapshot() == str(legacy_snapshot)
    expected_kwargs = {
        "cache_dir": str(tmp_path / "hf_home" / "hub"),
        "local_files_only": True,
        "allow_patterns": grounding_dino.BERT_SNAPSHOT_ALLOW_PATTERNS,
    }
    assert snapshot_download.call_args_list == [
        call(repo_id="google-bert/bert-base-uncased", **expected_kwargs),
        call(repo_id="bert-base-uncased", **expected_kwargs),
    ]


def test_interactive_sam3_declares_bpe_as_required_cache_artifact() -> None:
    repository_root = Path(__file__).parents[4]
    artifacts = _literal_artifact_list(
        source_path=repository_root
        / "inference"
        / "models"
        / "sam3"
        / "visual_segmentation.py",
        class_name="Sam3ForInteractiveImageSegmentation",
    )

    assert "weights.pt" in artifacts
    assert "bpe_simple_vocab_16e6.txt.gz" in artifacts


def test_sam3_3d_declares_dinov2_checkpoint_as_required_cache_artifact() -> None:
    repository_root = Path(__file__).parents[4]
    artifacts = _literal_artifact_list(
        source_path=repository_root
        / "inference"
        / "models"
        / "sam3_3d"
        / "segment_anything_3d.py",
        class_name="SegmentAnything3_3D_Objects",
    )

    assert "dinov2_vitl14_reg4_pretrain.pth" in artifacts
