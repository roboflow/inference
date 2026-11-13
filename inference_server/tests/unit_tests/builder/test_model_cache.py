import json

import pytest

from inference_models.models.auto_loaders.model_cache_paths import (
    slugify_model_id_to_os_safe_format,
)
from inference_server import configuration
from inference_server.builder import model_cache


@pytest.fixture
def roots(tmp_path, monkeypatch):
    server_root = tmp_path / "server_cache"
    models_home = tmp_path / "models_home"
    server_root.mkdir()
    models_home.mkdir()
    monkeypatch.setattr(configuration, "MODEL_CACHE_DIR", str(server_root))
    monkeypatch.setattr(configuration, "INFERENCE_HOME", str(models_home))
    monkeypatch.setattr(configuration, "LEGACY_OFFLINE_MODE", False)
    return server_root, models_home


def _write_traditional(root, model_id, task_type, architecture, with_model_id=True):
    model_dir = root / model_id
    model_dir.mkdir(parents=True)
    metadata = {"project_task_type": task_type, "model_type": architecture}
    if with_model_id:
        metadata["model_id"] = model_id
    (model_dir / "model_type.json").write_text(json.dumps(metadata))
    (model_dir / "weights.onnx").write_bytes(b"x")
    return model_dir


def _write_inference_models(root, model_id, task_type, architecture):
    package_dir = (
        root
        / "models-cache"
        / slugify_model_id_to_os_safe_format(model_id=model_id)
        / "pkga"
    )
    package_dir.mkdir(parents=True)
    (package_dir / "model_config.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "task_type": task_type,
                "model_architecture": architecture,
            }
        )
    )
    return package_dir


def test_configured_roots_hold_both_settings(roots):
    server_root, models_home = roots

    assert model_cache.configured_cache_roots() == [
        str(server_root),
        str(models_home),
    ]


def test_configured_roots_are_deduplicated_when_the_settings_agree(roots, monkeypatch):
    server_root, _ = roots
    monkeypatch.setattr(configuration, "INFERENCE_HOME", str(server_root))

    assert model_cache.configured_cache_roots() == [str(server_root)]


def test_listing_scans_both_roots_and_both_layouts(roots):
    server_root, models_home = roots
    _write_traditional(server_root, "ws/trad-a/1", "object-detection", "yolov8n")
    _write_inference_models(server_root, "ws/im-a/1", "classification", "vit")
    _write_traditional(models_home, "ws/trad-b/2", "instance-segmentation", "yolov8s")
    _write_inference_models(models_home, "ws/im-b/2", "keypoint-detection", "yolo")

    listed = {entry["model_id"]: entry for entry in model_cache.list_cached_models()}

    assert set(listed) == {"ws/trad-a/1", "ws/im-a/1", "ws/trad-b/2", "ws/im-b/2"}
    assert listed["ws/trad-a/1"] == {
        "model_id": "ws/trad-a/1",
        "name": "ws/trad-a/1",
        "task_type": "object-detection",
        "model_architecture": "yolov8n",
        "is_foundation": False,
    }
    assert listed["ws/im-a/1"]["task_type"] == "classification"
    assert listed["ws/trad-b/2"]["model_architecture"] == "yolov8s"
    assert listed["ws/im-b/2"]["task_type"] == "keypoint-detection"


def test_listing_scans_the_second_root_when_only_it_holds_models(roots):
    _, models_home = roots
    _write_traditional(models_home, "ws/only-b/1", "object-detection", "yolov8n")

    listed = model_cache.list_cached_models()

    assert [entry["model_id"] for entry in listed] == ["ws/only-b/1"]


def test_listing_reads_a_historical_traditional_root_without_model_id(roots):
    server_root, _ = roots
    _write_traditional(
        server_root, "ws/old/1", "object-detection", "yolov8n", with_model_id=False
    )

    listed = model_cache.list_cached_models()

    assert [entry["model_id"] for entry in listed] == ["ws/old/1"]


def test_listing_does_not_descend_into_a_nested_second_root(roots, monkeypatch):
    server_root, _ = roots
    nested_home = server_root / "inner_home"
    nested_home.mkdir()
    monkeypatch.setattr(configuration, "INFERENCE_HOME", str(nested_home))
    _write_traditional(
        nested_home, "ws/nested/1", "object-detection", "yolov8n", with_model_id=False
    )

    listed = model_cache.list_cached_models()

    assert [entry["model_id"] for entry in listed] == ["ws/nested/1"]


def test_listing_skips_reserved_top_level_directories(roots):
    server_root, _ = roots
    _write_traditional(server_root / "workflow", "ws/hidden/1", "x", "y")

    assert model_cache.list_cached_models() == []


def test_listing_ignores_a_traditional_root_whose_metadata_names_another_model(roots):
    server_root, _ = roots
    model_dir = server_root / "ws" / "liar" / "1"
    model_dir.mkdir(parents=True)
    (model_dir / "model_type.json").write_text(
        json.dumps(
            {"model_id": "ws/other/1", "project_task_type": "a", "model_type": "b"}
        )
    )

    assert model_cache.list_cached_models() == []


def test_listing_drops_models_with_conflicting_metadata_across_roots(roots):
    server_root, models_home = roots
    _write_inference_models(server_root, "ws/same/1", "object-detection", "yolov8n")
    _write_inference_models(models_home, "ws/same/1", "classification", "vit")

    assert model_cache.list_cached_models() == []


def test_listing_keeps_identical_models_found_in_both_roots_once(roots):
    server_root, models_home = roots
    _write_inference_models(server_root, "ws/same/1", "object-detection", "yolov8n")
    _write_inference_models(models_home, "ws/same/1", "object-detection", "yolov8n")

    listed = model_cache.list_cached_models()

    assert [entry["model_id"] for entry in listed] == ["ws/same/1"]


def test_traditional_layout_under_the_server_root_marks_a_model_cached(roots):
    server_root, _ = roots
    _write_traditional(server_root, "ws/proj/3", "object-detection", "yolov8n")

    assert model_cache.is_model_cached("ws/proj/3") is True


def test_traditional_layout_with_an_uppercase_id_marks_a_model_cached(roots):
    server_root, _ = roots
    model_id = "clip/ViT-B-16"
    cache_key = model_cache.get_model_id_cache_path(
        model_id=model_id, cache_dir_root=str(server_root)
    )
    model_dir = server_root / cache_key
    model_dir.mkdir(parents=True)
    (model_dir / "textual.onnx").write_bytes(b"x")

    assert model_cache.is_model_cached(model_id) is True


def test_inference_models_layout_under_the_models_home_marks_a_model_cached(roots):
    _, models_home = roots
    _write_inference_models(models_home, "clip/ViT-B-16", "", "")

    assert model_cache.is_model_cached("clip/ViT-B-16") is True


def test_hidden_files_alone_do_not_mark_a_model_cached(roots):
    server_root, _ = roots
    model_dir = server_root / "ws" / "proj" / "3"
    model_dir.mkdir(parents=True)
    (model_dir / ".lock").write_bytes(b"")

    assert model_cache.is_model_cached("ws/proj/3") is False


def test_symlinked_model_directory_is_not_cached(roots, tmp_path):
    server_root, _ = roots
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "weights.onnx").write_bytes(b"x")
    (server_root / "ws").mkdir()
    (server_root / "ws" / "proj").symlink_to(outside, target_is_directory=True)

    assert model_cache.is_model_cached("ws/proj/3") is False


def test_unsafe_model_id_is_not_cached(roots):
    assert model_cache.is_model_cached("../escape") is False


def test_uncached_model_is_not_cached(roots):
    assert model_cache.is_model_cached("ws/missing/1") is False


def test_variant_check_is_true_when_any_variant_is_cached(roots):
    _, models_home = roots
    _write_inference_models(models_home, "clip/RN50", "", "")

    assert model_cache.has_cached_model_variant(["clip/RN101", "clip/RN50"]) is True


def test_variant_check_is_false_without_variants_or_cache(roots):
    assert model_cache.has_cached_model_variant(None) is False
    assert model_cache.has_cached_model_variant([]) is False
    assert model_cache.has_cached_model_variant(["clip/RN50"]) is False


def test_offline_mode_resolves_a_legacy_slugged_traditional_tree(roots, monkeypatch):
    server_root, _ = roots
    monkeypatch.setattr(configuration, "LEGACY_OFFLINE_MODE", True)
    model_id = "WS/Proj/3"
    legacy_key = model_cache.get_legacy_model_id_cache_path(
        model_id=model_id, cache_dir_root=str(server_root)
    )
    assert legacy_key is not None
    model_dir = server_root / legacy_key
    model_dir.mkdir(parents=True)
    (model_dir / "model_type.json").write_text(json.dumps({"model_id": model_id}))
    (model_dir / "weights.onnx").write_bytes(b"x")

    assert model_cache.is_model_cached(model_id) is True


def _write_many_traditional(root, count):
    for index in range(count):
        _write_traditional(root, f"ws/many{index}/1", "object-detection", "yolov8n")


def _scan_warnings(caplog):
    return [
        record
        for record in caplog.records
        if record.levelname == "WARNING" and "incomplete" in record.getMessage()
    ]


def test_scan_within_bounds_is_not_marked_incomplete(roots, caplog):
    server_root, _ = roots
    _write_many_traditional(server_root, 3)

    scan = model_cache.scan_cached_models(str(server_root))

    assert scan.truncated is False
    assert len(scan.models) == 3
    assert _scan_warnings(caplog) == []


def test_scan_stops_and_reports_when_the_entry_bound_is_exceeded(
    roots, monkeypatch, caplog
):
    server_root, _ = roots
    _write_many_traditional(server_root, 10)
    monkeypatch.setattr(model_cache, "MAX_SCAN_ENTRIES", 5)

    scan = model_cache.scan_cached_models(str(server_root))

    assert scan.truncated is True
    assert len(scan.models) < 10
    warnings = _scan_warnings(caplog)
    assert len(warnings) == 1
    assert str(server_root) not in warnings[0].getMessage()


def test_scan_does_not_descend_below_the_depth_bound(roots, monkeypatch, caplog):
    server_root, _ = roots
    _write_traditional(server_root, "ws/deep/1", "object-detection", "yolov8n")
    monkeypatch.setattr(model_cache, "MAX_SCAN_DEPTH", 2)

    scan = model_cache.scan_cached_models(str(server_root))

    assert scan.truncated is True
    assert scan.models == []
    assert len(_scan_warnings(caplog)) == 1


def test_scan_skips_and_reports_a_metadata_file_over_the_byte_bound(
    roots, monkeypatch, caplog
):
    server_root, _ = roots
    _write_traditional(server_root, "ws/big/1", "object-detection", "yolov8n")
    _write_traditional(server_root, "ws/small/1", "object-detection", "yolov8n")
    big_metadata = server_root / "ws" / "big" / "1" / "model_type.json"
    big_metadata.write_text(
        json.dumps(
            {
                "model_id": "ws/big/1",
                "project_task_type": "object-detection",
                "model_type": "yolov8n",
                "padding": "x" * 5000,
            }
        )
    )
    monkeypatch.setattr(model_cache, "MAX_METADATA_FILE_BYTES", 1000)

    scan = model_cache.scan_cached_models(str(server_root))

    assert scan.truncated is True
    assert [entry["model_id"] for entry in scan.models] == ["ws/small/1"]
    assert len(_scan_warnings(caplog)) == 1


def test_scan_does_not_follow_a_symlinked_directory(roots, tmp_path):
    server_root, _ = roots
    outside = tmp_path / "outside"
    _write_traditional(
        outside, "escaped/1", "object-detection", "yolov8n", with_model_id=False
    )
    (server_root / "linked").symlink_to(outside, target_is_directory=True)

    scan = model_cache.scan_cached_models(str(server_root))

    assert scan.models == []
    assert scan.truncated is False


def test_scan_does_not_follow_a_symlinked_models_cache_directory(roots, tmp_path):
    server_root, _ = roots
    outside_home = tmp_path / "outside_home"
    _write_inference_models(outside_home, "ws/escaped/1", "classification", "vit")
    (server_root / "models-cache").symlink_to(
        outside_home / "models-cache", target_is_directory=True
    )

    assert model_cache.scan_cached_models(str(server_root)).models == []
