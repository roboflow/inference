"""Offline registry storage accounting checks."""

import json
from unittest.mock import patch

import depot_ci_image as image
import pytest

ROOT = "sha256:" + "a" * 64
CHILD = "sha256:" + "b" * 64
LAYER = "sha256:" + "c" * 64
CONFIG = "sha256:" + "d" * 64


def descriptor(digest, size):
    return {"digest": digest, "size": size}


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    monkeypatch.setenv("IMAGE_FAMILY", "parallel-tests")
    monkeypatch.setenv("GITHUB_SHA", "e" * 40)
    monkeypatch.setenv("DEPOT_PROJECT_ID", "exampleproject")


def test_shared_layers_count_once_and_private_fields_are_excluded():
    manifest = {
        "config": descriptor(CONFIG, 10),
        "layers": [descriptor(LAYER, 100), descriptor(LAYER, 100)],
        "annotations": {"private": "do-not-export"},
    }
    index = {
        "manifests": [descriptor(CHILD, 20), descriptor(CHILD, 20)],
        "annotations": {"private": "do-not-export"},
    }
    with patch.object(image, "_run", return_value=json.dumps(manifest)) as run:
        result = image._storage_inventory(ROOT, index)
    assert sum(blob["size"] for blob in result["blobs"]) == 130
    assert "do-not-export" not in json.dumps(result)
    assert result["excluded_root_manifests"] == 1
    run.assert_called_once_with(
        [
            "docker",
            "buildx",
            "imagetools",
            "inspect",
            "--raw",
            f"registry.depot.dev/exampleproject@{CHILD}",
        ]
    )


@pytest.mark.parametrize("size", [-1, True, 1.5, "100", 10**12 + 1])
def test_invalid_size_is_rejected(size):
    manifest = {"config": descriptor(CONFIG, size), "layers": []}
    with pytest.raises(ValueError):
        image._storage_inventory(ROOT, manifest)


def test_invalid_digest_does_not_trigger_network_request():
    with patch.object(image, "_run") as run, pytest.raises(ValueError):
        image._storage_inventory(ROOT, {"manifests": [descriptor("https://other", 1)]})
    run.assert_not_called()


def test_conflicting_sizes_are_rejected():
    manifest = {
        "config": descriptor(CONFIG, 10),
        "layers": [descriptor(LAYER, 100), descriptor(LAYER, 101)],
    }
    with pytest.raises(ValueError, match="Inconsistent"):
        image._storage_inventory(ROOT, manifest)


def test_nested_index_includes_attestation_and_shared_config():
    inner = {"manifests": [descriptor(CONFIG, 20)]}
    leaf = {"config": descriptor(LAYER, 5), "layers": []}
    with patch.object(image, "_run", side_effect=[json.dumps(inner), json.dumps(leaf)]):
        result = image._storage_inventory(ROOT, {"manifests": [descriptor(CHILD, 30)]})
    assert sum(blob["size"] for blob in result["blobs"]) == 55


def test_transient_manifest_read_retries_without_exposing_errors(capsys):
    with patch.object(
        image,
        "_run",
        side_effect=[
            ValueError("private-diagnostic"),
            json.dumps({"schemaVersion": 2}),
        ],
    ) as run, patch.object(image.time, "sleep") as sleep:
        assert image._read_manifest(ROOT) == {"schemaVersion": 2}
    assert run.call_count == 2
    sleep.assert_called_once_with(1)
    assert capsys.readouterr().out == "CI_IMAGE_MANIFEST_RETRY attempt=1\n"


def test_manifest_read_exhaustion_is_bounded_and_invalid_json_is_not_retried(capsys):
    with patch.object(
        image, "_run", side_effect=ValueError("private-diagnostic")
    ) as run, patch.object(image.time, "sleep") as sleep, pytest.raises(ValueError):
        image._read_manifest(ROOT)
    assert run.call_count == 3
    assert [call.args for call in sleep.call_args_list] == [(1,), (2,)]
    assert "private-diagnostic" not in capsys.readouterr().out
    with patch.object(image, "_run", return_value="invalid-json") as run, pytest.raises(
        json.JSONDecodeError
    ):
        image._read_manifest(ROOT)
    assert run.call_count == 1
