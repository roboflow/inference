"""Unit tests for `_resolve_metadata_driven_workflow_parameters`.

The helper isolates the metadata pre-flight logic from
`_process_single_image_from_directory` so we can pin down its four real
branches (mapping absent, file missing, keys missing, happy path) without
mocking the workflow execution chain.
"""

import json

from inference_cli.lib.workflows.local_image_adapter import (
    _resolve_metadata_driven_workflow_parameters,
)


def _write_image_with_metadata(tmp_path, image_name: str, metadata: dict) -> str:
    image_path = tmp_path / image_name
    image_path.write_bytes(b"fake-image-bytes")
    metadata_path = tmp_path / f"{image_path.stem}.json"
    metadata_path.write_text(json.dumps(metadata))
    return str(image_path)


def test_resolve_returns_empty_params_when_mapping_is_none() -> None:
    # When the caller doesn't ask for metadata injection, the helper must
    # short-circuit — no filesystem lookup, no failure even if there's no
    # sibling .json file.
    params, error = _resolve_metadata_driven_workflow_parameters(
        image_path="/nonexistent/image.jpg",
        images_metadata_input_mapping=None,
    )
    assert params == {}
    assert error is None


def test_resolve_returns_empty_params_when_mapping_is_empty_dict() -> None:
    # Empty dict is the same as None — nothing to inject, no lookup needed.
    params, error = _resolve_metadata_driven_workflow_parameters(
        image_path="/nonexistent/image.jpg",
        images_metadata_input_mapping={},
    )
    assert params == {}
    assert error is None


def test_resolve_reports_error_when_metadata_file_is_missing(tmp_path) -> None:
    # Image exists, but no .json sibling — the helper must refuse.
    image_path = tmp_path / "img.jpg"
    image_path.write_bytes(b"fake")

    params, error = _resolve_metadata_driven_workflow_parameters(
        image_path=str(image_path),
        images_metadata_input_mapping={"workflow_input": "metadata_key"},
    )

    assert params is None
    assert error is not None
    assert str(image_path) in error
    assert "metadata file" in error


def test_resolve_reports_error_listing_missing_keys_when_metadata_incomplete(
    tmp_path,
) -> None:
    image_path = _write_image_with_metadata(
        tmp_path, "img.jpg", metadata={"present_key": "value"}
    )

    params, error = _resolve_metadata_driven_workflow_parameters(
        image_path=image_path,
        images_metadata_input_mapping={
            "wf_input_a": "present_key",
            "wf_input_b": "missing_key",
        },
    )

    assert params is None
    assert error is not None
    assert "missing_key" in error
    # The present key should not be listed as missing.
    assert "present_key" not in error.split("Missing keys:")[1]


def test_resolve_projects_metadata_through_mapping_on_happy_path(tmp_path) -> None:
    image_path = _write_image_with_metadata(
        tmp_path,
        "img.jpg",
        metadata={
            "lat": 51.5,
            "lon": -0.1,
            "captured_at": "2026-01-01",
        },
    )

    params, error = _resolve_metadata_driven_workflow_parameters(
        image_path=image_path,
        images_metadata_input_mapping={
            "latitude": "lat",
            "longitude": "lon",
        },
    )

    # Keys come from the mapping's keys (workflow input names); values come
    # from the metadata via the mapping's values (metadata field names).
    # `captured_at` is in the metadata but unmapped, so it's dropped.
    assert params == {"latitude": 51.5, "longitude": -0.1}
    assert error is None


def test_resolve_finds_metadata_for_image_without_extension(tmp_path) -> None:
    # `replace_file_extension` handles the no-existing-extension case;
    # this test pins down that the helper composes with it correctly so
    # extension-less image paths still resolve to a sibling .json.
    image_path = tmp_path / "img"  # no extension
    image_path.write_bytes(b"fake")
    (tmp_path / "img.json").write_text(json.dumps({"k": "v"}))

    params, error = _resolve_metadata_driven_workflow_parameters(
        image_path=str(image_path),
        images_metadata_input_mapping={"wf": "k"},
    )

    assert params == {"wf": "v"}
    assert error is None
