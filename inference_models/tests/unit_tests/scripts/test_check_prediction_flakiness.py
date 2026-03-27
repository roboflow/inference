"""Unit tests for scripts/check_prediction_flakiness.py (mocked; no network or real models)."""

import importlib.util
from pathlib import Path
from unittest import mock

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "check_prediction_flakiness.py"
)


@pytest.fixture(scope="module")
def flakiness_mod():
    spec = importlib.util.spec_from_file_location(
        "check_prediction_flakiness", SCRIPT_PATH
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def minimal_config():
    return {"ws-a": {"proj-1": ["model-a"]}}


@pytest.fixture
def fake_image_paths(tmp_path):
    p = tmp_path / "img.jpg"
    p.write_bytes(b"x")
    return [p]


def test_refetch_artifact_stable_when_outputs_identical(
    flakiness_mod, minimal_config, fake_image_paths, tmp_path
):
    """Same normalized outputs every iteration -> status stable."""
    cache_dir = tmp_path / "cache"
    streams_dir = tmp_path / "streams"
    img_cache = tmp_path / "img_cache"

    out = [[{"pred": 1.0}]]

    with mock.patch.object(
        flakiness_mod, "fetch_roboflow_test_images", return_value=fake_image_paths
    ), mock.patch.object(
        flakiness_mod, "run_model_once", return_value=(out, "load")
    ), mock.patch.object(
        flakiness_mod, "clear_cache"
    ) as clear_mock:
        report = flakiness_mod.run_flakiness_check(
            models_by_workspace=minimal_config,
            iterations=3,
            cache_dir=cache_dir,
            api_key="k",
            float_precision=6,
            streams_output_dir=streams_dir,
            sample_different_images_limit=10,
            num_test_images=1,
            roboflow_images_cache_dir=img_cache,
            use_roboflow_image_cache=True,
            scenario=flakiness_mod.REFETCH_ARTIFACT,
        )

    assert report["results"]["ws-a"]["proj-1"]["model-a"]["status"] == "stable"
    assert report["results"]["ws-a"]["proj-1"]["model-a"]["mismatch_iterations"] == []
    assert clear_mock.call_count == 3


def test_refetch_artifact_flaky_when_second_iteration_differs(
    flakiness_mod, minimal_config, fake_image_paths, tmp_path
):
    """Different output on iteration 2 -> flaky, mismatch at iteration 2."""
    cache_dir = tmp_path / "cache"
    streams_dir = tmp_path / "streams"
    img_cache = tmp_path / "img_cache"

    calls = {"n": 0}

    def fake_run_model_once(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return [[{"pred": 1.0}]], "load-a"
        return [[{"pred": 2.0}]], "load-b"

    with mock.patch.object(
        flakiness_mod, "fetch_roboflow_test_images", return_value=fake_image_paths
    ), mock.patch.object(
        flakiness_mod, "run_model_once", side_effect=fake_run_model_once
    ), mock.patch.object(flakiness_mod, "clear_cache"):
        report = flakiness_mod.run_flakiness_check(
            models_by_workspace=minimal_config,
            iterations=3,
            cache_dir=cache_dir,
            api_key="k",
            float_precision=6,
            streams_output_dir=streams_dir,
            sample_different_images_limit=10,
            num_test_images=1,
            roboflow_images_cache_dir=img_cache,
            use_roboflow_image_cache=True,
            scenario=flakiness_mod.REFETCH_ARTIFACT,
        )

    entry = report["results"]["ws-a"]["proj-1"]["model-a"]
    assert entry["status"] == "flaky"
    assert entry["mismatch_iterations"] == [2, 3]
    assert "2" in entry["iteration_diffs"]
    assert entry["iteration_diffs"]["2"]["num_images_with_differences"] >= 1
    assert report["summary"]["flaky_models"] == 1
    assert report["summary"]["flaky_by_workspace"]["ws-a"]["proj-1"] == ["model-a"]


def test_reload_local_artifact_flaky_when_second_iteration_differs(
    flakiness_mod, minimal_config, fake_image_paths, tmp_path
):
    """reload-local-artifact: no cache clear; still detects output drift."""
    cache_dir = tmp_path / "cache"
    streams_dir = tmp_path / "streams"
    img_cache = tmp_path / "img_cache"

    calls = {"n": 0}

    def fake_run_model_once(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return [[1]], "load-a"
        return [[2]], "load-b"

    with mock.patch.object(
        flakiness_mod, "fetch_roboflow_test_images", return_value=fake_image_paths
    ), mock.patch.object(
        flakiness_mod, "run_model_once", side_effect=fake_run_model_once
    ), mock.patch.object(flakiness_mod, "clear_cache") as clear_mock:
        report = flakiness_mod.run_flakiness_check(
            models_by_workspace=minimal_config,
            iterations=2,
            cache_dir=cache_dir,
            api_key="k",
            float_precision=6,
            streams_output_dir=streams_dir,
            sample_different_images_limit=10,
            num_test_images=1,
            roboflow_images_cache_dir=img_cache,
            use_roboflow_image_cache=True,
            scenario=flakiness_mod.RELOAD_LOCAL_ARTIFACT,
        )

    assert clear_mock.call_count == 0
    assert report["results"]["ws-a"]["proj-1"]["model-a"]["status"] == "flaky"


def test_same_model_repeated_inference_flaky_when_second_inference_differs(
    flakiness_mod, minimal_config, fake_image_paths, tmp_path
):
    """Load once; second inference returns different list -> flaky."""
    cache_dir = tmp_path / "cache"
    streams_dir = tmp_path / "streams"
    img_cache = tmp_path / "img_cache"

    inference_calls = {"n": 0}

    def fake_inference_only(model, image_paths, float_precision):
        inference_calls["n"] += 1
        if inference_calls["n"] == 1:
            return [[0.0]]
        return [[1.0]]

    fake_model = object()

    with mock.patch.object(
        flakiness_mod, "fetch_roboflow_test_images", return_value=fake_image_paths
    ), mock.patch.object(
        flakiness_mod, "load_model", return_value=(fake_model, "load-stream")
    ), mock.patch.object(
        flakiness_mod, "run_inference_only", side_effect=fake_inference_only
    ), mock.patch.object(flakiness_mod, "clear_cache") as clear_mock:
        report = flakiness_mod.run_flakiness_check(
            models_by_workspace=minimal_config,
            iterations=3,
            cache_dir=cache_dir,
            api_key="k",
            float_precision=6,
            streams_output_dir=streams_dir,
            sample_different_images_limit=10,
            num_test_images=1,
            roboflow_images_cache_dir=img_cache,
            use_roboflow_image_cache=True,
            scenario=flakiness_mod.SAME_MODEL_REPEATED_INFERENCE,
        )

    assert clear_mock.call_count == 0
    entry = report["results"]["ws-a"]["proj-1"]["model-a"]
    assert entry["status"] == "flaky"
    assert entry["mismatch_iterations"] == [2, 3]


def test_compute_diff_summary_lists_mismatched_paths(flakiness_mod, fake_image_paths):
    """Sanity check: diff summary ties mismatches to image paths."""
    baseline = [[1], [2]]
    candidate = [[1], [99]]
    summary = flakiness_mod.compute_diff_summary(
        baseline=baseline,
        candidate=candidate,
        image_paths=fake_image_paths * 2,
        sample_different_images_limit=10,
    )
    assert summary["num_images_compared"] == 2
    assert summary["num_images_with_differences"] == 1
    assert len(summary["sample_different_images"]) == 1
