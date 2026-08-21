from unittest.mock import MagicMock

import pytest

from inference.core.interfaces.http_worker.entities import ArtifactTarget, TimeBase
from inference.core.interfaces.http_worker.sinks import ArtifactWriter


def test_artifact_writer_mints_signed_url_puts_gcs_and_posts_commit(
    monkeypatch,
) -> None:
    calls = []

    def fake_put(url, data=None, json=None, headers=None, timeout=None):
        calls.append(("PUT", url, data, json, headers))
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {}
        return response

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(("POST", url, json, headers))
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "uploadUrl": "https://storage.googleapis.com/signed/chunk-000000.json"
        }
        return response

    monkeypatch.setattr(
        "inference.core.interfaces.http_worker.sinks.requests.put",
        fake_put,
    )
    monkeypatch.setattr(
        "inference.core.interfaces.http_worker.sinks.requests.post",
        fake_post,
    )

    writer = ArtifactWriter(
        app_base_url="https://app.roboflow.com",
        video_id="video-1",
        workspace_id="ws-1",
        dataset_id="ds-1",
        revision_id="rev-1",
        api_key="rf_key",
    )
    writer.checkpoint_chunk(
        track_id="7",
        chunk_index=0,
        total_chunks=1,
        samples=[{"trackId": 7, "className": "forklift"}],
    )
    writer.commit_revision(
        track_id="7",
        start_frame_index=0,
        end_frame_index=4,
        start_pts=0,
        end_pts=4,
        video_time_base=TimeBase(numerator=1, denominator=30),
        class_name="forklift",
        tracker_id=7,
        sample_count=5,
        chunk_count=1,
    )

    assert calls[0][0] == "POST"
    assert calls[0][1].endswith(
        "query/video/video-1/tracks/7/artifact-chunks/upload-url"
    )
    assert calls[0][2]["revisionId"] == "rev-1"
    assert calls[0][3]["Authorization"] == "Bearer rf_key"
    assert calls[1][0] == "PUT"
    assert calls[1][1].startswith("https://storage.googleapis.com/signed/")
    assert b'"samples"' in calls[1][2]
    assert calls[1][4] == {"Content-Type": "application/json"}
    assert "Authorization" not in (calls[1][4] or {})
    assert calls[2][0] == "POST"
    assert "artifact-revisions/rev-1/commit" in calls[2][1]
    assert calls[2][2]["sampleCount"] == 5
    assert calls[2][3]["Authorization"] == "Bearer rf_key"


def test_artifact_writer_rejects_untrusted_app_host() -> None:
    with pytest.raises(ValueError, match="Roboflow app host"):
        ArtifactWriter(
            app_base_url="https://evil.example",
            video_id="video-1",
            workspace_id="ws-1",
            dataset_id="ds-1",
            revision_id="rev-1",
            api_key="rf_key",
        )


def test_session_request_rejects_untrusted_app_host() -> None:
    with pytest.raises(ValueError, match="Roboflow app host"):
        ArtifactTarget(
            app_base_url="https://evil.example",
            video_id="video-1",
            workspace_id="ws-1",
            dataset_id="ds-1",
            revision_id="rev-1",
        )
