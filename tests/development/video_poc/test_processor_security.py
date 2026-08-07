import sys
from pathlib import Path

import pytest

PROCESSOR_DIR = (
    Path(__file__).resolve().parents[3] / "development" / "video_poc" / "processor"
)
sys.path.insert(0, str(PROCESSOR_DIR))

from security import (  # noqa: E402
    DiagnosticRing,
    JobSecurityRegistry,
    MissingJobAccessToken,
    extract_access_token,
    format_inference_error,
    sanitize_diagnostic,
    validate_job_id,
)


def test_required_registry_rejects_claim_without_access_token():
    registry = JobSecurityRegistry(require_tokens=True)

    with pytest.raises(MissingJobAccessToken):
        registry.register_job("job-a")


def test_job_tokens_are_tenant_scoped():
    registry = JobSecurityRegistry(require_tokens=True)
    registry.register_job("job-a", "token-a")
    registry.register_job("job-b", "token-b")

    assert registry.authorize("job-a", "token-a")
    assert registry.authorize("job-b", "token-b")
    assert not registry.authorize("job-a", "token-b")
    assert not registry.authorize("job-b", "token-a")
    assert not registry.authorize("unknown-job", "token-a")


def test_tokenless_jobs_are_allowed_only_in_compatibility_mode():
    registry = JobSecurityRegistry(require_tokens=False)
    registry.register_job("local-job")

    assert registry.authorize("local-job", "")
    assert not registry.authorize("another-job", "")


def test_access_token_prefers_bearer_header_and_supports_media_query():
    assert (
        extract_access_token(
            "Bearer header-token", "/events/poll?job=job-a&access_token=query-token"
        )
        == "header-token"
    )
    assert (
        extract_access_token("", "/preview.mjpeg?job=job-a&access_token=query-token")
        == "query-token"
    )


def test_access_token_parsing_handles_untrusted_header_whitespace_linearly():
    header = "Bearer " + (" " * 10_000) + "header-token"

    assert extract_access_token(header, "/events?job=job-a") == "header-token"


@pytest.mark.parametrize(
    "job_id",
    ["../other-job", "job/child", "job%2Fchild", "", ".", "job id"],
)
def test_job_id_validation_rejects_path_syntax(job_id):
    with pytest.raises(ValueError):
        validate_job_id(job_id)


def test_results_resolve_only_from_recorder_registered_paths():
    registry = JobSecurityRegistry(require_tokens=True)
    registry.register_job("job-a", "token-a")
    registry.register_result_paths(
        "job-a",
        {
            "video.mp4": "/safe/job-a/video.mp4",
            "meta.json": "/safe/job-a/meta.json",
        },
    )

    assert registry.result_path("job-a", "video.mp4") == "/safe/job-a/video.mp4"
    assert registry.result_path("job-a", "../job-b/video.mp4") is None
    assert registry.result_path("../job-b", "video.mp4") is None


def test_failure_diagnostics_redact_credentials_and_signed_queries():
    line = (
        "source rtsp://camera-user:camera-pass@relay.example/live?token=stream-secret "
        "download https://storage.example/video.mp4?X-Goog-Signature=signed-secret "
        "Authorization: Bearer browser-secret"
    )

    sanitized = sanitize_diagnostic(line)

    for secret in (
        "camera-user",
        "camera-pass",
        "stream-secret",
        "signed-secret",
        "browser-secret",
    ):
        assert secret not in sanitized
    assert "[credentials-redacted]" in sanitized
    assert "Bearer [redacted]" in sanitized


def test_diagnostic_ring_is_bounded_and_sanitizes_each_job_independently():
    first = DiagnosticRing(capacity=2)
    second = DiagnosticRing(capacity=2)

    first.note("first job")
    first.note("Bearer first-secret")
    first.note("last first line")
    second.note("second job")

    assert first.tail() == ["Bearer [redacted]", "last first line"]
    assert second.tail() == ["second job"]


def test_format_inference_error_preserves_cause_and_redacts_credentials():
    result = format_inference_error(
        {
            "error_type": "ClientCausedStepExecutionError",
            "error_message": (
                "model failed at https://user:pass@example.com/weights?api_key=secret "
                "with Bearer another-secret"
            ),
        }
    )

    assert result.startswith("ClientCausedStepExecutionError: model failed")
    assert "user:pass" not in result
    assert "another-secret" not in result
    assert "[credentials-redacted]@example.com" in result
    assert "Bearer [redacted]" in result


def test_format_inference_error_handles_missing_payload():
    assert format_inference_error(None) == (
        "InferenceError: the workflow inference thread stopped unexpectedly"
    )
