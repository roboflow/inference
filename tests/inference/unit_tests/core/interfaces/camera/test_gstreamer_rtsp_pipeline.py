import pytest

from inference.core.interfaces.camera.gstreamer_rtsp_pipeline import (
    build_gstreamer_rtsp_pipeline,
    split_rtsp_credentials,
)
from inference.core.interfaces.camera.rtsp_tls import (
    RTSP_TLS_VALIDATION_FLAGS_ENV_VAR,
)


def test_split_rtsp_credentials_strips_userinfo() -> None:
    clean, user, password = split_rtsp_credentials(
        "rtsp://admin:secret@192.168.1.1:554/stream"
    )
    assert clean == "rtsp://192.168.1.1:554/stream"
    assert user == "admin"
    assert password == "secret"


def test_split_rtsp_credentials_password_with_at_sign() -> None:
    clean, user, password = split_rtsp_credentials(
        "rtsps://user:p@ss@host:554/stream"
    )
    assert clean == "rtsps://host:554/stream"
    assert user == "user"
    assert password == "p@ss"


def test_build_pipeline_uses_user_id_not_location_creds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(RTSP_TLS_VALIDATION_FLAGS_ENV_VAR, raising=False)
    pipeline = build_gstreamer_rtsp_pipeline(
        "rtsps://ent1544-user:ent1544-secret@127.0.0.1:8322/ent1544test"
    )
    location_value = pipeline.split('location="', 1)[1].split('"', 1)[0]
    assert location_value == "rtsps://127.0.0.1:8322/ent1544test"
    assert 'user-id="ent1544-user"' in pipeline
    assert 'user-pw="ent1544-secret"' in pipeline
    assert "ent1544-secret" not in location_value
    assert "rtspsrc" in pipeline
    assert "nvv4l2decoder" in pipeline


def test_build_pipeline_appends_tls_validation_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(RTSP_TLS_VALIDATION_FLAGS_ENV_VAR, "126")
    pipeline = build_gstreamer_rtsp_pipeline("rtsps://host:554/stream")
    assert " tls-validation-flags=126" in pipeline


def test_build_pipeline_rejects_non_rtsp_url() -> None:
    with pytest.raises(ValueError):
        build_gstreamer_rtsp_pipeline("/dev/video0")
