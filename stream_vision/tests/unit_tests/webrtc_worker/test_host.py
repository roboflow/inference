import pytest
from streamvision.stream_manager.manager_app import host as pipeline_host
from streamvision.stream_manager.manager_app.host import (
    PipelineHostDescriptor,
    PipelineHostNotConfiguredError,
)
from streamvision.webrtc_worker import host as worker_host

_BUILT = []


class _Host:
    def __init__(self, **settings) -> None:
        self.settings = settings


def _build_host(**settings) -> _Host:
    built = _Host(**settings)
    _BUILT.append(built)

    return built


@pytest.fixture(autouse=True)
def _clean_host_state(monkeypatch):
    monkeypatch.setattr(worker_host, "_HOST", None)
    monkeypatch.setattr(pipeline_host, "_DEFAULT_DESCRIPTOR", None)
    _BUILT.clear()


def test_get_webrtc_worker_host_requires_an_installed_descriptor() -> None:
    with pytest.raises(PipelineHostNotConfiguredError):
        worker_host.get_webrtc_worker_host()


def test_get_webrtc_worker_host_builds_the_installed_host_once() -> None:
    pipeline_host.install_default_host_descriptor(
        PipelineHostDescriptor(
            factory=f"{__name__}:_build_host", settings={"name": "probe"}
        )
    )

    first = worker_host.get_webrtc_worker_host()
    second = worker_host.get_webrtc_worker_host()

    assert first is second
    assert _BUILT == [first]
    assert first.settings == {"name": "probe"}
