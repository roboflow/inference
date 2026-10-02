import pytest

from tests.unit_tests.legacy.conftest import (  # noqa: F401
    FakeGateway,
    fake_stat,
    key_gated_stat,
    legacy_client,
)


@pytest.fixture(autouse=True)
def _active_learning_off(monkeypatch):
    monkeypatch.setattr("inference_server.configuration.ACTIVE_LEARNING_ENABLED", False)
