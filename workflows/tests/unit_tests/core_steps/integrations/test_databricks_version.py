from importlib.metadata import PackageNotFoundError

import pytest
from roboflow_workflows.core_steps.integrations.databricks import availability


@pytest.mark.parametrize("installed", ["4.0.0", "4.7.0", "4.8.1", "5.0.0"])
def test_unsupported_connector_is_not_advertised(monkeypatch, installed):
    """Verify unsupported connector is not advertised.

    Args:
        monkeypatch (pytest.MonkeyPatch): Scoped replacements restored after this test.
        installed (str): Connector distribution version reported by the test fixture.
    """
    monkeypatch.setattr(availability, "block_enabled", lambda: True)
    monkeypatch.setattr(availability, "connector_available", lambda: True)
    monkeypatch.setattr(availability, "version", lambda name: installed)

    reason = availability.databricks_unavailability_reason()

    assert installed in reason
    assert "requires 4.8.0" in reason
    assert availability.databricks_block_available() is False


def test_importable_connector_without_distribution_is_not_advertised(monkeypatch):
    """Verify importable connector without distribution is not advertised.

    Args:
        monkeypatch (pytest.MonkeyPatch): Scoped replacements restored after this test.
    """
    monkeypatch.setattr(availability, "block_enabled", lambda: True)
    monkeypatch.setattr(availability, "connector_available", lambda: True)

    def _missing(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(availability, "version", _missing)

    assert "distribution is missing" in availability.databricks_unavailability_reason()
