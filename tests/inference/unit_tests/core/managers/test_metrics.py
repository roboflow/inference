from unittest import mock

from inference.core.managers import metrics


def test_get_system_info_returns_info() -> None:
    info = metrics.get_system_info()
    assert isinstance(info, dict)
    assert "platform" in info


@mock.patch.object(metrics.platform, "system", side_effect=RuntimeError("fail"))
def test_get_system_info_returns_info_even_on_exception(system_mock: mock.MagicMock) -> None:
    info = metrics.get_system_info()
    assert isinstance(info, dict)
    assert info == {}
    system_mock.assert_called_once_with()
