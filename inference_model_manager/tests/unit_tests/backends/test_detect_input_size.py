from types import SimpleNamespace

from inference_model_manager.backends.base import detect_input_size


def _model_with_network_input(**network_input):
    return SimpleNamespace(
        _inference_config=SimpleNamespace(
            network_input=SimpleNamespace(**network_input)
        )
    )


def test_fixed_input_size_is_reported_as_height_and_width():
    model = _model_with_network_input(
        dynamic_spatial_size_supported=False,
        training_input_size=SimpleNamespace(height=480, width=640),
    )

    assert detect_input_size(model) == (480, 640)


def test_dynamic_spatial_size_reports_no_input_size():
    model = _model_with_network_input(
        dynamic_spatial_size_supported=True,
        training_input_size=SimpleNamespace(height=480, width=640),
    )

    assert detect_input_size(model) == (None, None)


def test_missing_config_reports_no_input_size():
    assert detect_input_size(SimpleNamespace()) == (None, None)
    assert detect_input_size(SimpleNamespace(_inference_config=SimpleNamespace())) == (
        None,
        None,
    )
    assert detect_input_size(None) == (None, None)


def test_non_int_input_size_reports_no_input_size():
    for size in (
        SimpleNamespace(height=None, width=640),
        SimpleNamespace(height="tall", width=640),
        SimpleNamespace(height=480),
        None,
    ):
        model = _model_with_network_input(training_input_size=size)

        assert detect_input_size(model) == (None, None)


def test_numeric_strings_are_coerced_to_int():
    model = _model_with_network_input(
        training_input_size=SimpleNamespace(height="320", width=320.0),
    )

    assert detect_input_size(model) == (320, 320)


def test_non_positive_input_size_reports_no_input_size():
    model = _model_with_network_input(
        training_input_size=SimpleNamespace(height=0, width=640),
    )

    assert detect_input_size(model) == (None, None)
