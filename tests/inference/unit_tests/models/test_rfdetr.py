import numpy as np

from inference.models.rfdetr.rfdetr import RFDETRObjectDetection


def test_sigmoid_stable_zero():
    # given
    model = RFDETRObjectDetection.__new__(RFDETRObjectDetection)

    # when
    result = model.sigmoid_stable(0.0)

    # then
    expected = 0.5
    assert np.isclose(result, expected)


def test_sigmoid_stable_positive_small():
    # given
    model = RFDETRObjectDetection.__new__(RFDETRObjectDetection)

    # when
    x = np.array([0.5, 1.0, 2.0])
    result = model.sigmoid_stable(x)

    # then
    expected = np.array([0.6224593312018546, 0.7310585786300049, 0.8807970779778823])

    assert np.allclose(result, expected)


def test_sigmoid_stable_negative_small():
    # given
    model = RFDETRObjectDetection.__new__(RFDETRObjectDetection)

    # when
    x = np.array([-0.5, -1.0, -2.0])
    result = model.sigmoid_stable(x)

    # then
    expected = np.array([0.3775406687981454, 0.2689414213699951, 0.11920292202211755])

    assert np.allclose(result, expected)


def test_sigmoid_stable_large_positive():
    # given
    model = RFDETRObjectDetection.__new__(RFDETRObjectDetection)

    # when
    x = np.array([50.0, 100.0, 500.0, 1000.0])
    result = model.sigmoid_stable(x)

    # then
    expected = np.ones_like(x)

    assert np.allclose(result, expected, atol=1e-15)


def test_sigmoid_stable_large_negative():
    # given
    model = RFDETRObjectDetection.__new__(RFDETRObjectDetection)

    # when
    x = np.array([-50.0, -100.0, -500.0, -1000.0])
    result = model.sigmoid_stable(x)

    # then
    expected = np.zeros_like(x)

    assert np.allclose(result, expected, atol=1e-15)


def test_sigmoid_stable_mixed_values():
    # given
    model = RFDETRObjectDetection.__new__(RFDETRObjectDetection)

    # when
    x = np.array([-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0])
    result = model.sigmoid_stable(x)

    # then
    expected = np.array([0.0, 0.0000453978687024, 0.2689414213699951, 0.5, 0.7310585786300049, 0.9999546021312976, 1])

    assert np.allclose(result, expected, atol=1e-15)


def test_sigmoid_stable_symmetry():
    # given
    model = RFDETRObjectDetection.__new__(RFDETRObjectDetection)

    # when
    x = np.array([1.0, 2.0, 5.0, 10.0])
    pos_result = model.sigmoid_stable(x)
    neg_result = model.sigmoid_stable(-x)

    expected_neg = 1.0 - pos_result

    assert np.allclose(neg_result, expected_neg)


def test_sigmoid_stable_single_scalar():
    # given
    model = RFDETRObjectDetection.__new__(RFDETRObjectDetection)

    # when
    result = model.sigmoid_stable(1.0)
    expected = 0.7310585786300049  # 1 / (1 + exp(-1))

    # then
    assert np.isclose(result, expected)


def test_sigmoid_stable_array_shapes():
    # given
    model = RFDETRObjectDetection.__new__(RFDETRObjectDetection)

    # when
    x_2d = np.array([[0.0, 1.0], [-1.0, 2.0]])
    result_2d = model.sigmoid_stable(x_2d)

    x_3d = np.array([[[0.0, 1.0], [-1.0, 2.0]], [[0.5, -0.5], [3.0, -3.0]]])
    result_3d = model.sigmoid_stable(x_3d)

    # then
    assert result_2d.shape == x_2d.shape
    assert result_3d.shape == x_3d.shape


def test_sigmoid_stable_edge_case_inf():
    # given
    model = RFDETRObjectDetection.__new__(RFDETRObjectDetection)

    # when
    result_pos_inf = model.sigmoid_stable(np.inf)
    result_neg_inf = model.sigmoid_stable(-np.inf)

    # then
    assert result_pos_inf == 1.0
    assert result_neg_inf == 0.0


def test_sigmoid_stable_edge_case_nan():
    # given
    model = RFDETRObjectDetection.__new__(RFDETRObjectDetection)

    # when
    result = model.sigmoid_stable(np.nan)

    # then
    assert np.isnan(result)
