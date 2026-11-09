import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from onnx import TensorProto, helper

from inference.core.utils.onnx import (
    _run_session_with_tensor_iobinding,
    run_session_via_iobinding,
)


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("mode", ["numpy", "cpu_tensor", "cpu_iobinding", "cuda"])
def test_onnx_predictions_preserve_shape_values_and_output_order(
    batch_size, dynamic, dtype, mode
) -> None:
    if mode == "cuda" and not torch.cuda.is_available():
        pytest.skip("Requires a CUDA device")

    provider = "CUDAExecutionProvider" if mode == "cuda" else "CPUExecutionProvider"
    assert provider in ort.get_available_providers()
    element_type = TensorProto.FLOAT16 if dtype == np.float16 else TensorProto.FLOAT
    shape = ["batch" if dynamic else batch_size, 2]
    graph = helper.make_graph(
        [
            helper.make_node("Cast", ["images"], ["unchanged"], to=TensorProto.FLOAT),
            helper.make_node("Neg", ["images"], ["negative"]),
        ],
        "prediction_fixture",
        [helper.make_tensor_value_info("images", element_type, shape)],
        [
            helper.make_tensor_value_info("unchanged", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info("negative", element_type, shape),
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9
    )
    onnx.checker.check_model(model)
    session = ort.InferenceSession(model.SerializeToString(), providers=[provider])
    assert provider in session.get_providers()
    expected = np.arange(batch_size * 2, dtype=dtype).reshape(2, batch_size).T
    if mode == "numpy":
        inputs = expected
    else:
        inputs = torch.from_numpy(expected).to("cuda" if mode == "cuda" else "cpu")

    runner = (
        _run_session_with_tensor_iobinding
        if mode == "cpu_iobinding"
        else run_session_via_iobinding
    )
    predictions = runner(session, input_name="images", input_data=inputs)

    assert len(predictions) == 2
    second_dtype = np.float32 if mode in {"cpu_iobinding", "cuda"} else dtype
    for prediction, expected_values, expected_dtype in zip(
        predictions, [expected, -expected], [np.float32, second_dtype]
    ):
        assert prediction.shape == (batch_size, 2)
        assert prediction.dtype == expected_dtype
        np.testing.assert_array_equal(prediction, expected_values)
