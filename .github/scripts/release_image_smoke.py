"""Execute a small numerical inference inside the exact release image."""

import importlib
import json
import os
from importlib.metadata import distributions, version
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper

profile = json.loads(os.environ["RELEASE_IMAGE_PROFILE"])
module = "inference_models" if profile["version_source"] == "models" else "inference"
importlib.import_module(module)
if profile["cpu_torch"]:
    assert torch.version.cuda is None
    assert torch.__version__.endswith("+cpu")
    assert version("torchvision").endswith("+cpu")
    packages = {item.metadata["Name"].lower() for item in distributions()}
    assert "triton" not in packages
    assert {name for name in packages if name.startswith("nvidia-")} == {"nvidia-ml-py"}

device = "cpu"
provider = "CPUExecutionProvider"
if profile["accelerator"] != "cpu":
    assert torch.cuda.is_available(), "A real compatible GPU is required"
    device = "cuda"
    provider = "CUDAExecutionProvider"
    assert provider in ort.get_available_providers()
x = torch.tensor([[1.0, 2.0]], device=device)
model = torch.nn.Linear(2, 1, bias=False).to(device)
with torch.no_grad():
    model.weight.copy_(torch.tensor([[3.0, 4.0]], device=device))
    assert model(x).cpu().tolist() == [[11.0]]
if profile["cpu_torch"]:
    from torchvision.ops import nms

    assert nms(
        torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0]]),
        torch.tensor([0.9, 0.8]),
        0.5,
    ).tolist() == [0]

graph = helper.make_graph(
    [helper.make_node("Add", ["x", "x"], ["y"])],
    "release-smoke",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
session = ort.InferenceSession(model.SerializeToString(), providers=[provider])
assert session.get_providers()[0] == provider
np.testing.assert_array_equal(
    session.run(None, {"x": np.array([[1.0, 2.0]], dtype=np.float32)})[0], [[2.0, 4.0]]
)
print("Image import, Torch and ONNX inference smoke checks passed")
