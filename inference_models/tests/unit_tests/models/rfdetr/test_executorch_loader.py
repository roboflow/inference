import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from inference_models.errors import (
    CorruptedModelPackageError,
    InvalidModelInitParameterError,
    ModelPackageRestrictedError,
)
from inference_models.models.rfdetr import rfdetr_object_detection_executorch
from inference_models.models.rfdetr.rfdetr_object_detection_executorch import (
    RFDetrForObjectDetectionExecuTorch,
)


class FakeMethod:
    """ExecuTorch method stand-in returning one confident box per image."""

    def __init__(self, num_logit_classes=3):
        self.num_logit_classes = num_logit_classes
        self.batch_sizes = []

    def execute(self, inputs, clone_outputs):
        (chunk,) = inputs
        self.batch_sizes.append(chunk.shape[0])
        boxes = torch.zeros((chunk.shape[0], 300, 4))
        boxes[:, 0] = torch.tensor([0.5, 0.5, 0.2, 0.2])
        logits = torch.full((chunk.shape[0], 300, self.num_logit_classes), -10.0)
        logits[:, 0, 1] = 10.0

        return [boxes, logits]


def _package(tmp_path, configured_size: int, with_data_file: bool = False) -> str:
    (tmp_path / "class_names.txt").write_text("egg\nchicken\n")
    (tmp_path / "inference_config.json").write_text(
        json.dumps(
            {
                "network_input": {
                    "training_input_size": {
                        "height": configured_size,
                        "width": configured_size,
                    },
                    "dynamic_spatial_size_supported": False,
                    "color_mode": "rgb",
                    "resize_mode": "stretch",
                    "input_channels": 3,
                    "scaling_factor": 255,
                    "normalization": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
                },
            }
        )
    )
    (tmp_path / "weights.pte").write_bytes(b"pte")
    if with_data_file:
        (tmp_path / "weights.ptd").write_bytes(b"ptd")

    return str(tmp_path)


class FakeProgram:
    """ExecuTorch program stand-in that records when its method is loaded."""

    def __init__(
        self,
        input_shape,
        input_device="cpu",
        input_dtype_name="float32",
        load_error=None,
        num_inputs=1,
        num_outputs=2,
    ):
        input_meta = SimpleNamespace(
            sizes=lambda: input_shape,
            device=lambda: input_device,
            dtype_name=lambda: input_dtype_name,
        )
        self.method = FakeMethod()
        self.loaded_methods = []
        self._load_error = load_error
        self._meta = SimpleNamespace(
            input_tensor_meta=lambda index: input_meta,
            num_inputs=lambda: num_inputs,
            num_outputs=lambda: num_outputs,
        )

    def metadata(self, method_name):
        return self._meta

    def load_method(self, method_name):
        self.loaded_methods.append(method_name)
        if self._load_error is not None:
            raise self._load_error
        return self.method


def _patch_loader(monkeypatch, program: FakeProgram) -> list:
    load_calls = []

    def load_executorch_program(program_path, *, data_path=None):
        load_calls.append((program_path, data_path))
        return program

    monkeypatch.setattr(
        rfdetr_object_detection_executorch,
        "load_executorch_program",
        load_executorch_program,
    )

    return load_calls


@pytest.mark.parametrize("with_data_file", [False, True])
def test_executorch_loader_uses_program_input_size_and_data_file(
    tmp_path, monkeypatch, with_data_file
) -> None:
    # given
    package_dir = _package(tmp_path, configured_size=640, with_data_file=with_data_file)
    program = FakeProgram(input_shape=(1, 3, 384, 384))
    load_calls = _patch_loader(monkeypatch, program)

    # when
    model = RFDetrForObjectDetectionExecuTorch.from_pretrained(
        package_dir, device=torch.device("cpu")
    )

    # then
    size = model._inference_config.network_input.training_input_size
    assert (size.height, size.width) == (384, 384)
    assert load_calls == [
        (
            str(tmp_path / "weights.pte"),
            str(tmp_path / "weights.ptd") if with_data_file else None,
        )
    ]
    assert program.loaded_methods == ["forward"]


@pytest.mark.parametrize(
    "configured_size, program_size, rejected",
    [(384, 1024, True), (1024, 384, False)],
)
def test_executorch_loader_checks_size_limit_against_program_before_loading_it(
    tmp_path, monkeypatch, configured_size, program_size, rejected
) -> None:
    # given
    package_dir = _package(tmp_path, configured_size=configured_size)
    program = FakeProgram(input_shape=(1, 3, program_size, program_size))
    _patch_loader(monkeypatch, program)

    # when
    if rejected:
        with pytest.raises(ModelPackageRestrictedError):
            RFDetrForObjectDetectionExecuTorch.from_pretrained(
                package_dir,
                device=torch.device("cpu"),
                rf_detr_max_input_resolution=800,
            )
    else:
        RFDetrForObjectDetectionExecuTorch.from_pretrained(
            package_dir,
            device=torch.device("cpu"),
            rf_detr_max_input_resolution=800,
        )

    # then
    assert program.loaded_methods == ([] if rejected else ["forward"])


@pytest.mark.parametrize(
    "input_shape, input_dtype_name, input_device, device, error",
    [
        ((1, 3, 384), "float32", "cpu", "cpu", CorruptedModelPackageError),
        ((1, 384, 384, 3), "float32", "cpu", "cpu", CorruptedModelPackageError),
        ((1, 3, 384, 384), "float16", "cpu", "cpu", CorruptedModelPackageError),
        ((1, 3, 384, 384), "float32", "cuda:0", "cpu", InvalidModelInitParameterError),
        (
            (1, 3, 384, 384),
            "float32",
            "cuda:0",
            "cuda:1",
            InvalidModelInitParameterError,
        ),
    ],
)
def test_executorch_loader_rejects_programs_it_cannot_feed(
    tmp_path, monkeypatch, input_shape, input_dtype_name, input_device, device, error
) -> None:
    # given
    package_dir = _package(tmp_path, configured_size=384)
    program = FakeProgram(
        input_shape=input_shape,
        input_device=input_device,
        input_dtype_name=input_dtype_name,
    )
    _patch_loader(monkeypatch, program)

    # when
    with pytest.raises(error):
        RFDetrForObjectDetectionExecuTorch.from_pretrained(
            package_dir, device=torch.device(device)
        )

    # then
    assert program.loaded_methods == []


@pytest.mark.parametrize("num_inputs, num_outputs", [(2, 2), (1, 3)])
def test_executorch_loader_rejects_programs_without_one_input_and_two_outputs(
    tmp_path, monkeypatch, num_inputs, num_outputs
) -> None:
    # given
    package_dir = _package(tmp_path, configured_size=384)
    program = FakeProgram(
        input_shape=(1, 3, 384, 384),
        num_inputs=num_inputs,
        num_outputs=num_outputs,
    )
    _patch_loader(monkeypatch, program)

    # when
    with pytest.raises(CorruptedModelPackageError):
        RFDetrForObjectDetectionExecuTorch.from_pretrained(
            package_dir, device=torch.device("cpu")
        )

    # then
    assert program.loaded_methods == []


@pytest.mark.gpu_only
@pytest.mark.trt_extras
@pytest.mark.parametrize("requested_device", ["cuda", "cuda:0"])
def test_executorch_loader_initializes_delegates_on_requested_gpu(
    tmp_path, monkeypatch, requested_device
) -> None:
    # given
    package_dir = _package(tmp_path, configured_size=384)
    program = FakeProgram(input_shape=(1, 3, 384, 384))
    _patch_loader(monkeypatch, program)
    active_device_indices = []
    device_indices_at_load = []
    cuda_device = torch.cuda.device

    class RecordingCudaDevice(cuda_device):
        def __enter__(self):
            active_device_indices.append(self.idx)
            return super().__enter__()

        def __exit__(self, *args):
            active_device_indices.pop()
            return super().__exit__(*args)

    load_method = program.load_method

    def recording_load_method(method_name):
        device_indices_at_load.append(list(active_device_indices))
        return load_method(method_name)

    monkeypatch.setattr(torch.cuda, "device", RecordingCudaDevice)
    program.load_method = recording_load_method

    # when
    model = RFDetrForObjectDetectionExecuTorch.from_pretrained(
        package_dir, device=torch.device(requested_device)
    )

    # then
    assert device_indices_at_load == [[0]]
    assert model._device == torch.device("cuda", 0)


def test_executorch_loader_explains_delegate_initialization_failures(
    tmp_path, monkeypatch
) -> None:
    # given
    package_dir = _package(tmp_path, configured_size=384)
    program = FakeProgram(
        input_shape=(1, 3, 384, 384),
        load_error=RuntimeError("Failed to load method forward, error: 0x14"),
    )
    _patch_loader(monkeypatch, program)

    # when
    with pytest.raises(CorruptedModelPackageError, match="torch_tensorrt") as error:
        RFDetrForObjectDetectionExecuTorch.from_pretrained(
            package_dir, device=torch.device("cpu")
        )

    # then
    assert "0x14" in str(error.value)


def test_executorch_model_runs_batches_larger_than_program_batch_size(
    tmp_path, monkeypatch
) -> None:
    # given
    package_dir = _package(tmp_path, configured_size=64)
    program = FakeProgram(input_shape=(1, 3, 64, 64))
    _patch_loader(monkeypatch, program)
    model = RFDetrForObjectDetectionExecuTorch.from_pretrained(
        package_dir, device=torch.device("cpu")
    )
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    # when
    predictions = model([image, image], confidence=0.5)

    # then
    assert program.method.batch_sizes == [1, 1]
    assert len(predictions) == 2
    for detections in predictions:
        assert detections.class_id.tolist() == [1]
        assert torch.allclose(
            detections.xyxy.float(), torch.tensor([[80.0, 40.0, 120.0, 60.0]])
        )


def test_executorch_model_runs_a_real_xnnpack_program(tmp_path) -> None:
    # given
    exir = pytest.importorskip("executorch.exir")
    xnnpack = pytest.importorskip(
        "executorch.backends.xnnpack.partition.xnnpack_partitioner"
    )

    class OneBoxDetector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            boxes = torch.zeros((1, 300, 4))
            boxes[:, 0] = torch.tensor([0.5, 0.5, 0.2, 0.2])
            logits = torch.full((1, 300, 3), -10.0)
            logits[:, 0, 1] = 10.0
            self.register_buffer("boxes", boxes)
            self.register_buffer("logits", logits)

        def forward(self, images):
            zero = images.mean() * 0

            batch_size = images.shape[0]

            return (
                self.boxes.expand(batch_size, -1, -1) + zero,
                self.logits.expand(batch_size, -1, -1) + zero,
            )

    exported = torch.export.export(OneBoxDetector(), (torch.zeros((2, 3, 64, 64)),))
    program = exir.to_edge_transform_and_lower(
        exported, partitioner=[xnnpack.XnnpackPartitioner()]
    ).to_executorch()
    package_dir = _package(tmp_path, configured_size=640)
    (tmp_path / "weights.pte").write_bytes(program.buffer)
    model = RFDetrForObjectDetectionExecuTorch.from_pretrained(
        package_dir, device=torch.device("cpu")
    )
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    # when
    pre_processed_images, pre_processing_meta = model.pre_process([image] * 3)
    bboxes, logits = model.forward(pre_processed_images)
    predictions = model.post_process(
        (bboxes, logits), pre_processing_meta, confidence=0.5
    )

    # then
    assert bboxes.shape == (3, 300, 4)
    assert logits.shape == (3, 300, 3)
    assert len(predictions) == 3
    for detections in predictions:
        assert detections.class_id.tolist() == [1]
        assert torch.allclose(
            detections.xyxy.float(), torch.tensor([[80.0, 40.0, 120.0, 60.0]])
        )
