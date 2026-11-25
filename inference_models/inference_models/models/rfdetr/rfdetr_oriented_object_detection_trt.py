"""Native RF-DETR OBB TensorRT consumer, for packages built by the TRT compilation job from the native OBB ONNX.

Preprocessing and decoding are shared with the ONNX backend (``RFDetrForOrientedObjectDetectionBase``); this class
loads ``engine.plan`` and runs it.
"""

import threading
from typing import List, Optional

import torch

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.errors import (
    CorruptedModelPackageError,
    MissingDependencyError,
    ModelRuntimeError,
)
from inference_models.models.common.cuda import (
    use_cuda_context,
    use_primary_cuda_context,
)
from inference_models.models.common.roboflow.model_packages import (
    TRTConfig,
    parse_trt_config,
)
from inference_models.models.common.streams import get_cuda_stream
from inference_models.models.common.trt import (
    TRTCudaGraphCache,
    establish_trt_cuda_graph_cache,
    get_trt_engine_inputs_and_outputs,
    infer_from_trt_engine,
    load_trt_model,
)
from inference_models.models.rfdetr.rfdetr_oriented_object_detection_common import (
    OUTPUT_NAMES,
    RFDetrForOrientedObjectDetectionBase,
    check_model_input,
    check_output_shapes,
    load_oriented_package,
)

try:
    try:
        import tensorrt as trt
    except ImportError:
        import tensorrt_lean as trt
except ImportError as import_error:
    raise MissingDependencyError(
        message="Could not import RFDetr oriented-box model with TRT backend - this error means that some "
        "additional dependencies are not installed in the environment. If you see this error running locally, "
        "please follow our installation guide: https://inference-models.roboflow.com/getting-started/installation/",
        help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror",
    ) from import_error

try:
    import pycuda.driver as cuda
except ImportError as import_error:
    raise MissingDependencyError(
        message="Running model RFDETR with TRT backend on GPU requires pycuda installation, which is brought with "
        "`trt-*` extras of `inference-models` library. If you see this error running locally, "
        "please follow our installation guide: https://inference-models.roboflow.com/getting-started/installation/",
        help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror",
    ) from import_error


def _shape(engine: "trt.ICudaEngine", name: str) -> tuple:
    shape = engine.get_tensor_shape(name)
    return tuple(int(shape[i]) for i in range(len(shape)))


class RFDetrForOrientedObjectDetectionTRT(RFDetrForOrientedObjectDetectionBase):
    """Run a compiled native OBB TensorRT engine (``dets``, ``labels``, ``heading``) on a CUDA device."""

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        engine_host_code_allowed: bool = False,
        trt_cuda_graph_cache: Optional[TRTCudaGraphCache] = None,
        default_trt_cuda_graph_cache_size: int = 8,
        **kwargs,
    ) -> "RFDetrForOrientedObjectDetectionTRT":
        """Load a native OBB TensorRT package with Core-owned frozen class semantics.

        Args:
            model_name_or_path: Package directory with ``class_names.txt``, ``inference_config.json``,
                ``trt_config.json`` and ``engine.plan``.
            device: CUDA device used for inference. CPU is not supported.
            engine_host_code_allowed: Whether TensorRT may execute engine host code.
            trt_cuda_graph_cache: Optional CUDA graph cache to reuse across models.
            default_trt_cuda_graph_cache_size: Cache size used when no cache is provided.
            **kwargs: oriented_box_metadata (required) and rf_detr_max_input_resolution, as for the ONNX backend.

        Returns:
            Initialized native model.

        Raises:
            ModelRuntimeError: If ``device`` is not a CUDA device.
            CorruptedModelPackageError: If the engine does not have one square image input and the three outputs.
            ValueError: For missing metadata or preprocessing other than the training one.
        """
        if device.type != "cuda":
            raise ModelRuntimeError(
                message=f"TRT engine only runs on CUDA device - {device} device detected.",
                help_url="https://inference-models.roboflow.com/errors/models-runtime/#modelruntimeerror",
            )
        maximum = kwargs.get("rf_detr_max_input_resolution")
        package = load_oriented_package(
            model_name_or_path,
            raw_metadata=kwargs.get("oriented_box_metadata"),
            weights_files=["trt_config.json", "engine.plan"],
            max_input_resolution=maximum,
        )
        trt_config = parse_trt_config(config_path=package.files["trt_config.json"])
        cuda.init()
        cuda_device = cuda.Device(device.index or 0)
        with use_primary_cuda_context(cuda_device=cuda_device) as cuda_context:
            engine = load_trt_model(
                model_path=package.files["engine.plan"],
                engine_host_code_allowed=engine_host_code_allowed,
            )
            execution_context = engine.create_execution_context()
        inputs, outputs = get_trt_engine_inputs_and_outputs(engine=engine)
        input_shape = _shape(engine, inputs[0]) if len(inputs) == 1 else ()
        if len(input_shape) != 4 or input_shape[2] != input_shape[3]:
            raise CorruptedModelPackageError(
                message="Native OBB engine must have one square BCHW image input.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        try:
            check_output_shapes(
                {name: _shape(engine, name) for name in outputs},
                package.output_indices,
            )
        except ValueError as error:
            raise CorruptedModelPackageError(
                message=f"Native OBB engine outputs: {error}",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            ) from error
        check_model_input(package, input_shape[2], maximum)
        return cls(
            engine=engine,
            input_name=inputs[0],
            trt_config=trt_config,
            device=device,
            cuda_context=cuda_context,
            execution_context=execution_context,
            trt_cuda_graph_cache=establish_trt_cuda_graph_cache(
                default_cuda_graph_cache_size=default_trt_cuda_graph_cache_size,
                cuda_graph_cache=trt_cuda_graph_cache,
            ),
            metadata=package.metadata,
            class_names=package.class_names,
            output_indices=package.output_indices,
            image_pre_processing=package.inference_config.image_pre_processing,
            network_input=package.inference_config.network_input,
            resolution=input_shape[2],
        )

    def __init__(
        self,
        *,
        engine: "trt.ICudaEngine",
        input_name: str,
        trt_config: TRTConfig,
        device: torch.device,
        cuda_context: "cuda.Context",
        execution_context: "trt.IExecutionContext",
        trt_cuda_graph_cache: Optional[TRTCudaGraphCache],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._engine = engine
        self._input_name = input_name
        self._trt_config = trt_config
        self._device = device
        self._cuda_context = cuda_context
        self._execution_context = execution_context
        self._trt_cuda_graph_cache = trt_cuda_graph_cache
        self._lock = threading.Lock()

    def forward(
        self, pre_processed_images, disable_cuda_graphs: bool = False, **kwargs
    ) -> List[torch.Tensor]:
        """Run the engine on the normalized BCHW batch from ``pre_process``.

        Args:
            pre_processed_images: Normalized float32 BCHW array or tensor.
            disable_cuda_graphs: If True, skip CUDA graph replay for this call.

        Returns:
            Raw ``dets``, ``labels`` and ``heading`` tensors before selection.
        """
        images = torch.as_tensor(pre_processed_images).to(
            device=self._device, dtype=torch.float32
        )
        cache = self._trt_cuda_graph_cache if not disable_cuda_graphs else None
        with self._lock:
            with use_cuda_context(context=self._cuda_context):
                return infer_from_trt_engine(
                    pre_processed_images=images.contiguous(),
                    trt_config=self._trt_config,
                    engine=self._engine,
                    context=self._execution_context,
                    device=self._device,
                    input_name=self._input_name,
                    outputs=list(OUTPUT_NAMES),
                    stream=get_cuda_stream(device=self._device, purpose="inference"),
                    trt_cuda_graph_cache=cache,
                )
