"""Image embeddings from pretrained vision backbones exported to ONNX.

One class serves DINOv2, DINOv3 and SigLIP2 packages. Each package holds the
graph (``model.onnx``, optionally with a ``model.onnx_data`` sidecar), the
verbatim Hugging Face ``preprocessor_config.json`` and an
``embedding_config.json`` naming the graph output that holds the vector.
"""

from threading import Lock
from typing import List, Optional, Tuple, Union

import numpy as np
import onnxruntime
import torch
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.entities import ColorFormat
from inference_models.errors import (
    CorruptedModelPackageError,
    EnvironmentConfigurationError,
    ModelInputError,
    ModelRuntimeError,
)
from inference_models.models.base.image_embeddings import ImageEmbeddingModel
from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.models.common.onnx import (
    align_device_with_onnx_session,
    create_onnx_inference_session,
    run_onnx_session_with_batch_size_limit,
    set_onnx_execution_provider_defaults,
)
from inference_models.models.common.streams import get_cuda_stream, use_cuda_stream
from inference_models.models.pretrained_image_embeddings.preprocessing import (
    CORRUPTED_PACKAGE_HELP_URL,
    ImageProcessorConfig,
    parse_image_processor_config,
    pre_process_images,
)
from inference_models.utils.file_system import read_json
from inference_models.utils.onnx_introspection import (
    get_selected_onnx_execution_providers,
)

FEATURE_VECTOR_OUTPUT_TYPE = "feature_vector"
MODEL_FILE = "model.onnx"
PREPROCESSOR_CONFIG_FILE = "preprocessor_config.json"
EMBEDDING_CONFIG_FILE = "embedding_config.json"


class EmbeddingSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repository: str
    revision: str
    file: str


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_name: str = Field(min_length=1)
    token_index: Optional[NonNegativeInt] = None
    dimension: PositiveInt
    feature_definition: str = Field(min_length=1)
    source: Optional[EmbeddingSource] = None


def parse_embedding_config(config_path: str) -> EmbeddingConfig:
    try:
        return EmbeddingConfig.model_validate(read_json(path=config_path))
    # pydantic's ValidationError is a ValueError, as are JSON decoding errors.
    except (IOError, OSError, ValueError) as error:
        raise CorruptedModelPackageError(
            message=f"`{EMBEDDING_CONFIG_FILE}` of the model package is invalid: {error}",
            help_url=CORRUPTED_PACKAGE_HELP_URL,
        ) from error


class PretrainedImageEmbeddingOnnx(ImageEmbeddingModel):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        onnx_execution_providers: Optional[List[Union[str, tuple]]] = None,
        default_onnx_trt_options: bool = True,
        device: torch.device = DEFAULT_DEVICE,
        max_batch_size: int = 32,
        **kwargs,
    ) -> "PretrainedImageEmbeddingOnnx":
        if onnx_execution_providers is None:
            onnx_execution_providers = get_selected_onnx_execution_providers()
        if not onnx_execution_providers:
            raise EnvironmentConfigurationError(
                message=f"Could not initialize model - selected backend is ONNX which requires execution provider to "
                f"be specified - explicitly in `from_pretrained(...)` method or via env variable "
                f"`ONNXRUNTIME_EXECUTION_PROVIDERS`. If you run model locally - adjust your setup, otherwise "
                f"contact the platform support.",
                help_url="https://inference-models.roboflow.com/errors/runtime-environment/#environmentconfigurationerror",
            )
        onnx_execution_providers = set_onnx_execution_provider_defaults(
            providers=onnx_execution_providers,
            model_package_path=model_name_or_path,
            device=device,
            default_onnx_trt_options=default_onnx_trt_options,
        )
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=[MODEL_FILE, PREPROCESSOR_CONFIG_FILE, EMBEDDING_CONFIG_FILE],
        )
        preprocessing = parse_image_processor_config(
            config_path=model_package_content[PREPROCESSOR_CONFIG_FILE]
        )
        embedding_config = parse_embedding_config(
            config_path=model_package_content[EMBEDDING_CONFIG_FILE]
        )
        # Loaded by path, so ONNX Runtime finds external weights next to the graph.
        session = create_onnx_inference_session(
            model_path=model_package_content[MODEL_FILE],
            providers=onnx_execution_providers,
        )
        device = align_device_with_onnx_session(session=session, device=device)
        input_name, static_batch_size = _resolve_graph_input(
            session=session, image_size=preprocessing.output_size
        )
        output_index = _resolve_graph_output(
            session=session, embedding_config=embedding_config
        )
        return cls(
            session=session,
            input_name=input_name,
            output_index=output_index,
            preprocessing=preprocessing,
            embedding_config=embedding_config,
            device=device,
            max_batch_size=static_batch_size or max_batch_size,
            min_batch_size=static_batch_size,
        )

    def __init__(
        self,
        session: onnxruntime.InferenceSession,
        input_name: str,
        output_index: int,
        preprocessing: ImageProcessorConfig,
        embedding_config: EmbeddingConfig,
        device: torch.device,
        max_batch_size: int,
        min_batch_size: Optional[int] = None,
    ):
        self._session = session
        self._input_name = input_name
        self._output_index = output_index
        self._preprocessing = preprocessing
        self._embedding_config = embedding_config
        self._device = device
        self._max_batch_size = max_batch_size
        self._min_batch_size = min_batch_size
        self._session_thread_lock = Lock()

    def prepare_image_embeddings(self, output_type: Optional[str] = None) -> None:
        self._resolve_embedding_output_type(output_type)

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        **kwargs,
    ) -> torch.Tensor:
        pre_process_stream = self._pre_process_stream
        with use_cuda_stream(pre_process_stream):
            pre_processed_images = pre_process_images(
                images=images,
                config=self._preprocessing,
                device=self._device,
                input_color_format=input_color_format,
            )
        if pre_process_stream is not None:
            pre_process_stream.synchronize()
        return pre_processed_images

    def forward_embedding(
        self, pre_processed_images: torch.Tensor, output_type: Optional[str] = None
    ) -> torch.Tensor:
        self._resolve_embedding_output_type(output_type)
        with self._session_thread_lock:
            outputs = run_onnx_session_with_batch_size_limit(
                session=self._session,
                inputs={self._input_name: pre_processed_images},
                min_batch_size=self._min_batch_size,
                max_batch_size=self._max_batch_size,
                stream=self._inference_stream,
            )
        embeddings = outputs[self._output_index]
        token_index = self._embedding_config.token_index
        if token_index is not None:
            if embeddings.ndim != 3 or token_index >= embeddings.shape[1]:
                raise ModelRuntimeError(
                    message=f"Model output `{self._embedding_config.output_name}` has shape "
                    f"{tuple(embeddings.shape)}, which has no token {token_index}.",
                    help_url="https://inference-models.roboflow.com/errors/models-runtime/#modelruntimeerror",
                )
            embeddings = embeddings[:, token_index, :]
        if (
            embeddings.ndim != 2
            or embeddings.shape[1] != self._embedding_config.dimension
        ):
            raise ModelRuntimeError(
                message=f"Model produced embeddings of shape {tuple(embeddings.shape)}, while the "
                f"package declares dimension {self._embedding_config.dimension}.",
                help_url="https://inference-models.roboflow.com/errors/models-runtime/#modelruntimeerror",
            )
        return embeddings.to(torch.float32).contiguous()

    def get_embedding_info(self, output_type: Optional[str] = None) -> dict:
        return {
            "feature_definition": self._embedding_config.feature_definition,
            "output_type": self._resolve_embedding_output_type(output_type),
            "normalization": "none",
        }

    @property
    def embedding_info(self) -> dict:
        return self.get_embedding_info()

    @property
    def embedding_preprocessing(self) -> dict:
        return self._preprocessing.describe()

    def _resolve_embedding_output_type(self, output_type: Optional[str]) -> str:
        if output_type not in (None, FEATURE_VECTOR_OUTPUT_TYPE):
            raise ModelInputError(
                message=f"Pretrained image embedding models produce only "
                f"`{FEATURE_VECTOR_OUTPUT_TYPE}` embeddings, requested `{output_type}`.",
                help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
            )
        return FEATURE_VECTOR_OUTPUT_TYPE

    @property
    def _pre_process_stream(self) -> Optional[torch.cuda.Stream]:
        return get_cuda_stream(device=self._device, purpose="pre-processing")

    @property
    def _inference_stream(self) -> Optional[torch.cuda.Stream]:
        return get_cuda_stream(device=self._device, purpose="inference")


def _resolve_graph_input(
    session: onnxruntime.InferenceSession, image_size: Tuple[int, int]
) -> Tuple[str, Optional[int]]:
    graph_inputs = session.get_inputs()
    if len(graph_inputs) != 1 or len(graph_inputs[0].shape) != 4:
        raise CorruptedModelPackageError(
            message="Pretrained image embedding graph must take a single NCHW image input.",
            help_url=CORRUPTED_PACKAGE_HELP_URL,
        )
    batch_size, _, height, width = graph_inputs[0].shape
    if isinstance(height, int) and isinstance(width, int):
        if (height, width) != tuple(image_size):
            raise CorruptedModelPackageError(
                message=f"Graph input expects images of {height}x{width}, while "
                f"`{PREPROCESSOR_CONFIG_FILE}` produces {image_size[0]}x{image_size[1]}.",
                help_url=CORRUPTED_PACKAGE_HELP_URL,
            )
    if not isinstance(batch_size, int):
        batch_size = None
    return graph_inputs[0].name, batch_size


def _resolve_graph_output(
    session: onnxruntime.InferenceSession, embedding_config: EmbeddingConfig
) -> int:
    output_names = [output.name for output in session.get_outputs()]
    if embedding_config.output_name not in output_names:
        raise CorruptedModelPackageError(
            message=f"Graph has no output `{embedding_config.output_name}` named in "
            f"`{EMBEDDING_CONFIG_FILE}`; available outputs: {output_names}.",
            help_url=CORRUPTED_PACKAGE_HELP_URL,
        )
    output_index = output_names.index(embedding_config.output_name)
    shape = session.get_outputs()[output_index].shape
    expected_rank = 2 if embedding_config.token_index is None else 3
    if len(shape) != expected_rank:
        raise CorruptedModelPackageError(
            message=f"Graph output `{embedding_config.output_name}` has rank {len(shape)}, "
            f"expected {expected_rank} for token_index {embedding_config.token_index}.",
            help_url=CORRUPTED_PACKAGE_HELP_URL,
        )
    if isinstance(shape[-1], int) and shape[-1] != embedding_config.dimension:
        raise CorruptedModelPackageError(
            message=f"Graph output `{embedding_config.output_name}` has dimension {shape[-1]}, "
            f"while `{EMBEDDING_CONFIG_FILE}` declares {embedding_config.dimension}.",
            help_url=CORRUPTED_PACKAGE_HELP_URL,
        )
    return output_index
