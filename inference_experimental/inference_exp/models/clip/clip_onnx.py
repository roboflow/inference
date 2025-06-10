from threading import Lock
from typing import List, Optional, Union

import clip
import numpy as np
import onnxruntime
import torch
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
)

from inference_exp.configuration import DEFAULT_DEVICE, ONNXRUNTIME_EXECUTION_PROVIDERS
from inference_exp.entities import ColorFormat
from inference_exp.errors import EnvironmentConfigurationError, ModelRuntimeError
from inference_exp.models.base.embeddings import TextImageEmbeddingModel
from inference_exp.models.common.model_packages import get_model_package_contents
from inference_exp.models.common.onnx import (
    run_session_via_iobinding,
    set_execution_provider_defaults,
)

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


class ClipOnnx(TextImageEmbeddingModel):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        onnx_execution_providers: Optional[List[Union[str, tuple]]] = None,
        default_onnx_trt_options: bool = True,
        device: torch.device = DEFAULT_DEVICE,
        max_batch_size: int = 32,
        **kwargs,
    ) -> "ClipOnnx":
        if onnx_execution_providers is None:
            onnx_execution_providers = ONNXRUNTIME_EXECUTION_PROVIDERS
        if not onnx_execution_providers:
            raise EnvironmentConfigurationError(
                f"Could not initialize model - selected backend is ONNX which requires execution provider to "
                f"be specified - explicitly in `from_pretrained(...)` method or via env variable "
                f"`ONNXRUNTIME_EXECUTION_PROVIDERS`. If you run model locally - adjust your setup, otherwise "
                f"contact the platform support."
            )
        onnx_execution_providers = set_execution_provider_defaults(
            providers=onnx_execution_providers,
            model_package_path=model_name_or_path,
            device=device,
            default_onnx_trt_options=default_onnx_trt_options,
        )
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=[
                "textual.onnx",
                "visual.onnx",
            ],
        )
        visual_onnx_session = onnxruntime.InferenceSession(
            path_or_bytes=model_package_content["visual.onnx"],
            providers=onnx_execution_providers,
        )
        textual_onnx_session = onnxruntime.InferenceSession(
            path_or_bytes=model_package_content["textual.onnx"],
            providers=onnx_execution_providers,
        )
        image_size = visual_onnx_session.get_inputs()[0].shape[2]
        visual_input_name = visual_onnx_session.get_inputs()[0].name
        textual_input_name = textual_onnx_session.get_inputs()[0].name
        return cls(
            visual_onnx_session=visual_onnx_session,
            textual_onnx_session=textual_onnx_session,
            image_size=image_size,
            visual_input_name=visual_input_name,
            textual_input_name=textual_input_name,
            device=device,
            max_batch_size=max_batch_size,
        )

    def __init__(
        self,
        visual_onnx_session: onnxruntime.InferenceSession,
        textual_onnx_session: onnxruntime.InferenceSession,
        image_size: int,
        visual_input_name: str,
        textual_input_name: str,
        device: torch.device,
        max_batch_size: int,
    ):
        self._visual_onnx_session = visual_onnx_session
        self._textual_onnx_session = textual_onnx_session
        self._image_size = image_size
        self._visual_input_name = visual_input_name
        self._textual_input_name = textual_input_name
        self._device = device
        self._max_batch_size = max_batch_size
        self._visual_session_thread_lock = Lock()
        self._textual_session_thread_lock = Lock()
        self._torch_transforms = Compose(
            [
                Resize(self._image_size, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(self._image_size),
                lambda x: x / 255,
                Normalize(MEAN, STD),
            ]
        )

    def embed_images(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        **kwargs,
    ) -> torch.Tensor:
        pre_processed_images = self._pre_process_images(
            images=images, input_color_format=input_color_format
        )
        with self._visual_session_thread_lock:
            if pre_processed_images.shape[0] <= self._max_batch_size:
                return run_session_via_iobinding(
                    session=self._visual_onnx_session,
                    input_name=self._visual_input_name,
                    inputs=pre_processed_images,
                )[0]
        results = []
        for i in range(0, pre_processed_images.shape[0], self._max_batch_size):
            batch_input = pre_processed_images[
                i : i + self._max_batch_size
            ].contiguous()
            batch_results = run_session_via_iobinding(
                session=self._visual_onnx_session,
                input_name=self._visual_input_name,
                inputs=batch_input,
            )[0]
            results.append(batch_results)
        return torch.cat(results, dim=0)

    def embed_text(self, texts: Union[str, List[str]], **kwargs) -> torch.Tensor:
        if not isinstance(texts, list):
            texts = [texts]
        tokenized_batch = clip.tokenize(texts)
        with self._textual_session_thread_lock:
            if tokenized_batch.shape[0] <= self._max_batch_size:
                return run_session_via_iobinding(
                    session=self._textual_onnx_session,
                    input_name=self._textual_input_name,
                    inputs=tokenized_batch,
                )[0]
        results = []
        for i in range(0, tokenized_batch.shape[0], self._max_batch_size):
            batch_input = tokenized_batch[i : i + self._max_batch_size].contiguous()
            batch_results = run_session_via_iobinding(
                session=self._textual_onnx_session,
                input_name=self._textual_input_name,
                inputs=batch_input,
            )[0]
            results.append(batch_results)
        return torch.cat(results, dim=0)

    def _pre_process_images(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat],
    ) -> torch.Tensor:
        if isinstance(images, torch.Tensor):
            input_color_format = input_color_format or "rgb"
            images = images.to(self._device).unsqueeze(dim=0)
            if input_color_format != "rgb":
                images = images[:, [2, 1, 0], :, :]
            return self._torch_transforms(images.float())
        if isinstance(images, np.ndarray):
            input_color_format = input_color_format or "bgr"
            if input_color_format != "rgb":
                images = np.ascontiguousarray(images[:, :, ::-1])
            images = torch.from_numpy(images).permute(2, 0, 1).to(self._device)
            return self._torch_transforms(images).unsqueeze(dim=0)
        if not len(images):
            raise ModelRuntimeError("Detected empty input to the model")
        if isinstance(images[0], np.ndarray):
            input_color_format = input_color_format or "bgr"
            results = []
            for image in images:
                if input_color_format != "rgb":
                    image = np.ascontiguousarray(image[:, :, ::-1])
                image = torch.from_numpy(image).permute(2, 0, 1).to(self._device)
                preprocessed_image = self._torch_transforms(image)
                results.append(preprocessed_image)
            return torch.stack(results, dim=0).contiguous()
        if isinstance(images[0], torch.Tensor):
            input_color_format = input_color_format or "rgb"
            results = []
            for image in images:
                image = image.to(device=self._device).unsqueeze(dim=0)
                if input_color_format != "rgb":
                    image = image[:, [2, 1, 0], :, :]
                preprocessed_image = self._torch_transforms(image.float())
                results.append(preprocessed_image)
            return torch.cat(results, dim=0).contiguous()
        raise ModelRuntimeError(
            f"Detected unknown input batch element: {type(images[0])}"
        )
