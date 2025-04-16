import os
from typing import Tuple

from tqdm import tqdm

import tensorrt as trt
import requests
import cv2
import numpy as np
import torch
from torchvision.transforms import functional

IMAGE_URL = "https://media.roboflow.com/dog.jpeg"
MODEL_PATH = os.environ["MODEL_PATH"]
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
DEVICE = torch.device("cuda:0")


def main() -> None:
    image = load_image(image_url=IMAGE_URL)
    engine = load_model(model_path=MODEL_PATH)
    context = engine.create_execution_context()
    pre_processed_image = pre_process_image(image=image, device=DEVICE, target_size=(224, 224), batch_size=64)
    for _ in tqdm(range(1000), total=1000):
        results = perform_inference(pre_processed_image, engine, context, DEVICE)
    print(results.shape)


def load_image(image_url: str) -> np.ndarray:
    response = requests.get(image_url)
    response.raise_for_status()
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)


def load_model(model_path: str):
    with open(model_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def pre_process_image(
    image: np.ndarray,
    device: torch.device,
    target_size: Tuple[int, int],
    batch_size: int,
) -> torch.Tensor:
    image = image[:, :, ::-1].copy()  # BGR to RGB
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # (C, H, W)
    image_tensor = image_tensor.to(device)
    image_tensor = functional.resize(image_tensor, list(target_size), interpolation=functional.InterpolationMode.BILINEAR)
    return image_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1).contiguous()


def perform_inference(
    pre_processed_image: torch.Tensor,
    engine,
    context,
    device: torch.device,
) -> torch.Tensor:
    batch_size = pre_processed_image.shape[0]
    output_tensor_shape = engine.get_tensor_shape("prediction")
    output_tensor_type = trt_dtype_to_torch(engine.get_tensor_dtype("prediction"))
    context.set_input_shape("images", tuple(pre_processed_image.shape))
    context.set_tensor_address("images", pre_processed_image.data_ptr())
    result = torch.empty((batch_size, ) + output_tensor_shape[1:], dtype=output_tensor_type, device=device)
    context.set_tensor_address("prediction", result.data_ptr())
    stream = torch.cuda.Stream(device=DEVICE)
    status = context.execute_async_v3(stream_handle=stream.cuda_stream)
    if not status:
        raise RuntimeError("Error with inference!")
    stream.synchronize()
    return result


def trt_dtype_to_torch(trt_dtype):
    return {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT8: torch.int8,
        trt.DataType.BOOL: torch.bool,
    }[trt_dtype]


if __name__ == '__main__':
    main()
