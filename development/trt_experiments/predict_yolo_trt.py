import os
from typing import Tuple, List

import torchvision
from tqdm import tqdm

import tensorrt as trt
import requests
import cv2
import supervision as sv
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
    pre_processed_image = pre_process_image(image=image, device=DEVICE, target_size=(640, 640), batch_size=8)
    for _ in tqdm(range(1000), total=1000):
        results = perform_inference(pre_processed_image, engine, context, DEVICE)
        after_nms = run_nms(output=results)
    detections = sv.Detections(
        xyxy=((after_nms[0][:, :4].cpu().numpy() - (140, 0, 140, 0)) * 2).astype(np.int32),
        confidence=after_nms[0][:, 4].cpu().numpy(),
        class_id=after_nms[0][:, 5].cpu().numpy().astype(np.int32),
    )
    box_annotator = sv.BoxAnnotator()
    annotated = box_annotator.annotate(image.copy(), detections)
    cv2.imwrite("annotated.jpg", annotated)


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
    _, h, w = image_tensor.shape
    target_h, target_w = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    image_tensor = functional.resize(image_tensor, [new_h, new_w], interpolation=functional.InterpolationMode.BILINEAR)
    pad_top = max((target_h - new_h) // 2, 0)
    pad_bottom = max(target_h - new_h - pad_top, 0)
    pad_left = max((target_w - new_w) // 2, 0)
    pad_right = max(target_w - new_w - pad_left, 0)
    image_tensor = functional.pad(image_tensor, [pad_left, pad_top, pad_right, pad_bottom], fill=0.5)
    return image_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1).contiguous()


def perform_inference(
    pre_processed_image: torch.Tensor,
    engine,
    context,
    device: torch.device,
) -> torch.Tensor:
    batch_size = pre_processed_image.shape[0]
    output_tensor_shape = engine.get_tensor_shape("output0")
    output_tensor_type = trt_dtype_to_torch(engine.get_tensor_dtype("output0"))
    context.set_input_shape("images", tuple(pre_processed_image.shape))
    context.set_tensor_address("images", pre_processed_image.data_ptr())
    result = torch.empty((batch_size, ) + output_tensor_shape[1:], dtype=output_tensor_type, device=device)
    context.set_tensor_address("output0", result.data_ptr())
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


def run_nms(
    output: torch.Tensor,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    max_detections: int = 100,
    class_agnostic: bool = False,
) -> List[torch.Tensor]:
    bs = output.shape[0]
    boxes = output[:, :4, :]      # (N, 4, 8400)
    scores = output[:, 4:, :]     # (N, 80, 8400)

    results = []

    for b in range(bs):
        bboxes = boxes[b].T                    # (8400, 4)
        class_scores = scores[b].T             # (8400, 80)

        class_conf, class_ids = class_scores.max(1)  # (8400,), (8400,)

        mask = class_conf > conf_thresh
        if mask.sum() == 0:
            results.append(torch.zeros((0, 6), device=output.device))
            continue

        bboxes = bboxes[mask]
        class_conf = class_conf[mask]
        class_ids = class_ids[mask]
        # Convert [x, y, w, h] -> [x1, y1, x2, y2]
        xyxy = torch.zeros_like(bboxes)
        xyxy[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2  # x1
        xyxy[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2  # y1
        xyxy[:, 2] = bboxes[:, 0] + bboxes[:, 2] / 2  # x2
        xyxy[:, 3] = bboxes[:, 1] + bboxes[:, 3] / 2  # y2
        # Class-agnostic NMS -> use dummy class ids
        nms_class_ids = torch.zeros_like(class_ids) if class_agnostic else class_ids
        keep = torchvision.ops.batched_nms(xyxy, class_conf, nms_class_ids, iou_thresh)
        keep = keep[:max_detections]
        detections = torch.cat([
            xyxy[keep],
            class_conf[keep].unsqueeze(1),
            class_ids[keep].unsqueeze(1).float()
        ], dim=1)  # [x1, y1, x2, y2, conf, cls]

        results.append(detections)
    return results


if __name__ == '__main__':
    main()
