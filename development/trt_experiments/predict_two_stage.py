import os
import time
from typing import Tuple, List

import torchvision
from torchvision.ops import roi_align
from tqdm import tqdm

import tensorrt as trt
import requests
import cv2
import supervision as sv
import numpy as np
import torch
from torchvision.transforms import functional

IMAGE_URL = "https://media.roboflow.com/dog.jpeg"
IMAGE_PATH = os.getenv("IMAGE_PATH")
DETECTOR_PATH = os.environ["DETECTOR_PATH"]
CLASSIFIER_PATH = os.environ["CLASSIFIER_PATH"]
DETECTOR_MAX_BATCH_SIZE = 16
CLASSIFIER_MAX_BATCH_SIZE = int(os.getenv("CLASSIFIER_MAX_BATCH_SIZE", "64"))
CLASSIFIER_INPUT_SIZE = int(os.getenv("CLASSIFIER_INPUT_SIZE", "224"))
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
DEVICE = torch.device("cuda:0")


def main() -> None:
    if IMAGE_PATH is None:
        image = load_image(image_url=IMAGE_URL)
    else:
        image = cv2.imread(IMAGE_PATH)
    images = [torch.from_numpy(image).to(device=DEVICE) for _ in range(16)]
    detector_engine = load_model(model_path=DETECTOR_PATH)
    detector_context = detector_engine.create_execution_context()
    classifier_engine = load_model(model_path=CLASSIFIER_PATH)
    classifier_context = classifier_engine.create_execution_context()
    results = None
    for _ in tqdm(range(100), total=100):
        results = run_processing(
            images=images,
            detector_engine=detector_engine,
            detector_context=detector_context,
            classifier_engine=classifier_engine,
            classifier_context=classifier_context,
            device=DEVICE
        )
    print(results)


def load_image(image_url: str) -> np.ndarray:
    response = requests.get(image_url)
    response.raise_for_status()
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)


def load_model(model_path: str):
    with open(model_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def run_processing(
    images: List[torch.Tensor],
    detector_engine,
    detector_context,
    classifier_engine,
    classifier_context,
    device: torch.device,
) -> List[torch.Tensor]:
    start_a = time.monotonic()
    pre_processed_images, images_metadata = preprocess_images_for_detector(images, (640, 640))
    end_a = time.monotonic()
    print(f"DETECTOR PRE-PROCESSING: {round((end_a - start_a) * 1000, 2)}ms")
    results = []
    for i in range(0, pre_processed_images.shape[0], DETECTOR_MAX_BATCH_SIZE):
        batch = pre_processed_images[i:i+DETECTOR_MAX_BATCH_SIZE]
        start_b = time.monotonic()
        batch_results = perform_inference_from_detector(
            batch,
            engine=detector_engine,
            context=detector_context,
            device=device,
        )
        end_b = time.monotonic()
        print(f"DETECTOR BATCH INFERENCE: {round((end_b - start_b) * 1000, 2)}ms - {len(batch)} items")
        results.append(batch_results)
    start_c = time.monotonic()
    detections = torch.cat(results, dim=0)
    end_c = time.monotonic()
    print(f"DETECTOR RESULTS CONSOLIDATION: {round((end_c - start_c) * 1000, 2)}ms")
    start_d = time.monotonic()
    detections_after_nms = run_nms(detections)
    end_d = time.monotonic()
    print(f"NMS: {round((end_d - start_d) * 1000, 2)}ms")
    start_e = time.monotonic()
    rescaled_detections = rescale_detections(detections_after_nms, images_metadata)
    end_e = time.monotonic()
    print(f"DETECTIONS RESCALING: {round((end_e - start_e) * 1000, 2)}ms")
    start_f = time.monotonic()
    crops, crops_metadata = crop_and_resize_fast(
        images=images,
        detections=rescaled_detections,
        target_size=(CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE),
    )
    end_f = time.monotonic()
    print(f"CLASSIFIER PRE-PROCESSING: {round((end_f - start_f) * 1000, 2)}ms")
    classifier_results = []
    for i in range(0, crops.shape[0], CLASSIFIER_MAX_BATCH_SIZE):
        batch = crops[i:i+CLASSIFIER_MAX_BATCH_SIZE]
        start_g = time.monotonic()
        batch_results = perform_inference_from_classifier(
            batch,
            engine=classifier_engine,
            context=classifier_context,
            device=device,
        )
        end_g = time.monotonic()
        print(f"CLASSIFIER BATCH INFERENCE: {round((end_g - start_g) * 1000, 2)}ms - {len(batch)} items")
        classifier_results.append(batch_results)
    start_h = time.monotonic()
    all_classification_results = torch.cat(classifier_results, dim=0)
    scaled_probabs = torch.nn.functional.softmax(all_classification_results, dim=1)
    max_probs, class_indices = torch.max(scaled_probabs, dim=1)
    concatenated = torch.stack((max_probs, class_indices), dim=1)
    end_h = time.monotonic()
    print(f"CLASSIFIER POST-PROCESSING: {round((end_h - start_h) * 1000, 2)}ms")
    start_i = time.monotonic()
    offset = 0
    for image_detections in rescaled_detections:
        image_detections[:, 4:] = concatenated[offset:offset+len(image_detections), :]
        offset += len(image_detections)
    end_i = time.monotonic()
    print(f"CLASSES REPLACEMENT: {round((end_i - start_i) * 1000, 2)}ms")
    return rescaled_detections


def preprocess_images_for_detector(
    images: List[torch.Tensor],
    target_size: Tuple[int, int],
) -> Tuple[torch.Tensor, List[dict]]:
    scaled_images = []
    metadata = []
    for image in images:
        image_rgb = image[..., [2, 1, 0]]
        rgb_chw = image_rgb.permute(2, 0, 1) / 255.0
        _, h, w = rgb_chw.shape
        target_h, target_w = target_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        image_tensor = functional.resize(rgb_chw, [new_h, new_w], interpolation=functional.InterpolationMode.BILINEAR)
        pad_top = max((target_h - new_h) // 2, 0)
        pad_bottom = max(target_h - new_h - pad_top, 0)
        pad_left = max((target_w - new_w) // 2, 0)
        pad_right = max(target_w - new_w - pad_left, 0)
        image_tensor = functional.pad(image_tensor, [pad_left, pad_top, pad_right, pad_bottom], fill=0.5)
        scaled_images.append(image_tensor)
        metadata.append({"pad_left": pad_left, "pad_top": pad_top, "scale": scale})
    return torch.stack(scaled_images).contiguous(), metadata


def perform_inference_from_detector(
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


def perform_inference_from_classifier(
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


def rescale_detections(detections: List[torch.Tensor], images_metadata: List[dict]) -> List[torch.Tensor]:
    for image_detections, metadata in zip(detections, images_metadata):
        offsets = torch.tensor(
            [metadata["pad_left"], metadata["pad_top"], metadata["pad_left"], metadata["pad_top"]],
            dtype=image_detections.dtype,
            device=image_detections.device
        )
        image_detections[:, :4] -= offsets
        image_detections[:, :4] *= 1 / metadata["scale"]
    return detections


def crop_and_resize(
    images: List[torch.Tensor],
    detections: List[torch.Tensor],
    target_size: Tuple[int, int],
) -> Tuple[torch.Tensor, List[dict]]:
    pre_processed_images = []
    metadata = []
    shapes = []
    for i, (image, image_detections) in enumerate(zip(images, detections)):
        image_rgb = (image[..., [2, 1, 0]]).permute(2, 0, 1) / 255.0
        for j, detection in enumerate(image_detections):
            x_min, y_min, x_max, y_max = detection[:4].to(torch.int)
            crop = image_rgb[:, y_min:y_max, x_min:x_max]
            scaled_crop = functional.resize(crop, list(target_size), interpolation=functional.InterpolationMode.BILINEAR)
            shapes.append(crop.shape)
            pre_processed_images.append(scaled_crop)
            metadata.append({"image_id": i, "detection_id": j})
    avg_h = round(sum([s[1] for s in shapes]) / len(shapes))
    avg_w = round(sum([s[2] for s in shapes]) / len(shapes))
    print(f"AVG DETECTION SIZE: {avg_h}x{avg_w}")
    return torch.stack(pre_processed_images).contiguous(), metadata


def crop_and_resize_fast(
    images: List[torch.Tensor],
    detections: List[torch.Tensor],
    target_size: Tuple[int, int],
) -> Tuple[torch.Tensor, List[dict]]:
    # Prepare data
    device = images[0].device
    all_images = []
    all_boxes = []
    batch_indices = []
    metadata = []

    for i, (image, image_detections) in enumerate(zip(images, detections)):
        image_rgb = (image[..., [2, 1, 0]]).permute(2, 0, 1) / 255.0
        num_detections = image_detections.shape[0]

        all_images.append(image_rgb.unsqueeze(0))  # add batch dim
        all_boxes.append(image_detections[:, :4])  # xyxy
        batch_indices.append(torch.full((num_detections,), i, dtype=torch.int, device=device))

        for j in range(num_detections):
            metadata.append({"image_id": i, "detection_id": j})

    all_images = torch.cat(all_images, dim=0)  # (B, 3, H, W)
    all_boxes = torch.cat(all_boxes, dim=0)    # (N, 4)
    batch_indices = torch.cat(batch_indices, dim=0)  # (N,)
    # Build rois: (batch_idx, x1, y1, x2, y2)
    rois = torch.cat([batch_indices.unsqueeze(1).float(), all_boxes], dim=1)  # (N, 5)
    # Apply roi_align
    crops = roi_align(
        input=all_images,
        boxes=rois,
        output_size=target_size,
        aligned=True
    )
    return crops.contiguous(), metadata


if __name__ == '__main__':
    main()
