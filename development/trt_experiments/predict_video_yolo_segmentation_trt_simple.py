import os
import time
from queue import Queue
from threading import Thread
from typing import Tuple, List, Optional, Union

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
import torchvision.transforms.functional as F
from datetime import datetime

from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.entities import AnyPrediction

MODEL_MAX_BATCH_SIZE = 1
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
DEVICE = torch.device("cuda:0")
MODEL_PATH = os.environ["MODEL_PATH"]
VIDEO_REFERENCE = os.environ["VIDEO_REFERENCE"]
WITH_NMS = bool(int(os.getenv("WITH_NMS", "1")))


def main() -> None:
    model = SingleStageModel(detector_path=MODEL_PATH)
    pipeline = InferencePipeline.init_with_custom_logic(
        video_reference=[VIDEO_REFERENCE] * 1,
        on_video_frame=model.on_video_frame,
        on_prediction=model.on_prediction,
        batch_collection_timeout=0.01,
    )
    pipeline.start()
    pipeline.join()


class SingleStageModel:

    def __init__(
        self,
        detector_path: str,
    ):
        self._detector_engine = load_model(model_path=detector_path)
        self._detector_context = self._detector_engine.create_execution_context()
        self._monitor = sv.FPSMonitor(sample_size=128)

    def on_video_frame(self, frames: List[VideoFrame]) -> list:
        frames_tensors = [torch.from_numpy(frame.image).to(device=DEVICE) for frame in frames]
        return run_processing(
            images=frames_tensors,
            detector_engine=self._detector_engine,
            detector_context=self._detector_context,
            device=DEVICE,
        )

    def on_prediction(
        self,
        predictions: Union[List[Optional[AnyPrediction]], AnyPrediction],
        frames: Union[List[Optional[VideoFrame]], VideoFrame],
    ) -> None:
        if not isinstance(frames, list):
            frames = [frames]
            predictions = [predictions]
        for _ in predictions:
            self._monitor.tick()
        print(f"FPS: {self._monitor.fps}, CROPS: {sum(len(p[0]) for p in predictions)} from {len(frames)} frames")


def run_processing(
    images: List[torch.Tensor],
    detector_engine,
    detector_context,
    device: torch.device,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    start_a = time.monotonic()
    pre_processed_images, images_metadata = preprocess_images_for_segmentation_model(images, (1280, 1280))
    end_a = time.monotonic()
    print(f"DETECTOR PRE-PROCESSING: {round((end_a - start_a) * 1000, 2)}ms")
    results, protos = [], []
    for i in range(0, pre_processed_images.shape[0], MODEL_MAX_BATCH_SIZE):
        start_b = time.monotonic()
        batch = pre_processed_images[i:i + MODEL_MAX_BATCH_SIZE].contiguous()
        end_b = time.monotonic()
        print(f"SLICING: {round((end_b - start_b) * 1000, 2)}ms - {len(batch)} items")
        start_b1 = time.monotonic()
        batch_results, batch_protos = perform_inference_from_segmentation_model(
            batch,
            engine=detector_engine,
            context=detector_context,
            device=device,
        )
        results.append(batch_results)
        protos.append(batch_protos)
        end_b1 = time.monotonic()
        print(f"DETECTOR BATCH INFERENCE: {round((end_b1 - start_b1) * 1000, 2)}ms - {len(batch)} items")
    start_c = time.monotonic()
    detections = torch.cat(results, dim=0)
    protos = torch.cat(protos, dim=0)
    end_c = time.monotonic()
    print(f"DETECTOR RESULTS CONSOLIDATION: {round((end_c - start_c) * 1000, 2)}ms")
    start_d = time.monotonic()
    if WITH_NMS:
        detections_after_nms = run_nms(detections)
    else:
        detections_after_nms = post_process_detections(detections)
    end_d = time.monotonic()
    print(f"NMS: {round((end_d - start_d) * 1000, 2)}ms")
    final_results = []
    start_e = time.monotonic()
    for image_bboxes, image_protos, image_meta in zip(detections_after_nms, protos, images_metadata):
        pre_processed_masks = preprocess_segmentation_masks(protos=image_protos, masks_in=image_bboxes[:, 6:])
        cropped_masks = crop_masks_to_boxes(image_bboxes[:, :4], pre_processed_masks)
        aligned_boxes, aligned_masks = align_results(
            image_bboxes,
            cropped_masks,
            (image_meta["pad_left"], image_meta["pad_top"], image_meta["pad_right"], image_meta["pad_bottom"]),
            image_meta["scale"],
            image_meta["original_size"],
            image_meta["inference_size"],
        )
        final_results.append((aligned_boxes, aligned_masks))
    end_e = time.monotonic()
    print(f"DETECTIONS RESCALING: {round((end_e - start_e) * 1000, 2)}ms")
    return final_results


def preprocess_segmentation_masks(
    protos: torch.Tensor,
    masks_in: torch.Tensor
) -> torch.Tensor:
    masks_sum = torch.einsum('chw,nc->nhw', protos, masks_in)
    masks_sigmoid = torch.sigmoid(masks_sum)
    return masks_sigmoid > 0.5


def crop_masks_to_boxes(
    boxes: torch.Tensor,
    masks: torch.Tensor,
    scaling: float = 0.25,
) -> torch.Tensor:
    n, h, w = masks.shape
    scaled_boxes = (boxes * scaling)
    x1, y1, x2, y2 = scaled_boxes[:, 0][:, None, None], scaled_boxes[:, 1][:, None, None], \
        scaled_boxes[:, 2][:, None, None], scaled_boxes[:, 3][:, None, None]

    rows = torch.arange(w)[None, None, :].to(DEVICE)  # shape: [1, 1, w]
    cols = torch.arange(h)[None, :, None].to(DEVICE)  # shape: [1, h, 1]

    crop_mask = (rows >= x1) & (rows < x2) & (cols >= y1) & (cols < y2)
    return masks * crop_mask


def align_results(
    image_bboxes: torch.Tensor,
    masks: torch.Tensor,
    padding: Tuple[int, int, int, int],
    scale: int,
    original_size: Tuple[int, int],
    inference_size: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    pad_left, pad_top, pad_right, pad_bottom = padding
    offsets = torch.tensor(
        [pad_left, pad_top, pad_left, pad_top],
        device=DEVICE,
    )
    image_bboxes[:, :4] -= offsets
    image_bboxes[:, :4] *= 1 / scale
    print(masks.shape)
    n, mh, mw = masks.shape
    inf_h, inf_w = inference_size
    mask_h_scale = mh / inf_h
    mask_w_scale = mw / inf_w
    mask_pad_top, mask_pad_bottom, mask_pad_left, mask_pad_right = \
        round(mask_h_scale * pad_top), round(mask_h_scale * pad_bottom), \
        round(mask_w_scale * pad_left), round(mask_w_scale * pad_right)
    masks = masks[:, mask_pad_top:mh-mask_pad_bottom, mask_pad_left:mw-mask_pad_right]
    masks = functional.resize(masks, list(original_size), interpolation=functional.InterpolationMode.BILINEAR)
    return image_bboxes, masks


def preprocess_images_for_segmentation_model(
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
        metadata.append({
            "pad_left": pad_left,
            "pad_top": pad_top,
            "pad_bottom": pad_bottom,
            "pad_right": pad_right,
            "scale": scale,
            "original_size": tuple(image.shape[:2]),
            "inference_size": target_size,
        })
    return torch.stack(scaled_images).contiguous(), metadata


def preprocess_images_optimized_pytorch(
    images: List[torch.Tensor],  # List of HWC, BGR, uint8/float32 tensors on GPU
    target_size: Tuple[int, int],
    device: torch.device = DEVICE,
) -> Tuple[torch.Tensor, List[dict]]:
    target_h, target_w = target_size
    num_images = len(images)

    # Pre-allocate the output batch tensor on the correct device, filled with the pad value
    # Your original code pads with 0.5 (normalized gray)
    # The input images are BGR, HWC. We need to convert to RGB, CHW, normalize, resize, then place.

    final_batch = torch.full(
        (num_images, 3, target_h, target_w),
        0.5,  # Normalized padding value
        dtype=torch.float32,
        device=device
    )
    metadata = []  # Still need this for per-image info

    # Batch calculate shapes, scales, and padding if possible (CPU part)
    # This reduces redundant calculations inside the loop if shapes are known.
    # If images is a list of tensors with varying shapes, this is a good approach.
    original_shapes = torch.tensor([[img.shape[0], img.shape[1]] for img in images],
                                   dtype=torch.float32)  # Nx2 tensor of [h, w]

    # Calculate scales
    # scale = min(target_w / w, target_h / h)
    scale_w = target_w / original_shapes[:, 1]
    scale_h = target_h / original_shapes[:, 0]
    scales = torch.minimum(scale_w, scale_h)

    # Calculate new dimensions after scaling
    new_ws = (original_shapes[:, 1] * scales).int()
    new_hs = (original_shapes[:, 0] * scales).int()

    # Calculate padding amounts (top-left for placement)
    pad_tops = ((target_h - new_hs) / 2).int()
    pad_lefts = ((target_w - new_ws) / 2).int()

    for i in range(num_images):
        image_hwc_bgr = images[i].to(device)  # Ensure on correct device

        # 1. Convert to float if it's uint8
        if image_hwc_bgr.dtype == torch.uint8:
            image_hwc_bgr = image_hwc_bgr.float()

        # 2. BGR to RGB
        image_hwc_rgb = image_hwc_bgr[..., [2, 1, 0]]

        # 3. HWC to CHW and Normalize
        image_chw_normalized = image_hwc_rgb.permute(2, 0, 1) / 255.0

        # Use pre-calculated new dimensions
        new_h_i, new_w_i = new_hs[i].item(), new_ws[i].item()

        # 4. Resize
        # Antialias=True is generally recommended for better quality when downscaling with BILINEAR.
        # In torchvision>=0.13, it's default for BILINEAR/BICUBIC when downscaling if not specified.
        # Explicitly set for clarity and compatibility.
        resized_chw = F.resize(
            image_chw_normalized,
            [new_h_i, new_w_i],
            interpolation=F.InterpolationMode.BILINEAR,
            antialias=True
        )

        # Use pre-calculated padding
        pad_top_i, pad_left_i = pad_tops[i].item(), pad_lefts[i].item()

        # 5. Place the resized image into the pre-allocated batch
        final_batch[i, :, pad_top_i: pad_top_i + new_h_i, pad_left_i: pad_left_i + new_w_i] = resized_chw

        metadata.append({
            "pad_left": pad_left_i,
            "pad_top": pad_top_i,
            "scale": scales[i].item(),
            "original_shape": (original_shapes[i, 0].item(), original_shapes[i, 1].item()),
            "inference_shape": (new_h_i, new_w_i),
        })

    return final_batch.contiguous(), metadata


def perform_inference_from_segmentation_model(
    pre_processed_image: torch.Tensor,
    engine,
    context,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = pre_processed_image.shape[0]
    output0_tensor_shape = engine.get_tensor_shape("output0")
    output1_tensor_shape = engine.get_tensor_shape("output1")
    output0_tensor_type = trt_dtype_to_torch(engine.get_tensor_dtype("output0"))
    output1_tensor_type = trt_dtype_to_torch(engine.get_tensor_dtype("output1"))
    context.set_input_shape("images", tuple(pre_processed_image.shape))
    context.set_tensor_address("images", pre_processed_image.data_ptr())
    result0 = torch.empty((batch_size, ) + output0_tensor_shape[1:], dtype=output0_tensor_type, device=device)
    result1 = torch.empty((batch_size,) + output1_tensor_shape[1:], dtype=output1_tensor_type, device=device)
    context.set_tensor_address("output0", result0.data_ptr())
    context.set_tensor_address("output1", result1.data_ptr())
    stream = torch.cuda.Stream(device=DEVICE)
    status = context.execute_async_v3(stream_handle=stream.cuda_stream)
    if not status:
        raise RuntimeError("Error with inference!")
    stream.synchronize()
    return result0, result1


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
    boxes = output[:, :4, :]  # (N, 4, 8400)
    scores = output[:, 4:-32, :]  # (N, 80, 8400)
    masks = output[:, -32:, :]
    results = []

    for b in range(bs):
        bboxes = boxes[b].T  # (8400, 4)
        class_scores = scores[b].T  # (8400, 80)
        box_masks = masks[b].T

        class_conf, class_ids = class_scores.max(1)  # (8400,), (8400,)

        mask = class_conf > conf_thresh
        if mask.sum() == 0:
            results.append(torch.zeros((0, 38), device=output.device))
            continue

        bboxes = bboxes[mask]
        class_conf = class_conf[mask]
        class_ids = class_ids[mask]
        box_masks = box_masks[mask]
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
            class_ids[keep].unsqueeze(1).float(),
            box_masks[keep]
        ], dim=1)  # [x1, y1, x2, y2, conf, cls]

        results.append(detections)
    return results


def post_process_detections(
    output: torch.Tensor,
    conf_thresh: float = 0.25,
    max_detections: int = 100,
) -> List[torch.Tensor]:
    results = []
    for batch_out in output:
        print(batch_out.shape)
        # Apply confidence threshold
        mask = batch_out[:, 4] > conf_thresh
        if mask.sum() == 0:
            results.append(torch.zeros((0, 6), device=output.device))
            continue
        filtered = batch_out[mask]

        # Limit to max detections
        filtered = filtered[:max_detections]
        results.append(filtered)
        # export bug all of the elements other than first are zeroed!
        return [filtered.clone() for _ in range(output.shape[0])]
    return results


def load_model(model_path: str):
    with open(model_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())



if __name__ == '__main__':
    main()