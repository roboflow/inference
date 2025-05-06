import os
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread
from typing import Tuple, List
from uuid import uuid4

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
IMAGE_PATH = os.getenv("IMAGE_PATH")
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
DEVICE = torch.device("cuda:0")


def main() -> None:
    if IMAGE_PATH is None:
        image = load_image(image_url=IMAGE_URL)
    else:
        image = cv2.imread(IMAGE_PATH)
    engine = load_model(model_path=MODEL_PATH)
    context = engine.create_execution_context()
    # print("SINGULAR OPERATIONS")
    # pre_processed_image = pre_process_image(image=image, device=DEVICE, target_size=(640, 640), batch_size=1)
    # pre_processing_times = []
    # for _ in tqdm(range(1000), total=1000):
    #     start = time.monotonic()
    #     pre_processed_image = pre_process_image(image=image, device=DEVICE, target_size=(640, 640), batch_size=1)
    #     end = time.monotonic()
    #     pre_processing_times.append((end - start))
    # print(f"PRE PROCESSING MEDIAN: {np.round((np.median(pre_processing_times) * 1000), 2)}ms")
    # inference_times = []
    # for _ in tqdm(range(1000), total=1000):
    #     start = time.monotonic()
    #     results = perform_inference(pre_processed_image, engine, context, DEVICE)
    #     end = time.monotonic()
    #     inference_times.append((end - start))
    # print(f"INFERENCE MEDIAN: {np.round((np.median(inference_times) * 1000), 2)}ms")
    # post_processing_time_times = []
    # for _ in tqdm(range(1000), total=1000):
    #     start = time.monotonic()
    #     after_nms = run_nms(output=results)
    #     end = time.monotonic()
    #     post_processing_time_times.append((end - start))
    # print(f"POST PROCESSING MEDIAN: {np.round((np.median(post_processing_time_times) * 1000), 2)}ms")
    #
    # print("OPERATIONS IN SEQUENCE")
    # for bs in [1, 8, 16]:
    #     print(f"FOR BATCH SIZE={bs}")
    #     pre_processing_times = []
    #     inference_times = []
    #     post_processing_time_times = []
    #     for _ in tqdm(range(200), total=200):
    #         start = time.monotonic()
    #         pre_processed_image = pre_process_image(image=image, device=DEVICE, target_size=(640, 640), batch_size=bs)
    #         end = time.monotonic()
    #         pre_processing_times.append((end - start))
    #         start = time.monotonic()
    #         results = perform_inference(pre_processed_image, engine, context, DEVICE)
    #         end = time.monotonic()
    #         inference_times.append((end - start))
    #         start = time.monotonic()
    #         after_nms = run_nms(output=results)
    #         end = time.monotonic()
    #         post_processing_time_times.append((end - start))
    #     print(f"PRE PROCESSING MEDIAN (per image): {np.round((np.median(pre_processing_times) * 1000 / bs), 2)}ms")
    #     print(f"INFERENCE MEDIAN (per image): {np.round((np.median(inference_times) * 1000 / bs), 2)}ms")
    #     print(f"POST PROCESSING MEDIAN (per image): {np.round((np.median(post_processing_time_times) * 1000 / bs), 2)}ms")
    # # detections = sv.Detections(
    # #     xyxy=((after_nms[0][:, :4].cpu().numpy() - (140, 0, 140, 0)) * 2).astype(np.int32),
    # #     confidence=after_nms[0][:, 4].cpu().numpy(),
    # #     class_id=after_nms[0][:, 5].cpu().numpy().astype(np.int32),
    # # )
    # print("WITH THREAD POOL EXECUTORS FOR PRE-/POST- PROCESSING")
    # with ThreadPoolExecutor(max_workers=16) as pre_processing_executor:
    #     with ThreadPoolExecutor(max_workers=16) as post_processing_executor:
    #         for bs in [1, 8, 16]:
    #             print(f"FOR BATCH SIZE={bs}")
    #             pre_processing_times = []
    #             inference_times = []
    #             post_processing_time_times = []
    #             for _ in tqdm(range(200), total=200):
    #                 _ = run_parallel_pipeline(
    #                     image=image,
    #                     engine=engine,
    #                     context=context,
    #                     batch_size=bs,
    #                     pre_processing_executor=pre_processing_executor,
    #                     post_processing_executor=post_processing_executor,
    #                     pre_processing_times=pre_processing_times,
    #                     inference_times=inference_times,
    #                     post_processing_time_times=post_processing_time_times,
    #                 )
    #             print(f"PRE PROCESSING MEDIAN (per image): {np.round((np.median(pre_processing_times) * 1000 / bs), 2)}ms")
    #             print(f"INFERENCE MEDIAN (per image): {np.round((np.median(inference_times) * 1000 / bs), 2)}ms")
    #             print(f"POST PROCESSING MEDIAN (per image): {np.round((np.median(post_processing_time_times) * 1000 / bs), 2)}ms")
    # # box_annotator = sv.BoxAnnotator()
    # # annotated = box_annotator.annotate(image.copy(), detections)
    # # cv2.imwrite("annotated.jpg", annotated)
    pre_processing_queue, inference_queue, post_processing_queue = Queue(maxsize=32), Queue(maxsize=32), Queue(maxsize=32)
    for bs in [1, 8, 16]:
        print(f"RUNNING PIPELINED PROCESSING BS={bs}")
        pre_processed_images = {}
        results_registry = {}
        preproc_thread = Thread(
            target=pre_processing_thread,
            args=(image, bs, pre_processed_images, pre_processing_queue, inference_queue)
        )
        preproc_thread.start()
        infer_thread = Thread(
            target=inference_thread,
            args=(inference_queue, post_processing_queue, engine, context, pre_processed_images, results_registry)
        )
        infer_thread.start()
        post_proc_thread = Thread(
            target=post_processing_thread,
            args=(post_processing_queue, results_registry)
        )
        post_proc_thread.start()
        for _ in range(1000):
            pre_processing_queue.put(True)
        pre_processing_queue.put(None)
        preproc_thread.join()
        infer_thread.join()
        post_proc_thread.join()



def load_image(image_url: str) -> np.ndarray:
    response = requests.get(image_url)
    response.raise_for_status()
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)


def load_model(model_path: str):
    with open(model_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def pre_processing_thread(
    input_image: np.ndarray,
    batch_size: int,
    pre_processed_images: dict[str, torch.Tensor],
    input_queue: Queue,
    output_queue: Queue,
) -> None:
    fps_monitor = sv.FPSMonitor(sample_size=128)
    while True:
        start = time.monotonic()
        sig = input_queue.get()
        end = time.monotonic()
        if not sig:
            output_queue.put(None)
            return None
        processing_start = time.monotonic()
        image_hash = str(uuid4())
        result = pre_process_image(
            image=input_image,
            device=DEVICE,
            target_size=(640, 640),
            batch_size=batch_size,
        )
        pre_processed_images[image_hash] = result
        for _ in range(batch_size):
            fps_monitor.tick()
        processing_end = time.monotonic()
        output_queue.put(image_hash)
        waiting_time = round((end - start) * 1000, 2)
        processing_time = round((processing_end - processing_start) * 1000, 2)
        # print(f"PRE-PROCESSING FPS: {fps_monitor.fps} IDLE: {waiting_time}ms PROCESSING: {processing_time}ms")


def inference_thread(
    input_queue: Queue,
    output_queue: Queue,
    engine, context,
    pre_processed_images: dict[str, torch.Tensor],
    results_registry: dict[str, torch.Tensor],
) -> None:
    fps_monitor = sv.FPSMonitor(sample_size=128)
    while True:
        start = time.monotonic()
        image_hash = input_queue.get()
        end = time.monotonic()
        if not image_hash:
            output_queue.put(None)
            return None
        processing_start = time.monotonic()
        image = pre_processed_images[image_hash]
        del pre_processed_images[image_hash]
        results = perform_inference(image, engine, context, DEVICE)
        results_hash = str(uuid4())
        results_registry[results_hash] = results
        processing_end = time.monotonic()
        output_queue.put(results_hash)
        for _ in range(image.shape[0]):
            fps_monitor.tick()
        waiting_time = round((end - start) * 1000, 2)
        processing_time = round((processing_end - processing_start) * 1000, 2)
        # print(f"INFERENCE FPS: {fps_monitor.fps} IDLE: {waiting_time}ms PROCESSING: {processing_time}ms")


def post_processing_thread(
    input_queue: Queue,
    results_registry: dict[str, torch.Tensor],
) -> None:
    fps_monitor = sv.FPSMonitor(sample_size=128)
    while True:
        start = time.monotonic()
        results_hash = input_queue.get()
        end = time.monotonic()
        if not results_hash:
            return None
        processing_start = time.monotonic()
        results = results_registry[results_hash]
        del results_registry[results_hash]
        nms_results = run_nms(results)
        processing_end = time.monotonic()
        for _ in range(len(nms_results)):
            fps_monitor.tick()
        waiting_time = round((end - start) * 1000, 2)
        processing_time = round((processing_end - processing_start) * 1000, 2)
        print(f"FPS: {fps_monitor.fps} IDLE: {waiting_time}ms PROCESSING: {processing_time}ms")


def run_parallel_pipeline(
    image: np.ndarray,
    engine,
    context,
    batch_size: int,
    pre_processing_executor: ThreadPoolExecutor,
    post_processing_executor: ThreadPoolExecutor,
    pre_processing_times: List[float],
    inference_times: List[float],
    post_processing_time_times: List[float],
):
    start = time.monotonic()
    pre_processed_image = pre_process_image_in_parallel(
        image=image,
        device=DEVICE,
        target_size=(640, 640),
        batch_size=batch_size,
        pre_processing_executor=pre_processing_executor,
    )
    end = time.monotonic()
    pre_processing_times.append((end - start))
    start = time.monotonic()
    results = perform_inference(pre_processed_image, engine, context, DEVICE)
    end = time.monotonic()
    inference_times.append((end - start))
    start = time.monotonic()
    result = run_nms_in_parallel(
        output=results,
        post_processing_executor=post_processing_executor,
    )
    end = time.monotonic()
    post_processing_time_times.append((end - start))
    return result


def pre_process_image_in_parallel(
    image: np.ndarray,
    device: torch.device,
    target_size: Tuple[int, int],
    batch_size: int,
    pre_processing_executor: ThreadPoolExecutor,
) -> torch.Tensor:
    futures = [
        pre_processing_executor.submit(pre_process_image, image, device, target_size, 1)
        for _ in range(batch_size)
    ]
    results = [
        f.result() for f in futures
    ]
    return torch.cat(results, dim=0).contiguous()


def pre_process_image(
    image: np.ndarray,
    device: torch.device,
    target_size: Tuple[int, int],
    batch_size: int,
) -> torch.Tensor:
    results = []
    for _ in range(batch_size):
        image_tensor = torch.from_numpy(image).to(device)
        image_rgb = image_tensor[..., [2, 1, 0]]
        rgb_chw = image_rgb.permute(2, 0, 1) / 255.0
        _, h, w = rgb_chw.shape
        target_h, target_w = target_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        image_tensor = functional.resize(rgb_chw, [new_h, new_w], interpolation=functional.InterpolationMode.NEAREST)
        pad_top = max((target_h - new_h) // 2, 0)
        pad_bottom = max(target_h - new_h - pad_top, 0)
        pad_left = max((target_w - new_w) // 2, 0)
        pad_right = max(target_w - new_w - pad_left, 0)
        image_tensor = functional.pad(image_tensor, [pad_left, pad_top, pad_right, pad_bottom], fill=0.5)
        img = image_tensor.unsqueeze(0)
        results.append(img)
    if len(results) == 1:
        return results[0].contiguous()
    return torch.cat(results, dim=0).contiguous()


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


def run_nms_in_parallel(
    output: torch.Tensor,
    post_processing_executor: ThreadPoolExecutor,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    max_detections: int = 100,
    class_agnostic: bool = False,
) -> List[torch.Tensor]:
    bs = output.shape[0]
    boxes = output[:, :4, :]      # (N, 4, 8400)
    scores = output[:, 4:, :]     # (N, 80, 8400)
    futures = [
        post_processing_executor.submit(
            run_single_nms_in_thread_pool, boxes, scores, b,
            conf_thresh, iou_thresh, max_detections, class_agnostic
        )
        for b in range(bs)
    ]
    return [
        f.result() for f in futures
    ]


def run_single_nms_in_thread_pool(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    b: int,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    max_detections: int = 100,
    class_agnostic: bool = False,
) -> torch.Tensor:
    bboxes = boxes[b].T  # (8400, 4)
    class_scores = scores[b].T  # (8400, 80)

    class_conf, class_ids = class_scores.max(1)  # (8400,), (8400,)

    mask = class_conf > conf_thresh
    if mask.sum() == 0:
        return torch.zeros((0, 6), device=DEVICE)

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
    return torch.cat([
        xyxy[keep],
        class_conf[keep].unsqueeze(1),
        class_ids[keep].unsqueeze(1).float()
    ], dim=1)  # [x1, y1, x2, y2, conf, cls]


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
