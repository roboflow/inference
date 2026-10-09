#!/usr/bin/env python3
"""Verify a local RF-DETR TRT package against a Jetson video/RTSP source.

Run with RUNS_ON_JETSON=true and ENABLE_TENSOR_DATA_REPRESENTATION=true.
The package must contain engine.plan, inference_config.json, trt_config.json,
and class_names.txt. Compile the engine on the target Jetson first.
"""

import argparse
import json

import torch

from inference.core.interfaces.camera.jetson_producer import JetsonVideoFrameProducer
from inference.core.interfaces.camera.video_source import VideoSource
from inference_models.models.rfdetr.optimization.execution_plan import (
    RFDetrExecutionPlan,
)
from inference_models.models.rfdetr.rfdetr_object_detection_trt import (
    RFDetrForObjectDetectionTRT,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-package", required=True)
    parser.add_argument("--video-source", required=True)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--expected-class")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")

    model = RFDetrForObjectDetectionTRT.from_pretrained(
        args.model_package,
        device=torch.device("cuda:0"),
        rfdetr_execution_plan=RFDetrExecutionPlan(
            preprocessor_id="triton-universal-v1",
            postprocessor_id="triton-fused-v1",
            allow_compatibility_fallback=False,
            allow_runtime_failure_fallback=False,
        ),
    )
    source = VideoSource.init(
        video_reference=args.video_source,
        allow_tensor_frames=True,
    )
    try:
        source.start()
        assert isinstance(source._video, JetsonVideoFrameProducer), type(source._video)
        for _ in range(args.frames):
            frame = source.read_frame(timeout=30.0)
            assert frame is not None
            assert isinstance(frame.image, torch.Tensor) and frame.image.is_cuda
            inputs, metadata = model.pre_process(frame.image, input_color_format="rgb")
            assert inputs.is_cuda
            raw = model.forward(inputs)
            assert all(output.is_cuda for output in raw)
            detections = model.post_process(raw, metadata, confidence=0.4)[0]
            assert all(
                value.is_cuda
                for value in (
                    detections.xyxy,
                    detections.class_id,
                    detections.confidence,
                )
            )
            # Only small final class IDs are copied for the test assertion;
            # decoded pixels, preprocessing and model outputs stay on CUDA.
            if args.expected_class:
                classes = [
                    model.class_names[int(index)] for index in detections.class_id.cpu()
                ]
                assert args.expected_class in classes, classes
        stats = source._video.tensor_bridge_stats
        assert stats["nvmm_frames"] >= args.frames
        for counter in (
            "host_pixel_maps",
            "host_to_device_copies",
            "device_to_host_copies",
            "array_flatten_copies",
        ):
            assert stats[counter] == 0, (counter, stats[counter])
        print(
            "JETSON_TRT_E2E_OK "
            + json.dumps(
                {
                    "frames": args.frames,
                    "model": type(model).__name__,
                    "preprocessor": model.preprocessor_implementation_id,
                    "postprocessor": model.postprocessor_implementation_id,
                    "bridge_stats": stats,
                }
            ),
            flush=True,
        )
    finally:
        source.terminate(wait_on_frames_consumption=False, purge_frames_buffer=True)


if __name__ == "__main__":
    main()
