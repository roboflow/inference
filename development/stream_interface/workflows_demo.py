import os
from threading import Thread

from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.sinks import render_boxes
from inference.core.interfaces.stream.watchdog import PipelineWatchDog, BasePipelineWatchDog

STOP = False


def main() -> None:
    global STOP
    watchdog = BasePipelineWatchDog()
    workflow_specification = {
        "specification": {
            "version": "1.0",
            "inputs": [
                {"type": "InferenceImage", "name": "image"},
            ],
            "steps": [
                {
                    "type": "ObjectDetectionModel",
                    "name": "step_1",
                    "image": "$inputs.image",
                    "model_id": "yolov8n-640",
                    "confidence": 0.5,
                }
            ],
            "outputs": [
                {"type": "JsonField", "name": "predictions", "selector": "$steps.step_1.*"},
            ],
        }
    }
    pipeline = InferencePipeline.init_with_workflow(
        video_reference=os.environ["VIDEO_REFERENCE"],
        workflow_specification=workflow_specification,
        watchdog=watchdog,
        on_prediction=console_sink,
    )
    control_thread = Thread(target=command_thread, args=(pipeline, watchdog))
    control_thread.start()
    pipeline.start()
    STOP = True
    pipeline.join()


def command_thread(pipeline: InferencePipeline, watchdog: PipelineWatchDog) -> None:
    global STOP
    while not STOP:
        key = input()
        if key == "i":
            print(watchdog.get_report())
        if key == "t":
            pipeline.terminate()
            STOP = True
        elif key == "p":
            pipeline.pause_stream()
        elif key == "m":
            pipeline.mute_stream()
        elif key == "r":
            pipeline.resume_stream()


def console_sink(
    predictions: dict,
    video_frame: VideoFrame,
) -> None:
    render_boxes(
        predictions["predictions"][0],
        video_frame,
        display_statistics=True,
    )


if __name__ == '__main__':
    main()
