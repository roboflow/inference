import os.path
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from rich.progress import Progress, TaskID

from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.entities import SinkHandler
from inference.core.interfaces.stream.sinks import multi_sink
from inference.core.utils.image_utils import load_image_bgr
from inference_cli.lib.utils import dump_jsonl
from inference_cli.lib.workflows.common import deduct_images, dump_objects_to_json
from inference_cli.lib.workflows.entities import OutputFileType, VideoProcessingDetails


def process_video_with_workflow(
    input_video_path: str,
    output_directory: str,
    output_file_type: OutputFileType,
    workflow_specification: Optional[dict] = None,
    workspace_name: Optional[str] = None,
    workflow_id: Optional[str] = None,
    workflow_parameters: Optional[Dict[str, Any]] = None,
    image_input_name: str = "image",
    max_fps: Optional[float] = None,
    save_image_outputs_as_video: bool = True,
    api_key: Optional[str] = None,
    on_prediction: Optional[SinkHandler] = None,
) -> VideoProcessingDetails:
    structured_sink = WorkflowsStructuredDataSink(
        output_directory=output_directory,
        output_file_type=output_file_type,
        numbers_of_streams=1,
    )
    progress_sink = ProgressSink.init(input_video_path=input_video_path)
    sinks = [structured_sink.on_prediction, progress_sink.on_prediction]
    if on_prediction:
        sinks.append(on_prediction)
    video_sink: Optional[WorkflowsVideoSink] = None
    if save_image_outputs_as_video:
        video_sink = WorkflowsVideoSink.init(
            input_video_path=input_video_path,
            output_directory=output_directory,
        )
        sinks.append(video_sink.on_prediction)
    pipeline = InferencePipeline.init_with_workflow(
        video_reference=[input_video_path],
        workflow_specification=workflow_specification,
        workspace_name=workspace_name,
        workflow_id=workflow_id,
        api_key=api_key,
        on_prediction=partial(multi_sink, sinks=sinks),
        workflows_parameters=workflow_parameters,
        serialize_results=True,
        image_input_name=image_input_name,
        max_fps=max_fps,
    )
    progress_sink.start()
    pipeline.start(use_main_thread=True)
    pipeline.join()
    progress_sink.stop()
    structured_results_file = structured_sink.flush()[0]
    video_outputs = None
    if video_sink is not None:
        video_outputs = video_sink.release()
    return VideoProcessingDetails(
        structured_results_file=structured_results_file,
        video_outputs=video_outputs,
    )


class WorkflowsStructuredDataSink:

    def __init__(
        self,
        output_directory: str,
        output_file_type: OutputFileType,
        numbers_of_streams: int = 1,
    ):
        self._output_directory = output_directory
        self._structured_results_buffer = defaultdict(list)
        self._output_file_type = output_file_type
        self._numbers_of_streams = numbers_of_streams

    def on_prediction(
        self,
        predictions: Union[Optional[dict], List[Optional[dict]]],
        video_frames: Union[Optional[VideoFrame], List[Optional[VideoFrame]]],
    ) -> None:
        if not isinstance(predictions, list):
            predictions = [predictions]
        for stream_idx, prediction in enumerate(predictions):
            if prediction is None:
                continue
            prediction = deduct_images(result=prediction)
            if self._output_file_type is OutputFileType.CSV:
                prediction = {
                    k: dump_objects_to_json(value=v) for k, v in prediction.items()
                }
            self._structured_results_buffer[stream_idx].append(prediction)

    def flush(self) -> List[Optional[str]]:
        stream_idx2file_path = {}
        for stream_idx, buffer in self._structured_results_buffer.items():
            file_path = self._flush_stream_buffer(stream_idx=stream_idx)
            stream_idx2file_path[stream_idx] = file_path
        return [
            stream_idx2file_path.get(stream_idx)
            for stream_idx in range(self._numbers_of_streams)
        ]

    def _flush_stream_buffer(self, stream_idx: int) -> Optional[str]:
        content = self._structured_results_buffer[stream_idx]
        if len(content) == 0:
            return None
        file_path = generate_results_file_name(
            output_directory=self._output_directory,
            results_log_type=self._output_file_type,
            stream_id=stream_idx,
        )
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if self._output_file_type is OutputFileType.CSV:
            data_frame = pd.DataFrame(content)
            data_frame.to_csv(file_path, index=False)
        else:
            dump_jsonl(path=file_path, content=content)
        self._structured_results_buffer[stream_idx] = []
        return file_path

    def __del__(self):
        self.flush()


def generate_results_file_name(
    output_directory: str,
    results_log_type: OutputFileType,
    stream_id: int,
) -> str:
    output_directory = os.path.abspath(output_directory)
    return os.path.join(
        output_directory,
        f"workflow_results_source_{stream_id}.{results_log_type.value}",
    )


class WorkflowsVideoSink:

    @classmethod
    def init(
        cls,
        input_video_path: str,
        output_directory: str,
    ) -> "WorkflowsVideoSink":
        source_video_info = sv.VideoInfo.from_video_path(video_path=input_video_path)
        return cls(
            source_video_info=source_video_info,
            output_directory=output_directory,
        )

    def __init__(self, source_video_info: sv.VideoInfo, output_directory: str):
        self._video_sinks: Dict[int, Dict[str, VideoSink]] = defaultdict(dict)
        self._source_video_info = source_video_info
        self._output_directory = output_directory

    def on_prediction(
        self,
        predictions: Union[Optional[dict], List[Optional[dict]]],
        video_frames: Union[Optional[VideoFrame], List[Optional[VideoFrame]]],
    ) -> None:
        if not isinstance(predictions, list):
            predictions = [predictions]
        for stream_idx, prediction in enumerate(predictions):
            if prediction is None:
                continue
            stream_sinks = self._video_sinks[stream_idx]
            for key, value in prediction.items():
                if (
                    not isinstance(value, dict)
                    or "value" not in value
                    or value.get("type") != "base64"
                ):
                    continue
                if key not in stream_sinks:
                    video_target_path = _generate_target_path_for_video(
                        output_directory=self._output_directory,
                        source_id=stream_idx,
                        field_name=key,
                    )
                    stream_sinks[key] = VideoSink(
                        target_path=video_target_path,
                        video_info=self._source_video_info,
                    )
                    stream_sinks[key].start()
                image = load_image_bgr(value)
                stream_sinks[key].write_frame(frame=image)

    def release(self) -> Optional[Dict[str, str]]:
        stream_idx2keys_videos: Dict[int, Dict[str, str]] = defaultdict(dict)
        for stream_idx, stream_sinks in self._video_sinks.items():
            for key, sink in stream_sinks.items():
                sink.release()
                stream_idx2keys_videos[stream_idx][key] = sink.target_path
        self._video_sinks = defaultdict(dict)
        return stream_idx2keys_videos.get(0)

    def __del__(self):
        self.release()


class ProgressSink:

    @classmethod
    def init(
        cls,
        input_video_path: str,
    ) -> "ProgressSink":
        source_video_info = sv.VideoInfo.from_video_path(video_path=input_video_path)
        return cls(total_frames=source_video_info.total_frames)

    def __init__(self, total_frames: Optional[int]):
        self._total_frames = total_frames
        self._progress_bar = Progress()
        self._task: Optional[TaskID] = None

    def start(self) -> None:
        self._progress_bar.start()
        self._task = self._progress_bar.add_task(
            description="Processing video...",
            total=self._total_frames,
        )

    def on_prediction(
        self,
        predictions: Union[Optional[dict], List[Optional[dict]]],
        video_frames: Union[Optional[VideoFrame], List[Optional[VideoFrame]]],
    ) -> None:
        if video_frames is None:
            return None
        if isinstance(video_frames, list):
            raise NotImplementedError(
                "ProgressSink is only to be used against single video file"
            )
        self._progress_bar.update(
            self._task,
            completed=video_frames.frame_id,
        )

    def stop(self) -> None:
        self._progress_bar.stop()

    def __del__(self):
        self.stop()


class VideoSink:

    def __init__(self, target_path: str, video_info: sv.VideoInfo, codec: str = "mp4v"):
        self.target_path = target_path
        self.video_info = video_info
        self.__codec = codec
        self.__writer = None

    def start(self) -> None:
        try:
            self.__fourcc = cv2.VideoWriter_fourcc(*self.__codec)
        except TypeError as e:
            print(str(e) + ". Defaulting to mp4v...")
            self.__fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.__writer = cv2.VideoWriter(
            self.target_path,
            self.__fourcc,
            self.video_info.fps,
            self.video_info.resolution_wh,
        )

    def write_frame(self, frame: np.ndarray):
        self.__writer.write(frame)

    def release(self) -> None:
        self.__writer.release()


def _generate_target_path_for_video(
    output_directory: str, source_id: int, field_name: str
) -> str:
    os.makedirs(os.path.abspath(output_directory), exist_ok=True)
    return os.path.join(
        os.path.abspath(output_directory),
        f"source_{source_id}_output_{field_name}_preview.mp4",
    )
