import os.path
from collections import defaultdict
from typing import List, Literal, Union, Optional, Dict

import cv2
import numpy as np
from supervision import VideoInfo

from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData


def process_video_with_workflow() -> None:
    pass


class WorkflowsStructuredDataSink:

    @classmethod
    def init(
        cls,
        output_directory: str,
        results_log_type: Literal["csv", "jsonl"],
        max_entries_in_logs_chunk: int,
    ) -> "WorkflowsStructuredDataSink":
        return cls(
            output_directory=output_directory,
            structured_results_buffer=[],
            results_log_type=results_log_type,
            max_entries_in_logs_chunk=max_entries_in_logs_chunk,
        )

    def __init__(
        self,
        output_directory: str,
        structured_results_buffer: List[dict],
        results_log_type: Literal["csv", "jsonl"],
        max_entries_in_logs_chunk: int,
    ):
        self._output_directory = output_directory
        self._structured_results_buffer = structured_results_buffer
        self._results_log_type = results_log_type
        self._max_entries_in_logs_chunk = max_entries_in_logs_chunk

    def on_prediction(
        self,
        predictions: Union[Optional[dict], List[Optional[dict]]],
        video_frames: Union[Optional[VideoFrame], List[Optional[VideoFrame]]],
    ) -> None:
        if not isinstance(predictions, list):
            predictions = [predictions]
        for prediction in predictions:
            if prediction is None:
                continue

    def __del__(self):
        pass


def dump_content() -> None:
    pass


class WorkflowsVideoSink:

    @classmethod
    def init(
        cls,
        input_video_path: str,
        output_directory: str,
    ) -> "WorkflowsVideoSink":
        source_video_info = VideoInfo.from_video_path(video_path=input_video_path)
        return cls(
            source_video_info=source_video_info,
            output_directory=output_directory,
        )

    def __init__(
        self,
        source_video_info: VideoInfo,
        output_directory: str
    ):
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
                if not isinstance(value, WorkflowImageData):
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
                stream_sinks[key].write_frame(frame=value.numpy_image)

    def __del__(self):
        for stream_sinks in self._video_sinks.values():
            for sink in stream_sinks.values():
                sink.release()



class VideoSink:

    def __init__(self, target_path: str, video_info: VideoInfo, codec: str = "mp4v"):
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
        """
        Writes a single video frame to the target video file.

        Args:
            frame (np.ndarray): The video frame to be written to the file. The frame
                must be in BGR color format.
        """
        self.__writer.write(frame)

    def release(self) -> None:
        self.__writer.release()


def _generate_target_path_for_video(output_directory: str, source_id: int, field_name: str) -> str:
    os.makedirs(os.path.abspath(output_directory), exist_ok=True)
    return os.path.join(os.path.abspath(output_directory), f"source_{source_id}_output_{field_name}_preview.mp4")