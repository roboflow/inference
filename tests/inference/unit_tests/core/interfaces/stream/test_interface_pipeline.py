from collections import defaultdict
from concurrent.futures import Future
from datetime import datetime
from functools import partial
from queue import Queue
from threading import Lock
from typing import Any, List, Optional, Tuple, Union
from unittest.mock import MagicMock

import numpy as np
import pytest

from inference.core.active_learning.middlewares import ThreadingActiveLearningMiddleware
from inference.core.cache import MemoryCache
from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
)
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.camera.exceptions import (
    EndOfStreamError,
    SourceConnectionError,
)
from inference.core.interfaces.camera.video_source import (
    SourceMetadata,
    SourceProperties,
    StreamState,
    VideoSource,
    lock_state_transition,
)
from inference.core.interfaces.stream import inference_pipeline
from inference.core.interfaces.stream.entities import (
    InferenceHandlerResult,
    ModelConfig,
)
from inference.core.interfaces.stream.inference_pipeline import (
    InferencePipeline,
    _resolve_prediction_futures,
)
from inference.core.interfaces.stream.model_handlers.roboflow_models import (
    default_process_frame,
)
from inference.core.interfaces.stream.model_handlers.workflows import (
    StreamLookaheadDrainError,
)
from inference.core.interfaces.stream.sinks import active_learning_sink, multi_sink
from inference.core.interfaces.stream.watchdog import BasePipelineWatchDog


class VideoSourceStub:
    def __init__(
        self, frames_number: int, is_file: bool, rounds: int = 0, source_id: int = 0
    ):
        self._frames_number = frames_number
        self._is_file = is_file
        self._current_round = 0
        self._emissions_in_current_round = 0
        self._rounds = rounds
        self._calls = []
        self._frame_id = 0
        self._state_change_lock = Lock()
        self.on_end = None
        self.source_id = source_id

    @lock_state_transition
    def restart(
        self, wait_on_frames_consumption: bool = True, purge_frames_buffer: bool = False
    ) -> None:
        self._calls.append("restart")
        if self._current_round == self._rounds:
            self.on_end()
            raise SourceConnectionError()
        self._current_round += 1
        self._emissions_in_current_round = 0

    @lock_state_transition
    def start(self) -> None:
        self._calls.append("start")
        self._current_round += 1
        self._emissions_in_current_round = 0

    @lock_state_transition
    def terminate(
        self, wait_on_frames_consumption: bool = True, purge_frames_buffer: bool = False
    ) -> None:
        self._calls.append("terminate")

    def describe_source(self) -> SourceMetadata:
        return SourceMetadata(
            source_properties=SourceProperties(
                is_file=self._is_file,
                width=100,
                height=100,
                fps=25,
                total_frames=10,
            ),
            source_reference="dummy",
            buffer_size=32,
            state=StreamState.RUNNING,
            buffer_filling_strategy=None,
            buffer_consumption_strategy=None,
            source_id=self.source_id,
        )

    def read_frame(self, timeout: Optional[float] = None) -> VideoFrame:
        self._calls.append("read_frame")
        if self._emissions_in_current_round == self._frames_number:
            raise EndOfStreamError()
        self._frame_id += 1
        self._emissions_in_current_round += 1
        return VideoFrame(
            image=np.zeros((128, 128, 3), dtype=np.uint8),
            frame_id=self._frame_id,
            frame_timestamp=datetime.now(),
            source_id=self.source_id,
        )

    def __iter__(self) -> "VideoSourceStub":
        return self

    def __next__(self) -> VideoFrame:
        try:
            return self.read_frame()
        except EndOfStreamError:
            raise StopIteration()


class ModelStub:
    def __init__(self):
        self.api_key = None

    def infer(self, image: Any, **kwargs) -> List[ObjectDetectionInferenceResponse]:
        return [
            ObjectDetectionInferenceResponse(
                predictions=[
                    ObjectDetectionPrediction(
                        **{
                            "x": 10,
                            "y": 20,
                            "width": 30,
                            "height": 40,
                            "confidence": 0.9,
                            "class": "car",
                            "class_id": 3,
                        }
                    )
                ],
                image=InferenceResponseImage(width=1920, height=1080),
            )
        ] * len(image)


class _PredictionReadyWatchdog:
    def __init__(self) -> None:
        self.ready_frames = []

    def on_model_prediction_ready(self, frames):
        self.ready_frames.append(frames)


class _FlushableInferenceHandler:
    def __init__(self, results):
        self.results = results
        self.flush_calls = 0
        self.close_calls = 0

    def flush(self):
        self.flush_calls += 1
        return self.results

    def close(self) -> None:
        self.close_calls += 1


def test_inference_pipeline_drain_enqueues_flush_results_with_bound_frames() -> None:
    frame_1 = VideoFrame(
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        frame_id=1,
        frame_timestamp=datetime.now(),
        source_id=0,
    )
    frame_2 = VideoFrame(
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        frame_id=2,
        frame_timestamp=datetime.now(),
        source_id=0,
    )
    handler = _FlushableInferenceHandler(
        results=[
            InferenceHandlerResult(predictions=["p1"], video_frames=[frame_1]),
            InferenceHandlerResult(predictions=["p2"], video_frames=[frame_2]),
        ]
    )
    watchdog = _PredictionReadyWatchdog()
    pipeline = object.__new__(InferencePipeline)
    pipeline._on_video_frame = handler
    pipeline._watchdog = watchdog
    pipeline._predictions_queue = Queue()
    pipeline._status_update_handlers = []

    pipeline._drain_inference_handler()

    assert handler.flush_calls == 1
    assert pipeline._predictions_queue.get_nowait() == (["p1"], [frame_1])
    assert pipeline._predictions_queue.get_nowait() == (["p2"], [frame_2])
    assert watchdog.ready_frames == [[frame_1], [frame_2]]


def test_inference_pipeline_drain_dispatches_partial_results_and_reraises() -> None:
    # given - a failed tail frame must not disappear as a normal empty drain:
    # the frames that drained are dispatched, the failure propagates
    frame_1 = VideoFrame(
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        frame_id=1,
        frame_timestamp=datetime.now(),
        source_id=0,
    )
    drained = [InferenceHandlerResult(predictions=["p1"], video_frames=[frame_1])]
    handler = _FlushableInferenceHandler(results=None)
    handler.flush = lambda: (_ for _ in ()).throw(
        StreamLookaheadDrainError("final frame failed", drained_results=drained)
    )
    pipeline = object.__new__(InferencePipeline)
    pipeline._on_video_frame = handler
    pipeline._watchdog = _PredictionReadyWatchdog()
    pipeline._predictions_queue = Queue()
    pipeline._status_update_handlers = []

    # when / then
    with pytest.raises(StreamLookaheadDrainError):
        pipeline._drain_inference_handler()
    assert pipeline._predictions_queue.get_nowait() == (["p1"], [frame_1])
    assert pipeline._predictions_queue.empty()


def test_resolve_prediction_futures_recursively_resolves_nested_values() -> None:
    inner = Future()
    inner.set_result("resolved")
    outer = Future()
    outer.set_result({"detections": [inner]})

    assert _resolve_prediction_futures((outer, {"raw": inner})) == (
        {"detections": ["resolved"]},
        {"raw": "resolved"},
    )


def test_inference_pipeline_close_calls_handler_close_hook() -> None:
    handler = _FlushableInferenceHandler(results=[])
    pipeline = object.__new__(InferencePipeline)
    pipeline._on_video_frame = handler

    pipeline._close_inference_handler()

    assert handler.close_calls == 1


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_inference_pipeline_works_correctly_against_video_file(
    local_video_path: str,
) -> None:
    # given
    model = ModelStub()
    video_source = VideoSource.init(video_reference=local_video_path)
    watchdog = BasePipelineWatchDog()
    watchdog.register_video_sources(video_sources=[video_source])
    predictions = []

    def on_prediction(prediction: dict, video_frame: VideoFrame) -> None:
        predictions.append((video_frame, prediction))

    status_update_handlers = [watchdog.on_status_update]
    inference_config = ModelConfig.init(confidence=0.5, iou_threshold=0.5)
    process_frame_func = partial(
        default_process_frame, model=model, inference_config=inference_config
    )
    predictions_queue = Queue(maxsize=512)
    inference_pipeline = InferencePipeline(
        on_video_frame=process_frame_func,
        video_sources=[video_source],
        on_prediction=on_prediction,
        max_fps=100,
        predictions_queue=predictions_queue,
        watchdog=watchdog,
        status_update_handlers=status_update_handlers,
    )

    # when
    inference_pipeline.start()
    inference_pipeline.join()
    inference_pipeline.start()
    inference_pipeline.join()

    # then
    assert len(predictions) == 431 * 2, "Not all video frames processed"
    assert [p[0].frame_id for p in predictions] == list(
        range(1, 431 * 2 + 1)
    ), "Order of prediction frames violated"


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_inference_pipeline_works_correctly_against_multiple_video_files(
    local_video_path: str,
) -> None:
    # given
    model = ModelStub()
    watchdog = BasePipelineWatchDog()
    accumulator = []

    def on_prediction(predictions: List[dict], video_frames: List[VideoFrame]) -> None:
        for frame_prediction, frame_prediction in zip(predictions, video_frames):
            if frame_prediction is None:
                continue
            accumulator.append((frame_prediction, frame_prediction))

    inference_config = ModelConfig.init(confidence=0.5, iou_threshold=0.5)
    process_frame_func = partial(
        default_process_frame, model=model, inference_config=inference_config
    )
    inference_pipeline = InferencePipeline.init_with_custom_logic(
        video_reference=[local_video_path, local_video_path],
        on_video_frame=process_frame_func,
        on_prediction=on_prediction,
        max_fps=200,
        watchdog=watchdog,
    )

    # when
    inference_pipeline.start()
    inference_pipeline.join()
    inference_pipeline.start()
    inference_pipeline.join()

    # then
    assert len(accumulator) == 431 * 2 * 2, "Not all video frames processed"
    frames_by_sources = defaultdict(list)
    for p in accumulator:
        frames_by_sources[p[0].source_id].append(p[0].frame_id)
    assert (
        len(frames_by_sources) == 2
    ), "Expected to register frames from exactly 2 video sources"
    assert frames_by_sources[0] == sorted(
        frames_by_sources[0]
    ), "Order of prediction frames violated for source 0"
    assert frames_by_sources[1] == sorted(
        frames_by_sources[1]
    ), "Order of prediction frames violated for source 1"


@pytest.mark.parametrize("use_main_thread", [True, False])
def test_inference_pipeline_works_correctly_against_stream_including_reconnections(
    use_main_thread: bool,
) -> None:
    # given
    model = ModelStub()
    video_source = VideoSourceStub(frames_number=100, is_file=False, rounds=2)
    watchdog = BasePipelineWatchDog()
    watchdog.register_video_sources(video_sources=[video_source])
    predictions = []

    def on_prediction(prediction: dict, video_frame: VideoFrame) -> None:
        predictions.append((video_frame, prediction))

    status_update_handlers = [watchdog.on_status_update]
    inference_config = ModelConfig.init(confidence=0.5, iou_threshold=0.5)
    process_frame_func = partial(
        default_process_frame, model=model, inference_config=inference_config
    )
    predictions_queue = Queue(maxsize=512)
    inference_pipeline = InferencePipeline(
        on_video_frame=process_frame_func,
        video_sources=[video_source],
        on_prediction=on_prediction,
        max_fps=None,
        predictions_queue=predictions_queue,
        watchdog=watchdog,
        status_update_handlers=status_update_handlers,
    )

    def stop() -> None:
        inference_pipeline._stop = True

    video_source.on_end = stop

    # when
    inference_pipeline.start(use_main_thread=use_main_thread)
    inference_pipeline.join()

    # then
    assert (
        0 < len(predictions) <= 200
    ), "Expected to process some frames, but not more than max number of emitted frames"
    frame_ids = [p[0].frame_id for p in predictions]
    assert frame_ids == sorted(frame_ids), "Order of prediction frames violated"
    assert (
        max(frame_ids) > 100
    ), "Expected to process at least one frame after reconnection"


@pytest.mark.parametrize("use_main_thread", [True, False])
def test_inference_pipeline_works_correctly_against_multiple_streams_including_reconnections(
    use_main_thread: bool,
) -> None:
    # given
    model = ModelStub()
    video_source_1 = VideoSourceStub(
        frames_number=100, is_file=False, rounds=2, source_id=0
    )
    video_source_2 = VideoSourceStub(
        frames_number=130, is_file=False, rounds=2, source_id=1
    )
    watchdog = BasePipelineWatchDog()
    watchdog.register_video_sources(video_sources=[video_source_1, video_source_2])
    accumulator = []

    def on_prediction(predictions: List[dict], video_frames: List[VideoFrame]) -> None:
        for frame_prediction, video_frame in zip(predictions, video_frames):
            if frame_prediction is None:
                continue
            accumulator.append((video_frame, frame_prediction))

    status_update_handlers = [watchdog.on_status_update]
    inference_config = ModelConfig.init(confidence=0.5, iou_threshold=0.5)
    process_frame_func = partial(
        default_process_frame, model=model, inference_config=inference_config
    )
    predictions_queue = Queue(maxsize=512)
    inference_pipeline = InferencePipeline(
        on_video_frame=process_frame_func,
        video_sources=[video_source_1, video_source_2],
        on_prediction=on_prediction,
        max_fps=None,
        predictions_queue=predictions_queue,
        watchdog=watchdog,
        status_update_handlers=status_update_handlers,
    )

    stop_counter = []
    stop_counter_lock = Lock()

    def stop() -> None:
        with stop_counter_lock:
            stop_counter.append(1)
            if len(stop_counter) == 2:
                inference_pipeline._stop = True

    video_source_1.on_end = stop
    video_source_2.on_end = stop

    # when
    inference_pipeline.start(use_main_thread=use_main_thread)
    inference_pipeline.join()

    # then
    assert (
        0 < len(accumulator) <= 100 * 2 + 130 * 2
    ), "Expected to process some frames, but not more than max number of emitted frames"
    frames_by_sources = defaultdict(list)
    for p in accumulator:
        frames_by_sources[p[0].source_id].append(p[0].frame_id)
    assert (
        len(frames_by_sources) == 2
    ), "Expected to register frames from exactly 2 video sources"
    assert frames_by_sources[0] == sorted(
        frames_by_sources[0]
    ), "Order of prediction frames violated for source 0"
    assert frames_by_sources[1] == sorted(
        frames_by_sources[1]
    ), "Order of prediction frames violated for source 1"
    assert (
        max(frames_by_sources[0]) > 100
    ), "Expected to process at least one frame after reconnection to source 1"
    assert (
        max(frames_by_sources[1]) > 130
    ), "Expected to process at least one frame after reconnection to source 2"


@pytest.mark.parametrize("use_main_thread", [True, False])
def test_inference_pipeline_works_correctly_against_stream_including_dispatching_errors(
    use_main_thread: bool,
) -> None:
    # given
    model = ModelStub()
    video_source = VideoSourceStub(frames_number=100, is_file=False, rounds=1)
    watchdog = BasePipelineWatchDog()
    watchdog.register_video_sources(video_sources=[video_source])
    predictions = []

    def on_prediction(prediction: dict, video_frame: VideoFrame) -> None:
        predictions.append((video_frame, prediction))
        raise Exception()

    status_update_handlers = [watchdog.on_status_update]
    inference_config = ModelConfig.init(confidence=0.5, iou_threshold=0.5)
    process_frame_func = partial(
        default_process_frame, model=model, inference_config=inference_config
    )

    predictions_queue = Queue(maxsize=512)
    inference_pipeline = InferencePipeline(
        on_video_frame=process_frame_func,
        video_sources=[video_source],
        on_prediction=on_prediction,
        max_fps=None,
        predictions_queue=predictions_queue,
        watchdog=watchdog,
        status_update_handlers=status_update_handlers,
    )

    def stop() -> None:
        inference_pipeline._stop = True

    video_source.on_end = stop

    # when
    inference_pipeline.start(use_main_thread=use_main_thread)
    inference_pipeline.join()

    # then
    assert (
        0 < len(predictions) <= 100
    ), "Expected to process some frames, but not more than max number of emitted frames"
    frames_ids = [p[0].frame_id for p in predictions]
    assert frames_ids == sorted(frames_ids), "Order of prediction frames violated"


@pytest.mark.parametrize("use_main_thread", [True])
def test_inference_pipeline_works_correctly_against_multiple_streams_including_dispatching_errors(
    use_main_thread: bool,
) -> None:
    # given
    model = ModelStub()
    video_source_1 = VideoSourceStub(
        frames_number=100, is_file=False, rounds=2, source_id=0
    )
    video_source_2 = VideoSourceStub(
        frames_number=130, is_file=False, rounds=2, source_id=1
    )
    watchdog = BasePipelineWatchDog()
    watchdog.register_video_sources(video_sources=[video_source_1, video_source_2])
    accumulator = []

    def on_prediction(predictions: List[dict], video_frames: List[VideoFrame]) -> None:
        for frame_prediction, video_frame in zip(predictions, video_frames):
            if frame_prediction is None:
                continue
            accumulator.append((video_frame, frame_prediction))
        raise Exception()

    status_update_handlers = [watchdog.on_status_update]
    inference_config = ModelConfig.init(confidence=0.5, iou_threshold=0.5)
    process_frame_func = partial(
        default_process_frame, model=model, inference_config=inference_config
    )
    predictions_queue = Queue(maxsize=1024)
    inference_pipeline = InferencePipeline(
        on_video_frame=process_frame_func,
        video_sources=[video_source_1, video_source_2],
        on_prediction=on_prediction,
        max_fps=None,
        predictions_queue=predictions_queue,
        watchdog=watchdog,
        status_update_handlers=status_update_handlers,
    )

    stop_counter = []
    stop_counter_lock = Lock()

    def stop() -> None:
        with stop_counter_lock:
            stop_counter.append(1)
            if len(stop_counter) == 2:
                inference_pipeline._stop = True

    video_source_1.on_end = stop
    video_source_2.on_end = stop

    # when
    inference_pipeline.start(use_main_thread=use_main_thread)
    inference_pipeline.join()

    # then
    assert (
        0 < len(accumulator) <= 100 * 2 + 130 * 2
    ), "Expected to process some frames, but not more than max number of emitted frames"
    frames_by_sources = defaultdict(list)
    for p in accumulator:
        frames_by_sources[p[0].source_id].append(p[0].frame_id)
    assert (
        len(frames_by_sources) == 2
    ), "Expected to register frames from exactly 2 video sources"
    assert frames_by_sources[0] == sorted(
        frames_by_sources[0]
    ), "Order of prediction frames violated for source 0"
    assert frames_by_sources[1] == sorted(
        frames_by_sources[1]
    ), "Order of prediction frames violated for source 1"
    assert (
        max(frames_by_sources[0]) > 100
    ), "Expected to process at least one frame after reconnection to source 1"
    assert (
        max(frames_by_sources[1]) > 130
    ), "Expected to process at least one frame after reconnection to source 2"


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_inference_pipeline_works_correctly_against_video_file_with_active_learning_enabled(
    local_video_path: str,
) -> None:
    # given
    model = ModelStub()
    video_source = VideoSource.init(video_reference=local_video_path)
    watchdog = BasePipelineWatchDog()
    watchdog.register_video_sources(video_sources=[video_source])
    predictions = []
    al_datapoints = []
    active_learning_middleware = ThreadingActiveLearningMiddleware(
        api_key="xxx",
        configuration=MagicMock(),
        cache=MemoryCache(),
        task_queue=Queue(),
    )

    def execute_registration_mock(
        inference_input: Any,
        prediction: dict,
        prediction_type: str,
        disable_preproc_auto_orient: bool = False,
    ) -> None:
        al_datapoints.append((inference_input, prediction))

    active_learning_middleware._execute_registration = execute_registration_mock
    al_sink = partial(
        active_learning_sink,
        active_learning_middleware=active_learning_middleware,
        model_type="object-detection",
        disable_preproc_auto_orient=False,
    )

    def on_prediction(
        prediction: Union[dict, List[Optional[dict]]],
        video_frame: Union[VideoFrame, List[Optional[VideoFrame]]],
    ) -> None:
        predictions.append((video_frame, prediction))

    prediction_handler = partial(multi_sink, sinks=[on_prediction, al_sink])

    status_update_handlers = [watchdog.on_status_update]
    inference_config = ModelConfig.init(confidence=0.5, iou_threshold=0.5)
    process_frame_func = partial(
        default_process_frame, model=model, inference_config=inference_config
    )
    predictions_queue = Queue(maxsize=512)
    inference_pipeline = InferencePipeline(
        on_video_frame=process_frame_func,
        video_sources=[video_source],
        on_prediction=prediction_handler,
        max_fps=100,
        predictions_queue=predictions_queue,
        watchdog=watchdog,
        status_update_handlers=status_update_handlers,
        on_pipeline_start=lambda: active_learning_middleware.start_registration_thread(),
        on_pipeline_end=lambda: active_learning_middleware.stop_registration_thread(),
    )

    # when
    inference_pipeline.start()
    inference_pipeline.join()
    inference_pipeline.start()
    inference_pipeline.join()

    # then
    assert len(predictions) == 431 * 2, "Not all video frames processed"
    assert (
        len(al_datapoints) == 431 * 2
    ), "Not all video frames and predictions registered in AL"
    assert [p[0].frame_id for p in predictions] == list(
        range(1, 431 * 2 + 1)
    ), "Order of prediction frames violated"
    assert all(
        [(p[0].image == al_dp[0]).all() for p, al_dp in zip(predictions, al_datapoints)]
    ), "The same images must be registered for explicit sink and Active Learning sink"


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_inference_pipeline_works_correctly_against_multiple_video_files_with_active_learning_enabled(
    local_video_path: str,
) -> None:
    # given
    model = ModelStub()
    video_source_1 = VideoSource.init(video_reference=local_video_path, source_id=0)
    video_source_2 = VideoSource.init(video_reference=local_video_path, source_id=1)
    watchdog = BasePipelineWatchDog()
    watchdog.register_video_sources(video_sources=[video_source_1, video_source_2])
    accumulator = []
    al_datapoints = []
    active_learning_middleware = ThreadingActiveLearningMiddleware(
        api_key="xxx",
        configuration=MagicMock(),
        cache=MemoryCache(),
        task_queue=Queue(),
    )

    def execute_registration_mock(
        inference_input: Any,
        prediction: dict,
        prediction_type: str,
        disable_preproc_auto_orient: bool = False,
    ) -> None:
        al_datapoints.append((inference_input, prediction))

    active_learning_middleware._execute_registration = execute_registration_mock
    al_sink = partial(
        active_learning_sink,
        active_learning_middleware=active_learning_middleware,
        model_type="object-detection",
        disable_preproc_auto_orient=False,
    )

    def on_prediction(
        predictions: Union[dict, List[Optional[dict]]],
        video_frames: Union[VideoFrame, List[Optional[VideoFrame]]],
    ) -> None:
        for frame_prediction, video_frame in zip(predictions, video_frames):
            if frame_prediction is None:
                continue
            accumulator.append((video_frame, frame_prediction))

    prediction_handler = partial(multi_sink, sinks=[on_prediction, al_sink])

    status_update_handlers = [watchdog.on_status_update]
    inference_config = ModelConfig.init(confidence=0.5, iou_threshold=0.5)
    process_frame_func = partial(
        default_process_frame, model=model, inference_config=inference_config
    )
    predictions_queue = Queue(maxsize=512)
    inference_pipeline = InferencePipeline(
        on_video_frame=process_frame_func,
        video_sources=[video_source_1, video_source_2],
        on_prediction=prediction_handler,
        max_fps=100,
        predictions_queue=predictions_queue,
        watchdog=watchdog,
        status_update_handlers=status_update_handlers,
        on_pipeline_start=lambda: active_learning_middleware.start_registration_thread(),
        on_pipeline_end=lambda: active_learning_middleware.stop_registration_thread(),
    )

    # when
    inference_pipeline.start()
    inference_pipeline.join()
    inference_pipeline.start()
    inference_pipeline.join()

    # then
    assert len(accumulator) == 431 * 2 * 2, "Not all video frames processed"
    assert (
        len(al_datapoints) == 431 * 2 * 2
    ), "Not all video frames and predictions registered in AL"
    frames_by_sources = defaultdict(list)
    for p in accumulator:
        frames_by_sources[p[0].source_id].append(p[0].frame_id)
    assert (
        len(frames_by_sources) == 2
    ), "Expected to register frames from exactly 2 video sources"
    assert frames_by_sources[0] == list(
        range(1, 431 * 2 + 1)
    ), "Order of prediction frames violated for source 0"
    assert frames_by_sources[1] == list(
        range(1, 431 * 2 + 1)
    ), "Order of prediction frames violated for source 1"


def test_buffered_stream_dispatch_keys_on_actual_handler_not_env() -> None:
    # given - buffered dispatch must follow whether the handler actually
    # buffers (exposes flush()), NOT whether a depth env var is set: a depth
    # flag on a workflow that does not qualify for pipelining falls back to a
    # plain WorkflowRunner, which must still use the normal dispatch path
    class _BufferingRunner:
        def __call__(self, video_frames):
            return None

        def flush(self):
            return None

    class _PlainRunner:
        def __call__(self, video_frames):
            return []

    # when / then
    assert inference_pipeline._uses_buffered_stream_dispatch(_BufferingRunner())
    assert not inference_pipeline._uses_buffered_stream_dispatch(_PlainRunner())
    assert not inference_pipeline._uses_buffered_stream_dispatch(lambda frames: [])


def test_init_with_custom_logic_caps_queue_only_for_buffering_handler() -> None:
    # given - the predictions queue is capped to 4 only when the handler
    # buffers; a non-buffering handler keeps the full queue even if a depth
    # env var happens to be set (the bug: cap + buffered path with nothing
    # buffered starves throughput)
    class _BufferingRunner:
        def __call__(self, video_frames):
            return None

        def flush(self):
            return None

    class _PlainRunner:
        def __call__(self, video_frames):
            return []

    buffering_pipeline = object.__new__(InferencePipeline)
    buffering_pipeline.__init__(
        on_video_frame=_BufferingRunner(),
        video_sources=[],
        predictions_queue=Queue(),
        watchdog=BasePipelineWatchDog(),
        status_update_handlers=[],
    )
    plain_pipeline = object.__new__(InferencePipeline)
    plain_pipeline.__init__(
        on_video_frame=_PlainRunner(),
        video_sources=[],
        predictions_queue=Queue(),
        watchdog=BasePipelineWatchDog(),
        status_update_handlers=[],
    )

    # then
    assert buffering_pipeline._buffered_stream_dispatch is True
    assert plain_pipeline._buffered_stream_dispatch is False
