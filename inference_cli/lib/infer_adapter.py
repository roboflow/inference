import base64
import os.path
from functools import partial
from glob import glob
from typing import Callable, List, Optional, Union

import cv2
import numpy as np
from supervision import (
    BlurAnnotator,
    BoundingBoxAnnotator,
    BoxCornerAnnotator,
    ByteTrack,
    CircleAnnotator,
    ColorAnnotator,
    Detections,
    DotAnnotator,
    EllipseAnnotator,
    HaloAnnotator,
    HeatMapAnnotator,
    LabelAnnotator,
    MaskAnnotator,
    PixelateAnnotator,
    PolygonAnnotator,
    TraceAnnotator,
    TriangleAnnotator,
    VideoInfo,
    VideoSink,
)
from supervision.annotators.base import BaseAnnotator
from supervision.utils.file import read_yaml_file
from tqdm import tqdm

from inference_cli.lib.env import ROBOFLOW_API_KEY
from inference_cli.lib.logger import CLI_LOGGER
from inference_cli.lib.utils import dump_json, initialise_client
from inference_sdk.http.utils.encoding import bytes_to_opencv_image
from inference_sdk.http.utils.loaders import load_image_from_string

CONFIGS_DIR_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "configs",
    )
)

ANNOTATOR_TYPE2CLASS = {
    "bounding_box": BoundingBoxAnnotator,
    "mask": MaskAnnotator,
    "polygon": PolygonAnnotator,
    "color": ColorAnnotator,
    "halo": HaloAnnotator,
    "ellipse": EllipseAnnotator,
    "box_corner": BoxCornerAnnotator,
    "circle": CircleAnnotator,
    "dot": DotAnnotator,
    "label": LabelAnnotator,
    "blur": BlurAnnotator,
    "trace": TraceAnnotator,
    "heat_map": HeatMapAnnotator,
    "pixelate": PixelateAnnotator,
    "triangle": TriangleAnnotator,
}

VIDEO_EXTENSIONS = {
    "mp4",
    "webm",
    "mkv",
    "flv",
    "vob",
    "ogg",
    "ogv",
    "avi",
    "mov",
    "qt",
    "wmv",
}


def infer(
    input_reference: Union[str, int],
    model_id: str,
    api_key: Optional[str],
    host: str,
    output_location: Optional[str],
    display: bool,
    visualise: bool,
    visualisation_config: Optional[str],
    model_configuration: Optional[str],
) -> None:
    if api_key is None:
        api_key = ROBOFLOW_API_KEY
    if input_reference.split(".")[-1] in VIDEO_EXTENSIONS:
        infer_on_video(
            input_reference=input_reference,
            model_id=model_id,
            api_key=api_key,
            host=host,
            output_location=output_location,
            display=display,
            visualise=visualise,
            visualisation_config=visualisation_config,
            model_configuration=model_configuration,
        )
        return None
    if os.path.isdir(input_reference):
        infer_on_directory(
            input_reference=input_reference,
            model_id=model_id,
            api_key=api_key,
            host=host,
            output_location=output_location,
            display=display,
            visualise=visualise,
            visualisation_config=visualisation_config,
            model_configuration=model_configuration,
        )
        return None
    infer_on_image(
        input_reference=input_reference,
        model_id=model_id,
        api_key=api_key,
        host=host,
        output_location=output_location,
        display=display,
        visualise=visualise,
        visualisation_config=visualisation_config,
        model_configuration=model_configuration,
    )


class NullVideoSink:
    def write_frame(self, **kwargs) -> None:
        pass

    def __enter__(self) -> "NullVideoSink":
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        pass


def infer_on_video(
    input_reference: Union[str, int],
    model_id: str,
    api_key: Optional[str],
    host: str,
    output_location: Optional[str],
    display: bool,
    visualise: bool,
    visualisation_config: Optional[str],
    model_configuration: Optional[str],
) -> None:
    if not is_something_to_do(
        output_location=output_location, display=display, visualise=visualise
    ):
        print(
            "Inference from video requires `output_location` to be given or both "
            "`display` and `visualise` options to be requested."
        )
        return None
    client = initialise_client(
        host=host,
        api_key=api_key,
        model_configuration=model_configuration,
    )
    on_frame_visualise = None
    if visualise:
        on_frame_visualise = build_visualisation_callback(
            visualisation_config=visualisation_config,
        )
    input_reference_extension = os.path.basename(input_reference).split(".")[-1]
    video_sink = NullVideoSink()
    if visualise and output_location is not None:
        video_info = VideoInfo.from_video_path(video_path=input_reference)
        video_sink = VideoSink(
            target_path=os.path.join(
                output_location, f"visualisation.{input_reference_extension}"
            ),
            video_info=video_info,
        )
    with video_sink:
        for reference, frame, prediction in tqdm(
            client.infer_on_stream(input_uri=input_reference, model_id=model_id),
            desc=f"Inference on video: {input_reference}",
        ):
            visualised = None
            if visualise:
                visualised = on_frame_visualise(frame, prediction)
            if display and visualised is not None:
                cv2.imshow("Visualisation", visualised)
                cv2.waitKey(1)
            if visualised is not None:
                video_sink.write_frame(frame=visualised)
            if output_location is not None:
                save_prediction(
                    reference=reference,
                    prediction=prediction,
                    output_location=output_location,
                )
    if display:
        cv2.destroyAllWindows()


def infer_on_directory(
    input_reference: str,
    model_id: str,
    api_key: Optional[str],
    host: str,
    output_location: Optional[str],
    display: bool,
    visualise: bool,
    visualisation_config: Optional[str],
    model_configuration: Optional[str],
) -> None:
    if not is_something_to_do(
        output_location=output_location, display=display, visualise=visualise
    ):
        print(
            "Inference from directory requires `output_location` to be given or both "
            "`display` and `visualise` options to be requested."
        )
        return None
    client = initialise_client(
        host=host,
        api_key=api_key,
        model_configuration=model_configuration,
    )
    on_frame_visualise = None
    if visualise:
        on_frame_visualise = build_visualisation_callback(
            visualisation_config=visualisation_config,
        )
    for reference, frame, prediction in tqdm(
        client.infer_on_stream(input_uri=input_reference, model_id=model_id),
        desc=f"Inference from directory: {input_reference}",
    ):
        visualised = None
        if visualise:
            visualised = on_frame_visualise(frame, prediction)
        if display and visualised is not None:
            cv2.imshow("Visualisation", visualised)
            cv2.waitKey(1)
        if output_location is not None:
            save_prediction(
                reference=reference,
                prediction=prediction,
                output_location=output_location,
            )
            if visualised is not None:
                save_visualisation_image(
                    reference=reference,
                    visualisation=visualised,
                    output_location=output_location,
                )


def infer_on_image(
    input_reference: str,
    model_id: str,
    api_key: Optional[str],
    host: str,
    output_location: Optional[str],
    display: bool,
    visualise: bool,
    visualisation_config: Optional[str],
    model_configuration: Optional[str],
) -> None:
    client = initialise_client(
        host=host,
        api_key=api_key,
        model_configuration=model_configuration,
    )
    on_frame_visualise = None
    if visualise:
        on_frame_visualise = build_visualisation_callback(
            visualisation_config=visualisation_config,
        )
    prediction = client.infer(inference_input=input_reference, model_id=model_id)
    visualised = None
    if visualise:
        frame_base64 = load_image_from_string(reference=input_reference)[0]
        frame_bytes = base64.b64decode(frame_base64)
        frame = bytes_to_opencv_image(payload=frame_bytes)
        visualised = on_frame_visualise(frame, prediction)
    print(prediction)
    if display and visualised is not None:
        cv2.imshow("Visualisation", visualised)
        cv2.waitKey(0)
    if output_location is not None:
        save_prediction(
            reference=input_reference,
            prediction=prediction,
            output_location=output_location,
        )
        if visualised is not None:
            save_visualisation_image(
                reference=input_reference,
                visualisation=visualised,
                output_location=output_location,
            )


def is_something_to_do(
    output_location: Optional[str],
    display: bool,
    visualise: bool,
) -> bool:
    return output_location is not None or (display is True and visualise is True)


def build_visualisation_callback(
    visualisation_config: Optional[str],
) -> Callable[[np.ndarray, dict], Optional[np.ndarray]]:
    annotators = [BoundingBoxAnnotator()]
    byte_tracker = None
    if visualisation_config is not None:
        raw_configuration = retrieve_visualisation_config(
            visualisation_config=visualisation_config,
        )
        annotators = initialise_annotators(
            annotators_config=raw_configuration["annotators"]
        )
        byte_tracker = initialise_byte_track(config=raw_configuration.get("tracking"))
    return partial(create_visualisation, annotators=annotators, tracker=byte_tracker)


def retrieve_visualisation_config(visualisation_config: str) -> dict:
    if os.path.isfile(visualisation_config):
        return read_yaml_file(file_path=visualisation_config)
    all_configs = glob(os.path.join(CONFIGS_DIR_PATH, "*.yml"))
    config_name2path = {os.path.basename(path): path for path in all_configs}
    if f"{visualisation_config}.yml" not in config_name2path:
        raise ValueError(f"Could not find config with reference {visualisation_config}")
    return read_yaml_file(file_path=config_name2path[f"{visualisation_config}.yml"])


def initialise_annotators(
    annotators_config: List[dict],
) -> List[BaseAnnotator]:
    annotators = []
    for annotator_config in annotators_config:
        annotator_type = annotator_config["type"]
        annotator_parameters = annotator_config["params"]
        if annotator_type not in ANNOTATOR_TYPE2CLASS:
            raise ValueError(
                f"Could not recognise annotator type: {annotator_type}. "
                f"Allowed values: {list(ANNOTATOR_TYPE2CLASS.keys())}."
            )
        annotator = ANNOTATOR_TYPE2CLASS[annotator_type](**annotator_parameters)
        annotators.append(annotator)
    return annotators


def initialise_byte_track(config: Optional[dict]) -> Optional[ByteTrack]:
    if config is None:
        return None
    return ByteTrack(**config)


def create_visualisation(
    frame: np.ndarray,
    prediction: dict,
    annotators: List[BaseAnnotator],
    tracker: Optional[ByteTrack],
) -> Optional[np.ndarray]:
    try:
        detections = Detections.from_inference(prediction)
        if tracker is not None:
            detections = tracker.update_with_detections(detections=detections)
        frame_copy = frame.copy()
        for annotator in annotators:
            frame_copy = annotator.annotate(scene=frame_copy, detections=detections)
        return frame_copy
    except KeyError:
        CLI_LOGGER.warning(
            "Could not visualise prediction. Probably visualisation was requested against model that does "
            "not produce detections."
        )
        return None


def save_prediction(
    reference: Union[str, int],
    prediction: dict,
    output_location: str,
) -> None:
    target_path = prepare_target_path(
        reference=reference, output_location=output_location, extension="json"
    )
    dump_json(path=target_path, content=prediction)


def save_visualisation_image(
    reference: Union[str, int],
    visualisation: np.ndarray,
    output_location: str,
) -> None:
    target_path = prepare_target_path(
        reference=reference, output_location=output_location, extension="jpg"
    )
    cv2.imwrite(target_path, visualisation)


def prepare_target_path(
    reference: Union[str, int],
    output_location: str,
    extension: str,
) -> str:
    if issubclass(type(reference), int):
        reference_number = str(reference).zfill(6)
        file_name = f"frame_{reference_number}.{extension}"
    else:
        file_name = ".".join(os.path.basename(reference).split(".")[:-1])
        file_name = f"{file_name}_prediction.{extension}"
    return os.path.join(output_location, file_name)
