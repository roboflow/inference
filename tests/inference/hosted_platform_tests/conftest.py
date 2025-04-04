import logging
import os
from enum import Enum
from functools import partial
from typing import Any, Tuple

import numpy as np
import pytest

from inference_sdk import InferenceHTTPClient


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if not issubclass(type(value), str):
        raise ValueError(
            f"Expected a boolean environment variable (true or false) but got '{value}'"
        )
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    else:
        raise ValueError(
            f"Expected a boolean environment variable (true or false) but got '{value}'"
        )


MAXIMUM_CONSECUTIVE_INVOCATION_ERRORS = int(
    os.getenv("HOSTED_PLATFORM_TESTS_MAX_WARMUP_CONSECUTIVE_ERRORS", 10)
)
MINIMUM_NUMBER_OF_SUCCESSFUL_RESPONSES = int(
    os.getenv("HOSTED_PLATFORM_TESTS_MIN_WARMUP_SUCCESS_RESPONSES", 5)
)
SKIP_WARMUP = str2bool(os.getenv("SKIP_WARMUP", False))
IMAGE_URL = "https://media.roboflow.com/inference/dog.jpeg"


class PlatformEnvironment(Enum):
    ROBOFLOW_STAGING = "roboflow-staging"
    ROBOFLOW_PLATFORM = "roboflow-platform"
    ROBOFLOW_STAGING_LOCALHOST = "roboflow-staging-localhost"
    ROBOFLOW_PLATFORM_LOCALHOST = "roboflow-platform-localhost"


SERVICES_URLS = {
    PlatformEnvironment.ROBOFLOW_PLATFORM: {
        "object-detection": "https://detect.roboflow.com",
        "instance-segmentation": "https://outline.roboflow.com",
        "classification": "https://classify.roboflow.com",
        "core-models": "https://infer.roboflow.com",
    },
    PlatformEnvironment.ROBOFLOW_PLATFORM_LOCALHOST: {
        "object-detection": "http://127.0.0.1:9001",
        "instance-segmentation": "http://127.0.0.1:9001",
        "classification": "http://127.0.0.1:9001",
        "core-models": "http://127.0.0.1:9001",
    },
    PlatformEnvironment.ROBOFLOW_STAGING: {
        "object-detection": "https://lambda-object-detection.staging.roboflow.com",
        "instance-segmentation": "https://lambda-instance-segmentation.staging.roboflow.com",
        "classification": "https://lambda-classification.staging.roboflow.com",
        "core-models": "https://3hkaykeh3j.execute-api.us-east-1.amazonaws.com",
    },
}
SERVICES_URLS[PlatformEnvironment.ROBOFLOW_STAGING_LOCALHOST] = SERVICES_URLS[
    PlatformEnvironment.ROBOFLOW_PLATFORM_LOCALHOST
]

MODELS_TO_BE_USED = {
    PlatformEnvironment.ROBOFLOW_PLATFORM: {
        "object-detection": "coin-counting/137",
        "instance-segmentation": "asl-poly-instance-seg/53",
        "classification": "catdog-w9i9e/18",
        "multi_class_classification": "vehicle-classification-eapcd/2",
        "yolov8n-640": "yolov8n-640",
        "yolov8n-pose-640": "yolov8n-pose-640",
    },
    PlatformEnvironment.ROBOFLOW_STAGING: {
        "object-detection": "eye-detection/35",
        "instance-segmentation": "asl-instance-seg/116",
        "classification": "catdog/28",
        "multi_class_classification": "car-classification/23",
        "yolov8n-640": "microsoft-coco-obj-det/8",
        "yolov8n-pose-640": "microsoft-coco-pose/1",
    },
}
MODELS_TO_BE_USED[PlatformEnvironment.ROBOFLOW_STAGING_LOCALHOST] = MODELS_TO_BE_USED[
    PlatformEnvironment.ROBOFLOW_STAGING
]
MODELS_TO_BE_USED[PlatformEnvironment.ROBOFLOW_PLATFORM_LOCALHOST] = MODELS_TO_BE_USED[
    PlatformEnvironment.ROBOFLOW_PLATFORM
]

TARGET_PROJECTS_TO_BE_USED = {
    PlatformEnvironment.ROBOFLOW_PLATFORM: "active-learning-demo",
    PlatformEnvironment.ROBOFLOW_STAGING: "coin-counting",
}
TARGET_PROJECTS_TO_BE_USED[PlatformEnvironment.ROBOFLOW_STAGING_LOCALHOST] = (
    TARGET_PROJECTS_TO_BE_USED[PlatformEnvironment.ROBOFLOW_STAGING]
)
TARGET_PROJECTS_TO_BE_USED[PlatformEnvironment.ROBOFLOW_PLATFORM_LOCALHOST] = (
    TARGET_PROJECTS_TO_BE_USED[PlatformEnvironment.ROBOFLOW_PLATFORM]
)

INTERFACE_DISCOVERING_WORKFLOW = {
    PlatformEnvironment.ROBOFLOW_STAGING: ("paul-guerrie", "staging-test-workflow"),
    PlatformEnvironment.ROBOFLOW_PLATFORM: (
        "paul-guerrie-tang1",
        "prod-test-workflow",
    ),
}
INTERFACE_DISCOVERING_WORKFLOW[PlatformEnvironment.ROBOFLOW_STAGING_LOCALHOST] = (
    INTERFACE_DISCOVERING_WORKFLOW[PlatformEnvironment.ROBOFLOW_STAGING]
)
INTERFACE_DISCOVERING_WORKFLOW[PlatformEnvironment.ROBOFLOW_PLATFORM_LOCALHOST] = (
    INTERFACE_DISCOVERING_WORKFLOW[PlatformEnvironment.ROBOFLOW_PLATFORM]
)

ROBOFLOW_API_KEY = os.environ["HOSTED_PLATFORM_TESTS_API_KEY"]
OPENAI_KEY = os.getenv("OPENAI_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY")


@pytest.fixture(scope="session")
def platform_environment() -> PlatformEnvironment:
    return PlatformEnvironment(os.environ["HOSTED_PLATFORM_TESTS_PROJECT"])


@pytest.fixture(scope="session")
def classification_service_url(platform_environment: PlatformEnvironment) -> str:
    return SERVICES_URLS[platform_environment]["classification"]


@pytest.fixture(scope="session")
def object_detection_service_url(platform_environment: PlatformEnvironment) -> str:
    return SERVICES_URLS[platform_environment]["object-detection"]


@pytest.fixture(scope="session")
def instance_segmentation_service_url(platform_environment: PlatformEnvironment) -> str:
    return SERVICES_URLS[platform_environment]["instance-segmentation"]


@pytest.fixture(scope="session")
def core_models_service_url(platform_environment: PlatformEnvironment) -> str:
    return SERVICES_URLS[platform_environment]["core-models"]


@pytest.fixture(scope="session")
def classification_model_id(platform_environment: PlatformEnvironment) -> str:
    return MODELS_TO_BE_USED[platform_environment]["classification"]


@pytest.fixture(scope="session")
def multi_class_classification_model_id(
    platform_environment: PlatformEnvironment,
) -> str:
    return MODELS_TO_BE_USED[platform_environment]["multi_class_classification"]


@pytest.fixture(scope="session")
def detection_model_id(platform_environment: PlatformEnvironment) -> str:
    return MODELS_TO_BE_USED[platform_environment]["object-detection"]


@pytest.fixture(scope="session")
def yolov8n_640_model_id(platform_environment: PlatformEnvironment) -> str:
    return MODELS_TO_BE_USED[platform_environment]["yolov8n-640"]


@pytest.fixture(scope="session")
def yolov8n_pose_640_model_id(platform_environment: PlatformEnvironment) -> str:
    return MODELS_TO_BE_USED[platform_environment]["yolov8n-pose-640"]


@pytest.fixture(scope="session")
def segmentation_model_id(platform_environment: PlatformEnvironment) -> str:
    return MODELS_TO_BE_USED[platform_environment]["instance-segmentation"]


@pytest.fixture(scope="session")
def target_project(platform_environment: PlatformEnvironment) -> str:
    return TARGET_PROJECTS_TO_BE_USED[platform_environment]


@pytest.fixture(scope="session")
def interface_discovering_workflow(
    platform_environment: PlatformEnvironment,
) -> Tuple[str, str]:
    return INTERFACE_DISCOVERING_WORKFLOW[platform_environment]


@pytest.fixture(scope="session", autouse=True)
def warm_up_classification_lambda(
    classification_service_url: str,
    classification_model_id: str,
) -> None:
    if SKIP_WARMUP:
        return None
    warm_up_service_with_roboflow_model(
        api_url=classification_service_url,
        model_id=classification_model_id,
        api_key=ROBOFLOW_API_KEY,
    )


@pytest.fixture(scope="session", autouse=True)
def warm_up_instance_segmentation_lambda(
    instance_segmentation_service_url: str,
    segmentation_model_id: str,
) -> None:
    if SKIP_WARMUP:
        return None
    warm_up_service_with_roboflow_model(
        api_url=instance_segmentation_service_url,
        model_id=segmentation_model_id,
        api_key=ROBOFLOW_API_KEY,
    )


@pytest.fixture(scope="session", autouse=True)
def warm_up_detection_lambda(
    object_detection_service_url: str,
    detection_model_id: str,
) -> None:
    if SKIP_WARMUP:
        return None
    warm_up_service_with_roboflow_model(
        api_url=object_detection_service_url,
        model_id=detection_model_id,
        api_key=ROBOFLOW_API_KEY,
    )


@pytest.fixture(scope="session", autouse=True)
def warm_up_ocr_model(
    core_models_service_url: str,
) -> None:
    if SKIP_WARMUP:
        return None
    client = InferenceHTTPClient(
        api_url=core_models_service_url,
        api_key=ROBOFLOW_API_KEY,
    ).select_api_v0()
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    function = partial(client.ocr_image, inference_input=image)
    for _ in range(MINIMUM_NUMBER_OF_SUCCESSFUL_RESPONSES):
        retry_at_max_n_times(
            function=function,
            n=MAXIMUM_CONSECUTIVE_INVOCATION_ERRORS,
            function_description=f"warm up of OCR model",
        )


@pytest.fixture(scope="session", autouse=True)
def warm_up_clip_model(
    core_models_service_url: str,
) -> None:
    if SKIP_WARMUP:
        return None
    client = InferenceHTTPClient(
        api_url=core_models_service_url,
        api_key=ROBOFLOW_API_KEY,
    ).select_api_v0()
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    function = partial(client.clip_compare, subject=image, prompt=["cat", "dog"])
    for _ in range(MINIMUM_NUMBER_OF_SUCCESSFUL_RESPONSES):
        retry_at_max_n_times(
            function=function,
            n=MAXIMUM_CONSECUTIVE_INVOCATION_ERRORS,
            function_description=f"warm up of CLIP model",
        )


@pytest.fixture(scope="session", autouse=True)
def warm_up_yolo_world_model(
    core_models_service_url: str,
) -> None:
    if SKIP_WARMUP:
        return None
    client = InferenceHTTPClient(
        api_url=core_models_service_url,
        api_key=ROBOFLOW_API_KEY,
    ).select_api_v0()
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    function = partial(
        client.infer_from_yolo_world, inference_input=image, class_names=["cat", "dog"]
    )
    for _ in range(MINIMUM_NUMBER_OF_SUCCESSFUL_RESPONSES):
        retry_at_max_n_times(
            function=function,
            n=MAXIMUM_CONSECUTIVE_INVOCATION_ERRORS,
            function_description=f"warm up of CLIP model",
        )


def warm_up_service_with_roboflow_model(
    api_url: str,
    model_id: str,
    api_key: str,
) -> None:
    client = InferenceHTTPClient(
        api_url=api_url,
        api_key=api_key,
    ).select_api_v0()
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    function = partial(client.infer, inference_input=image, model_id=model_id)
    for _ in range(MINIMUM_NUMBER_OF_SUCCESSFUL_RESPONSES):
        retry_at_max_n_times(
            function=function,
            n=MAXIMUM_CONSECUTIVE_INVOCATION_ERRORS,
            function_description=f"warm up of {api_url}",
        )


def retry_at_max_n_times(function: callable, n: int, function_description: str) -> None:
    attempts = 1
    while attempts <= n:
        try:
            function()
        except Exception as e:
            logging.warning(
                f"Retrying function call. Error: {e}. Attempts: {attempts}/{n}"
            )
        else:
            return None
        attempts += 1
    raise Exception(f"Could not achieve success of {function_description}")
