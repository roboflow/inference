import os

import pytest
import requests

api_key = os.environ.get("API_KEY")
port = os.environ.get("PORT", 9001)
base_url = os.environ.get("BASE_URL", "http://localhost")

DOG_IMAGE_URL = "https://media.roboflow.com/dog.jpeg"

# Keep in sync with RFDETR_ALIASES in inference/models/aliases.py.
# The legacy /{dataset_id}/{version_id} route cannot carry a slash-less alias —
# clients (inference_sdk) resolve aliases before calling the server — so the
# legacy tests hit the resolved model IDs, while the v1 /infer/* tests pass the
# raw alias to exercise server-side alias resolution.
RFDETR_DETECTION_ALIASES = {
    "rfdetr-base": "coco/36",
    "rfdetr-nano": "coco/38",
    "rfdetr-small": "coco/39",
    "rfdetr-medium": "coco/40",
    "rfdetr-large": "coco/50",
    "rfdetr-xlarge": "coco/47",
    "rfdetr-2xlarge": "coco/48",
}
RFDETR_SEGMENTATION_ALIASES = {
    "rfdetr-seg-preview": "coco-dataset-vdnr1/26",
    "rfdetr-seg-nano": "coco-dataset-vdnr1/41",
    "rfdetr-seg-small": "coco-dataset-vdnr1/36",
    "rfdetr-seg-medium": "coco-dataset-vdnr1/37",
    "rfdetr-seg-large": "coco-dataset-vdnr1/38",
    "rfdetr-seg-xlarge": "coco-dataset-vdnr1/39",
    "rfdetr-seg-2xlarge": "coco-dataset-vdnr1/40",
}


def bool_env(val):
    if isinstance(val, bool):
        return val
    return val.lower() in ["true", "1", "t", "y", "yes"]


pytestmark = pytest.mark.skipif(
    bool_env(os.getenv("SKIP_RFDETR_TEST", False)),
    reason="Skipping RFDETR test",
)


def assert_instances_detected(response: requests.Response) -> None:
    response.raise_for_status()
    data = response.json()
    assert "predictions" in data, f"Response lacks 'predictions' key: {data}"
    assert (
        len(data["predictions"]) > 0
    ), f"Expected at least one instance detected in {DOG_IMAGE_URL}"


@pytest.mark.parametrize("alias", list(RFDETR_DETECTION_ALIASES))
def test_rfdetr_detection_via_v1_endpoint(
    alias: str, clean_loaded_models_every_test_fixture
) -> None:
    payload = {
        "model_id": alias,
        "image": {"type": "url", "value": DOG_IMAGE_URL},
        "api_key": api_key,
    }

    response = requests.post(
        f"{base_url}:{port}/infer/object_detection",
        json=payload,
    )

    assert_instances_detected(response)


@pytest.mark.parametrize(
    "model_id",
    list(RFDETR_DETECTION_ALIASES.values()),
    ids=list(RFDETR_DETECTION_ALIASES),
)
def test_rfdetr_detection_via_legacy_endpoint(
    model_id: str, clean_loaded_models_every_test_fixture
) -> None:
    response = requests.post(
        f"{base_url}:{port}/{model_id}",
        params={
            "api_key": api_key,
            "image": DOG_IMAGE_URL,
        },
    )

    assert_instances_detected(response)


@pytest.mark.parametrize("alias", list(RFDETR_SEGMENTATION_ALIASES))
def test_rfdetr_segmentation_via_v1_endpoint(
    alias: str, clean_loaded_models_every_test_fixture
) -> None:
    payload = {
        "model_id": alias,
        "image": {"type": "url", "value": DOG_IMAGE_URL},
        "api_key": api_key,
    }

    response = requests.post(
        f"{base_url}:{port}/infer/instance_segmentation",
        json=payload,
    )

    assert_instances_detected(response)


@pytest.mark.parametrize(
    "model_id",
    list(RFDETR_SEGMENTATION_ALIASES.values()),
    ids=list(RFDETR_SEGMENTATION_ALIASES),
)
def test_rfdetr_segmentation_via_legacy_endpoint(
    model_id: str, clean_loaded_models_every_test_fixture
) -> None:
    response = requests.post(
        f"{base_url}:{port}/{model_id}",
        params={
            "api_key": api_key,
            "image": DOG_IMAGE_URL,
        },
    )

    assert_instances_detected(response)
