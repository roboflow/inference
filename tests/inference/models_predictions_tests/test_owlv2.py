import gc
import os
from threading import RLock
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from inference.core.cache.model_artifacts import get_cache_file_path
from inference.core.entities.requests.inference import ObjectDetectionInferenceRequest
from inference.core.entities.requests.owlv2 import OwlV2InferenceRequest
from inference.core.env import OWLV2_VERSION_ID
from inference.models.owlv2.owlv2 import (
    LazyImageRetrievalWrapper,
    OwlV2,
    Owlv2Singleton,
    SerializedOwlV2,
)


@pytest.mark.slow
def test_owlv2():
    image = {
        "type": "url",
        "value": "https://media.roboflow.com/inference/seawithdock.jpeg",
    }

    # test we can handle a single positive prompt
    request = OwlV2InferenceRequest(
        image=image,
        training_data=[
            {
                "image": image,
                "boxes": [
                    {
                        "x": 223,
                        "y": 306,
                        "w": 40,
                        "h": 226,
                        "cls": "post",
                        "negative": False,
                    },
                ],
            }
        ],
        visualize_predictions=True,
        confidence=0.9,
    )

    response = OwlV2(model_id=f"owlv2/{OWLV2_VERSION_ID}").infer_from_request(request)
    # we assert that we're finding all of the posts in the image
    assert len(response.predictions) == 5
    # next we check the x coordinates to force something about localization
    # the exact value here is sensitive to:
    # 1. the image interpolation mode used
    # 2. the data type used in the model, ie bfloat16 vs float16 vs float32
    # 3. the size of the model itself, ie base vs large
    # 4. the specific hardware used to run the model
    # we set a tolerance of 1.5 pixels from the expected value, which should cover most of the cases
    # first we sort by x coordinate to make sure we're getting the correct post
    posts = [p for p in response.predictions if p.class_name == "post"]
    posts.sort(key=lambda x: x.x)
    assert abs(223 - posts[0].x) < 1.5
    assert abs(248 - posts[1].x) < 1.5
    assert abs(264 - posts[2].x) < 1.5
    assert abs(532 - posts[3].x) < 1.5
    assert abs(572 - posts[4].x) < 1.5


def test_owlv2_serialized():
    image = {
        "type": "url",
        "value": "https://media.roboflow.com/inference/seawithdock.jpeg",
    }

    training_data = [
        {
            "image": image,
            "boxes": [
                {
                    "x": 223,
                    "y": 306,
                    "w": 40,
                    "h": 226,
                    "cls": "post",
                    "negative": False,
                },
            ],
        }
    ]
    model_id = "test/test_id"
    request = ObjectDetectionInferenceRequest(
        model_id=model_id,
        image=image,
        visualize_predictions=True,
        confidence=0.9,
    )

    SerializedOwlV2.download_model_artifacts_from_roboflow_api = MagicMock()
    serialized_pt = SerializedOwlV2.serialize_training_data(
        training_data=training_data,
        hf_id=f"google/{OWLV2_VERSION_ID}",
    )
    assert os.path.exists(serialized_pt)
    pt_path = get_cache_file_path(
        file=SerializedOwlV2.weights_file_path, model_id=model_id
    )
    os.makedirs(os.path.dirname(pt_path), exist_ok=True)
    os.rename(serialized_pt, pt_path)
    serialized_owlv2 = SerializedOwlV2(model_id=model_id)

    # Get the image hash before inference
    image_wrapper = LazyImageRetrievalWrapper(request.image)
    image_hash = image_wrapper.image_hash
    assert image_hash in serialized_owlv2.owlv2.cpu_image_embed_cache

    response = serialized_owlv2.infer_from_request(request)

    assert len(response.predictions) == 5
    posts = [p for p in response.predictions if p.class_name == "post"]
    posts.sort(key=lambda x: x.x)
    assert abs(223 - posts[0].x) < 1.5
    assert abs(248 - posts[1].x) < 1.5
    assert abs(264 - posts[2].x) < 1.5
    assert abs(532 - posts[3].x) < 1.5
    assert abs(572 - posts[4].x) < 1.5

    pt_path = serialized_owlv2.save_small_model_without_image_embeds()
    assert os.path.exists(pt_path)
    pt_dict = torch.load(pt_path)
    assert len(pt_dict["image_embeds"]) == 0


@pytest.mark.slow
def test_owlv2_multiple_prompts():
    image = {
        "type": "url",
        "value": "https://media.roboflow.com/inference/seawithdock.jpeg",
    }

    # test we can handle multiple (positive and negative) prompts for the same image
    request = OwlV2InferenceRequest(
        image=image,
        training_data=[
            {
                "image": image,
                "boxes": [
                    {
                        "x": 223,
                        "y": 306,
                        "w": 40,
                        "h": 226,
                        "cls": "post",
                        "negative": False,
                    },
                    {
                        "x": 247,
                        "y": 294,
                        "w": 25,
                        "h": 165,
                        "cls": "post",
                        "negative": True,
                    },
                    {
                        "x": 264,
                        "y": 327,
                        "w": 21,
                        "h": 74,
                        "cls": "post",
                        "negative": False,
                    },
                ],
            }
        ],
        visualize_predictions=True,
        confidence=0.9,
    )

    response = OwlV2().infer_from_request(request)
    assert len(response.predictions) == 4
    posts = [p for p in response.predictions if p.class_name == "post"]
    posts.sort(key=lambda x: x.x)
    assert abs(223 - posts[0].x) < 1.5
    assert abs(264 - posts[1].x) < 1.5
    assert abs(532 - posts[2].x) < 1.5
    assert abs(572 - posts[3].x) < 1.5


@pytest.mark.slow
def test_owlv2_image_without_prompts():
    image = {
        "type": "url",
        "value": "https://media.roboflow.com/inference/seawithdock.jpeg",
    }

    # test that we can handle an image without any prompts
    request = OwlV2InferenceRequest(
        image=image,
        training_data=[
            {
                "image": image,
                "boxes": [
                    {
                        "x": 223,
                        "y": 306,
                        "w": 40,
                        "h": 226,
                        "cls": "post",
                        "negative": False,
                    }
                ],
            },
            {
                "image": image,
                "boxes": [],
            },
        ],
        visualize_predictions=True,
        confidence=0.9,
    )

    response = OwlV2().infer_from_request(request)
    assert len(response.predictions) == 5


@pytest.mark.slow
def test_owlv2_bad_prompt():
    image = {
        "type": "url",
        "value": "https://media.roboflow.com/inference/seawithdock.jpeg",
    }

    # test that we can handle a bad prompt
    request = OwlV2InferenceRequest(
        image=image,
        training_data=[
            {
                "image": image,
                "boxes": [
                    {
                        "x": 1,
                        "y": 1,
                        "w": 1,
                        "h": 1,
                        "cls": "post",
                        "negative": False,
                    }
                ],
            }
        ],
        visualize_predictions=True,
        confidence=0.9,
    )

    response = OwlV2().infer_from_request(request)
    assert len(response.predictions) == 0


@pytest.mark.slow
def test_owlv2_bad_prompt_hidden_among_good_prompts():
    image = {
        "type": "url",
        "value": "https://media.roboflow.com/inference/seawithdock.jpeg",
    }

    # test that we can handle a bad prompt
    request = OwlV2InferenceRequest(
        image=image,
        training_data=[
            {
                "image": image,
                "boxes": [
                    {
                        "x": 1,
                        "y": 1,
                        "w": 1,
                        "h": 1,
                        "cls": "post",
                        "negative": False,
                    },
                    {
                        "x": 223,
                        "y": 306,
                        "w": 40,
                        "h": 226,
                        "cls": "post",
                        "negative": False,
                    },
                ],
            }
        ],
        visualize_predictions=True,
        confidence=0.9,
    )

    response = OwlV2().infer_from_request(request)
    assert len(response.predictions) == 5


@pytest.mark.slow
def test_owlv2_no_training_data():
    image = {
        "type": "url",
        "value": "https://media.roboflow.com/inference/seawithdock.jpeg",
    }

    # test that we can handle no training data
    request = OwlV2InferenceRequest(
        image=image,
        training_data=[],
    )

    response = OwlV2().infer_from_request(request)
    assert len(response.predictions) == 0


@pytest.mark.slow
def test_owlv2_multiple_training_images():
    image = {
        "type": "url",
        "value": "https://media.roboflow.com/inference/seawithdock.jpeg",
    }
    second_image = {
        "type": "url",
        "value": "https://media.roboflow.com/inference/dock2.jpg",
    }

    request = OwlV2InferenceRequest(
        image=image,
        training_data=[
            {
                "image": image,
                "boxes": [
                    {
                        "x": 223,
                        "y": 306,
                        "w": 40,
                        "h": 226,
                        "cls": "post",
                        "negative": False,
                    }
                ],
            },
            {
                "image": second_image,
                "boxes": [
                    {
                        "x": 3009,
                        "y": 1873,
                        "w": 289,
                        "h": 811,
                        "cls": "post",
                        "negative": True,
                    }
                ],
            },
        ],
        visualize_predictions=True,
        confidence=0.9,
    )

    response = OwlV2().infer_from_request(request)
    assert len(response.predictions) == 5


@pytest.mark.slow
def test_owlv2_multiple_training_images_repeated_inference():
    image = {
        "type": "url",
        "value": "https://media.roboflow.com/inference/seawithdock.jpeg",
    }
    second_image = {
        "type": "url",
        "value": "https://media.roboflow.com/inference/dock2.jpg",
    }

    request = OwlV2InferenceRequest(
        image=image,
        training_data=[
            {
                "image": image,
                "boxes": [
                    {
                        "x": 223,
                        "y": 306,
                        "w": 40,
                        "h": 226,
                        "cls": "post",
                        "negative": False,
                    }
                ],
            },
            {
                "image": second_image,
                "boxes": [
                    {
                        "x": 3009,
                        "y": 1873,
                        "w": 289,
                        "h": 811,
                        "cls": "post",
                        "negative": True,
                    }
                ],
            },
        ],
        visualize_predictions=True,
        confidence=0.9,
    )

    model = OwlV2()
    first_response = model.infer_from_request(request)
    second_response = model.infer_from_request(request)
    for p1, p2 in zip(first_response.predictions, second_response.predictions):
        assert p1.class_name == p2.class_name
        assert p1.x == p2.x
        assert p1.y == p2.y
        assert p1.width == p2.width
        assert p1.height == p2.height
        assert p1.confidence == p2.confidence


@pytest.mark.slow
def test_owlv2_model_unloaded_when_garbage_collected():
    model = OwlV2()
    del model
    gc.collect()
    assert len(Owlv2Singleton._instances) == 0


def test_infer_with_numpy_image_uses_image_after_sizing() -> None:
    """Ensure numpy images persist through compute_image_size and embed_image."""

    class DummyOwl:
        def __init__(self):
            self.image_size_cache = {}
            self.class_embeddings_cache = {}
            self.image_embed_cache = {}
            self.cpu_image_embed_cache = {}
            self.before_unload_image_none = False
            self.after_unload = False
            self.owlv2_lock = RLock()

        compute_image_size = OwlV2.compute_image_size
        infer = OwlV2.infer
        infer_from_embedding_dict = OwlV2.infer_from_embedding_dict

        def embed_image(self, image):
            # Image should still be loaded when embed_image is called
            self.before_unload_image_none = (
                image.image is None or image._image_as_numpy is None
            )
            # simulate embedding
            _ = image.image_as_numpy
            image.unload_numpy_image()
            self.after_unload = image.image is None and image._image_as_numpy is None
            return "hash"

        def infer_from_embed(self, *args, **kwargs):
            return []

        def make_response(self, predictions, image_sizes, class_names):
            return []

        def make_class_embeddings_dict(self, *args, **kwargs):
            return {}

    owl = DummyOwl()
    image_as_numpy = np.zeros((192, 168, 3), dtype=np.uint8)
    result = owl.infer(
        image_as_numpy, training_data=[{"image": image_as_numpy, "boxes": []}]
    )

    assert result == []
    assert owl.before_unload_image_none is False
    assert owl.after_unload is True


if __name__ == "__main__":
    test_owlv2()
