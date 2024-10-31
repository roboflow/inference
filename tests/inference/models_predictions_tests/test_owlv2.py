from inference.core.entities.requests.owlv2 import OwlV2InferenceRequest
from inference.models.owlv2.owlv2 import OwlV2


def test_owlv2():
    image = {
        "type": "url",
        "value": "https://media.roboflow.com/inference/seawithdock.jpeg",
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
    # we assert that we're finding all of the posts in the image
    assert len(response.predictions) == 4
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
    assert abs(264 - posts[1].x) < 1.5
    assert abs(532 - posts[2].x) < 1.5
    assert abs(572 - posts[3].x) < 1.5
