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
                "boxes": [{"x": 223, "y": 306, "w": 40, "h": 226, "cls": "post"}],
            }
        ],
        visualize_predictions=True,
    )

    response = OwlV2().infer_from_request(request)
    # the exact value here is highly sensitive to the image interpolation mode used
    # as well as the data type used in the model, ie bfloat16 vs float16 vs float32
    # and of course the size of the model itself, ie base vs large
    # we set a tolerance of 1.5 pixels from the expected value, which should cover most of the cases
    assert abs(223 - response.predictions[0].x) < 1.5
