from inference.core.entities.requests.owlv2 import OwlV2InferenceRequest
from inference.core.entities.responses.inference import ObjectDetectionInferenceResponse
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
                "boxes": [{"x": 223, "y": 306, "w": 40, "h": 226, "cls": "post", "negative": False}],
            }
        ],
        visualize_predictions=True,
    )

    response = OwlV2().infer_from_request(request)
    assert abs(221.4 - response.predictions[0].x) < 0.1
