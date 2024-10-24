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
                "boxes": [{"x": 223, "y": 306, "w": 40, "h": 226, "cls": "post"}],
            }
        ],
        visualize_predictions=True,
    )

    response = OwlV2().infer_from_request(request)
    # the exact value here is highly sensitive to the interpolation mode used
    # as well as the data type used in the model, ie bfloat16 vs float16 vs float32
    assert abs(222.4 - response.predictions[0].x) < 0.1


if __name__ == "__main__":
    # run a simple latency test
    image_via_url = {
        "type": "url",
        "value": "https://media.roboflow.com/inference/seawithdock.jpeg",
    }

    import requests
    import tempfile
    from PIL import Image
    from inference.core.utils.image_utils import load_image_rgb
    import time

    # Download the image
    response = requests.get(image_via_url["value"])
    response.raise_for_status()

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_file.write(response.content)
        temp_file_path = temp_file.name

    img = Image.open(temp_file_path)
    img = img.convert("RGB")
    
    request_dict = dict(
        image=img,
        training_data=[
            {
                "image": img,
                "boxes": [{"x": 223, "y": 306, "w": 40, "h": 226, "cls": "post"}],
            }
        ],
        visualize_predictions=False,
    )

    model = OwlV2()

    for _ in range(10):
        print("pre cache fill try")
        time_start = time.time()
        response = model.infer(**request_dict)
        time_end = time.time()
        print(f"Time taken: {time_end - time_start} seconds")

        print("post cache fill try")
        time_start = time.time()
        response = model.infer(**request_dict)
        time_end = time.time()
        print(f"Time taken: {time_end - time_start} seconds")

        model.reset_cache()
