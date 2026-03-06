from inference import get_model
from PIL import Image

def test_qwen_base_model():
    # Load the base Qwen model
    model = get_model("qwen25-vl-7b", api_key = "Zx9aEOJ2pRDCbkkZTSD9")

    # Load the image
    image = Image.open("banner.png")

    # Run inference with the prompt
    result = model.predict(image, prompt="What's in this image?")

    # Print the result
    print("Model response:")
    print(result)

    # Assert that we got a non-empty response
    assert result is not None
    assert len(result) > 0
    assert isinstance(result[0], str)
    assert len(result[0]) > 0

    print("\nTest passed!")

if __name__ == "__main__":
    test_qwen_base_model()
