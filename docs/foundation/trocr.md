TrOCR is a transformer-based model for text recognition, otherwise known as Optical Character Recognition (OCR).

TrOCR works best on focused, single-line printed text.

Be sure to use with cropped images since unlike some other OCR models, TrOCR will not perform well on uncropped or multi-line text.

Let's try running TrOCR on this image:

![Serial number](https://media.roboflow.com/serial_number.png)

To use TrOCR with Inference, you will need a Roboflow API key. If you don't already have a Roboflow account, <a href="https://app.roboflow.com" target="_blank">sign up for a free Roboflow account</a>.

Then, retrieve your API key from the Roboflow dashboard. <a href="https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key" target="_blank">Learn how to retrieve your API key</a>.

Run the following command to set your API key in your coding environment:

```
export ROBOFLOW_API_KEY=<your api key>
```

```python
import os
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://infer.roboflow.com",
    api_key=os.environ["ROBOFLOW_API_KEY"]
)

result = CLIENT.ocr_image(inference_input="./serial_number.png", model_id="trocr")  # single image request
print(result)
```
