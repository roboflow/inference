TrOCR is a transformer-based model for text recognition, otherwise known as Optical Character Recognition (OCR).

TrOCR works best on focused, single-line printed text.

Be sure to use with cropped images since unlike some other OCR models, TrOCR will not perform well on uncropped or multi-line text.

Let's try running TrOCR on this image:

![Serial number](https://media.roboflow.com/serial_number.png)

!!! note
    
    TROCR model is only supported in `inference` Python package and `inference` server deployed locally (excluding
    Roboflow Hosted Platform).
    
    To run the example, start `inference` server locally:

    ```bash
    inference server start
    ```

    Make sure you have `inference-cli` installed - if that's not the case run:

    ```bash
    pip install inference-cli
    ```

```python
import os
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(api_url="http://127.0.0.1:9001")

result = CLIENT.ocr_image(inference_input="./serial_number.png", model="trocr")  # single image request
print(result)
```
