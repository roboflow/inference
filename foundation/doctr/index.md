<a href="https://github.com/mindee/doctr" target="_blank">DocTR</a> is an Optical Character Recognition (OCR) model.

You can use DocTR to read the text in an image.

To use DocTR with Inference, you will need a Roboflow API key. If you don't already have a Roboflow account, <a href="https://app.roboflow.com" target="_blank">sign up for a free Roboflow account</a>. 

Then, retrieve your API key from the Roboflow dashboard. <a href="https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key" target="_blank">Learn how to retrieve your API key</a>.

Run the following command to set your API key in your coding environment:

```
export ROBOFLOW_API_KEY=<your api key>
```

Let's retrieve the text in the following image:

![A shipping container](https://lh7-us.googleusercontent.com/rBXP1ngqRAfez18KyFjSPHX5Keo_hgb3La72sV5npNTf_Te63_pSSdpUnq_OeD5teh9RFg17yftljNSCuyURdNRRstKMtq-eolVEHhQF0XwnVgyqq6vaj4WbrNa0VUXmBic89jlJbHDnTUT4sT1i-bw)

Create a new Python file and add the following code:

```python
import os
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://infer.roboflow.com",
    api_key=os.environ["ROBOFLOW_API_KEY"]
)

result = CLIENT.ocr_image(inference_input="./container.jpg")  # single image request
print(result)
```

Above, replace `container.jpeg` with the path to the image in which you want to detect objects.

The results of DocTR will appear in your terminal:

```
{'result': '', 'time': 3.98263641900121, 'result': 'MSKU 0439215', 'time': 3.870879542999319}
```

### Benchmarking

We ran 100 inferences on an NVIDIA T4 GPU to benchmark the performance of DocTR.

DocTR ran 100 inferences in 365.22 seconds (3.65 seconds per inference, on average).

## See Also

- <a href="https://blog.roboflow.com/ocr-api/" target="_blank">How to detect text in images with OCR</a>
