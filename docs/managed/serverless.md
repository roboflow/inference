# Serverless Hosted API

You can run fine-tuned computer vision models hosted on Roboflow and Workflows built in Roboflow using the Roboflow Serverless API.

The Serverless API requires no server management and scales as you go. Whether you are running one hundred or a million inferences a month, the Serverless API is ready to use.

You may want to use the Serverless API if any of the following are true:

- You are running inference on single images.
- You need infrastructure that automatically scales.
- You only need the results from models and plan to write code to process the results.

If you want to build multi-stage vision applications that run a model without writing code for each step in your application, check out [Workflows](/workflows/about.md).

APIs are automatically set up for the following model types:

- Object detection
- Segmentation
- Classification
- Keypoint detection

## Find Your API Endpoint

To find your API endpoint, go to your project in Roboflow and click Deployments in the sidebar.

Then, click on the "Hosted Image Inference" deployment option:

![](https://media.roboflow.com/inference/deployment_list.png)

A window will appear with code snippets in various programming languages that you can use to run your model.

The Python code snippet runs the Inference SDK. This SDK lets you make requests to the Serverless API in a few lines of code.

Here is an example of what the code snippet will look like:

```python
# import the inference-sdk
from inference_sdk import InferenceHTTPClient

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="YOUR_API_KEY"
)

# infer on a local image
result = CLIENT.infer("YOUR_IMAGE.jpg", model_id="counting-screws/3")
```

This code will call the Roboflow Serverless API and return results from the inference.

We can then plot predictions with the Roboflow supervision package:

```python
detections = sv.Detections.from_inference(results)

# create supervision annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# annotate the image with our inference results
annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

# display the image
sv.plot_image(annotated_image)
```
