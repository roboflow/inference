# Deploy on Azure

You can run Roboflow Inference on machines hosted on Azure.

This is ideal if you want to benefit from all of the features Inference has to offer but also want to manage your own cloud infrastructure.

## Set up an Azure Cloud Compute VM

To get started, you will need a cloud compute instance running on Azure.

For deploying instances, we recommend SkyPilot, a tool designed to help you set up cloud instances for AI projects.

To get started, run the following command on your own machine:

```
pip install inference "skypilot[azure]"
```

Follow the [SkyPilot Azure documentation](https://docs.skypilot.co/en/latest/getting-started/installation.html#cloud-account-setup) to authenticate with AWS

Then, run:

```
inference cloud deploy --provider azure --compute-type gpu
```

This will provision a GPU-capable instance in Azure.

The latest version of Roboflow Inference will be automatically installed on the machine.

When the command has run, you should see a message like:

```
Deployed Roboflow Inference to azure on gpu, deployment name is ...
To get a list of your deployments: inference status
To delete your deployment: inference undeploy ...
To ssh into the deployed server: ssh ...
The Roboflow Inference Server is running at http://34.66.116.66:9001
```

You can then use the API endpoint for your server for use in running models.

You can run any model that Inference supports, including object detection, segmentation, classification, and keypoint models that you have available on Roboflow, and foundation models like CLIP, PaliGemma, SAM-2, and more.

## Run inference

You can run inference with Roboflow models using the following code:

```python
from inference_sdk import InferenceHTTPClient

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://yourendpoint.com"
    api_key="YOUR_API_KEY"
)

# infer on a local image
result = CLIENT.infer("YOUR_IMAGE.jpg", model_id="counting-screws/3")
```

Replace:

1. The API URL with your Azure instance IP and port.
2. API Key with your Roboflow API key.
3. YOUR_IMAGE with the image you want to run inference on.
4. Your model ID with your [Roboflow model ID](https://docs.roboflow.com/api-reference/workspace-and-project-ids).

You can also run Roboflow Workflows on your deployment.

[Learn how to run Workflows with Inference](/start/getting-started/#install-the-sdk).