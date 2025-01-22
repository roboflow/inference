# Dedicated Deployments

Dedicated Deployments are private cloud servers managed by Roboflow, specifically designed to run your computer vision models.

Dedicated Deployments run Roboflow Inference.

You can set up two types of Dedicated Deployments:

1. Development, which spins up a machine that you can access for three hours for use in testing your model and logic.
2. Production, which spins up a machine that you can use for production applications. This machine will run until you turn it off.

Dedicated Deployments can both models trained or hosted on Roboflow, foundation models, and Roboflow Workflows.

Here are a few model types available:

- Object detection
- Image segmentation
- Classification
- Keypoint detection
- CLIP
- SAM-2
- Florence-2
- PaliGemma
- Roboflow Workflows

You can create Dedicated Deployments in the Roboflow web application or using the Roboflow Command Line Interface.

## Why Use Dedicated Deployments?

1. **Focus on your machine vision business problem, leave the infrastructure to us**: Spin up inference serving infrastructure with a few clicks and without having to signup with cloud providers, installing and securing servers, managing TLS certificates or worrying about server management, patching, updates etc.
2. **Dedicated Resources**: Get cloud servers allocated specifically for your use, ensuring consistent performance for your models. 
3. **Secure Access**: Dedicated Deployments are accessible with your workspace's unique API key and utilize HTTPS for secure communication.
4. **Easy Integration**: Each deployment receives a subdomain within roboflow.cloud, simplifying integration with your applications.

!!! note
    All dedicated deployments are currently hosted in US-based data centers; users from other Geographies may see higher latencies. Please contact us for a customized solution if you are outside of US, we can help you to reduce the network latency.

## Create a Dedicated Deployment from the Web App

Go to your Roboflow Dashboard and click "Deployments" in the right sidebar.

This will take you to your deployments dashboard.

![](https://media.roboflow.com/inference/deployments_tab.png)

Click "New Deployment" to create a new Dedicated Deployment.

![](https://media.roboflow.com/inference/deployments_window.png)

A modal will appear in which you can configure your server.

You can configure the following values:

| **Property**      | **Description**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Name**          | Choose a unique name (5–15 characters) to identify your Dedicated Deployment. This name will also become the subdomain for your deployment endpoint (e.g., `[invalid URL removed]`).<br><br>**Easy to Remember**: Pick a name that clearly reflects your deployment's purpose (e.g., "prod-inference", "dev-testing").<br>**Unique within Workspace**: If your chosen name is already taken, a short random code will be added to create a unique subdomain.<br><br>**Tips**:<br>- Use lowercase letters, numbers, and hyphens (-) for your name.<br>- Avoid special characters or spaces. |
| **Machine Type**  | Whether a CPU-only or a GPU dedicated deployment is needed.                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| **Deployment Type** | This is the deployment environment—Development (**dev**) or Production (**prod**).                                                                                                                                                                                                                                                                                                                                                                                                              |
| **Duration**      | This is the time for which the dedicated deployment remains online.<br><br>- **For development environments** this can range from 1–6 hours. Fractional hours are permitted, and they will be rounded to the nearest minute for billing purposes.<br>- **For production** there is no expiration time; the deployment will run until the user deletes it explicitly.                                                                                                                                 |

## Create a Dedicated Deployment from Workflows

Dedicated deployments can also be created from within Roboflow Workflows. Roboflow Workflows is a low-code, web-based application builder for creating computer vision applications.

There is no difference between provisioning a deployment from Workflows or the web app: both methods work the same.

To create a Dedicated Deployment, first create a Roboflow Workflow. To do so, click on Workflows on the left tab in the Roboflow dashboard, then create a Workflow.

Then, click on the Running on Hosted API link in the top left corner:

![](https://media.roboflow.com/inference/hosted_api_image.png)

Click Dedicated Deployments to create and see your Dedicated Deployments, the dialog presented here is identical to the one described above:

![](https://media.roboflow.com/inference/deployments_dialog.png)

When your Deployment is ready, the status will be updated to Ready. You can then click Connect to use your deployment with your Workflow in the Workflows editor:

![](https://media.roboflow.com/inference/workflows_deploy_connect)

Connecting to your Dedicated Deployment will allow you to use the deployment in your Workflow. This means you can run models like SAM-2 and Florence-2 that are only supported when running Workflows on-device or on a Dedicated Deployment.

## Use a Dedicated Deployment

With a Dedicated Deployment ready, you can use it to run inference images, video frames, and with Workflows.

### Roboflow Models

To use your Dedicated Deployment with models hosted on Roboflow:

1. Navigate to the Deployment tab of any model in your Workspace.
2. Click on "Hosted Image Inference".
3. Copy the code snippet and replace the API (i.e. `https://detect.roboflow.com`) with the URL of your Dedicated Deployment.

Here is an example:

```python
# import the inference-sdk
from inference_sdk import InferenceHTTPClient

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://yourdeployment.roboflow.cloud"
    api_key="YOUR_API_KEY"
)

# infer on a local image
result = CLIENT.infer("YOUR_IMAGE.jpg", model_id="counting-screws/3")
```

### Workflows

To deploy with a Dedicated Deployment in Workflows, go to any Workflow in your Roboflow workspace. Then, click "Deploy Workflow".

A window will appear with several deployment options. Select the "Run on an Image (Local)" or "Run on a Video (Local)" option, depending on how you want to deploy.

![Dedicated deployments](https://media.roboflow.com/inference/workflow_run.png)

Copy the code snippet relevant to you, and replace the API URL with the URL of your Dedicated Deployment.

## Provision and Manage Dedicated Deployments (Roboflow CLI)

The roboflow deployment command provides a set of subcommands to manage your Roboflow Dedicated Deployments. These deployments allow you to run inference on your computer vision models on dedicated servers.

Subcommands

- `machine_type`: List available machine types for your Dedicated Deployments. Please be noted that the output is a combination of Deployment Type and Machine Type mentioned above, i.e., dev-cpu, dev-gpu, prod-cpu, prod-gpu.

- `add`: Create a new Dedicated Deployment.

- `get`: Get detailed information about a specific Dedicated Deployment.

- `list`: List all Dedicated Deployments in your workspace.

- `usage_workspace`: Get usage statistics for all Dedicated Deployments in your workspace.

- `usage_deployment`: Get usage statistics for a specific Dedicated Deployment.

- `delete`: Delete a Dedicated Deployment.

- `log`: View logs for a specific Dedicated Deployment.

### Subcommand Examples

#### Create a new deployment

```
roboflow deployment add my-deployment -m prod-gpu
```

#### Get deployment information

```
roboflow deployment get my-deployment
```

#### List all deployments

```
roboflow deployment list
```

#### Get workspace usage

```
roboflow deployment usage_workspace
```

#### Get deployment usage

```
roboflow deployment usage_deployment my-deployment
```

#### Delete a deployment

```
roboflow deployment delete my-deployment
```

####  View deployment logs

```
roboflow deployment log my-deployment -t 60 -n 20
```