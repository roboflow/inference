You can upload [supported weights](#supported_models) to Roboflow and deploy them to your device.

This is ideal if you have already trained a model outside of Roboflow that you want to deploy with Inference.

To upload weights to Roboflow, you will need:

1. A Roboflow account
2. A project with your dataset (that does not have a trained model)

To learn how to create a project and a dataset, refer to these guides:

- [Create a project](https://docs.roboflow.com/datasets/create-a-project)
- [Create a dataset](https://docs.roboflow.com/datasets/create-a-dataset-version)

Once you have a project with a dataset, you can upload your weights.

Install the Roboflow Python package:

```bash
pip install roboflow
```

Then, create a new Python file and add the following code:

```python
import roboflow

roboflow.login()

rf = roboflow.Roboflow()
project = rf.project("your-project-id")
version = project.version(1)
version.deploy("model-type", "path/to/training/results/")
```

<a name="supported_models">The following model types are supported:</a>

|Model Architecture|Task      |Model Type ID                 |
|------------------|----------------|-------------------|
|YOLOv5            |Object Detection|yolov5             |
|YOLOv5            |Segmentation    |yolov5-seg         |
|YOLOv7            |Object Detection|yolov7-seg         |
|YOLOv8            |Object Detection|yolov8             |
|YOLOv8            |Segmentation    |yolov8-seg         |
|YOLOv8            |Classification  |yolov8-cls         |
|YOLOv8            |Pose Estimation |yolov8-pose        |
|YOLOv9            |Object Detection|yolov9             |
|YOLOv9            |Segmentation    |yolov9             |
|YOLO-NAS          |Object Detection|yolonas            |
|YOLOv10           |Object Detection|yolov10            |
|PaliGemma         |Multimodal      |paligemma-3b-pt-224|
|PaliGemma         |Multimodal      |paligemma-3b-pt-448|
|PaliGemma         |Multimodal      |paligemma-3b-pt-896|
|Florence-2        |Multimodal      |florence-2-large   |
|Florence-2        |Multimodal      |florence-2-base    |

In the code above, replace:

1. `your-project-id` with the ID of your project. [Learn how to retrieve your Roboflow project ID](https://docs.roboflow.com/api-reference/workspace-and-project-ids).
2. `1` with the version number of your project.
3. `model-type` with the model type you want to deploy.
4. `path/to/training/results/` with the path to the weights you want to upload. This path will vary depending on what model architecture you are using.

Your model weights will be uploaded to Roboflow. It may take a few minutes for your weights to be processed. Once your weights have been processed, your dataset version page will be updated to say that a model is available with your weights.

You can then use the model with Inference following our [Run a Private, Fine-Tuned Model](/quickstart/explore_models.md#run-a-private-fine-tuned-model) model.
