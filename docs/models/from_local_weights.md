You can upload [supported weights](/models/supported_models/) to Roboflow and deploy them to your device.

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

The following model types are supported:

- yolov5, yolov5-seg
- yolov7, yolov7-seg
- yolov8, yolov8-seg, yolov8-cls

In the code above, replace:

1. `your-project-id` with the ID of your project. [Learn how to retrieve your Roboflow project ID](/docs/projects/where_is_my_project_id/).
2. `1` with the version number of your project.
3. `model-type` with the model type you want to deploy.
4. `path/to/training/results/` with the path to the weights you want to upload. This path will vary depending on what model architecture you are using.

Your model weights will be uploaded to Roboflow. It may take a few minutes for your weights to be processed. Once your weights have been processed, your dataset version page will be updated to say that a model is available with your weights.

You can then use the model with Inference following our [Run a Private, Fine-Tuned Model](https://inference.roboflow.com/quickstart/explore_models/#run-a-private-fine-tuned-model) model.