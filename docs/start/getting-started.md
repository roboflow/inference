# Build and Run a Workflow

Inference supports over 100 "blocks" with which you can make a Workflow.

A Workflow is a single or multi-stage computer vision application.

In this guide, we are going to make a Workflow that runs an object detection model and shows the bounding boxes and labels returned by the model.

We will use a pre-trained model to detect vehicles and other common objects in a video.

This Workflow can then be run on images, videos, RTSP streams, and more.

We will then add video tracking to our Workflow.

!!! note

    Before we get started, make sure you have [installed Inference](/install/start/).

## Step #1: Choose a Workflow Builder

You can build Workflows both on your own hardware and using the Roboflow web application. No matter where your Workflow is built, it can be deployed to the cloud or your own hardware.

=== "Build with Roboflow"

    To get started, create a [free Roboflow account](https://app.roboflow.com).

    Then, click "Workflows" in the right sidebar of the Roboflow web interface.

    Click "Create Workflow" to create a blank Workflow.

=== "Build on Device"

    You can run the Workflow builder on your own hardware.

    To get started, [install Inference and set up your server](/install/index/).

    Use `inference server start --dev` to set up your server.

    Then, go to `http://localhost:9001/build`.

    Click "Create Workflow" to create a blank Workflow.

You will then see a blank Workflow builder in which you can build your application:

![](https://media.roboflow.com/inference/get-started/blank.png)

## Step #2: Add a Detection Block

You can deploy both base models (i.e. RF-DETR Base) as well as models fine-tuned for a specific use case with Workflows.

Let's build an application that detects common objects in an image. For this purpose, we will use a base RF-DETR Base model pre-trained with the Microsoft COCO dataset. This model is free to use, so we don't need to do any set up.

Want to detect a custom object? [Learn how to fine-tune a model for your use case](https://blog.roboflow.com/getting-started-with-roboflow/).

Click "Add Model" in Workflows, then choose "Object Detection Model":

![](https://media.roboflow.com/inference/get-started/add.png)

A window will appear in which you can choose the model you want to use. Click "Public Models", then choose "RF-DETR Base":

![](https://media.roboflow.com/inference/get-started/choose-model.png)

Then, click "Save" to add the model to your Workflow.

## Step #3: Add a Visualizer

The Object Detection block runs an object detection model. By default, this block returns the predictions from the model: bounding boxes with their associated labels.

You can use Workflows to show bounding boxes and labels.

Click "Add Block", then add a "Bounding Box Visualization". Then, add a "Label Visualization". Your Workflow should now look like this:

![](https://media.roboflow.com/inference/get-started/with-visuals.png)

Every block in Workflows is configurable. For example, you can change the thickness of the bounding box lines, or the size of the text in the labels.

## Step #4: Test the Workflow

We now have a Workflow ready to test.

To test your Workflow, click the "Test Workflow" button.

If you are running Workflows using a local Inference server or a Dedicated Deployment, you can test your Workflow on a video with the browser. Otherwise, you can test with images.

Let's test with the following image that contains vehicles, one of the objects our pre-trained model can identify. You can also upload an image with commmon objects like cell phones, cups, glasses, chairs -- [the model we are using, trained on the Microsoft COCO dataset, can identify over 80 objects](https://blog.roboflow.com/microsoft-coco-classes/).

![](https://media.roboflow.com/inference/get-started/test-image.png)

## Step #5: Add a Tracker

Workflows has an extensive suite of tools for building applications designed to run on video.

Let's add a tracker that will let us track objects between frames.

Hover over the Object Detection model block, then click the "+" (plus) icon to add a block below. Choose "Byte Tracker".

![](https://media.roboflow.com/inference/get-started/add-tracker.png)

Your Workflow should look like this:

![](https://media.roboflow.com/inference/get-started/final-workflow.png)

Our Workflow can now track objects between frames.

## Step #5: Deploy the Workflow

With a video-capable Workflow ready, let's test our Workflow on our hardware.

Click "Deploy" in the top right corner, then copy the code snippet you see. The code snippet will look like this:

```python
# Import the InferencePipeline object
from inference import InferencePipeline
import cv2

def my_sink(result, video_frame):
    if result.get("output_image"): # Display an image from the workflow response
        cv2.imshow("Workflow Image", result["output_image"].numpy_image)
        cv2.waitKey(1)
    print(result) # do something with the predictions of each frame


# initialize a pipeline object
pipeline = InferencePipeline.init_with_workflow(
    api_key="",
    workspace_name="capjamesg",
    workflow_id="custom-workflow-84",
    video_reference=0, # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
    max_fps=30,
    on_prediction=my_sink
)
pipeline.start() #start the pipeline
pipeline.join() #wait for the pipeline thread to finish
```

Create a new Python file with this code, then run the file.

You will see your Workflow running live on your webcam.

Want to run on an RTSP stream? Replace the `video_reference` with the URL of your stream.

You have just built your first Workflow!

Ready to build more? [Check out our gallery of tutorials](/guides/written/).