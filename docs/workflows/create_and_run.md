# How to Create and Run a Workflow

In this example, we are going to build a Workflow from scratch that detect dogs, classify their breeds and
visualize results.

## Step 1: Create a Workflow

Open [https://app.roboflow.com/](https://app.roboflow.com/) in your browser, and navigate to Workflows tab to click 
Create Workflows button. Select Custom Workflow to start creation process.

![Workflow start](https://media.roboflow.com/inference/getting_started_workflows.png)


## Step 2: Add object detection model
We need to add block with object detection model to existing workflow. We will use `yolov8n-640` model.

![Add object detection model](https://media.roboflow.com/inference/adding_object_detection_model.png)

## Step 3: Crop each detected object to run breed classification

Next, we are going to add a block to our Workflow that crops the objects that our first model detects.

![Add crop](https://media.roboflow.com/inference/adding_crop.png)

## Step 4: Classify dog breeds with second stage model

We are then going to add an classification model thar runs on each crop to classify its content. We will use
Roboflow Universe model `dog-breed-xpaq6/1`. Please make sure that in block configuration, property `Image`
points to Dynamic Crop output named `crops`.


![Add OCR](https://media.roboflow.com/inference/adding_secondary_model.png)

## Step 5: Replace Bounding Boxes classes with classification model prediction

When each crop is classified, we would like to assign class predicted for each crop (dog breed) as a class 
of bounding box with dog. To do this we use Detections Classes Replacement block, which accepts 
reference to predictions of object detection model, as well as reference to classification results on crops.

![Add Classes Replacement](https://media.roboflow.com/inference/detections_classes_replacement.png)


## Step 6: Visualise predictions

As a final step of workflow, we would like to visualize our predictions. We will use two 
visualization blocks: Bounding Box Visualization and Label Visualization chained together.
At first, add Bounding Box Visualization referring to `$inputs.image` in Image property (that's the
image sent as your input to workflow), the second step (Label Visualization) however, should point to 
the output of Bounding Box Visualization step. Both visualization steps should refer to predictions 
from Detections Classes Replacement step.

![Add Visualisation](https://media.roboflow.com/inference/adding_visualization.png)

## Step 7: Construct output
You have everything ready to construct your workflow output. You can use any intermediate step output that you
need, but in this example we will only select bounding boxes with replaced classes (output from Detections 
Classes Replacement step) and visualisation (output from Label Visualization step).


## Step 8: Running the workflow
Now your workflow, is ready. You can click `Save` button and move to `Run Preview` panel.

We will run our workflow against the following example image `https://media.roboflow.com/inference/dog.jpeg`.
Here are the results

![Results](https://media.roboflow.com/inference/workflow_preview.png)

Clicking on `Show Visual` button you will find results of our visualisation efforts.
<center><img src="https://media.roboflow.com/inference/workflow_visualisation_result.png" width="50%"/></center>


## Different ways of running your workflow
Your workflow is now saved at Roboflow Platform. This means you can run it in multiple different ways, including:

- HTTP request to Roboflow Hosted API

- HTTP request to your local instance of `inference server`

- on video

To see code snippets, click `Deploy Workflow` button:
<center><img src="https://media.roboflow.com/inference/deploy_workflow.png" width="50%"/></center>

## Workflow definition for quick reproduction

To make it easier to reproduce the workflow, below you can find workflow definition you can copy-paste to UI editor.

??? Tip "Workflow definitiom"
    
    ```json
    {
      "version": "1.0",
      "inputs": [
        {
          "type": "InferenceImage",
          "name": "image"
        }
      ],
      "steps": [
        {
          "type": "roboflow_core/roboflow_object_detection_model@v1",
          "name": "model",
          "images": "$inputs.image",
          "model_id": "yolov8n-640"
        },
        {
          "type": "roboflow_core/dynamic_crop@v1",
          "name": "dynamic_crop",
          "images": "$inputs.image",
          "predictions": "$steps.model.predictions"
        },
        {
          "type": "roboflow_core/roboflow_classification_model@v1",
          "name": "model_1",
          "images": "$steps.dynamic_crop.crops",
          "model_id": "dog-breed-xpaq6/1"
        },
        {
          "type": "roboflow_core/detections_classes_replacement@v1",
          "name": "detections_classes_replacement",
          "object_detection_predictions": "$steps.model.predictions",
          "classification_predictions": "$steps.model_1.predictions"
        },
        {
          "type": "roboflow_core/bounding_box_visualization@v1",
          "name": "bounding_box_visualization",
          "predictions": "$steps.detections_classes_replacement.predictions",
          "image": "$inputs.image"
        },
        {
          "type": "roboflow_core/label_visualization@v1",
          "name": "label_visualization",
          "predictions": "$steps.detections_classes_replacement.predictions",
          "image": "$steps.bounding_box_visualization.image"
        }
      ],
      "outputs": [
        {
          "type": "JsonField",
          "name": "detections",
          "coordinates_system": "own",
          "selector": "$steps.detections_classes_replacement.predictions"
        },
        {
          "type": "JsonField",
          "name": "visualisation",
          "coordinates_system": "own",
          "selector": "$steps.label_visualization.image"
        }
      ]
    }
    ```


## Next Steps

Now that you have created and run your first workflow, you can explore our other supported blocks and create a more complex workflow.

Refer to our [Supported Blocks](/workflows/blocks/) documentation to learn more about what blocks are supported.
We also recommend reading [Understanding workflows](/workflows/understanding/) page.
