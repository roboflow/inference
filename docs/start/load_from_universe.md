With Inference, you can run any of the 50,000+ models available on Roboflow Universe.

All models run on your own hardware.

## Run a Model From Roboflow Universe

In the first example, we showed how to run a people detection model. This model was hosted on Universe. Let's find another model to try.

Go to the <a href="https://universe.roboflow.com" target="_blank">Roboflow Universe</a> homepage and use the search bar to find a model.

![](https://media.roboflow.com/inference/universe/search.png)

!!! tip

    Add "model" to your search query to only find models.

Browse the Universe search page to find a model:

![](https://media.roboflow.com/inference/universe/search-results.png)

When you have found a model, click "Model" in the sidebar. You can preview the model and try it out on your own data or images from the model test set from this page.

## Use the Universe Model in a Workflow

To use the model you have found, open your Workflows editor. Then, add the model block that corresponds with the type of model you found on Universe. For example, if you found an object detection model, choose an [Object Detection Model](/workflows/blocks/object_detection_model/) block.

When you add a model block, a window will appear from which you can choose a model.

Click "Public Models".

Then, go back to Universe and copy the model ID from the model you want to use:

![](https://media.roboflow.com/inference/universe/model-id.png)

Paste the model ID in the Public Models search bar:

![](https://media.roboflow.com/inference/universe/set-model.png)

Then, click the purple "Use model ID" button.

The model will then be added to your Workflow:

![](https://media.roboflow.com/inference/universe/result.png)

Let's add Bounding Box and Label Visualizer blocks so we can visualize the predictions from our model:

![](https://media.roboflow.com/inference/universe/workflow.png)

Now, let's test our model on an image. Click "Test Workflow" in the top right, and drag in an image you want to use in your Workflow. Then, click "Run" to test your Workflow.

![](https://media.roboflow.com/inference/universe/empty-test.png)

Here are the results from our Workflow:

![](https://media.roboflow.com/inference/universe/result.png)

The model successfully identified a defect in the image we used as an input to our Workflow.

To deploy your workflow, click "Deploy". A window will appear with the code snippets you need to run your Workflow on your hardware.

Ready to build more logic in your Workflow? [Check out our gallery of tutorials.](/guides/written/)