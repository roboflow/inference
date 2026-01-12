# Introduction to Roboflow Ecosystem

![Roboflow Inference banner](https://github.com/roboflow/inference/blob/main/banner.png?raw=true)

[Roboflow](https://roboflow.com/) provides everything you need to label, train, and deploy computer vision solutions. It helps you manage and refine datasets, provides tools to streamline and speed up data labelling, helps train and deploy models in the cloud and on edge devices.

Inference is what allows you to deploy and run computer vision models. It enables you to perform object detection, classification, instance segmentation and keypoint detection, and utilize foundation models like CLIP, Segment Anything, and YOLO-World, through a Python-native package, a self-hosted inference server, or a fully managed API.

Roboflow offers both a free tier, and paid [plans](https://roboflow.com/pricing), encompassing all of its products.

- Over half of Fortune 100 companies build with Roboflow. If you're an enterprise customer and interested in a custom solution, parallel processing, active learning or licenses for more than one cloud instance or edge device - **[Reach Out](https://roboflow.com/sales)** to our sales team!

## Related Products

Inference is commonly used with several other roboflow products.

### Roboflow App

[Roboflow App](https://app.roboflow.com/) This is your central dashboard. Here you can upload data, annotate images, define datasets, train and deploy models. You can find the API key (scoped to workspace)

- [Roboflow App](https://app.roboflow.com/)
- [Docs: Getting Started with Roboflow](https://blog.roboflow.com/getting-started-with-roboflow/)
- [App: API key](https://app.roboflow.com/linas-ws/settings/api)
- [Docs: How to Retrieve the API key](https://docs.roboflow.com/api-reference/authentication)

### `roboflow` Package

Your workspace can be managed via the dashboard UI. If you'd like to do it via Python, install the [`roboflow` package](https://docs.roboflow.com/api-reference/install-python-package). This lets you manage your workspace, upload datasets and model weights, and even run model inference (a bit outdated).

However, If all you need is to run a deployed model, you likely won't need `roboflow` at all.
Where possible, we recommend `inference`. That's what we use on our servers!

If you wish to use the `roboflow` package, instructions can be found in [Roboflow Python Package Docs](https://docs.roboflow.com/api-reference/install-python-package).

### Universe

[Universe](https://universe.roboflow.com/) is our space for sharing datasets and models.

Search for models, test out their performance on your images, track model versions, access in inference, build on top.

The `model_id` you pass into inference can be a `model_id` from Universe.

- [Run Model from Roboflow Universe](../quickstart/explore_models.md)

### Supervision

What happens when you infer some results from a model? [`supervision`](https://supervision.roboflow.com/latest/) lets you plot bounding boxes and segmentation masks, track objects, merge various detections. With supervision tools such as `InferenceSlicer` you can detect small objects in an image by running the model on smaller patches.

It also encodes model results from various sources - `inference`, Hugging Face, Ultralytics and more, into a common format.
https://supervision.roboflow.com/latest/

- [Get started with Supervision](https://supervision.roboflow.com/latest/)

### Workflows

[Workflows](../workflows/about.md) are an new `inference` feature. Instead of writing code, you may chain together blocks to build your computer vision algorithms from scratch. There's an expanding [library](/workflows/blocks/index.md) of blocks available - see if you find anything you like!
