# How to Choose a Model

Most Workflows built with Inference start with a model.

The following model types are supported:

- Object Detection: Find the location of objects in an image. 
- Classification: Assign one or more labels to an image.
- Keypoint Detection: Find key points of objects in an image.
- Multimodal: Ask an LLM with vision capabilities questions.
- Instance Segmentation: Find the location of objects in an image with pixel precision.
- Semantic Segmentation: Find the location of objects in an image with pixel precision.

In addition, "foundation" models are supported. These are models that do well at one or more of the tasks above and may not require fine-tuning (i.e. CLIP for zero-shot classification, Segment Anything for zero-shot segmentation).

You can deploy models that are:

1. [Trained on or uploaded to Roboflow](/quickstart/pretrained_models.md);
2. [Released to the public on Roboflow Universe](/quickstart/load_from_universe/), and;
3. [Stored on your local computer](/models/from_local_weights/).

