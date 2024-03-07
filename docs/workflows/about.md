# Inference Workflows

Inference Workflows allow you to define multi-step processes that run one or more models and returns a result based on the output of the models.

With Inference workflows, you can:

- Detect, classify, and segment objects in images.
- Apply filters (i.e. process detections in a specific region, filter detections by confidence).
- Use Large Multimodal Models (LMMs) to make determinations at any stage in a workflow.

You can build simple workflows in the Roboflow web interface that you can then deploy to your own device or the cloud using Inference.

You can build more advanced workflows for use on your own devices by writing a workflow configuration directly in JSON.

In this section of documentation, we describe what you need to know to create workflows.

!!! note
    We require a Roboflow Enterprise License to use this in production. See inference/enterpise/LICENSE.txt for details.