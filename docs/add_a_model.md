# Add a Model to the Inference Server

The Roboflow Inference Server comes with several state-of-the-art models available for inference out of the box ([see list of available models](https://github.com/roboflow/inference#Supported_Models)). If your model is not supported by the Inference Server, you can implement support for a custom model architecture.

!!! note

    Inference is designed to be extensible so you can add support for new models.

    Before you add a new model, check the [list of supported models](https://inference.roboflow.com/docs/library/models/) to see if your model is already supported.
    
    Roboflow may not accept contributions for new models to Inference depending on the model architecture, but you can maintain your own fork that supports your custom model.

    If you are contributing a model to Inference that you want other people to use, please open a [GitHub Issue](https://github.com/roboflow/inference/issues/new) to discuss your contribution.

In this guide, we are going to walk through how to add support for a new model architecture to the Roboflow Inference Server.

## Create Model Structure

To add a model to the inference server, there are a few files you need to create.

First, create a folder called `inference/models/MODEL_NAME`, where `MODEL_NAME` is the name of your model. Then, create the following files:

```
inference/models/MODEL_NAME/__init__.py
inference/models/MODEL_NAME/MODEL_NAME_model.py
inference/models/MODEL_NAME/LICENSE.txt
```

- `__init__.py` should contain an import to your model API. You can see examples of this in the CLIP and SAM models.
- `MODEL_NAME_model.py` should contain the implementation of your model API. You can see examples of this in the CLIP and SAM models.
- `LICENSE.txt` should contain the license for your model. You must include the license of the model you are using.

## Implement Model API

To implement a model, you will need to work across multiple parts of the Inference codebase. Here are the main files with which you will need to work:

- `inference/models/MODEL_NAME/MODEL_NAME_model.py`: This file contains the implementation of your model API. You can see examples of this in the CLIP and SAM models.
- `inference/core/data_models.py`: This is where you will store your Request and Response objects. These will be consumed by FastAPI for creating a HTTP interface for your model.
- `inference/core/env.py`: This is where you should declare any configuration variables you want to infer from the environment. These will be available when you import `inference.core.env.VARIABLE`, where `VARIABLE` is the name of the variable with which you want to work.

Inference has support for adding PyTorch and ONNX models.

Every model needs to have:

1. An `__init__` function that loads the model.
2. A `preprocess_image` function that runs any preprocessing required before inference is run on an image.
3. An `infer` function through which inference is run.

This code should be written in your `inference/models/MODEL_NAME/MODEL_NAME_model.py` file.

You can implement any other functions needed for your model to work.

### PyTorch Models

TODO

### ONNX Models

TODO

### Request and Response Objects

Every model has a Request and a Response object. These objects are used to define the structure of the input and output that your model API accepts.

You must define these objects in `inference/core/data_models.py`.

TODO

## Write Documentation

All contributions to Roboflow Inference for new models must have documentation.

Create a file in `inference/docs/library/models/MODEL_NAME.md`. This file should contain auto-generated API documentation for your model. You can see an example of how this works in the CLIP documentation.

FastAPI, used to generate the HTTP interfaces for Inference, will automatically create OpenAPI documentation for your model.

Add your model to the `Supported Models` list in the Roboflow Inference `README`. In addition, update the `license` section in the project README to include a link to the model you have implemented, its license, and its location in the Inference codebase.

Next, add a link to your model in the Models section of the `mkdocs.yml` file in the root folder of the Inference project. This will ensure your model is linked in the sidebar of the [official Inference documentation](https://inference.roboflow.com).

## Open a Pull Request

If you want to contribute your model for others to use, open a pull request to the [Roboflow Inference repository](https://github.com/roboflow/inference) with your contribution. Your contribution will be reviewed by a member of the Roboflow team.