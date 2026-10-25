# Roboflow Models in Workflows

This directory contains the Workflow blocks that power models
you can deploy (and often fine-tune) via Roboflow's cloud platform.

## Generic Models

The pre-trained models listed in our [/docs/aliases.md](model aliases list)
may be used freely without a Roboflow account or API Key.

## Fine-Tuned Models

The Workflow blocks in this directory are Apache 2.0 licensed,
but loading a fine-tuned model from the Roboflow platform requires
a Roboflow account and API key. Running that model on your own
hardware does not use credits. The Serverless Cloud API bills
each image at the model's rate.

## Model Licenses

These Blocks can be used to load a variety of architectures. Each
underlying model has its own license which are listed in
[the `models` directory](/inference/models).

[Roboflow Cloud products](https://docs.roboflow.com/deployment/roboflow-cloud/serverless-api), including the Serverless Cloud API, include a commercial license for the models Roboflow can relicense, for every user. A commercial license for self-hosted deployment of those models is available as an [Enterprise add-on](https://roboflow.com/licensing). Without that add-on, self-hosted deployment follows the model's own license. An AGPL-3.0 model must be used under AGPL-3.0. It is your responsibility to ensure your code's compliance with the models you use.
