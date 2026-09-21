# Roboflow Models in Workflows

This directory contains the Workflow blocks that power models
you can deploy (and often fine-tune) via Roboflow's cloud platform.

## Generic Models

The pre-trained models listed in our [/docs/aliases.md](model aliases list)
may be used freely without a Roboflow account or API Key.

## Fine-Tuned Models

The Workflow blocks in this directory are Apache 2.0 licensed,
but if you're loading a fine-tuned model from the Roboflow cloud
platform you'll need a Roboflow account and a Roboflow API Key
and usage will be metered according to your Roboflow plan's limits.

## Model Licenses

These Blocks can be used to load a variety of architectures. Each
underlying model has its own license which are listed in
[the `models` directory](/inference/models).

Roboflow's paid plans [offer commercial licenses for some models](https://roboflow.com/licensing),
but outside of that it is your responsibility to ensure your code's
compliance with the models you use.