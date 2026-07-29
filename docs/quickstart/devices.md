# What Devices Can I Use?

You can deploy Inference on the edge, in your own cloud, or using the Roboflow hosted inference option.

## Supported Edge Devices

You can set up a server to use computer vision models with Inference on the following devices:

- ARM CPU (macOS, Raspberry Pi)
- x86 CPU (macOS, Linux, Windows)
- NVIDIA GPU
- NVIDIA Jetson — JetPack 7.2 (Orin and Thor) recommended; JetPack 6.2 and 5.1.x are also supported, but support for both ends in 2027 (5.1.x is deprecated)

## Self-Hosted on Cloud

You can deploy Inference Server on any cloud platform such as AWS, GCP, or Azure. See [Deploy in Your Own Cloud](/install/cloud).

## Use Hosted Inference from Roboflow

You can also run your models in the cloud with the <a href="https://docs.roboflow.com/deploy/serverless-hosted-api-v2" target="_blank">Roboflow hosted inference offering</a>. The Roboflow hosted inference solution enables you to deploy your models in the cloud without having to manage your own infrastructure. Roboflow's hosted solution does not support all features available in Inference that you can run on your own infrastructure.
