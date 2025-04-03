# What Devices Can I Use?

You can deploy Inference on the edge, in your own cloud, or using the Roboflow hosted inference option.

## Supported Edge Devices

You can set up a server to use computer vision models with Inference on the following devices:

- ARM CPU (macOS, Raspberry Pi)
- x86 CPU (macOS, Linux, Windows)
- NVIDIA GPU
- NVIDIA Jetson (JetPack 4.5.x, JetPack 4.6.x, JetPack 5.x, JetPack 6.x)

## Model Compatability

The table below shows on what devices you can deploy models supported by Inference.

See our [Docker Getting Started](./docker.md) guide for more information on how to deploy Inference on your device.

Table key:

- ✅ Fully supported
- 🚫 Not supported
- 🚧 On roadmap, not currently supported

| Model                  | CPU | GPU | Jetson 4.5.x | Jetson 4.6.x | Jetson 5.x | Roboflow Hosted Inference |
|------------------------|-----|-----|--------------|--------------|------------|---------------------------|
| YOLOv8 Object Detection| ✅   | ✅   | 🚫           | 🚫           | ✅         | ✅                        |
| YOLOv8 Classification  | ✅   | ✅   | 🚫           | 🚫           | ✅         | ✅                        |
| YOLOv8 Segmentation    | ✅   | ✅   | 🚫           | 🚫           | ✅         | ✅                        |
| YOLOv5 Object Detection| ✅   | ✅   | ✅           | ✅           | ✅         | ✅                        |
| YOLOv5 Classification  | ✅   | ✅   | ✅           | ✅           | ✅         | ✅                        |
| YOLOv5 Segmentation    | ✅   | ✅   | ✅           | ✅           | ✅         | ✅                        |
| CLIP                   | ✅   | ✅   | ✅           | ✅           | ✅         | ✅                        |
| DocTR                  | ✅   | ✅   | ✅           | ✅           | ✅         | ✅                        |
| Gaze                   | ✅   | ✅   | 🚫           | 🚫           | 🚫         | ✅                        |
| SAM                    | ✅   | ✅   | 🚫           | 🚫           | 🚫         | 🚫                       |
| ViT Classification     | ✅   | ✅   | ✅           | ✅           | ✅         | ✅                        |
| YOLACT                 | ✅   | ✅   | ✅           | ✅           | ✅         | ✅                        |

## Cloud Platform Support

You can deploy Inference on any cloud platform such as AWS, GCP, or Azure.

The installation and setup instructions are the same as for any edge device, once you have installed the relevant drivers on your cloud platform. We recommend deploying with an official "Deep Learning" image from your cloud provider if you are running inference on a GPU device. "Deep Learning" images should have the relevant drivers pre-installed so you can set up Inference without configuring GPU drivers manually

## Use Hosted Inference from Roboflow

You can also run your models in the cloud with the <a href="https://docs.roboflow.com/deploy/hosted-api" target="_blank">Roboflow hosted inference offering</a>. The Roboflow hosted inference solution enables you to deploy your models in the cloud without having to manage your own infrastructure. Roboflow's hosted solution does not support all features available in Inference that you can run on your own infrastructure.

To learn more about device compatability with different models, refer to the [model compatability matrix](./compatability_matrix.md).