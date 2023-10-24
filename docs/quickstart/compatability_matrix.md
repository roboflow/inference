# Model Compatability

The table below shows on what devices you can deploy models supported by Inference.

| Model | CPU | GPU | TensorRT | Jetson 4.5.x | Jetson 4.6.x | Jetson 5.x | Roboflow Hosted Inference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv8 Object Detection | ✅ | ✅ | ✅ | 🚫 | 🚫 | ✅ | ✅ |
| YOLOv8 Classification | ✅ | ✅ | ✅ | 🚫 | 🚫 | ✅ | ✅ |
| YOLOv8 Segmentation | ✅ | ✅ | ✅ | 🚫 | 🚫 | ✅ | ✅ |
| YOLOv5 Object Detection | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| YOLOv5 Classification | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| YOLOv5 Segmentation | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| DocTR | ✅ | ✅ | 🟡 | ✅ | ✅ | ✅ | 🚧 |
| CLIP | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| SAM | ✅ | ✅ | 🟡 | 🚫 | 🚫 | 🚫 | 🚫 |
| ViT Classification | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| YOLACT | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| ✅ Fully supported |
| 🟡 Not TRT accelerated |
| 🚫 Not supported |
| 🚧 On roadmap |

See our [Docker Getting Started](/docs/quickstart/docker) guide for more information on how to deploy Inference on your device.