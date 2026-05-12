# Model Compatability

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
| Gaze ⚠ deprecated      | 🚫   | 🚫   | 🚫           | 🚫           | 🚫         | 🚫                        |
| SAM                    | ✅   | ✅   | 🚫           | 🚫           | 🚫         | 🚫                       |
| ViT Classification     | ✅   | ✅   | ✅           | ✅           | ✅         | ✅                        |
| YOLACT                 | ✅   | ✅   | ✅           | ✅           | ✅         | ✅                        |
