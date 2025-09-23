# Welcome to Inference

![](https://github.com/roboflow/inference/raw/main/banner.png?raw=true)

Roboflow Inference is a fast, on-device computer vision inference server. With Inference, you can build multi-step "Workflows" that use state-of-the-art models. These Workflows can transform images, run and chain models, ask questions to VLMs, connect to external APIs, and more. Workflows can run on your own hardware, or in the cloud.

Inference runs as a microservice in Docker to which you can make web requests.

Inference supports essential video features, including object tracking, outlier frame detection (powered by embeddings), and tracking the time an object spends in a zone.

Inference runs both on the edge (i.e. on an NVIDIA Jetson) and in the cloud (i.e. AWS, GCP, Roboflow).

Inference is licensed under an [Apache 2.0 license](https://github.com/roboflow/inference/). Models supported by Inference are subject to their own licenses.

Ready to get started? [Let's build your first vision application with Inference](/start/getting-started.md).

Need inspiration? Check out these guides:

- [Fine-tune and deploy SmolVLM2 for document understanding](https://www.youtube.com/watch?v=qLPInUmH9xE)
- [Deploy the Qwen2.5-VL VLM](https://www.youtube.com/watch?v=3PIDMhvwZd8)
- [Detect people in danger zones](https://www.youtube.com/watch?v=1N8JKCqR5Xg)
- [Measure the dimensions of objects](https://www.youtube.com/watch?v=FQY7TSHfZeI)
- [Build a vision app in 10 minutes with Roboflow Instant and a Workflow](https://www.youtube.com/watch?v=aPxlImNxj5A)
- [Deploy Florence-2 for object detection](https://www.youtube.com/watch?v=_u53TxShLsk)

## Supported Models
Inference supports a wide range of state-of-the-art models, including:

<div class="grid" markdown style="grid-template-columns: 1fr 1fr !important;">
<div markdown>
### Fine-tunable Models
* [RF-DETR](https://blog.roboflow.com/rf-detr/)
* [ViT](https://blog.roboflow.com/train-vision-transformer/)
* [ResNet](https://blog.roboflow.com/how-to-train-a-resnet-50-model/)
* [SmolVLM2](//workflows/blocks/smol_vlm2/)
* [Qwen2.5-VL](/workflows/blocks/qwen2.5_vl/)
* [YOLOv12](/workflows/blocks/object_detection_model/)
* [YOLO11](/workflows/blocks/object_detection_model/)
* [YOLOv8](/workflows/blocks/object_detection_model/)
</div>
<div markdown>
### Foundation Models 
* [Perception Encoder](/workflows/blocks/perception_encoder_embedding_model)
* [Segment Anything 2](/workflows/blocks/segment_anything2_model/)
* [CLIP](/workflows/blocks/clip_embedding_model)
* [Florence-2](/workflows/blocks/florence2_model)
* [Moondream2](/workflows/blocks/moondream2)
* [Apple DepthPro](/workflows/blocks/depth_estimation)
</div>
</div>

Workflows supports 100+ "blocks" that you can use to build on top of vision models.

With Workflows, you can:

* Run object detection, segmentation, classification, and multimodal models
* Run supported foundation models like Segment Anything 2 and Perception Encoder
* Run stateful video tracking
* Implement conditional logic
* Transform images
* Run detections consensus algorithms
* And more

[See a full list of what you can do with Workflows](/workflows/tutorials).