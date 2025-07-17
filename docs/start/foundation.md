# Use a Foundation Model

Foundation models are machine learning models that have been trained on vast amounts of data to accomplish a specific task.

For example, OpenAI trained CLIP, a foundation model. CLIP enables you to classify images. You can also compare the similarity of images and text with CLIP.

The CLIP training process, which was run using over 400 million pairs of images and text, allowed the model to build an extensive range of knowledge, which can be applied to a range of domains.

Foundation models are being built for a range of vision tasks, from image segmentation to classification to zero-shot object detection.

You can use several foundation models in Workflows, including:

- [CLIP](/workflows/blocks/clip_embedding_model), ideal for zero-shot image classification and generating embeddings.
- [Perception Encoder](/workflows/blocks/perception_encoder_embedding_model), ideal for zero-shot image classification and generating embeddings.
- [Segment Anything 2](/workflows/blocks/segment_anything_2), ideal for zero-shot image segmentation.
- [Moondream 2](/workflows/blocks/moondream_2_model), ideal for zero-shot object detection.
- [YOLO-World](/workflows/blocks/yolo_world_model), ideal for zero-shot object detection.
- [Florence-2](/workflows/blocks/florence_2_model), ideal for zero-shot object detection, OCR, and more.
- [Depth Estimation](/workflows/blocks/depth_estimation), ideal for estimating the depth of objects in an image.

## How Are Foundation Models Used?

Use cases vary depending on the foundation model with which you are working. For example, CLIP has been used extensively in the field of computer vision for tasks such as:

1. Clustering images to identify groups of similar images and outliers;
2. Classifying images;
3. Moderating image content;
4. Identifying if two images are too similar or too different, ideal for dataset management and cleaning;
5. Building dataset search experiences,
6. And more.

Grounding DINO, on the other hand, can be used out-of-the-box to detect a range of objects. Or you can use Grounding DINO to automatically label data for use in training a smaller, faster object detection model that is fine-tuned to your use case.