Foundation models are machine learning models that have been trained on vast amounts of data to accomplish a specific task.

For example, OpenAI trained CLIP, a foundation model. CLIP enables you to classify images. You can also compare the similarity of images and text with CLIP.

The CLIP training process, which was run using over 400 million pairs of images and text, allowed the model to build an extensive range of knowledge, which can be applied to a range of domains.

Foundation models are being built for a range of vision tasks, from image segmentation to classification to zero-shot object detection.

Inference supports the following foundation models:

- Gaze (LC2S-Net): Detect the direction in which someone is looking.
- CLIP: Classify images and compare the similarity of images and text.
- DocTR: Read characters in images.
- Grounding DINO: Detect objects in images using text prompts.
- Segment Anything (SAM): Segment objects in images.

All of these models can be used over a HTTP request with Inference. This means you don't need to spend time setting up and configuring each model.

## How Are Foundation Models Used?

Use cases vary depending on the foundation model with which you are working. For example, CLIP has been used extensively in the field of computer vision for tasks such as:

1. Clustering images to identify groups of similar images and outliers;
2. Classifying images;
3. Moderating image content;
4. Identifying if two images are too similar or too different, ideal for dataset management and cleaning;
5. Building dataset search experiences,
6. And more.

Grounding DINO, on the other hand, can be used out-of-the-box to detect a range of objects. Or you can use Grounding DINO to automatically label data for use in training a smaller, faster object detection model that is fine-tuned to your use case.

## How to Use Foundation Models

The guides in this section walk through how to use each of the foundation models listed above with Inference. No machine learning experience is required to use each model. Our code snippets and accompanying reference material provide the knowledge you need to get started working with foundation models.