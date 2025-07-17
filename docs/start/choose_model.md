# How to Choose a Model

Most Workflows built with Inference start with a model.

The following model types are supported:

- [Object Detection](https://blog.roboflow.com/object-detection/): Get the coordinates of objects in an image. 
- [Classification](https://blog.roboflow.com/image-classification/): Assign one or more labels to an image.
- [Keypoint Detection](https://blog.roboflow.com/what-is-keypoint-detection/): Find key points of objects in an image.
- [Multimodal (VLM)](https://blog.roboflow.com/what-is-a-vision-language-model/): Ask questions to a Vision Language Model (VLM) in natural language.
- [Instance Segmentation](https://blog.roboflow.com/instance-segmentation/): Find the location of objects in an image with pixel precision.
- [Semantic Segmentation](https://blog.roboflow.com/what-is-semantic-segmentation/): Find the location of objects in an image with pixel precision.

If you want to know where an object is in an image, use object detection. Object detection models run quickly and return boxes that correspond with regions of interest in an image. For example, a model tuned to identify packages on an assembly line will return the coordinates of all the packages on an assembly line.

If you need to know precisely where an object is in an image, use instance segmentation. Instance segmentation models return polygons that outline an object.

If you want to assign a label to a whole image (for example, "damaged package" or "not damaged package"), use classification.

If you need to know "key points" of regions of an image -- for example, if you want to know different points of a person -- use keypoint detection.

If you want to be able to ask a model natural language questions -- for example "does this image contain a cat?" -- and retrieve text-based answers, use a Multimodal model.

In addition, "[foundation](/start/foundation/)" models are supported. These are models that do well at one or more of the tasks above and may not require fine-tuning (i.e. CLIP for zero-shot classification, Segment Anything for zero-shot segmentation).

## Start Training a Model

To train a model, you will need labeled data. This data is then used to teach a model how to do a specific task (i.e. object detection, classification).

We have guides that show how to label data for different tasks:

- General advice: [Annotate an image in Roboflow](https://docs.roboflow.com/annotate/use-roboflow-annotate)
- Object detection and segmentation: [Use AI-powered labeling to quickly label polygons for segmentation (that can also be converted to bounding boxes for object detection)](docs.roboflow.com/annotate/use-roboflow-annotate/enhanced-smart-polygon-with-sam)
- Keypoint detection: [Set keypoint skeletons for a keypoint detection model](https://docs.roboflow.com/annotate/edit-keypoint-skeletons), then [annotate keypoint data](https://docs.roboflow.com/annotate/annotate-keypoints)
- Multimodal: [Annotate multimodal data in Roboflow](https://docs.roboflow.com/annotate/annotate-multimodal-data)

We have written several guides that show how to train models with each of the above task types:

- [How to train a model with labeled data on Roboflow](https://docs.roboflow.com/train/train)
- [Train an RF-DETR object detection model](https://blog.roboflow.com/train-deploy-rf-detr/)
- [Train a ResNet-50 classification model](https://blog.roboflow.com/how-to-train-a-resnet-50-model/)
- [Train a keypoint detection model](https://blog.roboflow.com/keypoint-detection-on-roboflow/)
- [Train a SmolVLM2 multimodal model](https://blog.roboflow.com/train-smolvlm2/)