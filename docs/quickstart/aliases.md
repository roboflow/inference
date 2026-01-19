Inference supports running any of the 50,000+ pre-trained public models hosted on [Roboflow Universe](https://universe.roboflow.com), as well as fine-tuned models.

We have defined IDs for common models for ease of use. These models do not require an API key for use unlike other public or private models.

Using it in `inference` is as simple as:

```python
from inference import get_model

model = get_model(model_id="yolov8n-640")

results = model.infer("https://media.roboflow.com/inference/people-walking.jpg")
```

!!! Tip

    See the [Use a fine-tuned model](./explore_models.md) guide for an example on how to deploy your own model.

## Supported Pre-Trained Models

You can click the link associated with a model below to test the model in your browser, and use the ID with Inference to deploy the model to the edge.

<style>
table {
  width: 100%;
  border-collapse: collapse;
}
</style>
<table border="1">
<tr>
    <th>Model</th>
    <th>Size</th>
    <th>Task</th>
    <th>Model ID</th>
    <th>Test Model in Browser</th>
</tr>
<tr>
    <td>YOLOv8n</td>
    <td>640</td>
    <td>Object Detection</td>
    <td>yolov8n-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco/model/3">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv8n</td>
    <td>1280</td>
    <td>Object Detection</td>
    <td>yolov8n-1280</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco/model/9">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv8s</td>
    <td>640</td>
    <td>Object Detection</td>
    <td>yolov8s-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco/model/6">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv8s</td>
    <td>1280</td>
    <td>Object Detection</td>
    <td>yolov8s-1280</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco/model/10">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv8m 640</td>
    <td>640</td>
    <td>Object Detection</td>
    <td>yolov8m-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco/model/8">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv8m</td>
    <td>1280</td>
    <td>Object Detection</td>
    <td>yolov8m-1280</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco/model/11">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv8l</td>
    <td>640</td>
    <td>Object Detection</td>
    <td>yolov8l-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco/model/7">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv8l</td>
    <td>1280</td>
    <td>Object Detection</td>
    <td>yolov8l-1280</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco/model/12">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv8x</td>
    <td>640</td>
    <td>Object Detection</td>
    <td>yolov8x-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco/model/5">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv8x</td>
    <td>1280</td>
    <td>Object Detection</td>
    <td>yolov8x-1280</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco/model/13">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLO-NAS (small)</td>
    <td>640</td>
    <td>Object Detection</td>
    <td>yolo-nas-s-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco/model/14">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLO-NAS (medium)</td>
    <td>640</td>
    <td>Object Detection</td>
    <td>yolo-nas-m-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco/model/15">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLO-NAS (large)</td>
    <td>640</td>
    <td>Object Detection</td>
    <td>yolo-nas-l-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco/model/16">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv8n Instance Segmentation</td>
    <td>640</td>
    <td>Instance Segmentation</td>
    <td>yolov8n-seg-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco-dataset-vdnr1/model/2">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv8n Instance Segmentation</td>
    <td>1280</td>
    <td>Instance Segmentation</td>
    <td>yolov8n-seg-1280</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco-dataset-vdnr1/model/7">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv8s Instance Segmentation</td>
    <td>640</td>
    <td>Instance Segmentation</td>
    <td>yolov8s-seg-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco-dataset-vdnr1/model/4">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv8m Instance Segmentation</td>
    <td>1280</td>
    <td>Instance Segmentation</td>
    <td>yolov8s-seg-1280</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco-dataset-vdnr1/model/8">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv8m Instance Segmentation</td>
    <td>640</td>
    <td>Instance Segmentation</td>
    <td>yolov8m-seg-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco-dataset-vdnr1/model/5">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv8m Instance Segmentation</td>
    <td>1280</td>
    <td>Instance Segmentation</td>
    <td>yolov8m-seg-1280</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco-dataset-vdnr1/model/9">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv8l Instance Segmentation</td>
    <td>640</td>
    <td>Instance Segmentation</td>
    <td>yolov8l-seg-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco-dataset-vdnr1/model/6">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv8l Instance Segmentation</td>
    <td>1280</td>
    <td>Instance Segmentation</td>
    <td>yolov8l-seg-1280</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco-dataset-vdnr1/model/10">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv8x Instance Segmentation</td>
    <td>640</td>
    <td>Instance Segmentation</td>
    <td>yolov8x-seg-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco-dataset-vdnr1/model/3">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv8x Instance Segmentation</td>
    <td>1280</td>
    <td>Instance Segmentation</td>
    <td>yolov8x-seg-1280</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco-dataset-vdnr1/model/11">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv8x Keypoint Detection</td>
    <td>1280</td>[version.py](../../inference/core/version.py)
    <td>Keypoint Detection</td>
    <td>yolov8x-pose-1280</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco-pose-detection/6">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv8x Keypoint Detection</td>
    <td>640</td>
    <td>Keypoint Detection</td>
    <td>yolov8x-pose-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco-pose-detection/5">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv8l Keypoint Detection</td>
    <td>640</td>
    <td>Keypoint Detection</td>
    <td>yolov8l-pose-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco-pose-detection/4">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv8m Keypoint Detection</td>
    <td>640</td>
    <td>Keypoint Detection</td>
    <td>yolov8m-pose-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco-pose-detection/3">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv8s Keypoint Detection</td>
    <td>640</td>
    <td>Keypoint Detection</td>
    <td>yolov8s-pose-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco-pose-detection/2">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv8n Keypoint Detection</td>
    <td>640</td>
    <td>Keypoint Detection</td>
    <td>yolov8n-pose-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco-pose-detection/1">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv10n</td>
    <td>640</td>
    <td>Object Detection</td>
    <td>yolov10n-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco/model/19">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv10s</td>
    <td>640</td>
    <td>Object Detection</td>
    <td>yolov10s-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco/model/20">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv10m</td>
    <td>640</td>
    <td>Object Detection</td>
    <td>yolov10m-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco/model/21">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv10b</td>
    <td>640</td>
    <td>Object Detection</td>
    <td>yolov10b-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco/model/22">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv10l</td>
    <td>640</td>
    <td>Object Detection</td>
    <td>yolov10l-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco/model/23">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv10x</td>
    <td>640</td>
    <td>Object Detection</td>
    <td>yolov10x-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco/model/24">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv11n</td>
    <td>640</td>
    <td>Object Detection</td>
    <td>yolov11n-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco/model/25">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv11s</td>
    <td>640</td>
    <td>Object Detection</td>
    <td>yolov11s-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco/model/26">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv11m</td>
    <td>640</td>
    <td>Object Detection</td>
    <td>yolov11m-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco/model/27">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv11l</td>
    <td>640</td>
    <td>Object Detection</td>
    <td>yolov11l-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco/model/28">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv11x</td>
    <td>640</td>
    <td>Object Detection</td>
    <td>yolov11x-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco/model/29">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv11n</td>
    <td>1280</td>
    <td>Object Detection</td>
    <td>yolov11n-1280</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco/model/30">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv11s</td>
    <td>1280</td>
    <td>Object Detection</td>
    <td>yolov11s-1280</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco/model/31">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv11m</td>
    <td>1280</td>
    <td>Object Detection</td>
    <td>yolov11m-1280</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco/model/32">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv11l</td>
    <td>1280</td>
    <td>Object Detection</td>
    <td>yolov11l-1280</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco/model/33">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv11x</td>
    <td>1280</td>
    <td>Object Detection</td>
    <td>yolov11x-1280</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco/model/34">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv11n</td>
    <td>640</td>
    <td>Instance Segmentation</td>
    <td>yolov11n-seg-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco-dataset-vdnr1/model/19">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv11s</td>
    <td>640</td>
    <td>Instance Segmentation</td>
    <td>yolov11s-seg-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco-dataset-vdnr1/model/20">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv11m</td>
    <td>640</td>
    <td>Instance Segmentation</td>
    <td>yolov11m-seg-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco-dataset-vdnr1/model/21">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv11l</td>
    <td>640</td>
    <td>Instance Segmentation</td>
    <td>yolov11l-seg-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco-dataset-vdnr1/model/22">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv11x</td>
    <td>640</td>
    <td>Instance Segmentation</td>
    <td>yolov11x-seg-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco-dataset-vdnr1/model/23">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLO26n</td>
    <td>640</td>
    <td>Object Detection</td>
    <td>yolo26n-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco/model/41">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv26s</td>
    <td>640</td>
    <td>Object Detection</td>
    <td>yolo26s-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco/model/42">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv26m</td>
    <td>640</td>
    <td>Object Detection</td>
    <td>yolo26m-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco/model/43">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv26l</td>
    <td>640</td>
    <td>Object Detection</td>
    <td>yolo26l-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco/model/44">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv26x</td>
    <td>640</td>
    <td>Object Detection</td>
    <td>yolo26x-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco/model/45">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLO26n</td>
    <td>640</td>
    <td>Instance Segmentation</td>
    <td>yolo26n-seg-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco-dataset-vdnr1/model/27">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv26s</td>
    <td>640</td>
    <td>Instance Segmentation</td>
    <td>yolo26s-seg-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco-dataset-vdnr1/model/28">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv26m</td>
    <td>640</td>
    <td>Instance Segmentation</td>
    <td>yolo26m-seg-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco-dataset-vdnr1/model/29">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv26l</td>
    <td>640</td>
    <td>Instance Segmentation</td>
    <td>yolo26l-seg-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco-dataset-vdnr1/model/31">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv26x</td>
    <td>640</td>
    <td>Instance Segmentation</td>
    <td>yolo26x-seg-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco-dataset-vdnr1/model/34">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLO26n Keypoint Detection</td>
    <td>640</td>
    <td>Keypoint Detection</td>
    <td>yolo26n-pose-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco-pose-detection/12">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv26s Keypoint Detection</td>
    <td>640</td>
    <td>Keypoint Detection</td>
    <td>yolo26s-pose-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco-pose-detection/13">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv26m Keypoint Detection</td>
    <td>640</td>
    <td>Keypoint Detection</td>
    <td>yolo26m-pose-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco-pose-detection/14">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv26l Keypoint Detection</td>
    <td>640</td>
    <td>Keypoint Detection</td>
    <td>yolo26l-pose-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco-pose-detection/15">Test in Browser</a></td>
</tr>
<tr>
    <td>YOLOv26x Keypoint Detection</td>
    <td>640</td>
    <td>Keypoint Detection</td>
    <td>yolo26x-pose-640</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco-pose-detection/16">Test in Browser</a></td>
</tr>
</table>
