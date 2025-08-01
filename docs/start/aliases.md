# Load a Model from Roboflow Cloud

You can use any model trained on or uploaded to Roboflow in a Workflow.

You can also use one of [several pre-trained models](/start/aliases/#supported-pre-trained-models), ideal for experimentation, or any of the 50,000+ open source models hosted on [Roboflow Universe](/start/load_from_universe/).

To use a model trained on or uploaded to Roboflow, open your Workflows editor. Then, add a model block such as an [Object Detection Model](/workflows/blocks/object_detection_model/) or Instance Segmentation Model block.

When you add a model block, a window will appear from which you can choose a model.

You will see all the models in your Roboflow account:

![](https://media.roboflow.com/inference/workflows/select_model.png)

Select a model, then click "Save" to use the model.

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
    <td>1280</td>
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
</table>
