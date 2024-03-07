Inference supports running any of the 50,000+ pre-trained public models hosted on [Roboflow Universe](https://universe.roboflow.com), as well as fine-tuned models.

We have defined IDs for common models for ease of use. These models do not require an API key for use unlike other public or private models.

!!! Tip

    See the [Use a fine-tuned model](../guides/use-a-fine-tuned-model.md) guide for an example on how to deploy a model.

You can click the link associated with a model below to test the model in your browser, and use the ID with Inference to deploy the model to the edge.

## Supported Pre-Trained Models

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
    <td>640</td>
    <td>Instance Segmentation</td>
    <td>yolov8x-seg-1280</td>
    <td><a href="https://universe.roboflow.com/microsoft/coco-dataset-vdnr1/model/11">Test in Browser</a></td>
</tr>
</table>