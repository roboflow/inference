# Single-view 3D recontsruction with SAM-3D Objects

SAM-3D is 3D object generation model that converts 2D images with segmentation masks into 3D assets (meshes and Gaussian splats) and estimates their layout. You can learn more about it by visiting this [Roboflow Blog post](https://blog.roboflow.com/sam-3d/) and the [Meta AI project page](https://ai.meta.com/research/sam3d/).

![SAM-3D examples](https://blog.roboflow.com/content/images/size/w1000/2025/11/SAM3d_object_example.png)

## Requirements

To run these examples you will need a self-hosted inference server with a 32GB+ VRAM GPU and the `SAM3_3D_OBJECTS_ENABLED` flag turned on. We recommend using the Docker workflow below:

```bash
docker build -t roboflow/roboflow-inference-server-gpu-3d:dev -f docker/dockerfiles/Dockerfile.onnx.gpu.3d .
```

```bash
docker run --gpus all -p 9001:9001 -v ./inference:/app/inference roboflow/roboflow-inference-server-gpu-3d:dev
```

## Example notebooks

* [Single-view 3D reconstruction with SAM-3D Objects](sam-3d-detect.ipynb)
* [Monocular 3D object tracking with SAM-3D Objects](sam-3d-track.ipynb)