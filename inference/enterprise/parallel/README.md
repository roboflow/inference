# Parallel Roboflow Inference server

The Roboflow Inference Server supports concurrent processing. This version of the server accepts and processes requests asynchronously, running the web server, preprocessing, auto batching, inference, and post processing all in separate threads to increase server FPS throughput. Separate requests to the same model will be batched on the fly as allowed by `$MAX_BATCH_SIZE`, and then response handling will occurr independently. Images are passed via Python's SharedMemory module to maximize throughput.

These changes result in as much as a *76% speedup* on one measured workload.

> [!NOTE]
> Currently, only Object Detection, Instance Segmentation, and Classification models are supported by this module. Core models are not enabled.

> [!IMPORTANT] 
> We require a Roboflow Enterprise License to use this in production. See inference/enterpise/LICENSE.txt for details.

## How To Use Concurrent Processing
You can build the server using `./inference/enterprise/parallel/build.sh` and run it using `./inference/enterprise/parallel/run.sh`

We provide a container at Docker Hub that you can pull using `docker pull roboflow/roboflow-inference-server-gpu-parallel:latest`. If you are pulling a pinned tag, be sure to change the `$TAG` variable in `run.sh`.

This is a drop in replacement for the old server, so you can send requests using the [same API calls](https://inference.roboflow.com/quickstart/http_inference/#step-2-run-inference) you were using previously.


## Performance
We measure and report performance across a variety of different task types by selecting random models found on Roboflow Universe.

### Methodology

The following metrics are taken on a machine with eight cores and one gpu. The FPS metrics reflect best out of three trials. The column labeled 0.9.5.parallel reflects the latest concurrent FPS metrics. Instance segmentation metrics are calculated using `"mask_decode_mode": "fast"` in the request body. Requests are posted concurrently with a parallelism of 1000.

### Results
| Workspace | Model | Model Type | split | 0.9.5.rc FPS| 0.9.5.parallel FPS | 0.9.5.parallel (orjson) FPS |
| ----------|------ | ----------- |------|-------------| -------------------|-----------------------------|
| senior-design-project-j9gpp | nbafootage/3| object-detection | train | 30.2 fps | 44.03 fps | 54.5 fps |
| niklas-bommersbach-jyjff   | dart-scorer/8| object-detection | train | 26.6 fps | 47.0 fps | 52.3 fps |
| geonu  | water-08xpr/1 | instance-segmentation | valid | 4.7 fps | 6.1 fps | 10.4 fps |
| university-of-bradford | detecting-drusen_1/2 | instance-segmentation | train | 6.2 fps | 7.2 fps | 10.4 fps |
| fy-project-y9ecd | cataract-detection-viwsu/2 | classification | train | 48.5 fps | 65.4 fps | 64.9 fps |
| hesunyu | playing-cards-ir0wr/1 | classification | train | 44.6 fps | 57.7 fps | 57.7 fps |
