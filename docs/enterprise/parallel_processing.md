# Parallel Inference

!!! note

    This feature is only available for Roboflow Enterprise users. <a href="https://roboflow.com/sales" target="_blank">Contact our sales team</a> to learn more about Roboflow Enterprise.

You can run multiple models in parallel with Inference with parallel processing, a version of Roboflow Inference that processes inference requests asynchronously.

Inference Parallel supports all the same features as Roboflow Inference, with the exception that it does not support Core models (i.e. CLIP and SAM).

With Inference Parallel, preprocessing, auto batching, inference, and post processing all run in separate threads to increase server FPS throughput.

Separate requests to the same model will be batched on the fly as allowed by `$MAX_BATCH_SIZE`, and then response handling will occurr independently. Images are passed via Python's SharedMemory module to maximize throughput.

These changes result in as much as a *76% speedup* on one measured workload.

## How To Use Inference with Parallel Processing

You can run Inference with Parallel Processing in two ways: via the CLI or via Docker.

=== "Bash"

    First, build the parallel server

    ```bash
    ./inference/enterprise/parallel/build.sh
    ```

    Then, run the server:

    ```
    ./inference/enterprise/parallel/run.sh
    ```

    A message will appear in the terminal indicating that the server is running and ready for use.

=== "Docker"

    We provide a container at Docker Hub that you can pull using `docker pull roboflow/roboflow-inference-server-gpu-parallel:latest`. If you are pulling a pinned tag, be sure to change the `$TAG` variable in `run.sh`.


## Benchmarking

We evaluated the performance of Inference Parallel on a variety of models from <a href="https://universe.roboflow.com/" target="_blank">Roboflow Universe</a>. We compared the performance of Inference Parallel to the latest version of Inference Server (0.9.5.rc) on the same hardware.

We ran our tests on a computer with eight cores and one GPU. Instance segmentation metrics are calculated using `"mask_decode_mode": "fast"` in the request body. Requests are posted concurrently with a parallelism of 1000.

Here are the results of our tests:

| Workspace | Model | Model Type | split | 0.9.5.rc FPS| 0.9.5.parallel FPS |
| ----------|------ | ----------- |------|-------------| -------------------|
| senior-design-project-j9gpp | nbafootage/3| object-detection | train | 30.2 fps | 44.03 fps |
| niklas-bommersbach-jyjff   | dart-scorer/8| object-detection | train | 26.6 fps | 47.0 fps |
| geonu  | water-08xpr/1 | instance-segmentation | valid | 4.7 fps | 6.1 fps |
| university-of-bradford | detecting-drusen_1/2 | instance-segmentation | train | 6.2 fps | 7.2 fps |
| fy-project-y9ecd | cataract-detection-viwsu/2 | classification | train | 48.5 fps | 65.4 fps |
| hesunyu | playing-cards-ir0wr/1 | classification | train | 44.6 fps | 57.7 fps |

Inference with parallel processing enabled achieved higher FPS on every test. On eome models, the FPS increase by using Inference with parallel processing was greater than 10 FPS.