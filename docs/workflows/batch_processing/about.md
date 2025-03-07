# Roboflow Batch Processing

Roboflow Batch Processing is a fully managed solution powered by **Workflows** that allows you to process large 
volumes of videos and images without writing code. It offers an easy-to-use UI for quick tasks and a comprehensive API 
for automating data processing - fitting both small and large workloads.

With configurable processing workflows, real-time monitoring, and event-based notifications, Roboflow Batch Processing 
helps you efficiently manage data processing, track progress, and integrate with other systems — making it easy to 
achieve your goals.

!!! info "What is the nature of `batch` processing?"

    Batch processing means accepting large volumes of data to be processed in jobs that are scheduled to run 
    "in the background" - without external orchestation, but also with weak guarantees on when the job is started
    and when results will be available.

    When service is not busy, jobs start within 3-8 minutes after being scheduled, but this time may be longer 
    under high load.

    This service is suitable for use cases which does not require real-time responses, for example:

    * analysis of pre-recorded video files

    * making predictions from large pool of images saved in your storage

    * automatic data labeling

## Getting started

To get started with Roboflow Batch Processing, first, build and test your Workflow. Once it's ready, select the 
data you want to process and upload it using the UI or CLI tool. Then, initiate the processing and let Batch 
Processing handle the rest. Once the job is completed, retrieve the results and use them as needed. That's it - 
you no longer need to write code or run processing on your machine.

### UI Interface

Roboflow platform exposes UI interface to interact with Roboflow Batch Processing making it easy and accessible to
try out the feature or process **small to moderate** amount of data. 

![Batch Processing UI](https://media.roboflow.com/inference/batch-processing/bp-ui.jpg)


When creating a job, you can choose between image and video processing, select the appropriate Workflow, and adjust 
settings to fine-tune the system’s behavior. Key options include:

* **Machine type** – Choose between GPU and CPU based on processing needs. For Workflows using multiple or large models, 
a GPU is recommended, while smaller models or external API-based tasks can run efficiently on a CPU.

* **Predictions visualization & video outputs** – Enable this option if you want to generate and keep visual 
outputs of your Workflow.

* **Video frame rate sub-sampling** – Skip frames for faster, more cost-effective video processing.

* **Maximum job runtime** – Set a limit to help control costs and prevent excessive processing times.

### CLI

Installing `inference-cli` you are gaining access to `inference rf-cloud` command to interact with managed components 
of Roboflow Platform - including `batch-processing` and `data-staging` which are core components of Roboflow Batch 
Processing offering. 

The typical flow of interaction with CLI looks like that:

You ingest data to the platform - for images, use the following command

```bash
inference data-staging create-batch-of-images --images-dir <your-images-dir-path> --batch-id <your-batch-id>
```

for videos:
```bash
inference data-staging create-batch-of-videos --videos-dir <your-images-dir-path> --batch-id <your-batch-id>
```

!!! hint "Format of `<your-batch-id>`"

    Batch ID must be lower-cased string without special caraters, with letters and digits allowed.


You can inspect the details of staged batch of data:

```bash
inference rf-cloud data-staging show-batch-details --batch-id <your-batch-id>
```

Once the data is ingested - you can trigger batch job.

For images, use the following command:

```bash
inference rf-cloud batch-processing process-images-with-workflow --workflow-id <workflow-id> --batch-id <batch-id>
```

For videos:

```bash
inference rf-cloud batch-processing process-videos-with-workflow --workflow-id <workflow-id> --batch-id <batch-id>
```

!!! hint "How would I know `<workflow-id>`?"

    Workflow ID can be obtained in Roboflow App - open Workflow Editor of selected Workflow, hit "Deploy" button 
    and find identifier in code snippet.

Command will **print out the ID of the job**, which can be used to check job status: 

```bash
inference rf-cloud batch-processing show-job-details --job-id <your-job-id>
```

This way you can track progress of the job. What is also important, **the command will provide ID of output batch** 
which can be used to export results:

```bash
inference rf-cloud data-staging export-batch --target-dir <dir-to-export-result> --batch-id <output-batch-of-a-job>
```

That's it - you should be able to see your processing results now.

!!! hint "All configuration options"

    To discover all configuration options use the following help commands:

    ```bash
    inference rf-cloud --help
    inference rf-cloud data-staging --help 
    inference rf-cloud batch-processing --help 
    ```

## Service Pricing

The service charges usage based on the runtime of underlying compute machines, starting at **4 credits** per GPU hour 
and **1 credit** per CPU hour. You can find the specific rates for your workspace on our 
[pricing page](https://roboflow.com/pricing).

We cannot provide an exact cost estimate for processing 1,000 images or 1 hour of video, as this depends 
entirely on the **complexity of the chosen Workflow**. However, we offer benchmark results to help you 
better understand potential costs:

| Workflow Description                                               | Dataset Size                    | Machine Type | Charge                                              |
|--------------------------------------------------------------------|---------------------------------|--------------|-----------------------------------------------------|
| Single Model - YOLOv8 Nano `(image size = 640)`                    | 100k images                     | GPU          | 0.04  credit / 1k images                            |
| Single Model - YOLOv8 Nano `(image size = 1280)`                   | 100k images                     | GPU          | 0.06  credit / 1k images                            |
| Single Model - YOLOv8 Medium `(image size = 640)`                  | 100k images                     | GPU          | 0.06  credit / 1k images                            |
| Single Model - YOLOv8 Medium `(image size = 1280)`                 | 100k images                     | GPU          | 0.1  credit / 1k images                             |
| Single Model - YOLOv8 Large `(image size = 640)`                   | 100k images                     | GPU          | 0.08 credit / 1k images                             |
| Single Model - YOLOv8 Large `(image size = 1280)`                  | 100k images                     | GPU          | 0.18 credit / 1k images                             |
| Single Model - Roboflow Instant                                    | 30k images                      | GPU          | 0.33  credit / 1k images                            |
| Single Model - Florence-2 (detection + caption)                    | 30k images                      | GPU          | 0.5  credit / 1k images                             |
| Two stage - YoloV8 Nano + crop + YoloV8 Nano `(image size = 640)`  | 10k images                      | GPU          | 0.25  credit / 1k images                            |
| Two stage - YoloV8 Nano + crop + YoloV8 Large `(image size = 640)` | 10k images                      | GPU          | 0.30  credit / 1k images                            |
| Two stage - YoloV8 Nano + crop + CLIP `(image size = 640)`         | 10k images                      | GPU          | 0.25  credit / 1k images                            |
| Two stage - YoloV8 Nano + crop + Classifier `(image size = 640)`   | 10k images                      | GPU          | 0.20  credit / 1k images                            |
| Two stage - YoloV8 Nano + crop + SAM 2 `(image size = 640)`        | 10k images                      | GPU          | 0.40  credit / 1k images                            |
| Single Model - YOLOv8 Nano `(image size = 640)`                    | 4 videos, each 1h @ 30 fps 480p | GPU          | 1  credit / video hour, 0.01 credit / 1k frames     |
| Single Model - YOLOv8 Nano `(image size = 640)` + tracking         | 32 videos, each 1m @ 10 fps HD  | CPU          | 1.8 credit / video hour, 0.05 credit / 1k frames    |
| Two stage - YoloV8 Nano + crop + Classifier `(image size = 640)`   | 2 videos, each 1h @ 30 fps 480p | GPU          | 4.6  credits / video hour, 0.046 credit / 1k frames |


!!! Warning "Cost estimation in practice"

    Please find above results as reference values only - we advise to check the cost of smaller batches
    of data before running large processing jobs! Reported values are possible to be reproduced 
    once optimal settings for mahcine type and machine concurrency are set.

    Please acknowledge the technical nuances of the service (described bellow) to understand the pricing 
    better - in particular - since the service is sharding data under the hood (and executes parallel 
    processing on multiple machines at the same time) - **wall clock execution time usually does not 
    equall the time that is billed** - for instance if the job is using 4 GPU machines for an hour - 
    billed amount would be 4 GPU-hours - 16 credits.

## Known limitations

* Batch Processing service cannot run Custom Python blocks.

* Certain Workflow blocks requiring access to env variables and local storage (like File Sink and Environment 
Secret Store) are blacklisted and will not execute.

* Service only works with Workflows that define **singe** input image parameter.


## Technical details of Batch Processing

* Data is stored in `data-staging` component of the system - both your input images / videos are stored there, as 
well as the processing results. Expiry time for any piece of data submitted to data staging is **7 days**

* When you upload a data, you create a data batch in data staging, batch processing jobs accept input batches
pointed to be processing (one batch is processed by single job).

* Single batch processing job contains multiple stages (typically `processing` and `export`) - each stage would 
create output batch that you can retrieve later on. We **advise** to use `export` stage outputs, as those 
are adjusted to be transferred through the network (content is compressed / packed into the archive).

* Running job, when in `processing` stage` can be aborted using both UI and CLI

* Aborted / Failed job can be restarted using mentioned tools

* The service automatically shards the data and process them in parallel under the hood

* Parallelism is applied at two levels:

    * Service **automatically** adjust number of machines running the job based on data volume - ensuring 
    sufficient data throughput (values reaching 500k-1M images an hour should be achievable for certain workloads)

    * Single machine runs multiple workers processing tasks (chunks of data) that belong to the job - this option
    **can be configured** by user and should be adjusted to balance out processing speed and costs
