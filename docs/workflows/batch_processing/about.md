# Roboflow Batch Processing

Roboflow Batch Processing is a fully managed solution powered by **Workflows** that allows you to process large 
volumes of videos and images without writing code. It offers an easy-to-use UI for quick tasks and a comprehensive 
API for automating data processing—fitting both small and large workloads.

With configurable processing workflows, real-time monitoring, and event-based notifications, Roboflow Batch Processing 
helps you efficiently manage data processing, track progress, and integrate with other systems—making it easy to 
achieve your goals.


!!! info "What is the nature of `batch` processing?"

    Batch processing involves accepting large volumes of data to be processed in jobs that run  
    **in the background** - without external orchestration and with weak guarantees on when the job will start or 
    when results will be available.

    When the service is not busy, jobs typically start within 3–8 minutes after being scheduled, but this time may 
    be longer under high load.

    This service is suitable for use cases that do not require real-time responses, such as:

    * Analyzing pre-recorded video files

    * Making predictions from a large pool of images stored in your storage

    * Automatic data labeling

## Getting started

To get started with Roboflow Batch Processing, first build and test your Workflow. Once it's ready, select the data you 
want to process and upload it using the UI or CLI tool. Then, initiate the processing and let Batch Processing handle 
the rest. Once the job is completed, retrieve the results and use them as needed. That’s it — you no longer need to 
write code or run processing on your machine.

### UI Interface

The Roboflow platform provides a UI interface to interact with Roboflow Batch Processing, making it easy and accessible 
to try out the feature or process a  **small to moderate** amount of data. 

![Batch Processing UI](https://media.roboflow.com/inference/batch-processing/bp-ui.jpg)


When creating a job, you can choose between image and video processing, select the appropriate Workflow, and adjust 
settings to fine-tune the system’s behavior. Key options include:

* **Machine type** – Choose between GPU and CPU based on processing needs. For Workflows using multiple or large 
models, a GPU is recommended, while smaller models or external API-based tasks can run efficiently on a CPU.

* **Predictions visualization & video outputs** – Enable this option if you want to generate and retain visual 
outputs of your Workflow.

* **Video frame rate sub-sampling** – Skip frames for faster, more cost-effective video processing.

* **Maximum job runtime** – Set a limit to help control costs and prevent excessive processing times.

### CLI

By installing `inference-cli` you gain access to the `inference rf-cloud` command, which allows you to interact with 
managed components of the Roboflow Platform — including `batch-processing` and `data-staging`, the core components of 
the Roboflow Batch Processing offering.

The typical flow of interaction with the CLI is as follows:

First, ingest data into the platform. For images, use the following command:

```bash
inference rf-cloud data-staging create-batch-of-images --images-dir <your-images-dir-path> --batch-id <your-batch-id>
```

for videos:
```bash
inference rf-cloud data-staging create-batch-of-videos --videos-dir <your-images-dir-path> --batch-id <your-batch-id>
```

!!! hint "Format of `<your-batch-id>`"

    Batch ID must be lower-cased string without special caraters, with letters and digits allowed.


Then, you can inspect the details of staged batch of data:

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

    Workflow ID can be found in Roboflow App - open Workflow Editor of selected Workflow, hit "Deploy" button 
    and find identifier in code snippet.

Command will **display the ID of the job**, which can be used to check the job status: 

```bash
inference rf-cloud batch-processing show-job-details --job-id <your-job-id>
```

This allows you to track the progress of your job. Additionally, **the command will provide the ID of the output 
batch**, which can be used to export results.

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

The service charges usage based on the runtime of the underlying compute machines, starting at **4 credits** per GPU hour 
and **1 credit** per CPU hour. You can find the specific rates for your workspace on our 
[pricing page](https://roboflow.com/pricing).

We cannot provide an exact cost estimate for processing 1,000 images or 1 hour of video, as this depends entirely on 
the **complexity of the chosen Workflow**. However, we offer benchmark results to help you better understand 
potential costs.

| Workflow Description                                                            | Dataset Size                    | Machine Type | Charge                                              |
|---------------------------------------------------------------------------------|---------------------------------|--------------|-----------------------------------------------------|
| Single Model - YOLOv8 Nano `(image size = 640)` - Object Detection              | 100k images                     | GPU          | 0.04  credit / 1k images                            |
| Single Model - YOLOv8 Nano `(image size = 1280)`- Object Detection              | 100k images                     | GPU          | 0.06  credit / 1k images                            |
| Single Model - YOLOv8 Medium `(image size = 640)` - Object Detection            | 100k images                     | GPU          | 0.06  credit / 1k images                            |
| Single Model - YOLOv8 Medium `(image size = 1280)` - Object Detection           | 100k images                     | GPU          | 0.1  credit / 1k images                             |
| Single Model - YOLOv8 Large `(image size = 640)` - Object Detection             | 100k images                     | GPU          | 0.08 credit / 1k images                             |
| Single Model - YOLOv8 Large `(image size = 1280)` - Object Detection            | 100k images                     | GPU          | 0.18 credit / 1k images                             |
| Single Model - Roboflow Instant - Object Detection                              | 30k images                      | GPU          | 0.33  credit / 1k images                            |
| Single Model - Florence-2 - Object Detection + Region Captioning                | 30k images                      | GPU          | 0.5  credit / 1k images                             |
| Two stage - YoloV8 Nano + crop + YoloV8 Nano `(image size = 640)` - OD          | 10k images                      | GPU          | 0.25  credit / 1k images                            |
| Two stage - YoloV8 Nano + crop + YoloV8 Large `(image size = 640)` - OD + OD    | 10k images                      | GPU          | 0.30  credit / 1k images                            |
| Two stage - YoloV8 Nano + crop + CLIP `(image size = 640)` - OD + Embeddings    | 10k images                      | GPU          | 0.25  credit / 1k images                            |
| Two stage - YoloV8 Nano + crop + Classifier `(image size = 640)` - OD + CLS     | 10k images                      | GPU          | 0.20  credit / 1k images                            |
| Two stage - YoloV8 Nano + crop + SAM 2 `(image size = 640)` - OD + Segmentation | 10k images                      | GPU          | 0.40  credit / 1k images                            |
| Single Model - YOLOv8 Nano `(image size = 640)` - Object Detection              | 4 videos, each 1h @ 30 fps 480p | GPU          | 1  credit / video hour, 0.01 credit / 1k frames     |
| Single Model - YOLOv8 Nano `(image size = 640)` - Object Detection + tracking   | 32 videos, each 1m @ 10 fps HD  | CPU          | 1.8 credit / video hour, 0.05 credit / 1k frames    |
| Two stage - YoloV8 Nano + crop + Classifier `(image size = 640)` - OD + CLS     | 2 videos, each 1h @ 30 fps 480p | GPU          | 4.6  credits / video hour, 0.046 credit / 1k frames |


!!! Warning "Cost estimation in practice"

    Please consider the results above as reference values only—we advise checking the cost of smaller data batches 
    before running large processing jobs. Reported values can be reproduced once optimal settings for machine type 
    and machine concurrency are configured.

    Please take into account the technical nuances of the service (described below) to better understand the pricing. 
    In particular, since the service shards data under the hood and executes parallel processing on multiple machines
    simultaneously, **wall clock execution time usually does not equal the billed time.** For instance, if a job uses 
    four GPU machines for one hour, the billed amount would be **4 GPU-hours (16 credits)**.

## Known limitations

* Batch Processing service cannot run Custom Python blocks.

* Certain Workflow blocks requiring access to env variables and local storage (like File Sink and Environment 
Secret Store) are blacklisted and will not execute.

* Service only works with Workflows that define **singe** input image parameter.


## Technical Details of Batch Processing

* Data is stored in the `data-staging` component of the system — both your input images / videos and the processing 
results. The expiry time for any piece of data submitted to data staging is **7 days**.

* When you upload data, you create a data batch in data staging. Batch processing jobs accept input batches marked 
for processing (each batch is processed by a single job).

* A single batch processing job contains multiple stages (typically `processing` and `export`). Each stage creates an 
output batch that you can retrieve later. We **advise** using `export` stage outputs, as they are optimized for 
network transfer (content is compressed / packed into an archive).

* A running job in the  `processing` tage can be aborted using both the UI and CLI.

* An aborted or failed job can be restarted using the mentioned tools.

* The service automatically shards the data and processes it in parallel under the hood.

* Parallelism is applied at two levels:

    * The service **automatically** adjusts the number of machines running the job based on data volume, ensuring 
    sufficient throughput (values reaching 500k–1M images per hour should be achievable for certain workloads).

    * A single machine runs multiple workers processing tasks (chunks of data) that belong to the job. This option
    **can be configured** by the user and should be adjusted to balance processing speed and costs.
