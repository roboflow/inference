# Integration with Batch Processing

Batch processing is well-suited for task automation, especially when processes need to run on a 
recurring basis. A common question is how to integrate Roboflow Batch Processing with external 
systems on the client’s end.

This documentation provides a detailed guide on integration strategies, covering best practices 
and implementation approaches.

## Overview

The following bullet-points illustrate the interaction with Roboflow Batch Processing:

* **Workflow Creation:** A workflow is created to process the data.

* **Data Ingestion:** The data is ingested, creating a batch in Data Staging, which serves as ephemeral storage 
for both input and output batch processing data.

* **Processing:** The data undergoes processing, which involves multiple (typically two) stages. Usually `processing` 
stage is responsible for running the Workflow against ingested data producing predictions. This stage results in 
CSV / JSONL files being created. This stage is usually followed by `export`, which is responsible for creating archives 
with processing results for convenient extraction from Roboflow platform.

* **Data Export:** The data is exported from one of the output batches created during processing. Data is exposed 
through download links which can be used to pull the data. 


## Basic interactions with Batch Processing

We will begin by outlining the basic interactions with this feature, providing a foundational understanding. 
Building on this, we will explore more advanced concepts later in the documentation.

For demonstration, we will use `inference-cli` commands and cURL, though the same functionality is available through the UI.

The first step in batch processing is ingesting the data. Roboflow supports both video and image ingestion, 
with options for individual uploads and optimized bulk ingestion.

!!! hint "`inference-cli` installation"

    If you plan to use `inference-cli` snippets - please make sure that the package is installed in your environment
    
    ```bash
    pip install inference-cli
    ```

    For convenience, we recommend exporting Roboflow API key as env variable:

    ```bash
    export ROBOFLOW_API_KEY=YOUR_API_KEY
    ```


### Video ingestion
The simplest approach is uploading videos one by one. Once the request succeeds, the videos are ready for use. Use the 
following command to ingest videos:

=== "inference-cli"
    ```bash
    inference rf-cloud data-staging create-batch-of-videos --videos-dir <your-images-dir-path> --batch-id <your-batch-id>
    ```

=== "cURL"
    
    Ingesting video file is in fact requesting upload URL from Roboflow API and uploading the data 
    to the pointed location.
    
    ```bash
    curl -X POST "https://api.roboflow.com/data-staging/v1/external/{workspace}/batches/{batch_id}/upload/video" \
      -G \
      --data-urlencode "api_key=YOUR_API_KEY" \
      --data-urlencode "fileName=your_video.mp4"
    ```

    Response contains `"signedURLDetails"` key with the following details:
    
    * `"uploadURL"` - the URL to PUT the video

    * `"extensionHeaders"` - additional headers to include

    To upload the video, send the following request:
    
    ```bash
    curl -X PUT <url-from-the-response> -H "Name: value" --upload-file <path-to-your-video>
    ```
    
    with all headers from `"extensionHeaders"` response field.

    You can find full API reference [here](workflows/batch_processing/api_reference.md).


!!! warning "`batch_id` constraints"

    Currently, the client is in controll of `batch_id` and must meet the following 
    constraints:
    
    * only lower-cased ASCII letters, digits, hypen (`-`) and underscore (`_`) characters are allowed

    * `batch_id` must be at most 64 characters long


### Image Ingestion

Images can also be uploaded individually. However, for larger batches, a batch upload API is available for better 
performance. Clients can bundle up to 500 images into a single `*.tar` archive and send it to the API.

!!! note "Asynchronous indexing of batches"

    When performing bulk ingestion, Roboflow indexes the data in the background. This means that after the command 
    completes, there may be a short delay before the data is fully available, depending on batch size.

=== "inference-cli"
    
    ```bash
    inference rf-cloud data-staging create-batch-of-images --images-dir <your-images-dir-path> --batch-id <your-batch-id>
    ```

=== "cURL (single image)"
    
    The approach presented below is simplified version of bulk images ingest which let clients upload 
    images one-by-one. Due to speed limitations, **we recommend using this method for batches up to 5000 images.** 
    It is also important to note that **this method cannot be used along with bulk ingest for the same batch.**

    ```bash
    curl -X POST "https://api.roboflow.com/data-staging/v1/external/{workspace}/batches/{batch_id}/upload/image" \
      -G \
      --data-urlencode "api_key=YOUR_API_KEY" \
      --data-urlencode "fileName=your_image.jpg" \
      -F "your_image.jpg=@/path/to/your/image.jpg"
    ```

    You can find full API reference [here](workflows/batch_processing/api_reference.md).


=== "cURL (bulk upload)"
    
    To optimise ingest speed, we recommend using **bulk ingests for batches of size exceeding 5000 images.** 
    The procedure contains three steps:
    
    * requesting upload URL from Roboflow API

    * packing images to `*.tar` archive according to limits of size and files number dictated by API

    * uploading the archive to URL obtained from Roboflow API
    
    ```bash
    curl -X POST "https://api.roboflow.com/data-staging/v1/external/{workspace}/batches/{batch_id}/bulk-upload/image-files" \
      -G \
      --data-urlencode "api_key=YOUR_API_KEY"
    ```
    
    Response contains `"signedURLDetails"` key with the following details:
    
    * `"uploadURL"` - the URL to PUT the video

    * `"extensionHeaders"` - additional headers to include
    

    To upload the archive, send the following request:
    
    ```bash
    curl -X PUT <url-from-the-response> -H "Name: value" --upload-file <path-to-your-video>
    ```
    
    with all headers from `"extensionHeaders"` response field.

    Please remember, that bach created with bulk upload cannot be further filled with data ingested with simple upload 
    due to internal data organisation enforced by the upload method.
    
    You can find full API reference [here](workflows/batch_processing/api_reference.md).

!!! warning "`batch_id` constraints"

    Currently, the client is in controll of `batch_id` and must meet the following 
    constraints:
    
    * only lower-cased ASCII letters, digits, hypen (`-`) and underscore (`_`) characters are allowed

    * `batch_id` must be at most 64 characters long


### Before starting the job
Once data ingestion is complete, the next step is to start a job. However, since some background processing occurs, 
it's important to determine **when the data is fully ready.**

The simplest approach is to check the batch status by polling the system. This allows you to verify how many files 
have been successfully ingested before proceeding. Use the following command:

=== "inference-cli"
    ```bash
    inference rf-cloud data-staging show-batch-details --batch-id <your-batch-id>
    ```

=== "cURL"
    ```bash
    curl -X GET "https://api.roboflow.com/data-staging/v1/external/{workspace}/batches/{batch_id}/count" \
      -G \
      --data-urlencode "api_key=YOUR_API_KEY"
    ```

    You can find full API reference [here](workflows/batch_processing/api_reference.md).


To identify potential issues during data ingestion, you can fetch status updates using the following command:

=== "inference-cli"
    ```bash
    inference rf-cloud data-staging list-ingest-details --batch-id <your-batch-id>
    ```

=== "cURL"
    Endpoint to fetch shard upload details supports pagination, as the number of shards may be huge. Use 
    `nextPageToken` in consecutive requests, based on previous responses (first request do not need to have this 
    parameter attached).

    ```bash
    curl -X GET "https://api.roboflow.com/data-staging/v1/external/{workspace}/batches/{batch_id}/shards" \
      -G \
      --data-urlencode "api_key=YOUR_API_KEY" \
      --data-urlencode "nextPageToken=OptionalNextPageToken"
    ```

    You can find full API reference [here](workflows/batch_processing/api_reference.md).


Only after confirming that the expected number of files is available can the data be considered ready for processing.

### Job kick off

Once data ingestion is complete, you can start processing the batch. For image processing, use the following command:

=== "inference-cli"
    ```bash
    inference rf-cloud batch-processing process-images-with-workflow --workflow-id <workflow-id> --batch-id <batch-id>
    ```

=== "cURL"
    
    ```bash
    curl -X POST "https://api.roboflow.com/batch-processing/v1/external/{workspace}/jobs/{job_id}" \
      -G \
      --data-urlencode "api_key=YOUR_API_KEY" \
      -H "Content-Type: application/json" \
      -d '{
        "type": "simple-image-processing-v1",
        "jobInput": {
            "type": "staging-batch-input-v1",
            "batchId": "{batch_id}"
        },
        "computeConfiguration": {
            "type": "compute-configuration-v2",
            "machineType": "cpu",
            "workersPerMachine": 4
        },
        "processingTimeoutSeconds": 3600,
        "processingSpecification": {
            "type": "workflows-processing-specification-v1",
            "workspace": "{workspace}",
            "workflowId": "{workflow_id}",
            "aggregationFormat": "jsonl"
        }
    }'
    ```

    This example does not cover all of the parameters that can be used. 
    You can find full API reference [here](workflows/batch_processing/api_reference.md).


!!! warning "`job_id` constraints"

    Currently, the client is in controll of `job_id` and must meet the following 
    constraints:
    
    * only lower-cased ASCII letters, digits, hypen (`-`) and underscore (`_`) characters are allowed

    * `job_id` must be at most 20 characters long
    

Batch processing jobs can take time to complete. To determine when results are ready for export, you need to 
periodically check the job status:


=== "inference-cli"
    ```bash
    inference rf-cloud batch-processing show-job-details --job-id <your-job-id>
    ```

=== "cURL"
    
    Since job contains several stages, checking job status may be performed at different 
    level of granularity.

    To check the general status of a job:
    ```bash
    curl -X GET "https://api.roboflow.com/batch-processing/v1/external/{workspace}/jobs/{job_id}" \
      -G \
      --data-urlencode "api_key=YOUR_API_KEY"
    ```
    
    To list job stages:
    ```bash
    curl -X GET "https://api.roboflow.com/batch-processing/v1/external/{workspace}/jobs/{job_id}/stages" \
      -G \
      --data-urlencode "api_key=YOUR_API_KEY"
    ```

    To list job stage tasks (fundamental units of processing):
    ```bash
    curl -X GET "https://api.roboflow.com/batch-processing/v1/external/{workspace}/jobs/{job_id}/stages/{stage_id}/tasks" \
      -G \
      --data-urlencode "api_key=YOUR_API_KEY" \
      --data-urlencode "nextPageToken={next_page_token}"
    ```
    The endpoint supports pagination - use `nextPageToken` in consecutive requests, based on previous responses 
    (first request do not need to have this parameter attached).

    You can find full API reference [here](workflows/batch_processing/api_reference.md).



### Data export
Once the batch processing job is complete, you can download the results using the following command. 
**Figuring out `batch-id` would require listing job stages and choosing which stage output to export.** 
Typically `export` stage output is exported, as it contains archives that are easy to be transferred, but 
one may choose to export results of other stages if needed.

=== "inference-cli"
    ```bash
    inference rf-cloud data-staging export-batch --target-dir <dir-to-export-result> --batch-id <output-batch-of-a-job>
    ```

=== "cURL"
    
    The Batch Processing service creates "multipart" batches to store results. The term "multipart" refers to 
    nested nature of batch - namely it contains multiple "parts" of data, each being nested batch. To export such
    data batch, you may want to list its parts first:

    ```bash
    curl -X GET "https://api.roboflow.com/data-staging/v1/external/{workspace}/batches/{batch_id}/parts" \
      -G \
      --data-urlencode "api_key=YOUR_API_KEY"
    ```

    Then, for each part, list operation can be performed. As a result - download URL for each batch element will be 
    exposed:

    ```bash
    curl -X GET "https://api.roboflow.com/data-staging/v1/external/{workspace}/batches/{batch_id}/list" \
      -G \
      --data-urlencode "api_key=YOUR_API_KEY" \
      --data-urlencode "nextPageToken=YOUR_NEXT_PAGE_TOKEN" \
      --data-urlencode "partName=YOUR_PART_NAME"
    ```
    The endpoint supports pagination - use `nextPageToken` in consecutive requests, based on previous responses 
    (first request do not need to have this parameter attached).

    Having download URL for each batch element, the following `curl` command can be used to pull the data.
    
    ```bash
    curl <download-url> -o <downlaod-file-location>
    ```

    You can find full API reference [here](workflows/batch_processing/api_reference.md).



As seen in this workflow, multiple manual interactions are required to complete the process. To efficiently handle 
large volumes of data on a recurring basis, automation is essential. The next section will explore strategies for 
automating batch processing.

## Automation on top of Batch Processing

The approach outlined above presents a few challenges:

* **Local Data Dependency:** The process assumes that data is locally available, which is often not the case in 
enterprise environments where data is stored in the cloud.

* **Active Polling:** The process requires constant polling to verify statuses, which is inefficient and not scalable.

To address these issues, the system supports webhook notifications for both data staging and batch processing. 
Webhooks allow external systems to automatically react to status changes, eliminating the need for manual polling and 
enabling seamless automation. Additionally, system allows for data ingestion through **signed URLs** 
which should streamline integrations with cloud storage.

!!! important "API integration"

    In the document, we are presenting snippets from `inefrence-cli`, but under the hood there is 
    fully functional API that clients may integrate with (or use Python client embedded in CLI).

### Data ingestion

Both `inference rf-cloud data-staging create-batch-of-videos` and `inference rf-cloud data-staging create-batch-of-images` 
commands support additional parameters to enable webhook notifications and ingest data directly from cloud storage.

* `--data-source reference-file` instructs the CLI to process files referenced via signed URLs instead of requiring local data.

* `--references <path_or_url>` specifies either a local path to a JSONL document containing file URLs or a signed URL pointing to such a file.

* `--notifications-url <webhook_url>` defines the webhook URL where notifications about the ingestion process will be sent.

* `--notification-category <value>` - optionally allows filtering which notifications are sent. Options include:

    * `ingest-status` *(default)* - notifications about the overall ingestion process.
    
    * `files-status` - notifications for individual file processing.`

!!! important "API reference"

    Use [API reference](./api_reference.md) document to find out how to enforce 
    Batch Processing service notifications when integrating directly with API.


!!! important "Limited access to the feature"

    Currently, only Growth Plan customers and Enterprise customers can ingest data through signed URLs.


#### Structure of references file

!!! info "JSONL (JSON Lines)"

    The **JSONL (JSON Lines)** format consists of multiple JSON objects, each on a separate line within a single text file. 
    This format is commonly used to store large datasets and allows for efficient line-by-line processing.

Each entry in the references file contains two key attributes:

* `name`: A unique identifier for the file.
* `url`: A signed URL pointing to the file in cloud storage.

Here’s an example of the JSONL format:
```
{"name": "<your-unique-name-of-file-1>", "url": "https://<signed-url>"}
{"name": "<your-unique-name-of-file-2>", "url": "https://<signed-url>"}
```

You can store this references file locally or in cloud storage. If the file is stored in the cloud, simply provide the 
signed URL to the file when running the ingestion command.

#### Notifications

Notifications are delivered to clients via HTTP POST requests sent to the specified webhook endpoint. Each notification 
will include an Authorization header containing the Roboflow Publishable Key to authenticate the request.

#### Cloud storage
##### S3 Datasource
Using [AWS S3](https://aws.amazon.com/s3/) for ingesting data for batch processing can be achieved easily with this example script.
This also supports S3-compatible datasources such as Backblaze B2, Cloudflare R2, Oracle Cloud Infrastructure Object Storage and many more

```sh
insrt script
```

##### GCS Datasource
Using [google cloud storage (GCS)](https://cloud.google.com/storage) for ingesting data for batch processing can be easily achieved with this example script.

```sh
insert script
```



##### `ingest-status` notifications

Currently, the ingest-status category includes a single notification type, with the potential for more to be added in 
the future. This notification provides updates about the overall status of the data ingestion process.

Each ingest-status notification will contain details about the progress or completion of the data ingestion, 
allowing clients to track the status in real-time.

```json
{
    "type": "roboflow-data-staging-notification-v1",
    "event_id": "8c20f970-fe10-41e1-9ef2-e057c63c07ff",
    "ingest_id": "8cd48813430f2be70b492db67e07cc86",
    "batch_id": "test-batch-117",
    "shard_id": null,
    "notification": {
        "type": "ingest-status-notification-v1",
        "success": false,
        "error_details": {
            "type": "unsafe-url-detected",
            "reason": "Untrusted domain found: https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png"
        }
    },
    "delivery_attempt": 1
}
```

##### `files-status` notifications

In addition to the overall ingestion status, the system also allows clients to track the status of individual files 
ingested into batches. This provides granular insights into the progress of each file during the ingestion process.

Since multiple files can be ingested in a batch, notifications for each file are sent in bulk. However, the volume of 
notifications may become large depending on the number of files involved in the batch.

Despite this, the system optimizes the delivery of notifications, ensuring that you can stay informed about each file's 
status without overwhelming your system.

```json
{
    "type": "roboflow-data-staging-notification-v1",
    "event_id": "8f42708b-aeb7-4b73-9d83-cf18518b6d81",
    "ingest_id": "d5cb69aa-b2d1-4202-a1c1-0231f180bda9",
    "batch_id": "prod-batch-1",
    "shard_id": "0d40fa12-349e-439f-83f8-42b9b7987b33",
    "notification": {
        "type": "ingest-files-status-notification-v1",
        "success": true,
        "ingested_files": [
            "000000494869.jpg",
            "000000186042.jpg"
        ],
        "failed_files": [
            {
                "type": "file-size-limit-exceeded",
                "file_name": "1d3717ec-6e11-4fd5-a91d-7e1eda235aa2-big_image.png",
                "reason": "Max size of single image is 20971520B."
            }
        ],
        "content_truncated": false
    },
    "delivery_attempt": 1
}
```

### Job kick off
By using the ingestion method outlined above, your system will be notified when the data is ready to be processed, allowing you to react to this event and trigger the batch processing job automatically.

Batch jobs also support webhook notifications. Once the processing of the batch is complete, your system will be notified that the results are available. This notification enables you to pull the results and continue with any post-processing steps without manual intervention.

By leveraging webhook notifications for both the ingestion and job processing stages, you can automate the entire pipeline, ensuring a smooth, hands-off experience.

```bash
inference rf-cloud batch-processing process-images-with-workflow --workflow-id <workflow-id> --batch-id <batch-id> --notifications-url <webhook_url>
```

#### Format of notification

```json
{
  "type": "roboflow-batch-job-notification-v1",
  "event_id": "8f42708b-aeb7-4b73-9d83-cf18518b6d81",
  "job_id": "<your-batch-job-id>",
  "job_state": "success | fail",
  "delivery_attempt": 1
}
```

### Data export

Once the batch job is completed, your system will be notified via the configured webhook. In response to this event, you can automatically trigger the pulling of results and initiate any further processing required on your end.

This allows for a seamless, end-to-end automation, reducing manual intervention and ensuring that results are quickly integrated into your workflow once the job finishes.
