# `inference`

**Usage**:

```console
$ inference [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--version`
* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `benchmark`: Commands for running inference benchmarks.
* `cloud`: Commands for running the inference in...
* `infer`
* `rf-cloud`: Commands for interacting with Roboflow Cloud
* `server`: Commands for running the inference server...
* `workflows`: Commands for interacting with Roboflow...

## `inference benchmark`

Commands for running inference benchmarks.

**Usage**:

```console
$ inference benchmark [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `api-speed`
* `python-package-speed`

### `inference benchmark api-speed`

**Usage**:

```console
$ inference benchmark api-speed [OPTIONS]
```

**Options**:

* `-m, --model_id TEXT`: Model ID in format project/version.
* `-wid, --workflow-id TEXT`: Workflow ID.
* `-wn, --workspace-name TEXT`: Workspace Name.
* `-ws, --workflow-specification TEXT`: Workflow specification.
* `-wp, --workflow-parameters TEXT`: Model ID in format project/version.
* `-d, --dataset_reference TEXT`: Name of predefined dataset (one of ['coco']) or path to directory with images  [default: coco]
* `-h, --host TEXT`: Host to run inference on.  [default: http://localhost:9001]
* `-wr, --warm_up_requests INTEGER`: Number of warm-up requests  [default: 10]
* `-br, --benchmark_requests INTEGER`: Number of benchmark requests  [default: 1000]
* `-bs, --batch_size INTEGER`: Batch size of single request  [default: 1]
* `-c, --clients INTEGER`: Meaningful if `rps` not specified - number of concurrent threads that will send requests one by one  [default: 1]
* `-rps, --rps INTEGER`: Number of requests per second to emit. If not specified - requests will be sent one-by-one by requested number of client threads
* `-a, --api-key TEXT`: Roboflow API key for your workspace. If not given - env variable `ROBOFLOW_API_KEY` will be used
* `-mc, --model_config TEXT`: Location of yaml file with model config
* `-o, --output_location TEXT`: Location where to save the result (path to file or directory)
* `-L, --legacy-endpoints / -l, --no-legacy-endpoints`: Boolean flag to decide if legacy endpoints should be used (applicable for self-hosted API benchmark)  [default: no-legacy-endpoints]
* `-y, --yes / -n, --no`: Boolean flag to decide on auto `yes` answer given on user input required.  [default: no]
* `--max_error_rate FLOAT`: Max error rate for API speed benchmark - if given and the error rate is higher - command will return non-success error code. Expected percentage values in range 0.0-100.0
* `--help`: Show this message and exit.

### `inference benchmark python-package-speed`

**Usage**:

```console
$ inference benchmark python-package-speed [OPTIONS]
```

**Options**:

* `-m, --model_id TEXT`: Model ID in format project/version.  [required]
* `-d, --dataset_reference TEXT`: Name of predefined dataset (one of ['coco']) or path to directory with images  [default: coco]
* `-wi, --warm_up_inferences INTEGER`: Number of warm-up requests  [default: 10]
* `-bi, --benchmark_requests INTEGER`: Number of benchmark requests  [default: 1000]
* `-bs, --batch_size INTEGER`: Batch size of single request  [default: 1]
* `-a, --api-key TEXT`: Roboflow API key for your workspace. If not given - env variable `ROBOFLOW_API_KEY` will be used
* `-mc, --model_config TEXT`: Location of yaml file with model config
* `-o, --output_location TEXT`: Location where to save the result (path to file or directory)
* `--help`: Show this message and exit.

## `inference cloud`

Commands for running the inference in cloud with skypilot. 

Supported devices targets are x86 CPU and NVIDIA GPU VMs.

**Usage**:

```console
$ inference cloud [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `deploy`
* `start`
* `status`
* `stop`
* `undeploy`

### `inference cloud deploy`

**Usage**:

```console
$ inference cloud deploy [OPTIONS]
```

**Options**:

* `-p, --provider TEXT`: Cloud provider to deploy to. Currently aws or gcp.  [required]
* `-t, --compute-type TEXT`: Execution environment to deploy to: cpu or gpu.  [required]
* `-d, --dry-run`: Print out deployment plan without executing.
* `-c, --custom TEXT`: Path to config file to override default config.
* `-r, --roboflow-api-key TEXT`: Roboflow API key for your workspace.
* `-h, --help`: Print out help text.

### `inference cloud start`

**Usage**:

```console
$ inference cloud start [OPTIONS] CLUSTER_NAME
```

**Arguments**:

* `CLUSTER_NAME`: Name of cluster to start.  [required]

**Options**:

* `--help`: Show this message and exit.

### `inference cloud status`

**Usage**:

```console
$ inference cloud status [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

### `inference cloud stop`

**Usage**:

```console
$ inference cloud stop [OPTIONS] CLUSTER_NAME
```

**Arguments**:

* `CLUSTER_NAME`: Name of cluster to stop.  [required]

**Options**:

* `--help`: Show this message and exit.

### `inference cloud undeploy`

**Usage**:

```console
$ inference cloud undeploy [OPTIONS] CLUSTER_NAME
```

**Arguments**:

* `CLUSTER_NAME`: Name of cluster to undeploy.  [required]

**Options**:

* `--help`: Show this message and exit.

## `inference infer`

**Usage**:

```console
$ inference infer [OPTIONS]
```

**Options**:

* `-i, --input TEXT`: URL or local path of image / directory with images or video to run inference on.  [required]
* `-m, --model_id TEXT`: Model ID in format project/version.  [required]
* `-a, --api-key TEXT`: Roboflow API key for your workspace. If not given - env variable `ROBOFLOW_API_KEY` will be used
* `-h, --host TEXT`: Host to run inference on.  [default: http://localhost:9001]
* `-o, --output_location TEXT`: Location where to save the result (path to directory)
* `-D, --display / -d, --no-display`: Boolean flag to decide if visualisations should be displayed on the screen  [default: no-display]
* `-V, --visualise / -v, --no-visualise`: Boolean flag to decide if visualisations should be preserved  [default: visualise]
* `-c, --visualisation_config TEXT`: Location of yaml file with visualisation config
* `-mc, --model_config TEXT`: Location of yaml file with model config
* `--help`: Show this message and exit.

## `inference rf-cloud`

Commands for interacting with Roboflow Cloud

**Usage**:

```console
$ inference rf-cloud [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `batch-processing`: Commands for interacting with Roboflow...
* `data-staging`: Commands for interacting with Roboflow...

### `inference rf-cloud batch-processing`

Commands for interacting with Roboflow Batch Processing. THIS IS ALPHA PREVIEW OF THE FEATURE.

**Usage**:

```console
$ inference rf-cloud batch-processing [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `list-jobs`: List batch jobs in your workspace.
* `process-images-with-workflow`: Trigger batch job to process images with...
* `process-videos-with-workflow`: Trigger batch job to process videos with...
* `show-job-details`: Get job details.

#### `inference rf-cloud batch-processing list-jobs`

List batch jobs in your workspace.

**Usage**:

```console
$ inference rf-cloud batch-processing list-jobs [OPTIONS]
```

**Options**:

* `-a, --api-key TEXT`: Roboflow API key for your workspace. If not given - env variable `ROBOFLOW_API_KEY` will be used
* `-p, --max-pages INTEGER`: Number of pagination pages with batch jobs to display  [default: 1]
* `--debug-mode / --no-debug-mode`: Flag enabling errors stack traces to be displayed (helpful for debugging)  [default: no-debug-mode]
* `--help`: Show this message and exit.

#### `inference rf-cloud batch-processing process-images-with-workflow`

Trigger batch job to process images with Workflow

**Usage**:

```console
$ inference rf-cloud batch-processing process-images-with-workflow [OPTIONS]
```

**Options**:

* `-b, --batch-id TEXT`: Identifier of batch to be processed  [required]
* `-w, --workflow-id TEXT`: Identifier of the workflow  [required]
* `--workflow-params TEXT`: Path to JSON document with Workflow parameters - helpful when Workflow is parametrized and passing the parameters in CLI is not handy / impossible due to typing conversion issues.
* `--image-input-name TEXT`: Name of the Workflow input that defines placeholder for image to be processed
* `--save-image-outputs / --no-save-image-outputs`: Flag controlling persistence of Workflow outputs that are images  [default: no-save-image-outputs]
* `--image-outputs-to-save TEXT`: Use this option to filter out workflow image outputs you want to save
* `-p, --part-name TEXT`: Name of the batch part (relevant for multipart batches
* `-mt, --machine-type [cpu|gpu]`: Type of machine
* `-ms, --machine-size [xs|s|m|l|xl]`: Size of machine
* `--max-runtime-seconds INTEGER`: Max processing duration
* `--max-parallel-tasks INTEGER`: Max number of concurrent processing tasks
* `--aggregation-format [csv|jsonl]`: Format of results aggregation
* `-j, --job-id TEXT`: Identifier of job (if not given - will be generated)
* `-a, --api-key TEXT`: Roboflow API key for your workspace. If not given - env variable `ROBOFLOW_API_KEY` will be used
* `--debug-mode / --no-debug-mode`: Flag enabling errors stack traces to be displayed (helpful for debugging)  [default: no-debug-mode]
* `--notifications-url TEXT`: URL of the Webhook to be used for job state notifications.
* `--help`: Show this message and exit.

#### `inference rf-cloud batch-processing process-videos-with-workflow`

Trigger batch job to process videos with Workflow

**Usage**:

```console
$ inference rf-cloud batch-processing process-videos-with-workflow [OPTIONS]
```

**Options**:

* `-b, --batch-id TEXT`: Identifier of batch to be processed  [required]
* `-w, --workflow-id TEXT`: Identifier of the workflow  [required]
* `--workflow-params TEXT`: Path to JSON document with Workflow parameters - helpful when Workflow is parametrized and passing the parameters in CLI is not handy / impossible due to typing conversion issues.
* `--image-input-name TEXT`: Name of the Workflow input that defines placeholder for image to be processed
* `--save-image-outputs / --no-save-image-outputs`: Flag controlling persistence of Workflow outputs that are images  [default: no-save-image-outputs]
* `--image-outputs-to-save TEXT`: Use this option to filter out workflow image outputs you want to save
* `-p, --part-name TEXT`: Name of the batch part (relevant for multipart batches
* `-mt, --machine-type [cpu|gpu]`: Type of machine
* `-ms, --machine-size [xs|s|m|l|xl]`: Size of machine
* `--max-runtime-seconds INTEGER`: Max processing duration
* `--max-parallel-tasks INTEGER`: Max number of concurrent processing tasks
* `--aggregation-format [csv|jsonl]`: Format of results aggregation
* `--max-video-fps INTEGER`: Limit for FPS to process for video (subsampling predictions rate) - smaller FPS means faster processing and less accurate video analysis.
* `-j, --job-id TEXT`: Identifier of job (if not given - will be generated)
* `-a, --api-key TEXT`: Roboflow API key for your workspace. If not given - env variable `ROBOFLOW_API_KEY` will be used
* `--debug-mode / --no-debug-mode`: Flag enabling errors stack traces to be displayed (helpful for debugging)  [default: no-debug-mode]
* `--notifications-url TEXT`: URL of the Webhook to be used for job state notifications.
* `--help`: Show this message and exit.

#### `inference rf-cloud batch-processing show-job-details`

Get job details.

**Usage**:

```console
$ inference rf-cloud batch-processing show-job-details [OPTIONS]
```

**Options**:

* `-j, --job-id TEXT`: Identifier of job  [required]
* `-a, --api-key TEXT`: Roboflow API key for your workspace. If not given - env variable `ROBOFLOW_API_KEY` will be used
* `--debug-mode / --no-debug-mode`: Flag enabling errors stack traces to be displayed (helpful for debugging)  [default: no-debug-mode]
* `--help`: Show this message and exit.

### `inference rf-cloud data-staging`

Commands for interacting with Roboflow Data Staging. THIS IS ALPHA PREVIEW OF THE FEATURE.

**Usage**:

```console
$ inference rf-cloud data-staging [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `create-batch-of-images`: Create new batch with your images.
* `create-batch-of-videos`: Create new batch with your videos.
* `export-batch`: Export batch
* `list-batches`: List staging batches in your workspace.
* `show-batch-details`: Show batch details

#### `inference rf-cloud data-staging create-batch-of-images`

Create new batch with your images.

**Usage**:

```console
$ inference rf-cloud data-staging create-batch-of-images [OPTIONS]
```

**Options**:

* `-b, --batch-id TEXT`: Identifier of new batch (must be lower-cased letters with '-' and '_' allowed  [required]
* `-i, --images-dir TEXT`: Path to your images directory to upload  [required]
* `-bn, --batch-name TEXT`: Display name of your batch
* `-a, --api-key TEXT`: Roboflow API key for your workspace. If not given - env variable `ROBOFLOW_API_KEY` will be used
* `--debug-mode / --no-debug-mode`: Flag enabling errors stack traces to be displayed (helpful for debugging)  [default: no-debug-mode]
* `--help`: Show this message and exit.

#### `inference rf-cloud data-staging create-batch-of-videos`

Create new batch with your videos.

**Usage**:

```console
$ inference rf-cloud data-staging create-batch-of-videos [OPTIONS]
```

**Options**:

* `-b, --batch-id TEXT`: Identifier of new batch (must be lower-cased letters with '-' and '_' allowed  [required]
* `-v, --videos-dir TEXT`: Path to your videos directory to upload  [required]
* `-a, --api-key TEXT`: Roboflow API key for your workspace. If not given - env variable `ROBOFLOW_API_KEY` will be used
* `-bn, --batch-name TEXT`: Display name of your batch
* `--debug-mode / --no-debug-mode`: Flag enabling errors stack traces to be displayed (helpful for debugging)  [default: no-debug-mode]
* `--help`: Show this message and exit.

#### `inference rf-cloud data-staging export-batch`

Export batch

**Usage**:

```console
$ inference rf-cloud data-staging export-batch [OPTIONS]
```

**Options**:

* `-b, --batch-id TEXT`: Identifier of new batch (must be lower-cased letters with '-' and '_' allowed  [required]
* `-t, --target-dir TEXT`: Path to export directory  [required]
* `-a, --api-key TEXT`: Roboflow API key for your workspace. If not given - env variable `ROBOFLOW_API_KEY` will be used
* `--debug-mode / --no-debug-mode`: Flag enabling errors stack traces to be displayed (helpful for debugging)  [default: no-debug-mode]
* `--help`: Show this message and exit.

#### `inference rf-cloud data-staging list-batches`

List staging batches in your workspace.

**Usage**:

```console
$ inference rf-cloud data-staging list-batches [OPTIONS]
```

**Options**:

* `-a, --api-key TEXT`: Roboflow API key for your workspace. If not given - env variable `ROBOFLOW_API_KEY` will be used
* `-p, --pages INTEGER`: Number of pages to pull  [default: 1]
* `--page-size INTEGER`: Size of pagination page
* `--debug-mode / --no-debug-mode`: Flag enabling errors stack traces to be displayed (helpful for debugging)  [default: no-debug-mode]
* `--help`: Show this message and exit.

#### `inference rf-cloud data-staging show-batch-details`

Show batch details

**Usage**:

```console
$ inference rf-cloud data-staging show-batch-details [OPTIONS]
```

**Options**:

* `-b, --batch-id TEXT`: Identifier of new batch (must be lower-cased letters with '-' and '_' allowed  [required]
* `-a, --api-key TEXT`: Roboflow API key for your workspace. If not given - env variable `ROBOFLOW_API_KEY` will be used
* `--debug-mode / --no-debug-mode`: Flag enabling errors stack traces to be displayed (helpful for debugging)  [default: no-debug-mode]
* `--help`: Show this message and exit.

## `inference server`

Commands for running the inference server locally. 

Supported devices targets are x86 CPU, ARM64 CPU, and NVIDIA GPU.

**Usage**:

```console
$ inference server [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `start`
* `status`
* `stop`

### `inference server start`

**Usage**:

```console
$ inference server start [OPTIONS]
```

**Options**:

* `-p, --port INTEGER`: Port to run the inference server on (default is 9001).  [default: 9001]
* `-rfe, --rf-env TEXT`: Roboflow environment to run the inference server with (default is roboflow-platform).  [default: roboflow-platform]
* `-e, --env-file TEXT`: Path to file with env variables (in each line KEY=VALUE). Optional. If given - values will be overriden by any explicit parameter of this command.
* `-d, --dev`: Run inference server in development mode (default is False).
* `-k, --roboflow-api-key TEXT`: Roboflow API key (default is None).
* `--tunnel`: Start a tunnel to expose inference to external requests on a TLS-enabled https://<subdomain>.roboflow.run endpoint
* `--image TEXT`: Point specific docker image you would like to run with command (useful for development of custom builds of inference server)
* `--use-local-images / --not-use-local-images`: Flag to allow using local images (if set False image is always attempted to be pulled)  [default: not-use-local-images]
* `--metrics-enabled / --metrics-disabled`: Flag controlling if metrics are enabled (default is True)  [default: metrics-enabled]
* `--help`: Show this message and exit.

### `inference server status`

**Usage**:

```console
$ inference server status [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

### `inference server stop`

**Usage**:

```console
$ inference server stop [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

## `inference workflows`

Commands for interacting with Roboflow Workflows

**Usage**:

```console
$ inference workflows [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `process-image`: Process single image with Workflows...
* `process-images-directory`: Process whole images directory with...
* `process-video`: Process video file with your Workflow...

### `inference workflows process-image`

Process single image with Workflows (inference Package may be needed dependent on mode)

**Usage**:

```console
$ inference workflows process-image [OPTIONS]
```

**Options**:

* `-i, --image_path TEXT`: Path to image to be processed  [required]
* `-o, --output_dir TEXT`: Path to output directory  [required]
* `-pt, --processing_target [api|inference_package]`: Defines where the actual processing will be done, either in inference Python package running locally, or behind the API (which ensures greater throughput).  [default: api]
* `-ws, --workflow_spec TEXT`: Path to JSON file with Workflow definition (mutually exclusive with `workspace_name` and `workflow_id`)
* `-wn, --workspace_name TEXT`: Name of Roboflow workspace the that Workflow belongs to (mutually exclusive with `workflow_specification_path`)
* `-wid, --workflow_id TEXT`: Identifier of a Workflow on Roboflow platform (mutually exclusive with `workflow_specification_path`)
* `--workflow_params TEXT`: Path to JSON document with Workflow parameters - helpful when Workflow is parametrized and passing the parameters in CLI is not handy / impossible due to typing conversion issues.
* `--image_input_name TEXT`: Name of the Workflow input that defines placeholder for image to be processed  [default: image]
* `-a, --api-key TEXT`: Roboflow API key for your workspace. If not given - env variable `ROBOFLOW_API_KEY` will be used
* `--api_url TEXT`: URL of the API that will be used for processing, when API processing target pointed.  [default: https://detect.roboflow.com]
* `--allow_override / --no_override`: Flag to decide if content of output directory can be overridden.  [default: no_override]
* `--save_image_outputs / --no_save_image_outputs`: Flag controlling persistence of Workflow outputs that are images  [default: save_image_outputs]
* `--force_reprocessing / --no_reprocessing`: Flag to enforce re-processing of specific images. Images are identified by file name.  [default: no_reprocessing]
* `--debug_mode / --no_debug_mode`: Flag enabling errors stack traces to be displayed (helpful for debugging)  [default: no_debug_mode]
* `--help`: Show this message and exit.

### `inference workflows process-images-directory`

Process whole images directory with Workflows (inference Package may be needed dependent on mode)

**Usage**:

```console
$ inference workflows process-images-directory [OPTIONS]
```

**Options**:

* `-i, --input_directory TEXT`: Path to directory with images  [required]
* `-o, --output_dir TEXT`: Path to output directory  [required]
* `-pt, --processing_target [api|inference_package]`: Defines where the actual processing will be done, either in inference Python package running locally, or behind the API (which ensures greater throughput).  [default: api]
* `-ws, --workflow_spec TEXT`: Path to JSON file with Workflow definition (mutually exclusive with `workspace_name` and `workflow_id`)
* `-wn, --workspace_name TEXT`: Name of Roboflow workspace the that Workflow belongs to (mutually exclusive with `workflow_specification_path`)
* `-wid, --workflow_id TEXT`: Identifier of a Workflow on Roboflow platform (mutually exclusive with `workflow_specification_path`)
* `--workflow_params TEXT`: Path to JSON document with Workflow parameters - helpful when Workflow is parametrized and passing the parameters in CLI is not handy / impossible due to typing conversion issues.
* `--image_input_name TEXT`: Name of the Workflow input that defines placeholder for image to be processed  [default: image]
* `-a, --api-key TEXT`: Roboflow API key for your workspace. If not given - env variable `ROBOFLOW_API_KEY` will be used
* `--api_url TEXT`: URL of the API that will be used for processing, when API processing target pointed.  [default: https://detect.roboflow.com]
* `--allow_override / --no_override`: Flag to decide if content of output directory can be overridden.  [default: no_override]
* `--save_image_outputs / --no_save_image_outputs`: Flag controlling persistence of Workflow outputs that are images  [default: save_image_outputs]
* `--force_reprocessing / --no_reprocessing`: Flag to enforce re-processing of specific images. Images are identified by file name.  [default: no_reprocessing]
* `--aggregate / --no_aggregate`: Flag to decide if processing results for a directory should be aggregated to a single file at the end of processing.  [default: aggregate]
* `-af, --aggregation_format [jsonl|csv]`: Defines the format of aggregated results - either CSV of JSONL  [default: csv]
* `--threads INTEGER`: Defines number of threads that will be used to send requests when processing target is API. Default for Roboflow Hosted API is 32, and for on-prem deployments: 1.
* `--debug_mode / --no_debug_mode`: Flag enabling errors stack traces to be displayed (helpful for debugging)  [default: no_debug_mode]
* `--max-failures INTEGER`: Maximum number of Workflow executions for directory images which will be tolerated before give up. If not set - unlimited.
* `--help`: Show this message and exit.

### `inference workflows process-video`

Process video file with your Workflow locally (inference Python package required)

**Usage**:

```console
$ inference workflows process-video [OPTIONS]
```

**Options**:

* `-v, --video_path TEXT`: Path to video to be processed  [required]
* `-o, --output_dir TEXT`: Path to output directory  [required]
* `-ft, --output_file_type [jsonl|csv]`: Type of the output file  [default: csv]
* `-ws, --workflow_spec TEXT`: Path to JSON file with Workflow definition (mutually exclusive with `workspace_name` and `workflow_id`)
* `-wn, --workspace_name TEXT`: Name of Roboflow workspace the that Workflow belongs to (mutually exclusive with `workflow_specification_path`)
* `-wid, --workflow_id TEXT`: Identifier of a Workflow on Roboflow platform (mutually exclusive with `workflow_specification_path`)
* `--workflow_params TEXT`: Path to JSON document with Workflow parameters - helpful when Workflow is parametrized and passing the parameters in CLI is not handy / impossible due to typing conversion issues.
* `--image_input_name TEXT`: Name of the Workflow input that defines placeholder for image to be processed  [default: image]
* `--max_fps FLOAT`: Use the parameter to limit video FPS (additional frames will be skipped in processing).
* `--save_out_video / --no_save_out_video`: Flag deciding if image outputs of the workflow should be saved as video file  [default: save_out_video]
* `-a, --api-key TEXT`: Roboflow API key for your workspace. If not given - env variable `ROBOFLOW_API_KEY` will be used
* `--allow_override / --no_override`: Flag to decide if content of output directory can be overridden.  [default: no_override]
* `--debug_mode / --no_debug_mode`: Flag enabling errors stack traces to be displayed (helpful for debugging)  [default: no_debug_mode]
* `--help`: Show this message and exit.

