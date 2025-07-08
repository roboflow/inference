# Benchmarking `inference`

`inference benchmark` offers you an easy way to check the performance of `inference` in your setup. The command 
is capable of benchmarking `inference` server, `inference` Python package, and InferencePipeline for both models and workflows on video stream processing.

!!! Tip "Discovering command capabilities"

    To check detail of the command, run:
    
    ```bash
    inference benchmark --help
    ```

    Additionally, help guide is also available for each sub-command:

    ```bash
    inference benchmark api-speed --help
    ```

## Benchmarking `inference` Python package

!!! Important "`inference` needs to be installed"

    Running this command, make sure `inference` package is installed.

    ```bash
    pip install inference
    ```


Basic benchmark can be run using the following command: 

```bash
inference benchmark python-package-speed \
  -m {your_model_id} \
  -d {pre-configured dataset name or path to directory with images} \
  -o {output_directory}  
```
Command runs specified number of inferences using pointed model and saves statistics (including benchmark 
parameter, throughput, latency, errors and platform details) in pointed directory.


##  Benchmarking `inference` server

!!! note

    Before running API benchmark of your local `inference` server - make sure the server is up and running:
    
    ```bash
    inference server start
    ```

### Model Benchmarking

Basic model benchmark can be run using the following command: 

```bash
inference benchmark api-speed \
  -m {your_model_id} \
  -d {pre-configured dataset name or path to directory with images} \
  -o {output_directory}  
```

### Workflow Benchmarking

You can also benchmark workflows through the API:

```bash
# Remote workflow
inference benchmark api-speed \
  --workflow-id {workflow_id} \
  --workspace-name {workspace_name} \
  -d {dataset} \
  -o {output_directory}

# Local workflow specification
inference benchmark api-speed \
  --workflow-specification '{...}' \
  --workflow-parameters '{"param1": "value1"}' \
  -d {dataset} \
  -o {output_directory}
```

Command runs specified number of inferences using pointed model or workflow and saves statistics (including benchmark 
parameter, throughput, latency, errors and platform details) in pointed directory.

This benchmark has more configuration options to support different ways HTTP API profiling. In default mode,
single client will be spawned, and it will send one request after another sequentially. This may be suboptimal
in specific cases, so one may specify number of concurrent clients using `-c {number_of_clients}` option.
Each client will send next request once previous is handled. This option will also not cover all scenarios
of tests. For instance one may want to send `x` requests each second (which is closer to the scenario of
production environment where multiple clients are sending requests concurrently). In this scenario, `--rps {value}` 
option can be used (and `-c` will be ignored). Value provided in `--rps` option specifies how many requests 
are to be spawned **each second** without waiting for previous requests to be handled. In I/O intensive benchmark 
scenarios - we suggest running command from multiple separate processes and possibly multiple hosts.

## Benchmarking InferencePipeline

!!! note "InferencePipeline benchmarking"

    InferencePipeline is designed for continuous video stream processing and provides optimized 
    performance for real-time computer vision applications. It supports both model inference and 
    workflow execution benchmarking.

### Model Pipeline Benchmarking

Basic model pipeline benchmark can be run using the following command:

```bash
inference benchmark pipeline-speed \
  -m {your_model_id} \
  -v {video_source} \
  -o {output_directory}  
```

### Workflow Pipeline Benchmarking

InferencePipeline now supports benchmarking workflows for video processing:

```bash
# Remote workflow
inference benchmark pipeline-speed \
  --workflow-id {workflow_id} \
  --workspace-name {workspace_name} \
  -v {video_source} \
  -o {output_directory}

# Local workflow specification
inference benchmark pipeline-speed \
  --workflow-specification '{...}' \
  --workflow-parameters '{"param1": "value1"}' \
  -v {video_source} \
  -o {output_directory}
```

!!! important "Model vs Workflow Parameters"
    
    You must specify either `--model_id` OR workflow parameters (`--workflow-id` with `--workspace-name`, 
    or `--workflow-specification`), but not both. The benchmark will automatically use the appropriate
    InferencePipeline initialization method.

The pipeline benchmark is specifically designed to test video stream processing performance with features like:
- Multiple concurrent pipelines processing
- Various video sources (files, streams, cameras)
- Configurable FPS limits and processing modes
- Real-time performance metrics
- Support for both models and workflows

### Advanced Pipeline Benchmarking

You can benchmark multiple pipelines concurrently for both models and workflows:

```bash
# Model: Run 4 concurrent pipelines
inference benchmark pipeline-speed \
  -m yolov8n-640 \
  -v rtsp://camera.local/stream \
  --pipelines 4 \
  --duration 120 \
  --max_fps 30

# Workflow: Run 2 concurrent pipelines with parameters
inference benchmark pipeline-speed \
  --workflow-id object-tracking-workflow \
  --workspace-name my-workspace \
  --workflow-parameters '{"confidence": 0.7, "iou_threshold": 0.5}' \
  -v video.mp4 \
  --pipelines 2 \
  --duration 300
```

Pipeline-specific options:
- `--model_id/-m`: Model ID for model benchmarking (mutually exclusive with workflow options)
- `--workflow-id/-wid`: Workflow ID for workflow benchmarking
- `--workspace-name/-wn`: Workspace name (required with --workflow-id)
- `--workflow-specification/-ws`: Local workflow specification JSON
- `--workflow-parameters/-wp`: Additional workflow parameters JSON
- `--pipelines/-p`: Number of concurrent pipelines to run (default: 1)
- `--duration/-t`: Benchmark duration in seconds (default: 60)
- `--max_fps/-fps`: Maximum FPS limit for each pipeline
- `--video_reference/-v`: Video source (file path, RTSP URL, or camera index)

The benchmark will report:
- Benchmark type (Model or Workflow)
- Per-pipeline metrics (FPS, frames processed, errors)
- Aggregate performance across all pipelines
- Resource utilization patterns
- Frame processing latencies
