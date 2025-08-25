# How to Benchmark Inference

You can benchmark Inference on your machine using the `inference benchmark` command.

## Benchmark a Workflow (Recommended)

You can benchmark a Workflow using the following code:

```
python -m inference_cli.main benchmark api-speed -wid your-workflow-id -wn your-workspace-name -d path/to/dataset -rps 5 -br 500 -h http://localhost:9001 --yes --output_location test_results.json --max_error_rate 5.0
```

Above, replace:

- `your-workflow-id` with your Roboflow Workflow ID.
- `your-workspace-name` with your Roboflow Workspace Name.
- `path/to/dataset` with the path to a folder of images that you want to use as inputs to your Workflow.

If you want to benchmark your Workflow on the [Roboflow Serverless API](https://docs.roboflow.com/deploy/serverless-hosted-api-v2) or [Dedicated Deployment](https://docs.roboflow.com/deploy/dedicated-deployments), replace the `-h` URL with the URL to `https://serverless.roboflow.com` or your Dedicated Deployment URL.

Results will be printed to your terminal and saved to a file called `test_results.json`.

## Benchmark the `inference` Python SDK (Advanced) 

You can benchmark the raw Python code that is used to run a model. This is ideal if you are using a model outside of a Workflow.

You can benchmark a model using the following code:

```bash
inference benchmark python-package-speed \
  -m {your_model_id} \
  -d {pre-configured dataset name or path to directory with images} \
  -o {output_directory}  
```
Command runs specified number of inferences using pointed model and saves statistics (including benchmark 
parameter, throughput, latency, errors and platform details) in pointed directory.

##  Benchmark a Model with the Inference `inference` Server (Advanced)

You can benchmark the HTTP endpoint associated with a model running on Inference. This is ideal if you expect to make direct HTTP requests to the model endpoints that Inference makes available.

You can benchmark a model HTTP endpoint using the following code:

```bash
inference benchmark api-speed \
  -m {your_model_id} \
  -d {pre-configured dataset name or path to directory with images} \
  -o {output_directory}  
```

Command runs specified number of inferences using pointed model and saves statistics (including benchmark 
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
