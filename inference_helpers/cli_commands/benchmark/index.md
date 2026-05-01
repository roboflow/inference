# Benchmarking `inference`

`inference benchmark` offers you an easy way to check the performance of `inference` in your setup. The command 
is capable of benchmarking both `inference` server and `inference` Python package.

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
Basic benchmark can be run using the following command: 

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
