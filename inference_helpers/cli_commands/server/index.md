# Controlling `inference` server

`inference server` command provides a control layer around HTTP server exposing `inference`.

!!! Tip "Discovering command capabilities"

    To check detail of the command, run:
    
    ```bash
    inference server --help
    ```

    Additionally, help guide is also available for each sub-command:

    ```bash
    inference server start --help
    ```

## `inference server start`

Starts a local Inference server. It optionally takes a port number (default is 9001) and will only start the docker container if there is not already a container running on that port.

If you would rather run your server on a virtual machine in Google cloud or Amazon cloud, skip to the section titled "Deploy Inference on Cloud" below.

Before you begin, ensure that you have Docker installed on your machine. Docker provides a containerized environment,
allowing the Roboflow Inference Server to run in a consistent and isolated manner, regardless of the host system. If
you haven't installed Docker yet, you can get it from <a href="https://www.docker.com/get-started" target="_blank">Docker's official website</a>.

The CLI will automatically detect the device you are running on and pull the appropriate Docker image.

```bash
inference server start --port 9001 [-e {optional_path_to_file_with_env_variables}]
```

Parameter `--env-file` (or `-e`) is the optional path for .env file that will be loaded into your Inference server
in case that values of internal parameters needs to be adjusted. Any value passed explicitly as command parameter
is considered as more important and will shadow the value defined in `.env` file under the same target variable name.


### Development Mode

Use the `--dev` flag to start the Inference Server in development mode. Development mode enables the Inference Server's built in notebook environment for easy testing and development.

### Tunnel

Use the `--tunnel` flag to start the Inference Server with a tunnel to expose inference to external requests on a TLS-enabled endpoint.

The random generated address will be on server start output:

```
Tunnel to local inference running on https://somethingrandom-ip-192-168-0-1.roboflow.run
```

## inference server status

Checks the status of the local inference server.

```bash
inference server status
```

## inference server stop

Stops the inference server.

```bash
inference server stop
```