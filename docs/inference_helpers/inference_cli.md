<div align="center">
    <img
    width="100%"
    src="https://github.com/roboflow/inference/assets/6319317/9230d986-183d-4ab0-922b-4b497f16d937"
    />
</div>

## Roboflow Inference CLI

Roboflow Inference CLI offers a lightweight interface for running the Roboflow inference server locally or the Roboflow Hosted API.

To create custom inference server Docker images, go to the parent package, <a href="https://pypi.org/project/inference/" target="_blank">Roboflow Inference</a>.

<a href="https://roboflow.com" target="_blank">Roboflow</a> has everything you need to deploy a computer vision model to a range of devices and environments. Inference supports object detection, classification, and instance segmentation models, and running foundation models (CLIP and SAM).

## Examples

### inference server start

Starts a local inference server. It optionally takes a port number (default is 9001) and will only start the docker container if there is not already a container running on that port.

If you would rather run your server on a virtual machine in Google cloud or Amazon cloud, skip to the section titled "Deploy Inference on Cloud" below.

Before you begin, ensure that you have Docker installed on your machine. Docker provides a containerized environment,
allowing the Roboflow Inference Server to run in a consistent and isolated manner, regardless of the host system. If
you haven't installed Docker yet, you can get it from <a href="https://www.docker.com/get-started" target="_blank">Docker's official website</a>.

The CLI will automatically detect the device you are running on and pull the appropriate Docker image.

```bash
inference server start --port 9001 [-e {optional_path_to_file_with_env_variables}]
```

Parameter `--env-file` (or `-e`) is the optional path for .env file that will be loaded into inference server
in case that values of internal parameters needs to be adjusted. Any value passed explicitly as command parameter
is considered as more important and will shadow the value defined in `.env` file under the same target variable name.

#### Development Mode

Use the `--dev` flag to start the Inference Server in development mode. Development mode enables the Inference Server's built in notebook environment for easy testing and development.

### inference server status

Checks the status of the local inference server.

```bash
inference server status
```

### inference server stop

Stops the inference server.

```bash
inference server stop
```

## Deploy Inference on a Cloud VM

You can deploy Roboflow inference containers to virtual machines in the cloud. These VMs are configured to run CPU or GPU-based inference servers under the hood, so you don't have to deal with OS/GPU drivers/docker installations, etc! The inference cli currently supports deploying the Roboflow inference container images into a virtual machine running on Google (GCP) or Amazon cloud (AWS).

The Roboflow inference cli assumes the corresponding cloud cli is configured for the project you want to deploy the virtual machine into. Read instructions for setting up [Google/GCP - gcloud cli](https://cloud.google.com/sdk/docs/install) or the [Amazon/AWS aws cli](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html).

Roboflow inference cloud deploy is powered by the popular [Skypilot project](https://github.com/skypilot-org/skypilot).

### Cloud Deploy Examples

We illustrate inference cloud deploy with some examples, below.

*Deploy GPU or CPU inference to AWS or GCP*

```
# Deploy the roboflow inference GPU container into a GPU-enabled VM in AWS

inference cloud deploy --provider aws --compute-type gpu
```

```
# Deploy the roboflow inference CPU container into a CPU-only VM in GCP

inference cloud deploy --provider gcp --compute-type cpu

```

Note the "cluster name" printed after the deployment completes. This handle is used in many subsequent commands.
The deploy command also prints helpful debug and cost information about your VM.

Deploying inference into a cloud VM will also print out an endpoint of the form "http://1.2.3.4:9001"; you can now run inferences against this endpoint.

Note that the port 9001 is automatically opened - check with your security admin if this is acceptable for your cloud/project.

*List status on cloud deployments*

```
# List deployed cloud VMs

inference cloud status
```

*Stop/Start VM cloud deployments*

```
# Stop the VM, you only pay for disk storage while the VM is stopped
inference cloud stop <deployment_handle>

```

```
# Re-start the VM deployment
inference cloud start <deployment_handle>
```

*Undeploy (delete) the cloud deployment*
```
# !!This will delete the VM and the service

inference cloud undeploy <deployment_handle>
```

*SSH into a running deployment*
```
# The SSH key is automatically added to your .ssh/config, you dont need to configure this manually!
ssh <deployment_handle>
```



### Cloud Deploy Customization

Roboflow inference cloud deploy will create VMs based on internally tested templates.

For advanced usecases and to customize the template, you can use your [sky yaml](https://skypilot.readthedocs.io/en/latest/reference/yaml-spec.html) template on the command-line, like so

```
inference cloud deploy --custom /path/to/sky-template.yaml

```

If you want you can download the standard template stored in the roboflow cli and the modify it for your needs, this command will do that.

```
# This command will print out the standard gcp/cpu sky template.
inference cloud deploy --dry-run --provider gcp --compute-type cpu
```

Then you can deploy a custom template based off your changes.

As an aside, you can also use the [sky cli](https://skypilot.readthedocs.io/en/latest/reference/cli.html) to control your deployment(=s) and access some more advanced functionality.



Roboflow inference deploy currently supports AWS and GCP, please open an issue at https://github.com/roboflow/inference/issues if you would like to see other cloud providers supported.


### inference infer

Runs inference on a single image. It takes a path to an image, a Roboflow project name, model version, and API key, and will return a JSON object with the model's predictions. You can also specify a host to run inference on our hosted inference server.

#### Local image

```bash
inference infer ./image.jpg --project-id my-project --model-version 1 --api-key my-api-key
```

#### Hosted image

```bash
inference infer https://[YOUR_HOSTED_IMAGE_URL] --project-id my-project --model-version 1 --api-key my-api-key
```

###

#### Hosted API inference

```bash
inference infer ./image.jpg --project-id my-project --model-version 1 --api-key my-api-key --host https://detect.roboflow.com
```

## Supported Devices

Roboflow Inference CLI currently supports the following device targets:

- x86 CPU
- ARM64 CPU
- NVIDIA GPU

For Jetson specific inference server images, check out the <a href="https://pypi.org/project/inference/" target="_blank">Roboflow Inference</a> package, or pull the images directly following instructions in the official [Roboflow Inference documentation](/quickstart/docker/#pull-from-docker-hub).
