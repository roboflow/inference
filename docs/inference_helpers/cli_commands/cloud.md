# Deploying `inference` to Cloud

You can deploy Roboflow Inference containers to virtual machines in the cloud. These VMs are configured to run CPU or 
GPU-based Inference servers under the hood, so you don't have to deal with OS/GPU drivers/docker installations, etc! 
The Inference cli currently supports deploying the Roboflow Inference container images into a virtual machine running 
on Google (GCP) or Amazon cloud (AWS).

The Roboflow Inference CLI assumes the corresponding cloud CLI is configured for the project you want to deploy the 
virtual machine into. Read instructions for setting up [Google/GCP - gcloud cli](https://cloud.google.com/sdk/docs/install) or the [Amazon/AWS aws cli](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html).

Roboflow Inference cloud deploy is powered by the popular [Skypilot project](https://github.com/skypilot-org/skypilot).

!!! Important "Make sure `cloud-deploy` extras is installed"

    To run commands presented below, you need to have `cloud-deploy` extras installed:

    ```bash
    pip install "inference-cli[cloud-deploy]"
    ```

!!! Tip "Discovering command capabilities"

    To check detail of the command, run:
    
    ```bash
    inference cloud --help
    ```

    Additionally, help guide is also available for each sub-command:

    ```bash
    inference cloud deploy --help
    ```

## `inference cloud deploy`

We illustrate Inference cloud deploy with some examples, below.

*Deploy GPU or CPU inference to AWS or GCP*

```bash
# Deploy the roboflow Inference GPU container into a GPU-enabled VM in AWS

inference cloud deploy --provider aws --compute-type gpu
```

```bash
# Deploy the roboflow Inference CPU container into a CPU-only VM in GCP

inference cloud deploy --provider gcp --compute-type cpu
```

Note the "cluster name" printed after the deployment completes. This handle is used in many subsequent commands.
The deploy command also prints helpful debug and cost information about your VM.

Deploying Inference into a cloud VM will also print out an endpoint of the form "http://1.2.3.4:9001"; you can now run inferences against this endpoint.

Note that the port 9001 is automatically opened - check with your security admin if this is acceptable for your cloud/project.

## `inference cloud status`

To check the status of your deployment, run:

```bash
inference cloud status
```

## Stop and start deployments

You can start and stop your deployment using:

```bash
inference cloud start <deployment_handle>
```

and

```bash
# Stop the VM, you only pay for disk storage while the VM is stopped
inference cloud stop <deployment_handle>

```

## `inference cloud undeploy`

To delete (undeploy) your deployment, run:

```bash
inference cloud undeploy <deployment_handle>
```

## SSH into the cloud deployment

You can SSH into your cloud deployment with the following command:
```bash
ssh <deployment_handle>
```

The required SSH key is automatically added to your `~/.ssh/config`, you don't need to configure this manually.


## Cloud Deploy Customization

Roboflow Inference cloud deploy will create VMs based on internally tested templates.

For advanced usecases and to customize the template, you can use your [sky yaml](https://skypilot.readthedocs.io/en/latest/reference/yaml-spec.html) template on the command-line, like so:

```bash
inference cloud deploy --custom /path/to/sky-template.yaml
```

If you want you can download the standard template stored in the roboflow cli and the modify it for your needs, this command will do that.

```bash
# This command will print out the standard gcp/cpu sky template.
inference cloud deploy --dry-run --provider gcp --compute-type cpu
```

Then you can deploy a custom template based off your changes.

As an aside, you can also use the [sky cli](https://skypilot.readthedocs.io/en/latest/reference/cli.html) to control your deployment(s) and access some more advanced functionality.

Roboflow Inference deploy currently supports AWS and GCP, please open an issue on the [Inference GitHub repository](https://github.com/roboflow/inference/issues) if you would like to see other cloud providers supported.
