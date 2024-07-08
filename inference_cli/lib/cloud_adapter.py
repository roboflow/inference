import random
import string
import tempfile

YAML_DEFS = {
    "gcp_cpu": """
name: roboflow-gcp-inference-cpu
num_nodes: 1
resources:
  cloud: gcp
  instance_type: n2-standard-2
  use_spot: false
  disk_size: 100
  disk_tier: medium
  ports: 9001
  image_id: projects/ubuntu-os-cloud/global/images/ubuntu-2204-jammy-v20240112
setup: |
  sudo apt-get update
  sudo apt-get install -y uidmap
  curl -fsSL https://get.docker.com/rootless -o get-docker.sh
  sh get-docker.sh
  sudo loginctl enable-linger gcpuser
  systemctl --user start docker.service
  docker pull roboflow/roboflow-inference-server-cpu
  export PATH=/home/gcpuser/bin:$PATH
  echo
run: |
  export PATH=/home/gcpuser/bin:$PATH; docker run -d -p 9001:9001 PLACEHOLDER roboflow/roboflow-inference-server-cpu
""",
    "gcp_gpu": """
name: roboflow-gcp-inference-gpu
num_nodes: 1
resources:
  cloud: gcp
  accelerators: T4:1
  instance_type: n1-standard-4
  use_spot: false
  disk_size: 100
  disk_tier: medium
  ports: 9001
  image_id: projects/ml-images/global/images/c0-deeplearning-common-gpu-v20231209-debian-11
setup: |
  docker pull roboflow/roboflow-inference-server-gpu
run: |
  docker run -d --gpus all -p 9001:9001 PLACEHOLDER roboflow/roboflow-inference-server-gpu
""",
    "aws_cpu": """
name: roboflow-aws-inference-cpu
num_nodes: 1
resources:
  cloud: aws
  memory: 4+
  cpus: 2+
  use_spot: false
  disk_size: 100
  disk_tier: medium
  ports: 9001
setup: |
  sudo apt-get update
  docker pull roboflow/roboflow-inference-server-cpu
  export PATH=/home/gcpuser/bin:$PATH
  echo
run: |
  export PATH=/home/gcpuser/bin:$PATH; docker run -d -p 9001:9001 PLACEHOLDER roboflow/roboflow-inference-server-cpu
""",
    "aws_gpu": """
name: roboflow-aws-inference-gpu
num_nodes: 1
resources:
  cloud: aws
  accelerators: A10G:1
  memory: 4+
  cpus: 2+
  use_spot: false
  disk_size: 100
  disk_tier: medium
  ports: 9001
  image_id: skypilot:gpu-ubuntu-2004
setup: |
  docker pull roboflow/roboflow-inference-server-gpu
run: |
  docker run -d --gpus all -p 9001:9001 PLACEHOLDER roboflow/roboflow-inference-server-gpu
""",
}


def _random_char(y):
    return "".join(random.choice(string.ascii_lowercase) for x in range(y))


def cloud_status():
    import sky

    print("Getting status from skypilot...")
    print(sky.status())


def cloud_stop(cluster_name):
    import sky

    print(f"Stopping skypilot deployment {cluster_name}...")
    print(sky.stop(cluster_name))


def cloud_start(cluster_name):
    import sky

    print(f"Starting skypilot deployment {cluster_name}")
    print(sky.start(cluster_name))


def cloud_undeploy(cluster_name):
    import sky

    print(
        f"Undeploying Roboflow Inference and deleting {cluster_name}, this may take a few minutes."
    )
    sky.down(cluster_name)
    print(f"Undeployed Roboflow Inference complete: deleted {cluster_name}")


def cloud_deploy(provider, compute_type, dry_run, custom, help, roboflow_api_key):
    if help:
        print(
            """
              Deploy Roboflow Inference to a cloud provider.
              If your chosen cloud provider is configured on your terminal, inference 
              deploy will automatically use your default credentials. If you have not 
              configured your cloud provider, you can do so by following the instructions 
              for installing and configuring the cloud provider's CLI tools.

              Roboflow inference deploy  will create a new virtual machine 
              and deploy the Roboflow Inference container to it.

                Usage examples:
                # Deploy to GCP with CPU
                inference cloud deploy --provider gcp --compute-type cpu
                # Deploy to AWS with GPU
                inference cloud deploy --provider aws --compute-type gpu
                # Stop a deployment (without deleting it, you only pay for storage)
                inference cloud stop <deployment-name>
                # Start a deployment
                inference cloud start <deployment-name>
                # Get status of all deployments
                inference cloud status
                # Undeploy a deployment (delete it)
                inference cloud undeploy <deployment-name>
                # SSH into a deployment
                inference cloud ssh <deployment-name>
            
              Roboflow inference deploy uses sky (https://github.com/skypilot-org/skypilot) 
              to launch a new virtual machine in the cloud provider you specify,
              and configures it to run the Roboflow Inference container. If you want
              to customize the deployment with a custom sky configuration, 
              you can pass a custom config file with the --custom flag, like so:

                inference cloud deploy --custom ./custom.yaml
              
              To see a full list of sky configuration options, visit
              https://skypilot.readthedocs.io/en/latest/reference/yaml-spec.html

              Roboflow inference cli installs the sky CLI tool for you, so you can
              also use the sky cli for more advanced deployment options.

              Roboflow inference deploy currently supports AWS and GCP, please 
              open an issue at https://github.com/roboflow/inference/issues if 
              you would like to see other cloud providers supported.

        """
        )
        return

    if custom is None:
        try:
            yaml_string = YAML_DEFS[f"{provider}_{compute_type}"]
        except KeyError:
            print(f"Provider {provider} and compute type {compute_type} not supported.")
            print("Please open an issue at https://github.com/roboflow/inference")
            return
    else:
        yaml_string = open(custom, "r").read()

    if dry_run == True:
        print(yaml_string)
        return

    placeholder_string = (
        f" --env ROBOFLOW_API_KEY={roboflow_api_key} "
        if roboflow_api_key is not None
        else ""
    )

    # For later - when notebook becomes secure
    # if notebook:
    #     yaml_string.replace("9001", "[9001, 9002]")
    #     placeholder_string += " -e NOTEBOOK_ENABLED=true"

    yaml_string = yaml_string.replace("PLACEHOLDER", placeholder_string)

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        file_path = f.name
        f.write(yaml_string)
        f.close()
        task = sky.Task.from_yaml(file_path)

    cluster_name = f"roboflow-inference-{provider}-{compute_type}-{_random_char(5)}"

    print(
        f"Deploying Roboflow Inference to {provider} on {compute_type} using sky (https://github.com/skypilot-org/skypilot) "
    )
    print("Please be patient, this process can take up to 20 minutes.")
    sky.launch(task, cluster_name=cluster_name)

    print(
        f"Deployed Roboflow Inference to {provider} on {compute_type}, deployment name is {cluster_name} "
    )
    cluster_ip = sky.status(cluster_name)[0]["handle"].head_ip

    print(f"To get a list of your deployments: inference status")
    print(f"To delete your deployment: inference undeploy {cluster_name}")
    print(f"To ssh into the deployed server: ssh {cluster_name}")
    print(f"The Roboflow Inference Server is running at http://{cluster_ip}:9001")
