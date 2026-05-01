# Deploy in Your Own Cloud

You can run Roboflow Inference on major cloud platforms like AWS, Azure, or GCP.

Deploying in your own cloud is ideal if you want the scalability and flexibility of the cloud but have
technical or organizational constraints around where you can send your data.

Roboflow Inference has an integration with [SkyPilot](https://github.com/skypilot-org/skypilot) that makes
deploying a cloud instance to run Inference as quick as running one command after you have authenticated
with your cloud provider.

Read our deployment guides for more information:

- [Set up Inference on AWS](/install/cloud/aws.md)
- [Set up Inference on Azure](/install/cloud/azure.md)
- [Set up Inference on GCP](/install/cloud/gcp.md)

## Manual Setup

You can also use the [Linux setup guide](../../install/linux.md)
to manually configure a Docker container to run on your cloud VM.

[A Helm Chart](https://github.com/roboflow/inference/tree/main/inference/enterprise/helm-chart)
is available for enterprise cloud deployments via Kubernetes.
