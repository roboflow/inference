# Roboflow Inference Kubernetes Chart

This directory contains a Helm chart to deploy the [Roboflow Inference Server](https://inference.roboflow.com/) into a Kubernetes Cluster. The Helm Chart can be used for both [cpu](https://hub.docker.com/r/roboflow/roboflow-inference-server-cpu) and [gpu](https://hub.docker.com/r/roboflow/roboflow-inference-server-gpu) images published by Roboflow.

## Deployment Steps

The steps to deploy the helm chart into a Kubernetes cluster are listed below. A [minikube](https://minikube.sigs.k8s.io/docs/start/) Kubernetes cluster is used as an example.

### Infrastructure and Tooling Setup

1. Install [kubectl](https://kubernetes.io/docs/tasks/tools/#kubectl) and [helm](https://helm.sh/docs/intro/install/).

2. [Install minikube](https://minikube.sigs.k8s.io/docs/start/), or  alternatively, authenticate with your Kubernetes cluster so that the `kubectl` cli is connected to the target Kubernetes cluster. 

3. If you are working with minikube, start your Kubernetes cluster, like so
    ```
    minikube start --memory 6184 --cpus 3
    ```

4. Enable the metrics server on the cluster if you want the horizontal pod autoscaler to auto-scale the Roboflow inference service.
    ```
    minikube addons enable metrics-server
    ```

Note: You will need to pass additional flags to the `minikube start` command for GPU support if you plan to use `gpu` enabled Roboflow inference images. Refer to the minikube documentation for more information.


### Deploy your  Roboflow Inference Service

*(All paths in this README start assume the `inference-repo-root/inference/enterprise/helm-chart` as the working directory)*

First, customize the `roboflow-inference-server/values-cpu.yaml` for your deployment. For example, you can set options for gpu or cpu-based container images, node-selectors, auto-scaling, ingress etc.

We will show running the cpu Roboflow Inference service in this document. You can use the `roboflow-inference-server/values-gpu.yaml` file to start deploying a gpu-based Roboflow inference service.

Then, deploy the service using Helm

```
helm --namespace infer upgrade --create-namespace --install roboflow-infer roboflow-inference-server/ -f roboflow-inference-server/values-cpu.yaml

```

The output of the `helm` command has useful information on how to access the Roboflow inference service running inside your Kubernetes cluster.

Refer to the Helm documentation to learn how to manage your deployment for upgrades, rollbacks etc.

After a few minutes you should see the inference pod and its associated cluster-IP service running in the Kubernetes cluster:

```
kubectl -n infer get all

```

### Verification

You can now execute inference requests against the Roboflow inference service in your Kubernetes cluster. 

First, connect to the service, like so:

```
  export POD_NAME=$(kubectl get pods --namespace infer -l "app.kubernetes.io/name=roboflow-inference-server,app.kubernetes.io/instance=roboflow-infer" -o jsonpath="{.items[0].metadata.name}")
  export CONTAINER_PORT=$(kubectl get pod --namespace infer $POD_NAME -o jsonpath="{.spec.containers[0].ports[0].containerPort}")
  kubectl --namespace infer port-forward $POD_NAME 9001:$CONTAINER_PORT
```

Next, open a new terminal and send a curl request to infer on a Roboflow model version (see [this](https://docs.roboflow.com/deploy/legacy-deployment/enterprise-cpu) documentation for details).

For example,

```
base64 IMAGE.jpg | curl -d @-  "http://localhost:9001/project-id/version-number?api_key=XXXXXXXXXXX"
```

Verify that the returned JSON string corresponds to a successful inference on the image you passed into the Roboflow inference service running inside your Kubernetes cluster.


## Teardown

To remove the service from your Kubernetes cluster, run

```
helm --namespace infer delete roboflow-infer

```

If you used minikube, delete the minikube cluster

```
minikube delete
```

## License

The Roboflow Enterprise License (the “Enterprise License”)
Copyright (c) 2023 Roboflow Inc.

With regard to the Roboflow Software:

This software and associated documentation files (the "Software") may only be
used in production, if you (and any entity that you represent) have accepted
and are following the terms of a separate Roboflow Enterprise agreement
that governs how you use the software.

Subject to the foregoing sentence, you are free to modify this Software and publish
patches to the Software. You agree that Roboflow and/or its licensors (as applicable)
retain all right, title and interest in and to all such modifications and/or patches,
and all such modifications and/or patches may only be used, copied, modified,
displayed, distributed, or otherwise exploited with a valid Roboflow Enterprise
license for the correct number of seats, devices, inferences, and other 
usage metrics specified therein.

Notwithstanding the foregoing, you may copy and modify the Software for development
and testing purposes, without requiring a subscription. You agree that Roboflow and/or
its licensors (as applicable) retain all right, title and interest in and to all
such modifications. You are not granted any other rights beyond what is expressly
stated herein. Subject to the foregoing, it is forbidden to copy, merge, publish,
distribute, sublicense, and/or sell the Software.

The full text of this Enterprise License shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

For all third party components incorporated into the Roboflow Software, those
components are licensed under the original license provided by the owner of the
applicable component.