0. Get added to roboflow dockerhub
1. Bump verison in `inference/core/version.py`
2. `docker login`
3. `cd` to root
4. To deploy x86 images: `./deploy/cpu_http.sh` `./deploy/gpu_http.sh`
5. To deploy aws Lambda images to [staging|platform]: `source ~/aws_keys/[staging|platform].sh`
    - first time logging into staging ECR:
        - `aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 809383754475.dkr.ecr.us-east-1.amazonaws.com`
    - first time logging into prod ECR:
        - `aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 583392192139.dkr.ecr.us-east-1.amazonaws.com`
    - Upload container: `./deploy/onnx_lambda.sh roboflow-[staging|platform]`
    - Go to roboflow-infra and change Lambda tags
    - NOTE: Test out staging deploy before deploying prod
6. To deploy ARM images:
    - Go to ARM VM: `gcloud compute ssh --zone "us-central1-a" "arm-cpu-dev-machine" --project "roboflow-staging"`
    - `docker login`
    - pull code: `git clone git@github.com:roboflow/inference.git`
    - `./deploy/arm_cpu.http`