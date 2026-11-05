0. Get added to roboflow dockerhub
1. Bump verison in `inference/core/version.py`
2. `docker login`
3. `cd` to root
4. To deploy x86 images: `./deploy/cpu_http.sh` `./deploy/gpu_http.sh`
5. To deploy ARM images:
    - Go to ARM VM: `gcloud compute ssh --zone "us-central1-a" "arm-cpu-dev-machine" --project "roboflow-staging"`
    - `docker login`
    - pull code: `git clone git@github.com:roboflow/inference.git`
    - `./deploy/arm_cpu.http`