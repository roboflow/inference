#!/usr/bin/env bash
# One command, end to end: create a spot GPU VM on Google Compute Engine, run
# profile_cosmos3_action_recognition_call.py on it, copy the results back and
# delete the VM. Needs a local gcloud that is logged in with a default project.
#
# Usage:
#   ROBOFLOW_API_KEY=... development/benchmark_scripts/profile_cosmos3_on_gce.sh \
#       --model-id workspace/model/1 [--video clip.mp4] [-- <extra profiler args>]
#
# Example with a sweep:
#   ROBOFLOW_API_KEY=... development/benchmark_scripts/profile_cosmos3_on_gce.sh \
#       --model-id workspace/model/1 --video clip.mp4 -- \
#       --window-seconds 16,8,4 --sample-fps 4,2
#
# Environment overrides:
#   ZONE          us-central1-a            zone with L4 capacity
#   MACHINE_TYPE  g2-standard-8            one L4 (24 GB); T4: n1-standard-8 plus ACCELERATOR
#   ACCELERATOR   (empty)                  e.g. type=nvidia-tesla-t4,count=1 for n1 machines
#   IMAGE_FAMILY  common-cu129-ubuntu-2204-nvidia-580
#   BRANCH        ccr-b2702452-tch8eg      branch of roboflow/inference to clone on the VM
#   VM_NAME       cosmos3-profile-<epoch>
#   KEEP_VM       (unset)                  set to 1 to leave the VM running afterwards
#   OUT_DIR       ./profile-<VM_NAME>      where profile.json and profile.txt land
set -euo pipefail

ZONE="${ZONE:-us-central1-a}"
MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-8}"
ACCELERATOR="${ACCELERATOR:-}"
IMAGE_FAMILY="${IMAGE_FAMILY:-common-cu129-ubuntu-2204-nvidia-580}"
BRANCH="${BRANCH:-ccr-b2702452-tch8eg}"
VM_NAME="${VM_NAME:-cosmos3-profile-$(date +%s)}"
OUT_DIR="${OUT_DIR:-./profile-${VM_NAME}}"

MODEL_ID=""
VIDEO=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-id)
            MODEL_ID="$2"
            shift 2
            ;;
        --video)
            VIDEO="$2"
            shift 2
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ -z "${MODEL_ID}" ]]; then
    echo "--model-id is required" >&2
    exit 2
fi
if [[ -z "${ROBOFLOW_API_KEY:-}" ]]; then
    echo "ROBOFLOW_API_KEY must be set in the environment" >&2
    exit 2
fi
if [[ -n "${VIDEO}" && ! -f "${VIDEO}" ]]; then
    echo "video not found: ${VIDEO}" >&2
    exit 2
fi

WORK_DIR="$(mktemp -d)"

cleanup() {
    rm -rf "${WORK_DIR}"
    if [[ "${KEEP_VM:-}" == "1" ]]; then
        echo "KEEP_VM=1: leaving ${VM_NAME} running in ${ZONE}. Delete it with:"
        echo "  gcloud compute instances delete ${VM_NAME} --zone ${ZONE} --quiet"
        return
    fi
    echo "deleting ${VM_NAME}..."
    gcloud compute instances delete "${VM_NAME}" --zone "${ZONE}" --quiet >/dev/null 2>&1 || true
}
trap cleanup EXIT

CREATE_ARGS=(
    --zone "${ZONE}"
    --machine-type "${MACHINE_TYPE}"
    --image-family "${IMAGE_FAMILY}"
    --image-project deeplearning-platform-release
    --boot-disk-size 100GB
    --boot-disk-type pd-balanced
    --maintenance-policy TERMINATE
    --provisioning-model SPOT
    --instance-termination-action DELETE
    --metadata install-nvidia-driver=True
)
if [[ -n "${ACCELERATOR}" ]]; then
    CREATE_ARGS+=(--accelerator "${ACCELERATOR}")
fi

echo "creating ${VM_NAME} (${MACHINE_TYPE}, ${IMAGE_FAMILY}) in ${ZONE}..."
gcloud compute instances create "${VM_NAME}" "${CREATE_ARGS[@]}"

ssh_vm() {
    gcloud compute ssh "${VM_NAME}" --zone "${ZONE}" --quiet --command "$1" -- -o ConnectTimeout=15
}

echo "waiting for SSH..."
for _ in $(seq 1 40); do
    if ssh_vm "true" >/dev/null 2>&1; then
        break
    fi
    sleep 10
done
ssh_vm "true"

echo "waiting for the NVIDIA driver..."
for _ in $(seq 1 60); do
    if ssh_vm "nvidia-smi -L" 2>/dev/null; then
        break
    fi
    sleep 10
done
ssh_vm "nvidia-smi -L"

# The key travels in a 600 file rather than on a command line.
{
    printf 'export ROBOFLOW_API_KEY=%q\n' "${ROBOFLOW_API_KEY}"
    printf 'export MODEL_ID=%q\n' "${MODEL_ID}"
    printf 'export BRANCH=%q\n' "${BRANCH}"
    printf 'export HAS_VIDEO=%q\n' "${VIDEO:+1}"
    printf 'EXTRA_ARGS=('
    for arg in "${EXTRA_ARGS[@]}"; do
        printf '%q ' "${arg}"
    done
    printf ')\n'
} > "${WORK_DIR}/profile_env.sh"
chmod 600 "${WORK_DIR}/profile_env.sh"

cat > "${WORK_DIR}/profile_remote.sh" <<'REMOTE'
set -euo pipefail
source "${HOME}/profile_env.sh"
VIDEO_ARGS=()
if [[ -n "${HAS_VIDEO}" ]]; then
    VIDEO_ARGS=(--video "${HOME}/clip.mp4")
fi

sudo apt-get update -qq >/dev/null
sudo apt-get install -y -qq git libgl1 ffmpeg >/dev/null

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi
export PATH="${HOME}/.local/bin:${PATH}"

if [[ ! -d "${HOME}/inference" ]]; then
    git clone --depth 1 --branch "${BRANCH}" https://github.com/roboflow/inference.git "${HOME}/inference"
fi
cd "${HOME}/inference"

uv venv --python 3.11 .venv >/dev/null
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -q -e ./inference_models click opencv-python-headless
# cosmos3_edge needs transformers 5.15; fall back to git main until it is on PyPI.
if ! uv pip install -q "transformers>=5.15,<5.16"; then
    uv pip install -q "git+https://github.com/huggingface/transformers.git@main"
fi

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"

python development/benchmark_scripts/profile_cosmos3_action_recognition_call.py \
    --model-id "${MODEL_ID}" "${VIDEO_ARGS[@]}" --output "${HOME}/profile.json" "${EXTRA_ARGS[@]}" \
    | tee "${HOME}/profile.txt"
REMOTE

echo "copying files to the VM..."
gcloud compute scp --zone "${ZONE}" --quiet \
    "${WORK_DIR}/profile_env.sh" "${WORK_DIR}/profile_remote.sh" "${VM_NAME}":~/
if [[ -n "${VIDEO}" ]]; then
    gcloud compute scp --zone "${ZONE}" --quiet "${VIDEO}" "${VM_NAME}":~/clip.mp4
fi

echo "running the profiler (model download plus install takes a few minutes)..."
ssh_vm "bash ~/profile_remote.sh"

mkdir -p "${OUT_DIR}"
gcloud compute scp --zone "${ZONE}" --quiet \
    "${VM_NAME}":~/profile.json "${VM_NAME}":~/profile.txt "${OUT_DIR}/"
echo ""
echo "results in ${OUT_DIR}/profile.txt and ${OUT_DIR}/profile.json"
