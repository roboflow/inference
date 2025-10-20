#!/usr/bin/env bash
set -euo pipefail

# Configuration via env vars (override as needed when invoking):
#   API_KEY       - required; your Roboflow API key
#   REQUESTS      - total number of requests to send (default: 50)
#   CONCURRENCY   - number of requests to run in parallel (default: 10)
#   HOST          - server host with scheme (default: http://localhost:9001)
#   ENDPOINT_PATH - endpoint path (default: /sam2/segment_image)
#   IMAGE_URL     - image URL to segment (default: Pexels sample)
#   PRINT_BODIES  - if 1, print response bodies (default: 0)
#   IMAGE_ID      - optional: provide to reuse cached embedding for speed

REQUESTS="${REQUESTS:-50}"
CONCURRENCY="${CONCURRENCY:-10}"
API_KEY="${API_KEY:-}"
HOST="${HOST:-http://localhost:9001}"
ENDPOINT_PATH="${ENDPOINT_PATH:-/sam3/visual_segment}"
IMAGE_URL="${IMAGE_URL:-https://images.pexels.com/photos/18001210/pexels-photo-18001210.jpeg}"
PRINT_BODIES="${PRINT_BODIES:-0}"
IMAGE_ID="${IMAGE_ID:-}"

if [[ -z "${API_KEY}" ]]; then
  echo "ERROR: API_KEY env var is required." >&2
  exit 1
fi

make_payload() { # args: index
  local i="$1"
  # Vary prompt point positions slightly per request to avoid exact duplicates
  local x=$(( 50 + (i % 200) ))
  local y=$(( 50 + ((i * 7) % 200) ))
  if [[ -n "${IMAGE_ID}" ]]; then
    cat <<EOF
{
  "image": { "type": "url", "value": "${IMAGE_URL}" },
  "prompts": [ { "points": [ { "x": ${x}, "y": ${y}, "positive": true } ] } ],
  "image_id": "${IMAGE_ID}"
}
EOF
  else
    cat <<EOF
{
  "image": { "type": "url", "value": "${IMAGE_URL}" },
  "prompts": [ { "points": [ { "x": ${x}, "y": ${y}, "positive": true } ] } ]
}
EOF
  fi
}

do_request() { # args: index
  local i="$1"
  local url="${HOST%/}${ENDPOINT_PATH}"
  local payload
  payload="$(make_payload "$i")"
  if [[ "${PRINT_BODIES}" == "1" ]]; then
    curl -sS -X POST "${url}?api_key=${API_KEY}" \
      -H "Content-Type: application/json" \
      -d "${payload}" \
      -w "req=${i} status=%{http_code} time=%{time_total} sec\\n"
  else
    curl -sS -o /dev/null -X POST "${url}?api_key=${API_KEY}" \
      -H "Content-Type: application/json" \
      -d "${payload}" \
      -w "req=${i} status=%{http_code} time=%{time_total} sec\\n"
  fi
}

export -f make_payload
export -f do_request
export API_KEY HOST ENDPOINT_PATH IMAGE_URL PRINT_BODIES IMAGE_ID

seq 1 "${REQUESTS}" | xargs -n 1 -P "${CONCURRENCY}" -I {} bash -c 'do_request "$@"' _ {}


