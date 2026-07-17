variable "JETSON_SERVER_TAGS" {
  default = "roboflow/roboflow-inference-server-jetson-7.2.0:local"
}

variable "JETSON_WHEELS_TAGS" {
  default = "roboflow/roboflow-inference-wheels-jetson-7.2.0:local"
}

variable "JETSON_WHEELS_IMAGE" {
  default = "roboflow/roboflow-inference-wheels-jetson-7.2.0:1.4.0-torch213"
}

target "jetson-media-jp72" {
  context    = "."
  dockerfile = "docker/dockerfiles/Dockerfile.media.jetson.7.2.0"
  platforms  = ["linux/arm64"]
}

target "jetson-wheels-jp72" {
  context    = "."
  dockerfile = "docker/dockerfiles/Dockerfile.wheels.jetson.7.2.0"
  platforms  = ["linux/arm64"]
  tags       = split(",", JETSON_WHEELS_TAGS)
}

# Component stages of the wheels image, grouped by dependency chain so CI can
# fan the compiles out to separate builders; they meet in the shared project
# cache and jetson-wheels-jp72 assembles the result.
target "jetson-wheels-torchvision-jp72" {
  inherits = ["jetson-wheels-jp72"]
  target   = "torchvision-builder"
  tags     = []
}

target "jetson-wheels-flash-attn-jp72" {
  inherits = ["jetson-wheels-jp72"]
  target   = "flash-attn-builder"
  tags     = []
}

target "jetson-wheels-ort-jp72" {
  inherits = ["jetson-wheels-jp72"]
  target   = "ort-builder"
  tags     = []
}

target "jetson-wheels-triton-jp72" {
  inherits = ["jetson-wheels-jp72"]
  target   = "triton-fetcher"
  tags     = []
}

target "jetson-wheels-pycuda-jp72" {
  inherits = ["jetson-wheels-jp72"]
  target   = "pycuda-builder"
  tags     = []
}

group "jetson-wheels-torch-stack-jp72" {
  targets = ["jetson-wheels-torchvision-jp72", "jetson-wheels-flash-attn-jp72"]
}

group "jetson-wheels-ort-stack-jp72" {
  targets = ["jetson-wheels-ort-jp72", "jetson-wheels-triton-jp72", "jetson-wheels-pycuda-jp72"]
}

target "jetson-server-jp72" {
  context    = "."
  dockerfile = "docker/dockerfiles/Dockerfile.onnx.jetson.7.2.0"
  contexts = {
    jetson-media  = "target:jetson-media-jp72"
    jetson-wheels = "docker-image://${JETSON_WHEELS_IMAGE}"
  }
  platforms = ["linux/arm64"]
  tags      = split(",", JETSON_SERVER_TAGS)
}
