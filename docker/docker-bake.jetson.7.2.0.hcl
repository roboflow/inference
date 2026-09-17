variable "JETSON_SERVER_TAGS" {
  default = "roboflow/roboflow-inference-server-jetson-7.2.0:local"
}

variable "JETSON_WHEELS_TAGS" {
  default = "roboflow/roboflow-inference-wheels-jetson-7.2.0:local"
}

variable "JETSON_WHEELS_IMAGE" {
  # Keep the server coupled to the immutable Torch 2.13 wheel bundle.  Torch
  # 2.13 pins Triton 3.7.1, so the older torch213 bundle (which contains 3.6)
  # is intentionally not a valid server input.
  default = "roboflow/roboflow-inference-wheels-jetson-7.2.0:1.4.0-mvp-torch213-v4@sha256:38c7cef8af5f6076c0b17136c5e8982ed23dae22330ceca09f69552385788cba"
}

variable "JETSON_MEDIA_IMAGE" {
  # Empty by default: the media runtime (FFmpeg, GStreamer, OpenCV, and the
  # native SM87/SM110 tensor bridge) is built FROM THIS TREE so the compiled
  # bridge always matches this branch's Python-side ctypes contract.  The
  # native bridge version handshake ("7") CANNOT detect a struct-layout drift
  # between divergent branches, so never point this at a media image built
  # from another branch.  Set it to a digest built from the same ref (e.g. by
  # the rebuild_media path of docker.jetson.7.2.0.yml) to skip the media
  # compile.
  default = ""
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

# Component stages of the wheels image. CI calls these stages independently to
# populate Depot's shared project cache before jetson-wheels-jp72 assembles the
# publishable wheel bundle.
target "jetson-wheels-torch-jp72" {
  inherits = ["jetson-wheels-jp72"]
  target   = "torch-builder"
  tags     = []
}

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

target "jetson-server-jp72" {
  context    = "."
  dockerfile = "docker/dockerfiles/Dockerfile.onnx.jetson.7.2.0"
  contexts = {
    jetson-media  = JETSON_MEDIA_IMAGE != "" ? "docker-image://${JETSON_MEDIA_IMAGE}" : "target:jetson-media-jp72"
    jetson-wheels = "docker-image://${JETSON_WHEELS_IMAGE}"
  }
  platforms = ["linux/arm64"]
  tags      = split(",", JETSON_SERVER_TAGS)
}
