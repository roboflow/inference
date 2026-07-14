variable "JETSON_SERVER_TAGS" {
  default = "roboflow/roboflow-inference-server-jetson-7.2.0:local"
}

target "jetson-media-jp72" {
  context    = "."
  dockerfile = "docker/dockerfiles/Dockerfile.media.jetson.7.2.0"
  platforms  = ["linux/arm64"]
}

target "jetson-server-jp72" {
  context    = "."
  dockerfile = "docker/dockerfiles/Dockerfile.onnx.jetson.7.2.0"
  contexts = {
    jetson-media = "target:jetson-media-jp72"
  }
  platforms = ["linux/arm64"]
  tags      = split(",", JETSON_SERVER_TAGS)
}
