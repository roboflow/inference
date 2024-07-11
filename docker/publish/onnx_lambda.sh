set -e

PROJECT=${1:-roboflow-staging}
VERSION=${2:-no-slim}
if [ $PROJECT = 'roboflow-staging' ]
then
    echo "Deploying to staging"
    REPOSITORY=809383754475.dkr.ecr.us-east-1.amazonaws.com/roboflow-inference-server-lambda
elif [ $PROJECT = 'roboflow-platform' ]
then
    echo "Deploying to prod"
    REPOSITORY=583392192139.dkr.ecr.us-east-1.amazonaws.com/roboflow-inference-server-lambda
else
    echo "Project must be roboflow-staging or roboflow-platform"
    exit 1
fi
if [ $VERSION = 'slim' ]
then
  echo "Building slim image"
  REPOSITORY=$REPOSITORY-slim
  DOCKERFILE=docker/dockerfiles/Dockerfile.onnx.lambda.slim
else
  echo "Building standard image"
  DOCKERFILE=docker/dockerfiles/Dockerfile.onnx.lambda
fi
docker/publish/deploy_docker_image.sh $REPOSITORY $DOCKERFILE
