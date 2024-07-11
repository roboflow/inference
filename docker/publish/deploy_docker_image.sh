check_return_code () {
    if [ $1 != 0 ]
    then
        echo "$2 failed with return code $1"
        exit $1
    else
        echo "$2 succeeced"
    fi
}

VERSION=`python inference/core/version.py`
REPOSITORY=$1
DOCKERFILE=$2

docker buildx build --platform linux/amd64 --pull -t  $REPOSITORY:$VERSION -f $DOCKERFILE . --push
check_return_code $? "Docker build"

docker tag $REPOSITORY:$VERSION $REPOSITORY:latest
check_return_code $? "Docker tag"

docker push $REPOSITORY:$VERSION
check_return_code $? "Docker push $VERSION"
