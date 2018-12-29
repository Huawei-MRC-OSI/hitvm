#!/usr/bin/env bash

while test -n "$1" ; do
  case "$1" in
    -h|--help)
      echo "Usage: $0 [--map-sockets] [--noproxy] [--nvidia|--nv] [SUFFIX]" >&2
      exit 1
      ;;
    --map-sockets)
      MAPSOCKETS=y
      ;;
    --noproxy)
      NOPROXY=y
      ;;
    --nvidia|--nv)
      NVIDIA=y
      ;;
    *)
      SUFFIX="$1"
      ;;
  esac
  shift
done

set -x -e

test -z "$SUFFIX" && SUFFIX="dev"
if test -f "$SUFFIX" ; then
  DOCKERFILE_PATH="$SUFFIX"
  SUFFIX=`echo "$DOCKERFILE_PATH" | sed 's/.*\.//'`
else
  DOCKERFILE_PATH="./_docker/Dockerfile.${SUFFIX}"
fi

RUNDOCKER_UID=`id --user`
DOCKER_CONTEXT_PATH="./_docker"
test -z "$DOCKER_WORKSPACE" && DOCKER_WORKSPACE=`pwd`
test -z "$DOCKER_COMMAND" && DOCKER_COMMAND="/bin/bash"
mkdir _docker 2>/dev/null || true
rm -rf _docker/* 2>/dev/null || true
cp -R ./src/$USER/tvm/docker/* ./_docker/
cp -R ./docker/* ./_docker/
# FIXME: Prepare Dockerfile.dev from Dockerfile.ci_cpu rather then use our own version

# Prepare GPU docker from template
cp ./_docker/Dockerfile.ci_gpu ./_docker/Dockerfile.gpu

if test "$NOPROXY" != "y" ; then
  if test -n "$http_proxy" ; then
    # FIXME: JAVA_HOME is not set, so Java certs remain unitialized
    sed -i '/FROM.*/a \
      COPY proxy.sh /install/proxy.sh \
      RUN bash /install/proxy.sh' ./_docker/Dockerfile.gpu
  fi
fi

if test "$NVIDIA" = "y" -o "$SUFFIX" = "gpu" ; then
  DOCKER_BINARY="nvidia-docker"
else
  DOCKER_BINARY="docker"
fi

DOCKER_IMG_NAME="hitvm_$SUFFIX"

if test -z "$DOCKER_PROXY_ARGS" ; then
  if test -n "$https_proxy" ; then
    PROXY_HOST=`echo $https_proxy | sed 's@.*//\(.*\):.*@\1@'`
    PROXY_PORT=`echo $https_proxy | sed 's@.*//.*:\(.*\)@\1@'`
    DOCKER_PROXY_ARGS="--build-arg=http_proxy=$https_proxy --build-arg=https_proxy=$https_proxy --build-arg=ftp_proxy=$https_proxy"
  fi
fi

docker build ${DOCKER_PROXY_ARGS} -t ${DOCKER_IMG_NAME} \
  -f "${DOCKERFILE_PATH}" "${DOCKER_CONTEXT_PATH}"

echo "Remap Docker detach action to Ctrl+e,e"
mkdir /tmp/docker-$RUNDOCKER_UID || true
cat >/tmp/docker-$RUNDOCKER_UID/config.json <<EOF
{ "detachKeys": "ctrl-e,e" }
EOF
DOCKER_CFG="--config /tmp/docker-$RUNDOCKER_UID"

if test "$MAPSOCKETS" = "y"; then
  PORT_TENSORBOARD=`expr 6000 + $RUNDOCKER_UID - 1000`
  PORT_JUPYTER=`expr 8000 + $RUNDOCKER_UID - 1000`
  DOCKER_PORT_ARGS="-p 0.0.0.0:$PORT_TENSORBOARD:6006 -p 0.0.0.0:$PORT_JUPYTER:8888"
  echo
  echo "*****************************"
  echo "Your Jupyter port: ${PORT_JUPYTER}"
  echo "Your Tensorboard port: ${PORT_TENSORBOARD}"
  echo "*****************************"
fi

${DOCKER_BINARY} $DOCKER_CFG run --rm --pid=host \
  -v ${DOCKER_WORKSPACE}:/workspace \
  -w /workspace \
  -e "CI_BUILD_HOME=/workspace" \
  -e "CI_BUILD_USER=$(id -u -n)" \
  -e "CI_BUILD_UID=$(id -u)" \
  -e "CI_BUILD_GROUP=$(id -g -n)" \
  -e "CI_BUILD_GID=$(id -g)" \
  -e "DISPLAY=$DISPLAY" \
  ${DOCKER_PORT_ARGS} \
  -it \
  --cap-add SYS_PTRACE \
  ${DOCKER_IMG_NAME} \
  bash --login ./src/$USER/tvm/docker/with_the_same_user \
  ${DOCKER_COMMAND}

