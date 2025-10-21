Build a Docker container for the SGA project
============================================

Build a cross-platform docker container as documented [here](https://www.docker.com/blog/faster-multi-platform-builds-dockerfile-cross-compilation-guide), [here](https://blog.jaimyn.dev/how-to-build-multi-architecture-docker-images-on-an-m1-mac/), and [here](https://docs.nersc.gov/development/shifter/how-to-use/).

First, pull the latest container:
```
docker pull legacysurvey/legacypipe:gpu-1.4.6

export DOCKER_BUILDKIT=0
export COMPOSE_DOCKER_CLI_BUILD=0
```
and then either
```
docker buildx create --name SGA-build --node SGA --use
```
or
```
docker buildx use SGA-build
```
and then

```
docker buildx build --platform linux/amd64,linux/arm64/v8 --push -t legacysurvey/sga:0.5.6 .
docker buildx build --platform linux/amd64,linux/arm64/v8 --push -t legacysurvey/sga:latest .
```

To enter the container (with a shell prompt) on a laptop do:
```
docker pull legacysurvey/sga:latest
docker run -it legacysurvey/sga:latest
```

Or at NERSC:
```
shifterimg pull docker:legacysurvey/sga:0.5.6
shifter --image docker:legacysurvey/sga:0.5.6 bash
```
