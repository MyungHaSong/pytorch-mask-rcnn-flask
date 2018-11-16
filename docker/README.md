# Containerize
This dockerfile and deployment script is based off of [azadehkhojandi/gpu-maskrcnn-pytorch-notebook](https://github.com/Azadehkhojandi/gpu-jupyter-docker-stacks). The GPU architecture is hard coded to "sm_37" for a Tesla K80, which is the GPU in an Azure NC6 VM with GPU.

This container will install all of the dependencies for Mask R-CNN and launch the flask server defined by `application.py` in the parent directory. A pre-compiled version of this container, using GPU architecture "sm_37", can be pulled from Docker Hub by using:
```
docker pull jomalsan/gpu-maskrcnn-pytorch-flask
```

Then run the container using the following. Note: this command requires install nvidia-docker (instructions below). It will run using docker, however the inference will not use the GPU of the machine.
```
sudo nvidia-docker run --rm -p 5000:5000 jomalsan/gpu-maskrcnn-pytorch-flask
```

## To build
TODO: Add previous installing directions
After installing nvidia-docker

For all of the following commands replace 'jomalsan' with your username.
To build this docker image run the following from this directory. If you have already build the container and changed something in the repository that it is cloning from, used the `--no-cache` option to force rebuilding.
```
sudo nvidia-docker build -t jomalsan/gpu-maskrcnn-pytorch-flask -f gpu-maskrcnn-pytorch-flask.dockerfile .
```

To add a tag to your new container, use `sudo docker image list` to look up the image id and then run:
```
sudo nvidia-docker tag {imageid} jomalsan/gpu-maskrcnn-pytorch-flask:version1.0
```

To publish your new container to docker hub use:
```
sudo nvidia-docker push jomalsan/gpu-maskrcnn-pytorch-flask
```

To run the container use:
```
sudo nvidia-docker run --rm -p 5000:5000 jomalsan/gpu-maskrcnn-pytorch-flask
```