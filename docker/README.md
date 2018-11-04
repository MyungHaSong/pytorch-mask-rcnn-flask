# Containerize
This dockerfile and deployment script is based off of [azadehkhojandi/gpu-maskrcnn-pytorch-notebook](https://github.com/Azadehkhojandi/gpu-jupyter-docker-stacks). The GPU architecture is hard coded to "sm_37" for a Tesla K80, which is the GPU in an Azure NC6 VM with GPU.

## To build
TODO: Add previous installing directions
After installing nvidia-docker

Replace `jomalsan` with your docker hub username
```
sudo nvidia-docker build -t jomalsan/gpu-maskrcnn-pytorch-flask -f gpu-maskrcnn-pytorch-flask.dockerfile .
```

To run the container, where again `jomalsan` is replaced with your docker hub username:
```
sudo nvidia-docker run --rm --privileged -p 8888:8888 -p 5000:5000 -e GRANT_SUDO=yes --user root -v "$PWD":/home/jovyan/mywork jomalsan/gpu-maskrcnn-pytorch-flask:latest
```