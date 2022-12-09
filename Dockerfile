FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV NV_CUDNN_VERSION 8.6.0.163
ENV NV_CUDNN_PACKAGE_NAME "libcudnn8"

ENV NV_CUDNN_PACKAGE "libcudnn8=$NV_CUDNN_VERSION-1+cuda11.8"
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ${NV_CUDNN_PACKAGE} \
    && apt-mark hold ${NV_CUDNN_PACKAGE_NAME} \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update -y \
    && apt-get install -y python3-pip
RUN echo 'alias python=python3' >> ~/.bashrc

WORKDIR /app
COPY requirements.txt requirements.txt

# Activate conda environment for bash
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 --upgrade
RUN pip install -r requirements.txt

ENTRYPOINT [ "bash" ]
