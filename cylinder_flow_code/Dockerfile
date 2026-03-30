FROM ubuntu:20.04

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV HOME="/home/"
WORKDIR /home/

# Basic packages
RUN apt-get update && apt-get install -y \
    wget \
    git \
    vim \
    build-essential \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN mkdir -p ~/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
    rm ~/miniconda3/miniconda.sh

ENV PATH="/home/miniconda3/bin:$PATH"

# Accept ToS + create env ✅
RUN /home/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    /home/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    /home/miniconda3/bin/conda create -y -n env python=3.8

# Auto-activate
RUN echo "source /home/miniconda3/bin/activate env" >> ~/.bashrc

COPY requirements.txt /home/

# Install CUDA + cuDNN (correct versions)
RUN source /home/miniconda3/bin/activate env && \
    conda install -y -c conda-forge \
        cudatoolkit=11.0 \
        cudnn=8.0 && \
    pip install -r requirements.txt

# Force correct libs
ENV LD_LIBRARY_PATH=/home/miniconda3/envs/env/lib

CMD ["/bin/bash"]


# sudo docker run --rm -it --network host --env DISPLAY=$DISPLAY -v $PWD:/workspace --gpus all --ipc=host --name nnc bl_nnc:latest
# sudo docker build -t bl_nnc:latest .
