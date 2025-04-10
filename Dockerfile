# FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

# RUN apt-get update -y && apt-get install -y libgl1-mesa-glx libglib2.0-0 git gcc g++ && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y libgl1-mesa-glx libglib2.0-0 git gcc g++ build-essential wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install fairseq
RUN pip install git+https://github.com/openai/CLIP.git
RUN pip install git+https://github.com/kungfuai/CVlization.git
RUN pip install wandb
# this is to copy the model weights and code
# COPY . .
