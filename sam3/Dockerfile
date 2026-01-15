FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    git \
    wget \
    libgl1 \
    curl \
    ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

RUN python -m ensurepip --default-pip \
    && python -m pip install --upgrade pip

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

RUN sed -i '/torch/d' requirements.txt
RUN sed -i '/torchvision/d' requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install decord

RUN mkdir -p /usr/local/lib/python3.12/dist-packages/assets/ \
    && wget -O /usr/local/lib/python3.12/dist-packages/assets/bpe_simple_vocab_16e6.txt.gz \
    https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz

COPY run_sam.py /app/
COPY handler.py /app/

ENTRYPOINT ["python", "handler.py"]