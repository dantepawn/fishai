FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    ninja-build \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    python3-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone --depth 1 https://github.com/dantepawn/fishai.git /app/fishai \
    && git clone --depth 1 https://github.com/facebookresearch/sam2.git /app/sam2

COPY . /app/local

WORKDIR /app/local

RUN mkdir -p fishai \
    && ln -s /app/local/fishai_utils.py fishai/fishai_utils.py \
    && printf "from .fishai_utils import *\n" > fishai/__init__.py

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir \
    opencv-python-headless \
    pillow \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    tqdm \
    supervision>=0.21.0 \
    ultralytics>=8.2.0 \
    && pip install --no-cache-dir -e "/app/sam2[dev]"

RUN wget -O /app/sam2/sam2/configs/train.yaml "https://drive.usercontent.google.com/download?id=11cmbxPPsYqFyWq87tmLgBAQ6OZgEhPG3"

ENV PYTHONPATH=/app/local:/app/fishai

ENTRYPOINT ["python", "-m", "main.pipeline_cli"]
CMD ["--help"]
