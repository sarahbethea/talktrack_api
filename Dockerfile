# CUDA + cuDNN + PyTorch already included
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps: ffmpeg/sndfile + cuDNN 8
# System deps: add NVIDIA repo so libcudnn8 exists, then install
RUN apt-get update && apt-get install -y --no-install-recommends wget gnupg ca-certificates \
 && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
 && dpkg -i cuda-keyring_1.1-1_all.deb && rm -f cuda-keyring_1.1-1_all.deb \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      ffmpeg \
      libsndfile1 \
      libcudnn8 libcudnn8-dev \
 && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
