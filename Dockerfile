FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    unzip \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN python -m pip install --no-cache-dir uv

COPY requirements.txt /workspace/requirements.txt
RUN uv pip install --system --no-build-isolation -r /workspace/requirements.txt \
  && rm -rf /root/.cache/uv /root/.cache/pip

COPY . /workspace

CMD ["bash"]
