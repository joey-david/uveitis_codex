FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    unzip \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

COPY . /workspace

CMD ["bash"]
