FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    unzip \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    opencv-python-headless \
    pandas \
    numpy \
    requests \
    tqdm \
    kaggle \
    pyyaml \
    beautifulsoup4

# Copy source code
COPY . /app

# Default command
CMD ["bash"]
