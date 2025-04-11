# Base stage with common dependencies
FROM python:3.9-slim-buster AS base

# Set the working directory in the container
WORKDIR /workspace

# Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Update pip and install base Python packages with specific numpy version
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir "numpy>=1.23.5,<2"

# Install PyTorch with CUDA support via pip
RUN pip install --no-cache-dir \
    torch==2.4.0 torchvision==0.19.0 --extra-index-url https://download.pytorch.org/whl/cu121

# Install custom version of diffusers from Wan-AI GitHub repository and other dependencies
RUN pip install --no-cache-dir \
    huggingface_hub==0.30.1 \
    diffusers@git+https://github.com/huggingface/diffusers.git@6edb774b5e32f99987b89975b26f7c58b27ed111 \
    transformers==4.49.0 \
    accelerate==1.1.1 \
    einops \
    safetensors \
    tqdm \
    ipython \
    ftfy \
    imageio \
    imageio-ffmpeg

# Create directories for model cache
RUN mkdir -p /root/.cache/huggingface

# Pre-download the Wan2.1 model files during container build
RUN python -c "from diffusers import AutoencoderKLWan, WanPipeline; import torch; \
    vae = AutoencoderKLWan.from_pretrained('Wan-AI/Wan2.1-T2V-1.3B-Diffusers', subfolder='vae', torch_dtype=torch.float32); \
    pipe = WanPipeline.from_pretrained('Wan-AI/Wan2.1-T2V-1.3B-Diffusers', vae=vae, torch_dtype=torch.float16)"

# Create output directory
RUN mkdir -p /outputs && chmod 777 /outputs

# Production stage
FROM base AS production

# Copy the Python script into the container
COPY run_wan2.1.py /workspace/run_wan2.1.py
RUN chmod +x /workspace/run_wan2.1.py

# Set the entrypoint to run the Python script
ENTRYPOINT ["python", "/workspace/run_wan2.1.py"]

# Set default CMD (empty, as we're using environment variables now)
CMD []

# Development stage
FROM base AS development

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Keep the container running
CMD ["bash"]