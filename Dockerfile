# Base stage with common dependencies
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04 as base

# Set the working directory in the container
WORKDIR /workspace

# Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set Python alias
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Update pip and install base Python packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch separately with specific version requirements
RUN pip install --no-cache-dir \
    torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# Install custom version of diffusers from Wan-AI GitHub repository and other dependencies
RUN pip install --no-cache-dir \
    huggingface_hub==0.30.1 \
    diffusers@git+https://github.com/huggingface/diffusers.git@6edb774b5e32f99987b89975b26f7c58b27ed111 \
    transformers==4.46.3 \
    accelerate \
    einops \
    safetensors \
    tqdm \
    ipython \
    ftfy \
    imageio \
    imageio-ffmpeg

# Set default environment variables
ENV DEFAULT_PROMPT="A spaceship flying through a cosmic nebula"
ENV NEGATIVE_PROMPT=""

# Pre-download the Wan2.1 model files during container build
RUN python -c "from diffusers import AutoencoderKLWan, WanPipeline; import torch; \
    vae = AutoencoderKLWan.from_pretrained('Wan-AI/Wan2.1-T2V-1.3B-Diffusers', subfolder='vae', torch_dtype=torch.float32); \
    pipe = WanPipeline.from_pretrained('Wan-AI/Wan2.1-T2V-1.3B-Diffusers', vae=vae, torch_dtype=torch.float16)"

# Create output directory
RUN mkdir -p /outputs && chmod 777 /outputs

# Production stage
FROM base as production

# Create directories for model cache
RUN mkdir -p /root/.cache/huggingface

# Copy the Python script into the container
COPY run_wan2.py /workspace/run_wan2.py
RUN chmod +x /workspace/run_wan2.py

# Set the entrypoint to run the Python script and allow for command-line arguments
ENTRYPOINT ["python", "/workspace/run_wan2.py"]

# Set a default command that can be overridden
CMD ["${DEFAULT_PROMPT}"]

# Development stage
FROM base as development

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Keep the container running
CMD ["bash"]