#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

echo "🚀 Setting up development environment..."

# Upgrade pip
echo "📦 Upgrading pip..."
python -m pip install --upgrade pip

# Core dependencies
pip install packaging==24.2 typing-extensions==4.12.2

# PyTorch with CUDA 12.8 support
echo "🔥 Installing PyTorch with CUDA 12.8 support..."
pip install torch==2.10.0 torchvision==0.25.0 \
    --index-url https://download.pytorch.org/whl/cu128 --force-reinstall

# AI/ML libraries
pip install transformers==4.44.2 diffusers[torch]==0.30.3 \
    huggingface-hub==0.35.3 accelerate==0.33.0

# xformers (build from source for RTX 5090 / sm_120)
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST="12.0"
pip install --no-build-isolation -v \
    git+https://github.com/facebookresearch/xformers.git@main#egg=xformers

# Computer vision & image processing
pip install Pillow==10.4.0 opencv-python==4.10.0.84 \
    scikit-image==0.21.0 lpips==0.1.4

pip install facenet-pytorch --no-deps

# Background removal (installed separately to avoid dependency conflicts)
pip install rembg

# Scientific computing
pip install numpy==1.26.4 matplotlib==3.7.5 scikit-learn==1.3.2 \
    einops==0.8.0 jaxtyping==0.2.19 pytorch-msssim==1.0.0

# Utilities & configuration
pip install easydict==1.13 pyyaml==6.0.2 wandb==0.19.1 \
    termcolor==2.4.0 plyfile==1.0.3 tqdm gradio==5.49.1

# System dependencies
echo "🎬 Installing system dependencies..."
sudo apt update && sudo apt install -y ffmpeg

# Video processing
pip install videoio==0.3.0 ffmpeg-python==0.2.0

# Specialized libraries
pip install --no-build-isolation git+https://github.com/graphdeco-inria/diff-gaussian-rasterization

echo "✅ Setup complete!"
