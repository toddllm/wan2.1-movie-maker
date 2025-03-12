#!/bin/bash

# Setup script for R1-Omni

# Create directories
mkdir -p ~/models

# Clone repositories if they don't exist
if [ ! -d ~/HumanOmni ]; then
  echo "Cloning HumanOmni repository..."
  cd ~
  git clone https://github.com/HumanMLLM/HumanOmni.git
fi

if [ ! -d ~/R1-Omni ]; then
  echo "Cloning R1-Omni repository..."
  cd ~
  git clone https://github.com/HumanMLLM/R1-Omni.git
fi

# Add HumanOmni to Python path
if ! grep -q "PYTHONPATH.*HumanOmni" ~/.bashrc; then
  echo "Adding HumanOmni to Python path..."
  echo 'export PYTHONPATH=$PYTHONPATH:~/HumanOmni' >> ~/.bashrc
  source ~/.bashrc
fi

# Install dependencies
echo "Installing dependencies..."
pip install torch torchvision transformers tokenizers accelerate peft timm decord imageio imageio-ffmpeg moviepy scenedetect opencv-python pysubs2 scikit-learn huggingface_hub einops bitsandbytes pydantic requests openai uvicorn fastapi tensorboard wandb tabulate

# Download models if they don't exist
if [ ! -d ~/models/R1-Omni-0.5B ]; then
  echo "Downloading R1-Omni-0.5B model..."
  cd ~/models
  git lfs install
  git clone https://huggingface.co/StarJiaxing/R1-Omni-0.5B
fi

if [ ! -d ~/models/siglip-base-patch16-224 ]; then
  echo "Downloading siglip-base-patch16-224 model..."
  cd ~/models
  git clone https://huggingface.co/google/siglip-base-patch16-224
fi

if [ ! -d ~/models/whisper-large-v3 ]; then
  echo "Downloading whisper-large-v3 model..."
  cd ~/models
  git clone https://huggingface.co/openai/whisper-large-v3
fi

# Update config.json
echo "Updating config.json..."
sed -i 's|"mm_audio_tower": ".*"|"mm_audio_tower": "~/models/whisper-large-v3"|g' ~/models/R1-Omni-0.5B/config.json
sed -i 's|"mm_vision_tower": ".*"|"mm_vision_tower": "~/models/siglip-base-patch16-224"|g' ~/models/R1-Omni-0.5B/config.json

echo "Setup complete!"
echo "You can now run the test script with: python ~/movie_maker/r1_omni/test_r1_omni.py --video_path /path/to/your/video.mp4" 