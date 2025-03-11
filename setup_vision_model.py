#!/usr/bin/env python3
"""
Setup Vision-Language Model

This script downloads and sets up a vision-language model for evaluating video frames.
It supports several models including Idefics3-8B, OpenFlamingo, and LLaVA-OneVision.
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vision_model_setup.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("vision_model_setup")

# Define model information
MODELS = {
    "idefics3-8b": {
        "name": "Idefics3-8B",
        "huggingface_repo": "HuggingFaceM4/Idefics3-8B-Llama3",
        "description": "A state-of-the-art open-source vision-language model built on Llama 3.1 and SigLIP-SO400M.",
        "requirements": ["torch>=2.0.0", "transformers>=4.35.0", "accelerate", "pillow", "sentencepiece"]
    },
    "openflamingo": {
        "name": "OpenFlamingo",
        "huggingface_repo": "openflamingo/OpenFlamingo-9B-vitl-mpt7b",
        "description": "An open-source adaptation of DeepMind's Flamingo, combining CLIP ViT-L/14 with a 7B parameter language model.",
        "requirements": ["torch>=2.0.0", "transformers>=4.35.0", "accelerate", "pillow", "open_flamingo"]
    },
    "llava-onevision": {
        "name": "LLaVA-OneVision",
        "huggingface_repo": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "description": "A minimalist design that effectively leverages pre-trained capabilities of both LLM and visual models.",
        "requirements": ["torch>=2.0.0", "transformers>=4.35.0", "accelerate", "pillow", "llava"]
    },
    "cogvlm2": {
        "name": "CogVLM2",
        "huggingface_repo": "THUDM/cogvlm2-llama3-chat-19B",
        "description": "A family of open-source visual language models designed for image and video understanding.",
        "requirements": ["torch>=2.0.0", "transformers>=4.35.0", "accelerate", "pillow", "sentencepiece"]
    }
}

# Define paths
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vision_models")
os.makedirs(MODELS_DIR, exist_ok=True)

def install_requirements(model_key):
    """
    Install the required packages for the specified model
    
    Args:
        model_key: Key of the model in the MODELS dictionary
    
    Returns:
        success: Boolean indicating if installation was successful
    """
    if model_key not in MODELS:
        logger.error(f"Unknown model: {model_key}")
        return False
    
    model_info = MODELS[model_key]
    requirements = model_info["requirements"]
    
    logger.info(f"Installing requirements for {model_info['name']}: {', '.join(requirements)}")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade"] + requirements, check=True)
        logger.info(f"Successfully installed requirements for {model_info['name']}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing requirements: {e}")
        return False

def download_model(model_key, token=None):
    """
    Download the specified model using the Hugging Face transformers library
    
    Args:
        model_key: Key of the model in the MODELS dictionary
        token: Hugging Face API token for accessing private or gated models
    
    Returns:
        success: Boolean indicating if download was successful
        model_path: Path to the downloaded model
    """
    if model_key not in MODELS:
        logger.error(f"Unknown model: {model_key}")
        return False, None
    
    model_info = MODELS[model_key]
    model_path = os.path.join(MODELS_DIR, model_key)
    
    logger.info(f"Downloading {model_info['name']} from {model_info['huggingface_repo']}")
    
    try:
        # Import here to ensure requirements are installed first
        from transformers import AutoProcessor, AutoModelForCausalLM
        
        # Download model and processor
        processor = AutoProcessor.from_pretrained(
            model_info["huggingface_repo"],
            token=token
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_info["huggingface_repo"],
            device_map="auto",
            torch_dtype="auto",
            token=token
        )
        
        # Save model and processor
        os.makedirs(model_path, exist_ok=True)
        processor.save_pretrained(model_path)
        model.save_pretrained(model_path)
        
        logger.info(f"Successfully downloaded {model_info['name']} to {model_path}")
        return True, model_path
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return False, None

def test_model(model_key, model_path, token=None):
    """
    Test the downloaded model with a simple image
    
    Args:
        model_key: Key of the model in the MODELS dictionary
        model_path: Path to the downloaded model
        token: Hugging Face API token for accessing private or gated models
    
    Returns:
        success: Boolean indicating if test was successful
    """
    if model_key not in MODELS:
        logger.error(f"Unknown model: {model_key}")
        return False
    
    model_info = MODELS[model_key]
    
    logger.info(f"Testing {model_info['name']}")
    
    try:
        # Import here to ensure requirements are installed first
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM
        from PIL import Image
        
        # Create a simple test image (black square)
        test_image_path = os.path.join(MODELS_DIR, "test_image.jpg")
        Image.new('RGB', (224, 224), color='black').save(test_image_path)
        
        # Load model and processor
        processor = AutoProcessor.from_pretrained(model_path, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype="auto",
            token=token
        )
        
        # Load and process the test image
        image = Image.open(test_image_path)
        
        # Prepare a simple prompt
        prompt = "What is in this image?"
        
        # Process inputs
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
        
        # Generate a response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        # Decode the response
        response = processor.decode(outputs[0], skip_special_tokens=True)
        
        logger.info(f"Test response: {response}")
        logger.info(f"Successfully tested {model_info['name']}")
        return True
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        return False

def setup_model(model_key, token=None):
    """
    Set up the specified model:
    1. Install requirements
    2. Download model
    3. Test model
    
    Args:
        model_key: Key of the model in the MODELS dictionary
        token: Hugging Face API token for accessing private or gated models
    
    Returns:
        success: Boolean indicating if setup was successful
        model_path: Path to the set up model
    """
    if model_key not in MODELS:
        logger.error(f"Unknown model: {model_key}")
        return False, None
    
    model_info = MODELS[model_key]
    
    logger.info(f"Setting up {model_info['name']}")
    
    # Step 1: Install requirements
    if not install_requirements(model_key):
        logger.error(f"Failed to install requirements for {model_info['name']}")
        return False, None
    
    # Step 2: Download model
    success, model_path = download_model(model_key, token)
    if not success:
        logger.error(f"Failed to download {model_info['name']}")
        return False, None
    
    # Step 3: Test model
    if not test_model(model_key, model_path, token):
        logger.warning(f"Failed to test {model_info['name']}, but continuing anyway")
    
    logger.info(f"Successfully set up {model_info['name']} at {model_path}")
    return True, model_path

def list_models():
    """
    List all available models
    """
    logger.info("Available models:")
    for key, info in MODELS.items():
        logger.info(f"  {key}: {info['name']}")
        logger.info(f"    {info['description']}")
        logger.info(f"    Requirements: {', '.join(info['requirements'])}")
        logger.info("")

def main():
    parser = argparse.ArgumentParser(description="Set up a vision-language model for evaluation")
    parser.add_argument("--model", "-m", choices=list(MODELS.keys()), help="Model to set up")
    parser.add_argument("--list", "-l", action="store_true", help="List available models")
    parser.add_argument("--token", "-t", help="Hugging Face API token for accessing private or gated models")
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
        return
    
    if not args.model:
        parser.error("Please specify a model to set up using --model")
    
    setup_model(args.model, args.token)

if __name__ == "__main__":
    main() 