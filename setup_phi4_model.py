#!/usr/bin/env python3
"""
Setup Phi-4-Multimodal Model

This script downloads and sets up the Phi-4-multimodal-instruct model for enhancing prompts
and image/audio understanding.
"""

import os
import sys
import argparse
import logging
import subprocess
import torch
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("phi4_model_setup.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("phi4_model_setup")

# Define model information
MODEL_ID = "microsoft/Phi-4-multimodal-instruct"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phi4_models")

# Use the existing nexa_venv
VENV_PATH = os.path.expanduser("~/nexa_venv")
PYTHON_PATH = os.path.join(VENV_PATH, "bin", "python")

def install_requirements():
    """Install required packages for Phi-4-multimodal-instruct."""
    # Skip flash-attn if not on compatible hardware
    has_gpu, supports_flash_attn = check_gpu() if torch.cuda.is_available() else (False, False)
    
    # Base requirements without flash-attn
    requirements = [
        "transformers>=4.48.2",
        "accelerate>=1.3.0",
        "soundfile>=0.13.1",
        "pillow>=11.1.0",
        "scipy>=1.15.2",
        "torchvision>=0.21.0",
        "backoff>=2.2.1",
        "peft>=0.13.2"
    ]
    
    # Only add flash-attn if on compatible hardware
    if has_gpu and supports_flash_attn:
        logger.info("Compatible GPU detected, will attempt to install flash-attn")
        try:
            # Try installing flash-attn separately first
            subprocess.check_call([PYTHON_PATH, "-m", "pip", "install", "flash-attn>=2.7.4"])
            logger.info("Successfully installed flash-attn")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Could not install flash-attn: {e}")
            logger.warning("Will continue without flash-attn, using eager attention implementation instead")
    else:
        logger.info("Skipping flash-attn installation as no compatible GPU was detected")
    
    logger.info(f"Installing requirements: {', '.join(requirements)}")
    
    try:
        subprocess.check_call([PYTHON_PATH, "-m", "pip", "install"] + requirements)
        logger.info("Successfully installed requirements")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing requirements: {e}")
        return False

def check_gpu():
    """Check if a compatible GPU is available."""
    if not torch.cuda.is_available():
        logger.warning("No CUDA-compatible GPU detected. The model will run slowly on CPU.")
        return False, False
    
    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"Found GPU: {gpu_name}")
    
    # Check if the GPU supports Flash Attention
    supports_flash_attn = any(gpu in gpu_name.lower() for gpu in ["a100", "a6000", "h100"])
    if not supports_flash_attn:
        logger.warning(f"GPU {gpu_name} may not support Flash Attention. Will use eager attention implementation.")
    else:
        logger.info(f"GPU {gpu_name} supports Flash Attention.")
    
    return True, supports_flash_attn

def download_model(token=None):
    """Download the Phi-4-multimodal-instruct model."""
    logger.info(f"Downloading {MODEL_ID} model")
    
    try:
        # Import here to ensure requirements are installed first
        from transformers import AutoProcessor, AutoModelForCausalLM
        
        # Create model directory if it doesn't exist
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Download processor
        logger.info("Downloading processor...")
        processor_kwargs = {"trust_remote_code": True}
        if token:
            processor_kwargs["token"] = token
        
        processor = AutoProcessor.from_pretrained(MODEL_ID, **processor_kwargs)
        processor.save_pretrained(os.path.join(MODEL_DIR))
        
        # Download model
        logger.info("Downloading model (this may take a while)...")
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True
        }
        if token:
            model_kwargs["token"] = token
        
        # Check if GPU is available and supports Flash Attention
        has_gpu, supports_flash_attn = check_gpu() if torch.cuda.is_available() else (False, False)
        if has_gpu:
            if supports_flash_attn:
                try:
                    import flash_attn
                    model_kwargs["_attn_implementation"] = "flash_attention_2"
                    logger.info("Using flash attention implementation")
                except ImportError:
                    logger.warning("flash_attn module not available, falling back to eager implementation")
                    model_kwargs["_attn_implementation"] = "eager"
            else:
                model_kwargs["_attn_implementation"] = "eager"
                logger.info("Using eager attention implementation")
        
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
        model.save_pretrained(os.path.join(MODEL_DIR))
        
        logger.info(f"Successfully downloaded {MODEL_ID} to {MODEL_DIR}")
        return True
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_model(token=None):
    """Test the downloaded model with a simple prompt."""
    logger.info("Testing the downloaded model")
    
    try:
        # Import here to ensure requirements are installed first
        from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig
        import torch
        
        # Load processor and model
        processor_kwargs = {"trust_remote_code": True}
        if token:
            processor_kwargs["token"] = token
        
        processor = AutoProcessor.from_pretrained(os.path.join(MODEL_DIR), **processor_kwargs)
        
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True
        }
        if token:
            model_kwargs["token"] = token
        
        # Check if GPU is available and supports Flash Attention
        has_gpu, supports_flash_attn = check_gpu() if torch.cuda.is_available() else (False, False)
        if has_gpu:
            model_kwargs["device_map"] = "auto"
            if supports_flash_attn:
                try:
                    import flash_attn
                    model_kwargs["_attn_implementation"] = "flash_attention_2"
                except ImportError:
                    model_kwargs["_attn_implementation"] = "eager"
            else:
                model_kwargs["_attn_implementation"] = "eager"
        
        model = AutoModelForCausalLM.from_pretrained(os.path.join(MODEL_DIR), **model_kwargs)
        
        # Load generation config
        generation_config = GenerationConfig.from_pretrained(os.path.join(MODEL_DIR))
        
        # Define a simple prompt
        user_prompt = '<|user|>'
        assistant_prompt = '<|assistant|>'
        prompt_suffix = '<|end|>'
        
        prompt = f'{user_prompt}Enhance this prompt for video generation: "A cat playing with a ball of yarn"{prompt_suffix}{assistant_prompt}'
        
        # Process with the model
        inputs = processor(text=prompt, return_tensors='pt')
        if has_gpu:
            inputs = inputs.to('cuda:0')
        
        # Generate response
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            generation_config=generation_config,
        )
        
        # Decode the response
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        logger.info(f"Test prompt: {prompt}")
        logger.info(f"Model response: {response}")
        
        logger.info("Model test completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def setup_model(token=None):
    """Set up the Phi-4-multimodal-instruct model."""
    # Install requirements
    if not install_requirements():
        logger.warning("Some requirements installation failed, but continuing with model setup")
    
    # Download model
    if not download_model(token):
        logger.error("Failed to download model. Exiting.")
        return False
    
    # Test model
    if not test_model(token):
        logger.warning("Model test failed, but setup may still be usable.")
    
    logger.info(f"Successfully set up {MODEL_ID}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Set up Phi-4-multimodal-instruct model")
    parser.add_argument("--token", "-t", help="Hugging Face API token for accessing private or gated models")
    args = parser.parse_args()
    
    setup_model(args.token)

if __name__ == "__main__":
    main() 