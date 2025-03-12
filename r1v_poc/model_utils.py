#!/usr/bin/env python3
"""
Model Utilities - R1-V and R1-Omni Integration

This module provides utilities for loading and using the R1-V and R1-Omni models:
- Model loading and configuration
- Inference utilities
- Model caching for efficiency

It supports the R1-V and R1-Omni integration in the Movie Maker application.
"""

import os
import sys
import logging
import torch
import json
from pathlib import Path
from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForVision2Seq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_utils.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("model_utils")

class ModelCache:
    """
    Cache for loaded models to avoid reloading the same model multiple times.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelCache, cls).__new__(cls)
            cls._instance.models = {}
            cls._instance.processors = {}
        return cls._instance
    
    def get_model(self, model_name, device=None):
        """
        Get a model from the cache or load it if not cached.
        
        Args:
            model_name (str): The name or path of the model.
            device (str): Device to run the model on ('cuda' or 'cpu').
            
        Returns:
            tuple: (model, processor) for the model.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        cache_key = f"{model_name}_{device}"
        
        if cache_key in self.models and cache_key in self.processors:
            logger.info(f"Using cached model: {model_name} on {device}")
            return self.models[cache_key], self.processors[cache_key]
        
        logger.info(f"Loading model: {model_name} on {device}")
        try:
            # First try to load the processor
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            
            # Try loading with Vision2Seq first (for newer multimodal models)
            try:
                logger.info(f"Attempting to load {model_name} with AutoModelForVision2Seq")
                model = AutoModelForVision2Seq.from_pretrained(
                    model_name, 
                    trust_remote_code=True,
                    device_map=device
                )
                logger.info(f"Successfully loaded {model_name} with AutoModelForVision2Seq")
            except Exception as e:
                logger.warning(f"Failed to load with AutoModelForVision2Seq: {e}")
                # Fall back to CausalLM
                logger.info(f"Attempting to load {model_name} with AutoModelForCausalLM")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    trust_remote_code=True,
                    device_map=device
                )
                logger.info(f"Successfully loaded {model_name} with AutoModelForCausalLM")
            
            self.models[cache_key] = model
            self.processors[cache_key] = processor
            return model, processor
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise

def load_model(model_name, device=None):
    """
    Load a model and its processor.
    
    Args:
        model_name (str): The name or path of the model.
        device (str): Device to run the model on ('cuda' or 'cpu').
        
    Returns:
        tuple: (model, processor) for the model.
    """
    cache = ModelCache()
    return cache.get_model(model_name, device)

def get_r1v_model(model_name=None, device=None):
    """
    Get an R1-V model for visual analysis.
    
    Args:
        model_name (str): The name or path of the R1-V model. If None, use default.
        device (str): Device to run the model on ('cuda' or 'cpu').
        
    Returns:
        tuple: (model, processor) for the R1-V model.
    """
    # Use default model if not specified
    if model_name is None:
        model_name = "Qwen/Qwen-VL-Chat"
    
    return load_model(model_name, device)

def get_r1omni_model(model_name=None, device=None):
    """
    Get an R1-Omni model for emotion recognition.
    
    Args:
        model_name (str): The name or path of the R1-Omni model. If None, use default.
        device (str): Device to run the model on ('cuda' or 'cpu').
        
    Returns:
        tuple: (model, processor) for the R1-Omni model.
    """
    # Use default model if not specified
    if model_name is None:
        model_name = "StarJiaxing/R1-Omni-0.5B"
    
    return load_model(model_name, device)

def generate_with_model(model, processor, prompt, images=None, audio=None, max_new_tokens=512):
    """
    Generate text with a model.
    
    Args:
        model: The model to use for generation.
        processor: The processor for the model.
        prompt (str): The text prompt.
        images (list): Optional list of images (PIL.Image).
        audio (str): Optional path to an audio file.
        max_new_tokens (int): Maximum number of tokens to generate.
        
    Returns:
        str: The generated text.
    """
    try:
        # Prepare inputs
        inputs_kwargs = {"text": prompt, "return_tensors": "pt"}
        
        # Add images if provided
        if images is not None:
            inputs_kwargs["images"] = images
        
        # Add audio if provided
        if audio is not None:
            inputs_kwargs["audio"] = audio
        
        # Process inputs
        inputs = processor(**inputs_kwargs)
        
        # Move inputs to the same device as the model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        
        # Decode the response
        response = processor.decode(output[0], skip_special_tokens=True)
        
        # Extract the model's response (remove the prompt)
        if prompt in response:
            response = response.split(prompt)[-1].strip()
        
        return response
    
    except Exception as e:
        logger.error(f"Error generating with model: {e}")
        return f"Error: {str(e)}"

def save_model_config(config_path, r1v_model=None, r1omni_model=None):
    """
    Save model configuration to a JSON file.
    
    Args:
        config_path (str): Path to save the configuration.
        r1v_model (str): The name or path of the R1-V model.
        r1omni_model (str): The name or path of the R1-Omni model.
    """
    config = {
        "r1v_model": r1v_model or "Qwen/Qwen-VL-Chat",
        "r1omni_model": r1omni_model or "StarJiaxing/R1-Omni-0.5B",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Model configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Error saving model configuration: {e}")

def load_model_config(config_path):
    """
    Load model configuration from a JSON file.
    
    Args:
        config_path (str): Path to the configuration file.
        
    Returns:
        dict: The model configuration.
    """
    default_config = {
        "r1v_model": "Qwen/Qwen-VL-Chat",
        "r1omni_model": "StarJiaxing/R1-Omni-0.5B",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    if not os.path.exists(config_path):
        logger.warning(f"Configuration file {config_path} not found, using defaults")
        return default_config
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Model configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading model configuration: {e}")
        return default_config

# Main function for testing
def main():
    """Main function for testing model utilities."""
    import argparse
    from PIL import Image
    
    parser = argparse.ArgumentParser(description="Model Utilities Test")
    parser.add_argument("--r1v", help="Path to the R1-V model")
    parser.add_argument("--r1omni", help="Path to the R1-Omni model")
    parser.add_argument("--image", help="Path to an image file for testing")
    parser.add_argument("--audio", help="Path to an audio file for testing")
    parser.add_argument("--save_config", help="Path to save the model configuration")
    args = parser.parse_args()
    
    # Test model loading
    if args.r1v:
        logger.info("Testing R1-V model loading...")
        r1v_model, r1v_processor = get_r1v_model(args.r1v)
        logger.info("R1-V model loaded successfully")
    
    if args.r1omni:
        logger.info("Testing R1-Omni model loading...")
        r1omni_model, r1omni_processor = get_r1omni_model(args.r1omni)
        logger.info("R1-Omni model loaded successfully")
    
    # Test generation with image
    if args.r1v and args.image:
        logger.info(f"Testing generation with image: {args.image}")
        r1v_model, r1v_processor = get_r1v_model(args.r1v)
        
        # Load image
        image = Image.open(args.image)
        
        # Generate with image
        prompt = "Describe this image in detail."
        response = generate_with_model(r1v_model, r1v_processor, prompt, images=[image])
        
        logger.info(f"Generated response: {response[:100]}...")
    
    # Test generation with audio
    if args.r1omni and args.audio:
        logger.info(f"Testing generation with audio: {args.audio}")
        r1omni_model, r1omni_processor = get_r1omni_model(args.r1omni)
        
        # Generate with audio
        prompt = "<audio>\nDescribe the emotions in this audio."
        response = generate_with_model(r1omni_model, r1omni_processor, prompt, audio=args.audio)
        
        logger.info(f"Generated response: {response[:100]}...")
    
    # Save configuration
    if args.save_config:
        logger.info(f"Saving model configuration to {args.save_config}")
        save_model_config(args.save_config, args.r1v, args.r1omni)

if __name__ == "__main__":
    main() 