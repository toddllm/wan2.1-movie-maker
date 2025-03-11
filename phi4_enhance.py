#!/usr/bin/env python3
"""
Phi-4 Prompt Enhancement and Image Understanding

This script uses the Phi-4-multimodal-instruct model to enhance prompts for video generation
and to understand/analyze images.
"""

import os
import sys
import argparse
import logging
import json
import torch
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("phi4_enhance.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("phi4_enhance")

# Define model paths
PHI4_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phi4_models")

def load_model():
    """Load the Phi-4-multimodal-instruct model."""
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig
        
        logger.info("Loading Phi-4 model and processor...")
        
        # Check if model exists
        if not os.path.exists(PHI4_MODEL_DIR):
            logger.error(f"Model directory {PHI4_MODEL_DIR} not found. Please run setup_phi4_model.py first.")
            return None, None, None
        
        # Load processor
        processor = AutoProcessor.from_pretrained(PHI4_MODEL_DIR, trust_remote_code=True)
        
        # Check if GPU is available
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            logger.warning("No GPU available. Model will run slowly on CPU.")
        
        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if device == "cuda:0" else torch.float32,
            "low_cpu_mem_usage": True
        }
        
        # Set device map and attention implementation
        if device == "cuda:0":
            model_kwargs["device_map"] = "auto"
            gpu_name = torch.cuda.get_device_name(0)
            supports_flash_attn = any(gpu in gpu_name.lower() for gpu in ["a100", "a6000", "h100"])
            
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
        
        model = AutoModelForCausalLM.from_pretrained(PHI4_MODEL_DIR, **model_kwargs)
        
        # Load generation config
        generation_config = GenerationConfig.from_pretrained(PHI4_MODEL_DIR)
        
        logger.info(f"Successfully loaded Phi-4 model on {device}")
        return model, processor, generation_config
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None

def enhance_prompt(model, processor, generation_config, prompt, max_tokens=300):
    """Enhance a prompt for video generation using Phi-4."""
    logger.info(f"Enhancing prompt: {prompt}")
    
    try:
        # Define chat format
        user_prompt = '<|user|>'
        assistant_prompt = '<|assistant|>'
        prompt_suffix = '<|end|>'
        
        # Create the full prompt
        full_prompt = f'{user_prompt}Enhance this prompt for video generation. Make it more detailed, vivid, and cinematic. Add visual elements, lighting, camera angles, and atmosphere. Original prompt: "{prompt}"{prompt_suffix}{assistant_prompt}'
        
        # Process with the model
        inputs = processor(text=full_prompt, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = inputs.to('cuda:0')
        
        # Generate response
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            generation_config=generation_config,
        )
        
        # Decode the response
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        enhanced_prompt = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        logger.info(f"Enhanced prompt: {enhanced_prompt}")
        return enhanced_prompt
    except Exception as e:
        logger.error(f"Error enhancing prompt: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return prompt  # Return original prompt if enhancement fails

def analyze_image(model, processor, generation_config, image_path, max_tokens=300):
    """Analyze an image using Phi-4."""
    logger.info(f"Analyzing image: {image_path}")
    
    try:
        from PIL import Image
        
        # Check if image exists
        if not os.path.exists(image_path):
            logger.error(f"Image file {image_path} not found.")
            return None
        
        # Load image
        image = Image.open(image_path)
        
        # Define chat format
        user_prompt = '<|user|>'
        assistant_prompt = '<|assistant|>'
        prompt_suffix = '<|end|>'
        
        # Create the full prompt
        full_prompt = f'{user_prompt}<|image_1|>Describe this image in detail. Focus on visual elements, style, composition, and mood. What would be a good prompt to generate a similar image?{prompt_suffix}{assistant_prompt}'
        
        # Process with the model
        inputs = processor(text=full_prompt, images=image, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = inputs.to('cuda:0')
        
        # Generate response
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            generation_config=generation_config,
        )
        
        # Decode the response
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        analysis = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        logger.info(f"Image analysis: {analysis}")
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def batch_enhance_prompts(input_file, output_file, max_tokens=300):
    """Enhance a batch of prompts from a file."""
    logger.info(f"Batch enhancing prompts from {input_file} to {output_file}")
    
    # Load model
    model, processor, generation_config = load_model()
    if not model or not processor or not generation_config:
        logger.error("Failed to load model. Exiting.")
        return False
    
    try:
        # Read input prompts
        with open(input_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Found {len(prompts)} prompts to enhance")
        
        # Enhance each prompt
        enhanced_prompts = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{len(prompts)}")
            enhanced = enhance_prompt(model, processor, generation_config, prompt, max_tokens)
            enhanced_prompts.append(enhanced)
        
        # Write enhanced prompts to output file
        with open(output_file, 'w') as f:
            for prompt in enhanced_prompts:
                f.write(f"{prompt}\n")
        
        logger.info(f"Successfully enhanced {len(enhanced_prompts)} prompts and saved to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error in batch enhancement: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser(description="Phi-4 Prompt Enhancement and Image Analysis")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Enhance prompt command
    enhance_parser = subparsers.add_parser("enhance", help="Enhance a single prompt")
    enhance_parser.add_argument("prompt", help="Prompt to enhance")
    enhance_parser.add_argument("--max-tokens", type=int, default=300, help="Maximum tokens for generation")
    
    # Batch enhance command
    batch_parser = subparsers.add_parser("batch", help="Enhance prompts from a file")
    batch_parser.add_argument("input_file", help="Input file with prompts (one per line)")
    batch_parser.add_argument("output_file", help="Output file for enhanced prompts")
    batch_parser.add_argument("--max-tokens", type=int, default=300, help="Maximum tokens for generation")
    
    # Analyze image command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze an image")
    analyze_parser.add_argument("image_path", help="Path to the image file")
    analyze_parser.add_argument("--max-tokens", type=int, default=300, help="Maximum tokens for generation")
    
    args = parser.parse_args()
    
    if args.command == "enhance":
        model, processor, generation_config = load_model()
        if model and processor and generation_config:
            enhanced = enhance_prompt(model, processor, generation_config, args.prompt, args.max_tokens)
            print(enhanced)
    
    elif args.command == "batch":
        batch_enhance_prompts(args.input_file, args.output_file, args.max_tokens)
    
    elif args.command == "analyze":
        model, processor, generation_config = load_model()
        if model and processor and generation_config:
            analysis = analyze_image(model, processor, generation_config, args.image_path, args.max_tokens)
            print(analysis)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 