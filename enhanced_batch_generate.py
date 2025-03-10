#!/usr/bin/env python3
"""
Enhanced Batch Video Generation Script for Movie Maker

This script reads prompts from a file, enhances them using a language model,
and sends them to the Movie Maker API to generate videos in batch mode.
"""

import os
import sys
import time
import json
import argparse
import requests
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class PromptEnhancer:
    """Class to enhance prompts using a language model."""
    
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", device="cpu"):
        """Initialize the prompt enhancer with a language model."""
        self.model_name = model_name
        self.device = device  # Force CPU usage to avoid CUDA memory issues
        
        print(f"Loading prompt enhancement model: {model_name}")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map=self.device,
            torch_dtype=torch.float32,  # Use float32 for CPU
            trust_remote_code=True
        )
        
        print("Prompt enhancement model loaded successfully")
    
    def enhance_prompt(self, prompt):
        """Enhance a prompt to make it more detailed and descriptive."""
        system_message = (
            "You are an expert at enhancing text prompts for text-to-video generation. "
            "Your task is to expand the given prompt with rich visual details, including: "
            "- Specific visual elements and their attributes (colors, textures, materials) "
            "- Lighting conditions and atmosphere "
            "- Camera angles and movements "
            "- Temporal progression (what happens over time) "
            "- Background and environment details "
            "Make the prompt highly descriptive but keep it coherent and focused on the original concept. "
            "Respond ONLY with the enhanced prompt, without any explanations or additional text."
        )
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Enhance this prompt for text-to-video generation: '{prompt}'"}
        ]
        
        # Convert messages to model input format
        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize and generate
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        response = generated_text.split("ASSISTANT: ")[-1].strip()
        
        print(f"Original prompt: {prompt}")
        print(f"Enhanced prompt: {response}")
        
        return response

def generate_video(prompt, server_url, frame_count=160):
    """Send a request to generate a video with the given prompt."""
    url = f"{server_url}/generate"
    data = {"prompt": prompt}
    
    print(f"Generating video for prompt: {prompt}")
    print(f"Frame count: {frame_count} ({frame_count/16:.1f} seconds)")
    
    try:
        response = requests.post(url, data=data)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        result = response.json()
        if result.get("status") == "success":
            print(f"Success! Video clip: {result.get('clip')}")
            return True, result.get('clip')
        else:
            print(f"Error: {result.get('message')}")
            return False, result.get('message')
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return False, str(e)

def process_batch(prompt_file, server_url, frame_count=160, delay=0, enhance=False, model_name=None):
    """Process all prompts in the given file."""
    if not os.path.exists(prompt_file):
        print(f"Error: Prompt file '{prompt_file}' not found.")
        return False
    
    # Read prompts from file
    with open(prompt_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    if not prompts:
        print("Error: No prompts found in the file.")
        return False
    
    print(f"Found {len(prompts)} prompts to process.")
    
    # Initialize prompt enhancer if needed
    enhancer = None
    if enhance:
        try:
            # Always use CPU for prompt enhancement to avoid CUDA memory conflicts
            enhancer = PromptEnhancer(model_name=model_name, device="cpu")
        except Exception as e:
            print(f"Error initializing prompt enhancer: {e}")
            print("Continuing without prompt enhancement.")
    
    # Process each prompt
    successful = 0
    failed = 0
    
    # Create a directory to save enhanced prompts
    if enhance:
        os.makedirs("enhanced_prompts", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        enhanced_prompts_file = f"enhanced_prompts/enhanced_{timestamp}.txt"
        
    for i, prompt in enumerate(prompts, 1):
        print(f"\nProcessing prompt {i}/{len(prompts)}:")
        
        # Enhance the prompt if requested
        if enhance and enhancer:
            try:
                enhanced_prompt = enhancer.enhance_prompt(prompt)
                # Save enhanced prompt to file
                with open(enhanced_prompts_file, "a") as f:
                    f.write(f"Original: {prompt}\nEnhanced: {enhanced_prompt}\n\n")
                prompt = enhanced_prompt
            except Exception as e:
                print(f"Error enhancing prompt: {e}")
                print("Using original prompt instead.")
        
        success, _ = generate_video(prompt, server_url, frame_count)
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # Add delay between requests if specified
        if i < len(prompts) and delay > 0:
            print(f"Waiting {delay} seconds before next prompt...")
            time.sleep(delay)
    
    print(f"\nBatch processing complete!")
    print(f"Total prompts: {len(prompts)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if enhance:
        print(f"Enhanced prompts saved to: {enhanced_prompts_file}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Batch process video generation prompts with enhancement")
    parser.add_argument("prompt_file", help="File containing prompts (one per line)")
    parser.add_argument("--url", default="http://localhost:5001", help="Movie Maker server URL")
    parser.add_argument("--seconds", type=float, default=10.0, 
                        help="Length of each video in seconds (default: 10.0)")
    parser.add_argument("--delay", type=int, default=5, 
                        help="Delay in seconds between requests (default: 5)")
    parser.add_argument("--enhance", action="store_true", 
                        help="Enhance prompts using a language model")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", 
                        help="Model to use for prompt enhancement (default: smaller 1.5B model)")
    
    args = parser.parse_args()
    
    # Convert seconds to frames (16 FPS)
    frame_count = int(args.seconds * 16)
    
    # Ensure frame count is valid (minimum 16 frames, maximum 160 frames)
    if frame_count < 16:
        frame_count = 16
        print("Warning: Minimum clip length is 1 second (16 frames)")
    elif frame_count > 160:
        frame_count = 160
        print("Warning: Maximum clip length is 10 seconds (160 frames)")
    
    # Process the batch
    process_batch(
        args.prompt_file, 
        args.url, 
        frame_count, 
        args.delay,
        enhance=args.enhance,
        model_name=args.model
    )

if __name__ == "__main__":
    main() 