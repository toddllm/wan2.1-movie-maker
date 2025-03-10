#!/usr/bin/env python3
"""
Batch Video Generation Script for Movie Maker

This script reads prompts from a file (one per line) and sends them to the Movie Maker API
to generate videos in batch mode.
"""

import os
import sys
import time
import json
import argparse
import requests
import subprocess
from datetime import datetime

def check_gpu_usage():
    """Check if the GPU is already in use by another generation process."""
    try:
        # Check if nvidia-smi is available
        result = subprocess.run(["which", "nvidia-smi"], capture_output=True, text=True)
        if result.returncode != 0:
            return False, 0  # nvidia-smi not available, assume no GPU
        
        # Get GPU memory usage
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            return False, 0
        
        memory_used = int(result.stdout.strip())
        
        # Check if a generation process is running
        result = subprocess.run(
            ["ps", "-ef"],
            capture_output=True, text=True
        )
        
        # Only consider GPU in use if there's an actual generate.py process running
        if "generate.py" in result.stdout:
            return True, memory_used
        
        # Ignore baseline memory allocation completely
        # Only consider very high memory usage (>5GB) without a generate.py process
        if memory_used > 5000:
            return True, memory_used
            
        return False, memory_used
        
    except Exception as e:
        print(f"Error checking GPU usage: {e}")
        return False, 0

def wait_for_gpu(max_wait_time=3600, check_interval=30):
    """Wait for the GPU to be free."""
    start_time = time.time()
    
    while True:
        in_use, memory_used = check_gpu_usage()
        
        if not in_use:
            return True
        
        elapsed = time.time() - start_time
        if elapsed > max_wait_time:
            print(f"Timeout waiting for GPU to be free after {max_wait_time/60:.1f} minutes.")
            return False
        
        print(f"GPU is in use ({memory_used}MB). Waiting {check_interval} seconds...")
        time.sleep(check_interval)

def generate_video(prompt, server_url, frame_count=160, wait_for_completion=True):
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
            clip_path = result.get('clip')
            print(f"Success! Video clip: {clip_path}")
            
            if wait_for_completion:
                # Wait for the generation to complete by checking for GPU usage
                print("Waiting for generation to complete...")
                wait_for_gpu()
                
            return True, clip_path
        else:
            print(f"Error: {result.get('message')}")
            return False, result.get('message')
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return False, str(e)

def process_batch(prompt_file, server_url, frame_count=160, delay=0, wait_for_completion=True):
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
    
    # Check if GPU is in use before starting
    in_use, memory_used = check_gpu_usage()
    if in_use:
        print(f"GPU is currently in use ({memory_used}MB).")
        print("Waiting for GPU to be free before starting batch processing...")
        if not wait_for_gpu():
            print("Timeout waiting for GPU. Exiting.")
            return False
    
    # Process each prompt
    successful = 0
    failed = 0
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\nProcessing prompt {i}/{len(prompts)}:")
        success, _ = generate_video(prompt, server_url, frame_count, wait_for_completion)
        
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
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Batch process video generation prompts")
    parser.add_argument("prompt_file", help="File containing prompts (one per line)")
    parser.add_argument("--url", default="http://localhost:5001", help="Movie Maker server URL")
    parser.add_argument("--seconds", type=float, default=10.0, 
                        help="Length of each video in seconds (default: 10.0)")
    parser.add_argument("--delay", type=int, default=5, 
                        help="Delay in seconds between requests (default: 5)")
    parser.add_argument("--no-wait", action="store_true",
                        help="Don't wait for each generation to complete before starting the next one")
    
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
        not args.no_wait
    )

if __name__ == "__main__":
    main() 