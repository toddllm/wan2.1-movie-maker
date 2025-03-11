#!/usr/bin/env python3
"""
Direct Video Generation Script

This script reads prompts from a file and generates videos directly using the Wan2.1 model.
"""

import os
import sys
import time
import argparse
import logging
import subprocess
from datetime import datetime
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("direct_generate.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("direct_generate")

# Define paths
CLIPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clips")
WAN_DIR = os.path.join(os.path.expanduser("~"), "development", "wan-video", "Wan2.1")
MODEL_DIR = os.path.join(os.path.expanduser("~"), "development", "wan-video", "wan2.1", "models", "Wan2.1-T2V-1.3B")
VENV_PYTHON = os.path.join(os.path.expanduser("~"), "development", "wan-video", "wan2.1", "venv", "bin", "python")
WAN_REPO_PATH = os.path.join(os.path.expanduser("~"), "development", "wan-video", "wan2.1", "wan_repo")

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
        
        # Check if a Wan2.1 generation process is running
        result = subprocess.run(
            ["ps", "-ef"],
            capture_output=True, text=True
        )
        
        # Look for actual Wan2.1 generate.py process
        if "python generate.py --task" in result.stdout:
            logger.info(f"Found Wan2.1 generate.py process with {memory_used}MB GPU memory usage")
            return True, memory_used
        
        # Ignore baseline memory allocation completely
        # Only consider very high memory usage (>5GB) without a generate.py process
        if memory_used > 5000:
            logger.info(f"No generate.py process found, but high GPU memory usage: {memory_used}MB")
            return True, memory_used
            
        # If we get here, GPU is not in use (just baseline memory allocation)
        logger.info(f"GPU is not in use (baseline memory: {memory_used}MB)")
        return False, memory_used
        
    except Exception as e:
        logger.error(f"Error checking GPU usage: {e}")
        return False, 0

def wait_for_gpu(max_wait_time=3600, check_interval=30):
    """Wait for the GPU to be free."""
    start_time = time.time()
    
    while True:
        in_use, memory_used = check_gpu_usage()
        
        if not in_use:
            logger.info("GPU is now free to use.")
            return True
        
        elapsed = time.time() - start_time
        if elapsed > max_wait_time:
            logger.warning(f"Timeout waiting for GPU to be free after {max_wait_time/60:.1f} minutes.")
            return False
        
        logger.info(f"Waiting {check_interval} seconds for GPU to be free...")
        time.sleep(check_interval)

def generate_video(prompt, output_path, frame_count=160, size="832*480"):
    """Generate a video using the WAN2.1 model by calling the script directly."""
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Construct the command
        cmd = [
            VENV_PYTHON, "generate.py",
            "--task", "t2v-1.3B",
            "--size", size,
            "--ckpt_dir", MODEL_DIR,
            "--offload_model=True",
            "--frame_num", str(frame_count),
            "--prompt", prompt
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Set PYTHONPATH to include the wan_repo directory
        env = os.environ.copy()
        env["PYTHONPATH"] = WAN_REPO_PATH + ":" + env.get("PYTHONPATH", "")
        
        # Run the command in the Wan2.1 directory
        process = subprocess.Popen(
            cmd,
            cwd=WAN_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )
        
        # Capture output
        stdout, stderr = process.communicate()
        
        # Log the output
        if stdout:
            logger.info(f"Command output: {stdout}")
        if stderr:
            logger.error(f"Command error: {stderr}")
        
        # Check if the process was successful
        if process.returncode != 0:
            raise Exception(f"Command failed with return code {process.returncode}: {stderr}")
        
        # Find the generated video file
        # The WAN2.1 script generates files with a timestamp pattern
        timestamp = datetime.now().strftime("%Y%m%d")
        
        # Look for generated video files in the working directory
        generated_files = []
        for filename in os.listdir(WAN_DIR):
            if filename.startswith("t2v-1.3B") and filename.endswith(".mp4") and timestamp in filename:
                generated_files.append(filename)
        
        # Sort by modification time (newest first)
        generated_files.sort(key=lambda x: os.path.getmtime(os.path.join(WAN_DIR, x)), reverse=True)
        
        if not generated_files:
            raise Exception("No video file was generated")
        
        # Get the latest generated file
        latest_file = os.path.join(WAN_DIR, generated_files[0])
        logger.info(f"Found generated video: {latest_file}")
        
        # Copy the file to the desired output location
        shutil.copy2(latest_file, output_path)
        logger.info(f"Copied video to: {output_path}")
        
        return True, output_path
    except Exception as e:
        logger.error(f"Error generating video: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, str(e)

def process_batch(prompt_file, frame_count=160, width=832, height=480, delay=0, wait_for_completion=True):
    """Process all prompts in the given file."""
    if not os.path.exists(prompt_file):
        logger.error(f"Error: Prompt file '{prompt_file}' not found.")
        return False
    
    # Read prompts from file
    with open(prompt_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    if not prompts:
        logger.error("Error: No prompts found in the file.")
        return False
    
    logger.info(f"Found {len(prompts)} prompts to process.")
    
    # Check if GPU is in use before starting
    logger.info("Checking GPU status before starting batch processing...")
    in_use, memory_used = check_gpu_usage()
    if in_use:
        logger.info("Waiting for GPU to be free before starting batch processing...")
        if not wait_for_gpu():
            logger.error("Timeout waiting for GPU. Exiting.")
            return False
    else:
        logger.info("GPU is available for video generation.")
    
    # Process each prompt
    successful = 0
    failed = 0
    
    for i, prompt in enumerate(prompts, 1):
        logger.info(f"\nProcessing prompt {i}/{len(prompts)}:")
        
        # Generate a unique filename based on the prompt
        timestamp = int(time.time())
        safe_prompt = ''.join(c if c.isalnum() or c in ' _-' else '_' for c in prompt[:50])
        filename = f"{timestamp}_{safe_prompt}.mp4"
        output_path = os.path.join(CLIPS_DIR, filename)
        
        # Generate the video
        success, result = generate_video(prompt, output_path, frame_count, f"{width}*{height}")
        
        if success:
            successful += 1
            logger.info(f"Successfully generated video: {result}")
        else:
            failed += 1
            logger.error(f"Failed to generate video: {result}")
        
        # Add delay between requests if specified
        if i < len(prompts) and delay > 0:
            logger.info(f"Waiting {delay} seconds before next prompt...")
            time.sleep(delay)
    
    logger.info(f"\nBatch processing complete!")
    logger.info(f"Total prompts: {len(prompts)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Batch process video generation prompts directly")
    parser.add_argument("--input", "-i", required=True, help="Input file containing prompts (one per line)")
    parser.add_argument("--duration", "-d", type=float, default=10.0, 
                        help="Length of each video in seconds (default: 10.0)")
    parser.add_argument("--width", "-w", type=int, default=832,
                        help="Width of the video (default: 832)")
    parser.add_argument("--height", "-H", type=int, default=480,
                        help="Height of the generated video")
    parser.add_argument("--delay", type=int, default=5, 
                        help="Delay in seconds between requests (default: 5)")
    parser.add_argument("--no-wait", action="store_true",
                        help="Don't wait for each generation to complete before starting the next one")
    
    args = parser.parse_args()
    
    # Convert seconds to frames (16 FPS)
    frame_count = int(args.duration * 16)
    
    # Ensure frame count is valid (minimum 16 frames, maximum 160 frames)
    if frame_count < 16:
        frame_count = 16
        logger.warning("Warning: Minimum clip length is 1 second (16 frames)")
    elif frame_count > 160:
        frame_count = 160
        logger.warning("Warning: Maximum clip length is 10 seconds (160 frames)")
    
    # Process the batch
    process_batch(
        args.input, 
        frame_count,
        args.width,
        args.height,
        args.delay,
        not args.no_wait
    )

if __name__ == "__main__":
    main() 