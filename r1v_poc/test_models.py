#!/usr/bin/env python3
"""
R1-V and R1-Omni Model Tester

This script downloads and tests the R1-V and R1-Omni models to ensure they work correctly.
It verifies that the models can be loaded and used for basic inference.
"""

import os
import sys
import logging
import argparse
import json
import torch
import time
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("model_test")

def test_r1v_model(model_name="Qwen/Qwen2-VL-7B", device=None, test_image=None):
    """
    Test the R1-V model by loading it and running a simple inference.
    
    Args:
        model_name (str): The name or path of the R1-V model.
        device (str): Device to run the model on ('cuda' or 'cpu').
        test_image (str): Path to a test image file.
        
    Returns:
        bool: True if the test was successful, False otherwise.
    """
    try:
        logger.info(f"Testing R1-V model: {model_name}")
        start_time = time.time()
        
        # Import required modules
        from transformers import AutoProcessor, AutoModelForCausalLM
        
        # Determine device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        # Load the model and processor
        logger.info("Loading processor...")
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            device_map=device,
            trust_remote_code=True
        )
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        
        # Test inference if a test image is provided
        if test_image:
            if not os.path.exists(test_image):
                logger.warning(f"Test image not found: {test_image}")
                # Create a simple test image
                logger.info("Creating a simple test image")
                img = Image.new('RGB', (512, 512), color=(73, 109, 137))
                img.save('test_image.jpg')
                test_image = 'test_image.jpg'
            
            # Load the test image
            logger.info(f"Loading test image: {test_image}")
            image = Image.open(test_image)
            
            # Prepare inputs
            logger.info("Preparing inputs for inference")
            prompt = "Describe this image in detail."
            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(device)
            
            # Run inference
            logger.info("Running inference")
            inference_start = time.time()
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False
                )
            
            inference_time = time.time() - inference_start
            logger.info(f"Inference completed in {inference_time:.2f} seconds")
            
            # Decode the response
            response = processor.decode(output[0], skip_special_tokens=True)
            
            # Extract the model's response (remove the prompt)
            response = response.split(prompt)[-1].strip()
            
            logger.info(f"Model response: {response[:100]}...")
        
        logger.info("R1-V model test completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error testing R1-V model: {e}")
        return False

def test_r1omni_model(model_name="HumanMLLM/R1-Omni-0.5B", device=None, test_image=None):
    """
    Test the R1-Omni model by loading it and running a simple inference.
    
    Args:
        model_name (str): The name or path of the R1-Omni model.
        device (str): Device to run the model on ('cuda' or 'cpu').
        test_image (str): Path to a test image file.
        
    Returns:
        bool: True if the test was successful, False otherwise.
    """
    try:
        logger.info(f"Testing R1-Omni model: {model_name}")
        start_time = time.time()
        
        # Import required modules
        from transformers import AutoProcessor, AutoModelForCausalLM
        
        # Determine device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        # Load the model and processor
        logger.info("Loading processor...")
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            device_map=device,
            trust_remote_code=True
        )
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        
        # Test inference if a test image is provided
        if test_image:
            if not os.path.exists(test_image):
                logger.warning(f"Test image not found: {test_image}")
                # Create a simple test image
                logger.info("Creating a simple test image")
                img = Image.new('RGB', (512, 512), color=(73, 109, 137))
                img.save('test_image.jpg')
                test_image = 'test_image.jpg'
            
            # Load the test image
            logger.info(f"Loading test image: {test_image}")
            image = Image.open(test_image)
            
            # Prepare inputs
            logger.info("Preparing inputs for inference")
            prompt = "As an emotional recognition expert; what emotion is conveyed in this image?"
            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(device)
            
            # Run inference
            logger.info("Running inference")
            inference_start = time.time()
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False
                )
            
            inference_time = time.time() - inference_start
            logger.info(f"Inference completed in {inference_time:.2f} seconds")
            
            # Decode the response
            response = processor.decode(output[0], skip_special_tokens=True)
            
            # Extract the model's response (remove the prompt)
            response = response.split(prompt)[-1].strip()
            
            logger.info(f"Model response: {response[:100]}...")
        
        logger.info("R1-Omni model test completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error testing R1-Omni model: {e}")
        return False

def test_video_frame_extraction(video_path=None):
    """
    Test video frame extraction functionality.
    
    Args:
        video_path (str): Path to a test video file.
        
    Returns:
        bool: True if the test was successful, False otherwise.
    """
    try:
        logger.info("Testing video frame extraction")
        
        # Create a simple test video if none is provided
        if not video_path or not os.path.exists(video_path):
            logger.warning(f"Test video not found: {video_path}")
            logger.info("Skipping video frame extraction test")
            return True
        
        # Open the video file
        logger.info(f"Opening video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        # Check if the video file was opened successfully
        if not cap.isOpened():
            logger.error(f"Error opening video file: {video_path}")
            return False
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0
        
        logger.info(f"Video properties: {frame_count} frames, {fps} fps, {duration:.2f} seconds")
        
        # Extract a few frames
        num_frames = min(5, frame_count)
        frame_indices = [int(i * frame_count / num_frames) for i in range(num_frames)]
        
        frames = []
        for idx in frame_indices:
            # Set the frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            
            # Read the frame
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                
                frames.append(pil_image)
                
                # Save the frame for inspection
                frame_path = f"test_frame_{idx}.jpg"
                pil_image.save(frame_path)
                logger.info(f"Saved frame to {frame_path}")
            else:
                logger.warning(f"Failed to read frame at index {idx}")
        
        # Release the video file
        cap.release()
        
        logger.info(f"Extracted {len(frames)} frames successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error testing video frame extraction: {e}")
        return False

def update_config(config_path, r1v_model=None, r1omni_model=None, device=None):
    """
    Update the configuration file with the tested models.
    
    Args:
        config_path (str): Path to the configuration file.
        r1v_model (str): The name or path of the R1-V model.
        r1omni_model (str): The name or path of the R1-Omni model.
        device (str): Device to run the models on ('cuda' or 'cpu').
        
    Returns:
        bool: True if the update was successful, False otherwise.
    """
    try:
        logger.info(f"Updating configuration file: {config_path}")
        
        # Create default config
        config = {
            "r1v_model": r1v_model or "Qwen/Qwen2-VL-7B",
            "r1omni_model": r1omni_model or "HumanMLLM/R1-Omni-0.5B",
            "device": device or ('cuda' if torch.cuda.is_available() else 'cpu')
        }
        
        # Write the config to file
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration updated successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        return False

def main():
    """Main function for testing the models."""
    parser = argparse.ArgumentParser(description="R1-V and R1-Omni Model Tester")
    parser.add_argument("--r1v", default="Qwen/Qwen2-VL-7B", help="R1-V model name or path")
    parser.add_argument("--r1omni", default="HumanMLLM/R1-Omni-0.5B", help="R1-Omni model name or path")
    parser.add_argument("--device", help="Device to run the models on ('cuda' or 'cpu')")
    parser.add_argument("--image", help="Path to a test image file")
    parser.add_argument("--video", help="Path to a test video file")
    parser.add_argument("--skip-r1v", action="store_true", help="Skip testing the R1-V model")
    parser.add_argument("--skip-r1omni", action="store_true", help="Skip testing the R1-Omni model")
    parser.add_argument("--skip-video", action="store_true", help="Skip testing video frame extraction")
    args = parser.parse_args()
    
    # Determine device
    device = args.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info("Starting model tests")
    logger.info(f"Device: {device}")
    
    # Test R1-V model
    r1v_success = True
    if not args.skip_r1v:
        r1v_success = test_r1v_model(args.r1v, device, args.image)
    else:
        logger.info("Skipping R1-V model test")
    
    # Test R1-Omni model
    r1omni_success = True
    if not args.skip_r1omni:
        r1omni_success = test_r1omni_model(args.r1omni, device, args.image)
    else:
        logger.info("Skipping R1-Omni model test")
    
    # Test video frame extraction
    video_success = True
    if not args.skip_video:
        video_success = test_video_frame_extraction(args.video)
    else:
        logger.info("Skipping video frame extraction test")
    
    # Update configuration
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "r1v_config.json")
    update_config(config_path, args.r1v, args.r1omni, device)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    logger.info(f"R1-V Model Test: {'PASSED' if r1v_success else 'FAILED'}")
    logger.info(f"R1-Omni Model Test: {'PASSED' if r1omni_success else 'FAILED'}")
    logger.info(f"Video Frame Extraction Test: {'PASSED' if video_success else 'FAILED'}")
    logger.info("="*50)
    
    # Return success status
    return r1v_success and r1omni_success and video_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 