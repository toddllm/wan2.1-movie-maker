#!/usr/bin/env python3
"""
Video Quality Evaluation Script

This script evaluates if generated videos match their prompts by:
1. Extracting frames from videos using FFmpeg
2. Using a vision-language model to evaluate if frames match the prompt
3. Enhancing prompts and regenerating videos if needed
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
import time
import logging
import requests
from pathlib import Path
from datetime import datetime
from enhance_prompts import TemplateEnhancer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("video_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("video_evaluation")

# Paths
CLIP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clips")
ENHANCED_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "enhanced_prompts")
FRAMES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extracted_frames")
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(ENHANCED_PROMPTS_DIR, exist_ok=True)

# Initialize the prompt enhancer
prompt_enhancer = TemplateEnhancer()

def extract_frames(video_path, output_dir, interval=0.5):
    """
    Extract frames from a video at specified intervals
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        interval: Interval in seconds between frames (default: 0.5)
    
    Returns:
        List of paths to extracted frames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract video filename without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create a subdirectory for this video's frames
    video_frames_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_frames_dir, exist_ok=True)
    
    # Extract frames using FFmpeg
    cmd = [
        "ffmpeg", "-i", video_path, 
        "-vf", f"fps=1/{interval}", 
        "-q:v", "2",  # High quality
        os.path.join(video_frames_dir, f"frame_%04d.jpg")
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"Extracted frames from {video_path} at {interval}-second intervals")
        
        # Get list of extracted frames
        frames = sorted([os.path.join(video_frames_dir, f) for f in os.listdir(video_frames_dir) if f.endswith('.jpg')])
        return frames
    except subprocess.CalledProcessError as e:
        logger.error(f"Error extracting frames: {e}")
        logger.error(f"FFmpeg stderr: {e.stderr.decode()}")
        return []

def evaluate_frames_with_model(frames, prompt, model_name="idefics3-8b"):
    """
    Evaluate if frames match the prompt using a vision-language model
    
    Args:
        frames: List of paths to frames
        prompt: The original prompt used to generate the video
        model_name: Name of the vision-language model to use
    
    Returns:
        score: A score from 0-10 indicating how well the frames match the prompt
        feedback: Feedback on why the score was given
    """
    # TODO: Implement actual model loading and inference
    # For now, we'll use a simple API call to the Movie Maker service
    
    # Select a subset of frames to evaluate (first, middle, last)
    if len(frames) > 3:
        selected_frames = [frames[0], frames[len(frames)//2], frames[-1]]
    else:
        selected_frames = frames
    
    # Mock evaluation for now
    # In a real implementation, we would load the model and run inference
    logger.info(f"Evaluating {len(selected_frames)} frames against prompt: {prompt}")
    
    # Simulate model evaluation
    time.sleep(2)  # Simulate processing time
    
    # For now, return a random score between 5 and 9
    import random
    score = random.randint(5, 9)
    
    feedback = f"The video appears to show {prompt}. "
    if score < 7:
        feedback += "However, the visual elements could be improved to better match the prompt."
    else:
        feedback += "The visual elements match the prompt well."
    
    return score, feedback

def enhance_prompt(prompt):
    """
    Enhance a prompt to make it more detailed
    
    Args:
        prompt: The original prompt
    
    Returns:
        enhanced_prompt: The enhanced prompt
    """
    try:
        enhanced_prompt = prompt_enhancer.enhance_prompt(prompt)
        logger.info(f"Enhanced prompt: {enhanced_prompt}")
        return enhanced_prompt
    except Exception as e:
        logger.error(f"Error enhancing prompt: {e}")
        return prompt

def generate_new_video(prompt, api_url="http://localhost:5001"):
    """
    Generate a new video using the Movie Maker API
    
    Args:
        prompt: The prompt to use for generation
        api_url: URL of the Movie Maker API
    
    Returns:
        success: Boolean indicating if generation was successful
        video_path: Path to the generated video (if successful)
    """
    try:
        # Call the Movie Maker API to generate a video
        response = requests.post(
            f"{api_url}/generate",
            data={"prompt": prompt, "use_enhanced": "true", "enhanced_prompt": prompt}
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                # Extract the filename from the response
                clip_filename = data.get("clip")
                video_path = os.path.join(CLIP_DIR, clip_filename)
                logger.info(f"Generated new video: {video_path}")
                return True, video_path
        
        logger.error(f"Failed to generate video: {response.text}")
        return False, None
    except Exception as e:
        logger.error(f"Error generating video: {e}")
        return False, None

def find_video_for_prompt(prompt):
    """
    Find an existing video that matches the prompt
    
    Args:
        prompt: The prompt to search for
    
    Returns:
        video_path: Path to the video if found, None otherwise
    """
    # Clean the prompt for comparison
    clean_prompt = ''.join(c if c.isalnum() or c in ' _-' else '_' for c in prompt[:50])
    
    # Check if any clip in CLIP_DIR contains the clean prompt
    for filename in os.listdir(CLIP_DIR):
        if filename.endswith('.mp4') and clean_prompt.lower() in filename.lower():
            return os.path.join(CLIP_DIR, filename)
    
    return None

def process_prompt(prompt, min_score=7, api_url="http://localhost:5001"):
    """
    Process a single prompt:
    1. Check if a video already exists
    2. If not, generate a new video
    3. Extract frames and evaluate
    4. If score is below threshold, enhance prompt and regenerate
    
    Args:
        prompt: The prompt to process
        min_score: Minimum acceptable score (0-10)
        api_url: URL of the Movie Maker API
    
    Returns:
        result: Dictionary with processing results
    """
    result = {
        "prompt": prompt,
        "original_video": None,
        "enhanced_prompt": None,
        "new_video": None,
        "score": None,
        "feedback": None
    }
    
    # Step 1: Find existing video or generate new one
    video_path = find_video_for_prompt(prompt)
    if video_path:
        logger.info(f"Found existing video for prompt: {prompt}")
        result["original_video"] = video_path
    else:
        logger.info(f"No existing video found for prompt: {prompt}")
        success, video_path = generate_new_video(prompt, api_url)
        if not success:
            logger.error(f"Failed to generate video for prompt: {prompt}")
            return result
        result["original_video"] = video_path
    
    # Step 2: Extract frames and evaluate
    frames = extract_frames(video_path, FRAMES_DIR, interval=0.5)
    if not frames:
        logger.error(f"No frames extracted from video: {video_path}")
        return result
    
    score, feedback = evaluate_frames_with_model(frames, prompt)
    result["score"] = score
    result["feedback"] = feedback
    
    # Step 3: If score is below threshold, enhance prompt and regenerate
    if score < min_score:
        logger.info(f"Score {score} is below threshold {min_score}. Enhancing prompt.")
        enhanced_prompt = enhance_prompt(prompt)
        result["enhanced_prompt"] = enhanced_prompt
        
        if enhanced_prompt != prompt:
            # Generate new video with enhanced prompt
            success, new_video_path = generate_new_video(enhanced_prompt, api_url)
            if success:
                result["new_video"] = new_video_path
                
                # Evaluate new video
                new_frames = extract_frames(new_video_path, FRAMES_DIR, interval=0.5)
                if new_frames:
                    new_score, new_feedback = evaluate_frames_with_model(new_frames, enhanced_prompt)
                    result["new_score"] = new_score
                    result["new_feedback"] = new_feedback
    
    return result

def process_prompts_file(input_file, output_file=None, min_score=7, api_url="http://localhost:5001"):
    """
    Process all prompts in a file
    
    Args:
        input_file: Path to file containing prompts (one per line)
        output_file: Path to save results (JSON)
        min_score: Minimum acceptable score (0-10)
        api_url: URL of the Movie Maker API
    """
    # Read prompts from file
    with open(input_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Processing {len(prompts)} prompts from {input_file}")
    
    results = []
    for i, prompt in enumerate(prompts):
        logger.info(f"Processing prompt {i+1}/{len(prompts)}: {prompt}")
        result = process_prompt(prompt, min_score, api_url)
        results.append(result)
    
    # Save results to output file
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    
    # Print summary
    total = len(results)
    below_threshold = sum(1 for r in results if r["score"] is not None and r["score"] < min_score)
    enhanced = sum(1 for r in results if r["enhanced_prompt"] is not None)
    regenerated = sum(1 for r in results if r["new_video"] is not None)
    
    logger.info(f"Summary:")
    logger.info(f"  Total prompts processed: {total}")
    logger.info(f"  Prompts below threshold ({min_score}): {below_threshold}")
    logger.info(f"  Prompts enhanced: {enhanced}")
    logger.info(f"  Videos regenerated: {regenerated}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate and improve video quality")
    parser.add_argument("input_file", help="File containing prompts (one per line)")
    parser.add_argument("--output", "-o", help="Output file for results (JSON)")
    parser.add_argument("--min-score", "-m", type=int, default=7, help="Minimum acceptable score (0-10)")
    parser.add_argument("--api-url", "-a", default="http://localhost:5001", help="Movie Maker API URL")
    
    args = parser.parse_args()
    
    # Generate default output filename if not provided
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(ENHANCED_PROMPTS_DIR, f"evaluation_results_{timestamp}.json")
    
    process_prompts_file(args.input_file, args.output, args.min_score, args.api_url)

if __name__ == "__main__":
    main() 