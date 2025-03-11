#!/usr/bin/env python3
"""
Vision Analysis POC

This script demonstrates how to use a multimodal model to analyze
frames from generated videos and create a feedback loop for video generation.
"""

import os
import argparse
import logging
import torch
from PIL import Image
import cv2
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM
from pathlib import Path
import json
from difflib import SequenceMatcher
from datetime import datetime

# Import the scoring system
from scoring_system import ScoringSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vision_analysis_poc.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("vision_analysis_poc")

# Define paths
CLIPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clips")
MODEL_HF_REPO = "microsoft/Phi-4-multimodal-instruct"  # Using Phi-4 from HF directly
ACCURACY_THRESHOLD = 0.7  # Minimum similarity score to consider a video accurate

# Initialize the scoring system
scoring_system = ScoringSystem()

# Define threshold for acceptable score
SCORE_THRESHOLD = 0.6  # Minimum overall score to consider analysis acceptable

def calculate_similarity(text1, text2):
    """Calculate similarity between two text descriptions."""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def evaluate_clip_accuracy(analysis, target_description):
    """
    Evaluate how well a clip matches its intended description.
    Returns a score between 0 and 1, and feedback for improvement.
    """
    # Extract key elements from the target description
    target_elements = set(target_description.lower().split())
    
    # Calculate similarity scores
    frame_scores = []
    missing_elements = target_elements.copy()
    
    # Analyze each frame's description
    for frame in analysis:
        if isinstance(frame, str) and frame.startswith("Frame"):
            frame_text = frame.split(": ", 1)[1].lower()
            score = calculate_similarity(frame_text, target_description)
            frame_scores.append(score)
            
            # Track which elements were found
            frame_words = set(frame_text.split())
            missing_elements -= frame_words
    
    # Calculate overall accuracy
    frame_accuracy = sum(frame_scores) / len(frame_scores) if frame_scores else 0
    
    # Generate feedback
    feedback = []
    if missing_elements:
        feedback.append(f"Missing elements: {', '.join(missing_elements)}")
    if frame_accuracy < ACCURACY_THRESHOLD:
        feedback.append(f"Overall accuracy ({frame_accuracy:.2f}) below threshold ({ACCURACY_THRESHOLD})")
        
    return frame_accuracy, feedback

def generate_improved_prompt(original_prompt, feedback):
    """Generate an improved prompt based on feedback."""
    # Extract key elements that were missing
    missing_elements = []
    for item in feedback:
        if item.startswith("Missing elements:"):
            missing_elements = item.replace("Missing elements:", "").strip().split(", ")
    
    # Enhance the prompt with missing elements
    enhanced_prompt = original_prompt
    if missing_elements:
        enhanced_prompt += f" The scene should prominently include: {', '.join(missing_elements)}."
    
    return enhanced_prompt

def analyze_and_improve_clip(model, processor, video_path, target_description=None, iteration=0, max_iterations=3, user_preferences=None):
    """Analyze a clip and generate improvements if needed."""
    logger.info(f"\nIteration {iteration + 1}: Analyzing clip: {video_path}")
    
    # Extract target description from filename if not provided
    if target_description is None:
        target_description = os.path.basename(video_path).split('_', 1)[1].rsplit('.', 1)[0]
    
    # Analyze the video with more frames for better accuracy
    frame_descriptions = []
    frames = extract_frames(video_path, num_frames=5)
    for i, frame in enumerate(frames, 1):
        logger.info(f"Analyzing frame {i}/{len(frames)}")
        description = analyze_frame(model, processor, frame)
        frame_descriptions.append(description)
        logger.info(f"Frame {i}: {description}")
        
        # Release memory after each frame
        torch.cuda.empty_cache()
    
    # Generate overall analysis
    analysis_prompt = f"<|user|>Here are frame-by-frame descriptions of a short video clip. Provide an overall analysis of what's happening in the video:\n\n"
    for i, desc in enumerate(frame_descriptions, 1):
        analysis_prompt += f"Frame {i}: {desc}\n"
    analysis_prompt += "<|end|><|assistant|>"
    
    inputs = processor(
        text=analysis_prompt,
        return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_logits_to_keep=1000,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id
    )
    
    overall_analysis = processor.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"\nOverall Analysis:\n{overall_analysis}")
    
    # Release memory
    torch.cuda.empty_cache()
    
    # Save frame descriptions and overall analysis
    analysis_file = video_path.rsplit('.', 1)[0] + '_analysis.txt'
    with open(analysis_file, 'w') as f:
        f.write("=== Frame-by-Frame Analysis ===\n\n")
        for i, desc in enumerate(frame_descriptions, 1):
            f.write(f"Frame {i}: {desc}\n\n")
        
        f.write("\n=== Overall Analysis ===\n\n")
        f.write(overall_analysis + "\n\n")
    
    # Score the analysis using the scoring system
    frame_desc_text = [desc for desc in frame_descriptions]
    
    # Apply user preferences to scoring if provided
    if user_preferences and 'weights' in user_preferences:
        custom_weights = user_preferences['weights']
        temp_scoring_system = ScoringSystem(weights=custom_weights)
        overall_score, scores, feedback = temp_scoring_system.evaluate_analysis(
            target_description, frame_desc_text
        )
    else:
        overall_score, scores, feedback = scoring_system.evaluate_analysis(
            target_description, frame_desc_text
        )
    
    logger.info(f"\nAnalysis Score: {overall_score:.2f}")
    logger.info(f"Score Breakdown: {scores}")
    logger.info(f"Feedback: {feedback}")
    
    # Save the scoring results
    score_file = video_path.rsplit('.', 1)[0] + '_scores.json'
    with open(score_file, 'w') as f:
        json.dump({
            'overall_score': overall_score,
            'scores': scores,
            'feedback': feedback
        }, f, indent=2)
    
    # Compare with target description and generate improved prompt
    improvement_prompt = f"<|user|>I need to create a video showing: '{target_description}'\n\nThe current video's content is: {overall_analysis}\n\nPlease help me improve it by:\n1. Identifying what elements are missing or need improvement\n2. Creating a detailed, specific prompt for the Wan2.1 text-to-video model that would generate a better video with focus on the car, the winding road, the mountains, and scenic views\n3. Suggesting camera angles and lighting that would enhance the scene<|end|><|assistant|>"
    
    # If user has specific focus areas, add them to the prompt
    if user_preferences and 'focus_areas' in user_preferences:
        focus_areas = ', '.join(user_preferences['focus_areas'])
        improvement_prompt = improvement_prompt.replace(
            "with focus on the car, the winding road, the mountains, and scenic views",
            f"with focus on {focus_areas}"
        )
    
    inputs = processor(
        text=improvement_prompt,
        return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=1000,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_logits_to_keep=1000,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id
    )
    
    improved_prompt = processor.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"\nImproved Prompt:\n{improved_prompt}")
    
    # Release memory
    torch.cuda.empty_cache()
    
    # Extract just the prompt part from the improvement suggestions
    prompt_lines = []
    in_prompt = False
    for line in improved_prompt.split('\n'):
        # Look for lines that might contain the prompt
        if '"' in line and not in_prompt:
            in_prompt = True
            prompt_lines.append(line.strip('"'))
        elif in_prompt and '"' in line:
            prompt_lines.append(line.strip('"'))
            break
        elif in_prompt:
            prompt_lines.append(line.strip('"'))
    
    # If we couldn't extract a prompt, use a reasonable default
    if not prompt_lines:
        # Look for lines that might follow "prompt:" or similar
        for line in improved_prompt.split('\n'):
            if "prompt:" in line.lower() or "generate:" in line.lower():
                prompt_lines = [line.split(":", 1)[1].strip()]
                break
    
    final_prompt = ' '.join(prompt_lines) if prompt_lines else target_description
    
    # Ensure the final prompt isn't too long
    if len(final_prompt) > 200:
        final_prompt = final_prompt[:200]
    
    logger.info(f"\nExtracted final prompt: {final_prompt}")
    
    # Save improvement suggestions to a file
    iteration_file = video_path.rsplit('.', 1)[0] + f'_iteration_{iteration + 1}.json'
    with open(iteration_file, 'w') as f:
        json.dump({
            'original_descriptions': frame_descriptions,
            'overall_analysis': overall_analysis,
            'target_description': target_description,
            'improvement_analysis': improved_prompt,
            'final_prompt': final_prompt,
            'iteration': iteration + 1,
            'scores': scores,
            'feedback': feedback,
            'overall_score': overall_score
        }, f, indent=2)
    
    # If we haven't hit max iterations and the score is below threshold, generate a new video with the improved prompt
    should_regenerate = (
        iteration < max_iterations - 1 and 
        (overall_score < SCORE_THRESHOLD or 
         (user_preferences and 'min_score' in user_preferences and overall_score < user_preferences['min_score']))
    )
    
    if should_regenerate:
        # Import the proper video generation function from direct_generate
        from direct_generate import generate_video as wan_generate_video, wait_for_gpu
        
        # Wait for GPU to be free (if it's in use)
        logger.info("Checking GPU availability...")
        wait_for_gpu()
        
        # First generate a test frame if needed (skipping for Wan2.1 which doesn't support test frames)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{timestamp}_{final_prompt[:50]}.mp4"
        output_path = os.path.join("clips", output_filename)
        
        logger.info(f"Generating video using Wan2.1 with prompt: {final_prompt}")
        success, result = wan_generate_video(
            prompt=final_prompt,
            output_path=output_path,
            frame_count=160,  
            size="832*480"
        )
        
        if success:
            logger.info(f"New video generated: {result}")
            # Recursively analyze and improve with the new video
            return analyze_and_improve_clip(
                model, processor, result, 
                target_description, iteration + 1, max_iterations,
                user_preferences
            )
        else:
            logger.error(f"Failed to generate video: {result}")
            # Try with a more direct prompt
            direct_prompt = f"A car driving on a winding mountain highway with scenic views. The car is prominently featured on the road, with mountains in the background. Golden hour lighting creating dramatic shadows."
            logger.info(f"Using direct prompt: {direct_prompt}")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{timestamp}_{direct_prompt[:50]}.mp4"
            output_path = os.path.join("clips", output_filename)
            
            success, result = wan_generate_video(
                prompt=direct_prompt,
                output_path=output_path,
                frame_count=160,
                size="832*480"
            )
            
            if success:
                logger.info(f"New video generated with direct prompt: {result}")
                return analyze_and_improve_clip(
                    model, processor, result, 
                    target_description, iteration + 1, max_iterations,
                    user_preferences
                )
    
    return frame_descriptions, final_prompt, overall_score, scores, feedback

def load_model():
    """Load the vision model and processor."""
    logger.info(f"Loading model from {MODEL_HF_REPO}")
    
    try:
        # Load processor and model with correct configuration
        processor = AutoProcessor.from_pretrained(
            MODEL_HF_REPO,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_HF_REPO,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            use_flash_attention_2=True
        )
        
        logger.info("Model loaded successfully")
        return processor, model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

def extract_frames(video_path, num_frames=5):
    """Extract frames from a video."""
    logger.info(f"Extracting frames from {video_path}")
    
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps
        
        logger.info(f"Video has {frame_count} frames, {fps} fps, duration: {duration:.2f} seconds")
        
        # Compute frame indices to extract
        indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
        frames = []
        
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            else:
                logger.warning(f"Could not read frame {i}")
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames")
        return frames
    except Exception as e:
        logger.error(f"Error extracting frames: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def analyze_frame(model, processor, image, prompt="Describe what you see in this image with detail."):
    """Analyze a single frame using the model."""
    logger.info("Analyzing frame")
    
    try:
        # Format the prompt with chat format and special image token
        formatted_prompt = f"<|user|>Please describe what you see in this image in detail. <|endoftext10|><|end|><|assistant|>"
        
        # Process the image and text together
        inputs = processor(
            text=formatted_prompt,
            images=image,
            return_tensors="pt"
        ).to(model.device)
        
        # Generate response
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_logits_to_keep=1000,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id
        )
        
        # Decode and clean up the response
        response = processor.decode(outputs[0], skip_special_tokens=True)
        # Remove any repeated prompt text
        response = response.replace(prompt, "").strip()
        return response
    except Exception as e:
        logger.error(f"Error analyzing frame: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return "Error analyzing frame"

def analyze_video(model, processor, video_path, num_frames=3):
    """Analyze a video by extracting and analyzing frames."""
    logger.info(f"Analyzing video: {video_path}")
    
    # Extract frames
    frames = extract_frames(video_path, num_frames)
    logger.info(f"Extracted {len(frames)} frames")
    
    # Analyze each frame
    results = []
    for i, frame in enumerate(frames, 1):
        logger.info(f"Analyzing frame {i}/{len(frames)}")
        result = analyze_frame(model, processor, frame)
        results.append(f"Frame {i}: {result}")
    
    # Log frame-by-frame analysis
    logger.info("\n=== Frame-by-Frame Analysis ===")
    for result in results:
        logger.info(result)
    
    # Generate overall analysis
    logger.info("\nGenerating overall analysis of the video")
    overall_prompt = (
        f"<|system|>You are a helpful assistant that analyzes videos based on frame descriptions. "
        f"Provide a concise analysis focusing on the visual narrative, themes, and connections between frames. "
        f"Do not ask questions or repeat descriptions.<|end|>"
        f"<|user|>Here are frame-by-frame descriptions of a short video clip. "
        f"Provide an overall analysis of what's happening in the video:\n\n{chr(10).join(results)}<|end|>"
        f"<|assistant|>"
    )
    
    # Process the overall analysis
    inputs = processor(text=overall_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=1000,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_logits_to_keep=1000,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id
    )
    
    # Decode and clean up the overall analysis
    overall_analysis = processor.decode(outputs[0], skip_special_tokens=True)
    overall_analysis = overall_analysis.split("<|assistant|>")[-1].strip()
    # Remove any meta-questions or repeated content
    overall_analysis = overall_analysis.split("In your opinion")[0].strip()
    
    # Save the analysis to a file
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_file = os.path.join(os.path.dirname(video_path), f"{base_name}_analysis.txt")
    
    with open(output_file, "w") as f:
        f.write("=== Frame-by-Frame Analysis ===\n")
        for result in results:
            f.write(f"{result}\n")
        f.write("\n=== Overall Analysis ===\n")
        f.write(overall_analysis)
    
    logger.info(f"Analysis saved to {output_file}")
    return results, overall_analysis

def main():
    """Main function to process arguments and run analysis."""
    parser = argparse.ArgumentParser(description="Analyze video clips and improve prompts")
    
    # Add arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", help="Path to the video file to analyze")
    group.add_argument("--latest", action="store_true", help="Use the latest video in the clips directory")
    
    parser.add_argument("--target", default=None, help="Target description to compare against")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum number of improvement iterations")
    parser.add_argument("--min-score", type=float, default=SCORE_THRESHOLD, help="Minimum acceptable score (0.0-1.0)")
    parser.add_argument("--focus", nargs='+', default=None, help="Areas to focus on in the prompt (e.g. 'lighting' 'composition')")
    parser.add_argument("--weights", nargs='+', default=None, help="Custom weights for scoring metrics (format: metric=weight)")
    
    args = parser.parse_args()
    
    # Process user preferences
    user_preferences = {}
    
    if args.min_score:
        user_preferences['min_score'] = args.min_score
    
    if args.focus:
        user_preferences['focus_areas'] = args.focus
    
    if args.weights:
        weights = {}
        for weight_arg in args.weights:
            if '=' in weight_arg:
                metric, value = weight_arg.split('=')
                try:
                    weights[metric] = float(value)
                except ValueError:
                    logger.warning(f"Invalid weight value for {metric}: {value}")
        if weights:
            user_preferences['weights'] = weights
    
    # Load the model
    processor, model = load_model()
    if processor is None or model is None:
        logger.error("Failed to load model. Exiting.")
        return
    
    # Determine which video to analyze
    if args.latest:
        video_files = list(Path(CLIPS_DIR).glob("*.mp4"))
        if not video_files:
            # If no videos exist, generate the first one
            logger.info("No existing videos found. Generating initial video...")
            try:
                # Import the proper video generation function from direct_generate
                from direct_generate import generate_video as wan_generate_video, wait_for_gpu
                
                # Wait for GPU to be free (if it's in use)
                logger.info("Checking GPU availability...")
                wait_for_gpu()
                
                initial_prompt = args.target or "A car driving on a curvy highway scenic view"
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"{timestamp}_{initial_prompt[:50]}.mp4"
                output_path = os.path.join("clips", output_filename)
                
                logger.info(f"Generating initial video using Wan2.1 with prompt: {initial_prompt}")
                success, result = wan_generate_video(
                    prompt=initial_prompt,
                    output_path=output_path,
                    frame_count=160,
                    size="832*480"
                )
                
                if success:
                    video_path = result
                    logger.info(f"Initial video generated: {video_path}")
                else:
                    logger.error(f"Failed to generate initial video: {result}")
                    return
            except ImportError:
                logger.warning("Wan2.1 not available, falling back to generate_video.py")
                from generate_video import generate_video
                video_path = generate_video(
                    prompt=args.target or "A car driving on a curvy highway scenic view",
                    num_frames=24,
                    fps=12,
                    motion_bucket_id=150,
                    noise_aug_strength=0.1
                )
                if not video_path:
                    logger.error("Failed to generate initial video. Exiting.")
                    return
        else:
            video_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            video_path = str(video_files[0])
        logger.info(f"Using latest video: {video_path}")
    elif args.video:
        video_path = args.video
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return
    else:
        logger.error("No video specified. Use --video or --latest")
        return
    
    # Start the analysis and improvement loop
    frame_descriptions, improved_prompt, overall_score, scores, feedback = analyze_and_improve_clip(
        model, processor, video_path, args.target, max_iterations=args.max_iterations,
        user_preferences=user_preferences
    )
    
    # Save final results
    output_path = os.path.splitext(video_path)[0] + "_final_analysis.txt"
    with open(output_path, "w") as f:
        f.write(f"=== Final Analysis (Improved Prompt: {improved_prompt}) ===\n\n")
        f.write(f"Target Description: {args.target or 'Extracted from filename'}\n\n")
        f.write(f"Overall Score: {overall_score:.2f}\n\n")
        f.write("=== Score Breakdown ===\n\n")
        for metric, score in scores.items():
            if metric != "overall_score":
                f.write(f"{metric}: {score:.2f}\n")
        f.write("\n=== Feedback ===\n\n")
        for metric, fb in feedback.items():
            f.write(f"{metric}: {fb}\n")
        f.write("\n=== Frame-by-Frame Analysis ===\n\n")
        for frame_description in frame_descriptions:
            f.write(f"{frame_description}\n\n")
    
    logger.info(f"\nFinal improved prompt: {improved_prompt}")
    logger.info(f"Final analysis saved to {output_path}")

if __name__ == "__main__":
    main() 