#!/usr/bin/env python3
"""
Video Generation Script

This script generates videos using Stable Video Diffusion models
with specific prompts and parameters for optimal results.
"""

import torch
from diffusers import StableVideoDiffusionPipeline
from PIL import Image
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_video(
    prompt,
    output_dir="clips",
    num_frames=32,  # Increased for smoother motion
    fps=16,         # Increased for better temporal coherence
    motion_bucket_id=200,  # Higher value for more dynamic motion
    noise_aug_strength=0.05,  # Reduced for cleaner output
    test_frame_only=False
):
    """
    Generate a video using Stable Video Diffusion.
    
    Args:
        prompt (str): Text description of the desired video
        output_dir (str): Directory to save the output
        num_frames (int): Number of frames to generate
        fps (int): Frames per second
        motion_bucket_id (int): Controls the amount of motion (0-1000)
        noise_aug_strength (float): Controls the noise augmentation (0-1)
        test_frame_only (bool): If True, only generate the conditioning image for testing
    """
    logger.info(f"Generating {'test frame' if test_frame_only else 'video'} for prompt: {prompt}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Generate conditioning image using the text prompt
        from diffusers import DiffusionPipeline
        image_pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16"
        )
        image_pipe.enable_model_cpu_offload()
        
        # Generate the conditioning image
        logger.info("Generating conditioning image...")
        image = image_pipe(
            prompt=prompt,
            num_inference_steps=25,
            guidance_scale=7.5
        ).images[0]
        
        # If we only want to test the conditioning image, save it and return
        if test_frame_only:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            test_frame_path = os.path.join(output_dir, f"{timestamp}_{prompt[:50]}_test_frame.png")
            image.save(test_frame_path)
            logger.info(f"Test frame saved to: {test_frame_path}")
            return test_frame_path
            
        # Initialize the pipeline for video generation
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16,
            variant="fp16"
        )
        pipe.enable_model_cpu_offload()
        
        # Generate the video
        logger.info("Generating video frames...")
        frames = pipe(
            image,
            num_frames=num_frames,
            fps=fps,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength
        ).frames[0]
        
        # Save the video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"{timestamp}_{prompt[:50]}.mp4")
        
        # Convert frames to video using moviepy
        from moviepy.editor import ImageSequenceClip
        import numpy as np
        
        # Convert PIL images to numpy array
        frame_array = [np.array(frame) for frame in frames]
        clip = ImageSequenceClip(frame_array, fps=fps)
        clip.write_videofile(output_path, codec='libx264', fps=fps)
        
        logger.info(f"Video saved to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error generating video: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    # Enhanced prompt for better scene composition and motion
    prompt = (
        "A cinematic aerial shot of a luxury car driving on a scenic mountain highway, "
        "captured from a dynamic following angle. The sleek vehicle navigates sweeping curves "
        "through majestic mountain peaks, with dramatic cliffs and lush valleys visible in the "
        "background. Golden hour sunlight bathes the scene, casting long shadows across the "
        "winding road. The movement is smooth and continuous, emphasizing the car's journey "
        "through the breathtaking landscape. Photorealistic, 8K resolution, professional "
        "cinematography, motion blur, depth of field."
    )
    
    # Generate the video with optimized parameters
    video_path = generate_video(
        prompt=prompt,
        num_frames=32,     # More frames for smoother motion
        fps=16,           # Higher fps for better temporal quality
        motion_bucket_id=200,  # Increased motion for dynamic driving scene
        noise_aug_strength=0.05  # Reduced noise for cleaner output
    )
    
    if video_path:
        logger.info("Video generation completed successfully!")
    else:
        logger.error("Video generation failed.")

if __name__ == "__main__":
    main() 