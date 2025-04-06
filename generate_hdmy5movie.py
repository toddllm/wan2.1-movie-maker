#!/usr/bin/env python3
import os
import argparse
import time
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

def read_prompts(file_path):
    """Read prompts from the file, one per line."""
    with open(file_path, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts

def create_output_dirs(base_dir):
    """Create necessary output directories."""
    sections = [
        "01_opening_credits",
        "02_prologue",
        "03_act1",
        "04_interlude1",
        "05_act2",
        "06_interlude2",
        "07_act3",
        "08_epilogue",
        "09_credits"
    ]
    
    for section in sections:
        os.makedirs(os.path.join(base_dir, section), exist_ok=True)
    
    return os.path.join(base_dir, "progress.json")

def update_progress(progress_file, total, current, section, prompt):
    """Update the progress file with current generation status."""
    progress = {
        "total": total,
        "current": current,
        "percentage": round((current / total) * 100, 2),
        "last_updated": datetime.now().isoformat(),
        "current_section": section,
        "current_prompt": prompt
    }
    
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)

def assign_prompts_to_sections(prompts):
    """Assign prompts to their respective sections based on content."""
    # This is a simplified version - in a real implementation,
    # you would need more sophisticated logic to determine which
    # prompt belongs to which section
    
    sections = {
        "01_opening_credits": [],
        "02_prologue": [],
        "03_act1": [],
        "04_interlude1": [],
        "05_act2": [],
        "06_interlude2": [],
        "07_act3": [],
        "08_epilogue": [],
        "09_credits": []
    }
    
    # Simple distribution for demonstration
    total = len(prompts)
    section_size = total // 9  # Divide prompts roughly equally
    
    for i, section in enumerate(sections.keys()):
        start = i * section_size
        end = (i + 1) * section_size if i < 8 else total
        sections[section] = prompts[start:end]
    
    return sections

def generate_video(prompt, output_path, model_path=None, seed=None):
    """
    Generate a video using the specified prompt and model.
    Uses the direct_generate.py script for actual video generation.
    """
    print(f"Generating video for prompt: {prompt[:50]}...")
    print(f"Output will be saved to: {output_path}")
    
    # Create a temporary prompt file
    temp_prompt_file = os.path.join(os.path.dirname(output_path), "temp_prompt.txt")
    with open(temp_prompt_file, 'w') as f:
        f.write(prompt)
    
    try:
        # Get directory of output path
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Run the direct_generate.py script
        cmd = [
            "python3", 
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "direct_generate.py"),
            "--input", temp_prompt_file,
            "--duration", "10",  # 10 seconds video
            "--width", "832",    # Default width
            "--height", "480"    # Default height
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error generating video: {result.stderr}")
            # If direct_generate.py fails, create a placeholder file
            with open(output_path, 'w') as f:
                f.write(f"Error generating video for prompt: {prompt}")
            return False
        
        # The direct_generate.py script saves videos to the clips directory
        # We need to find the most recently created video and move it to our output path
        clips_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clips")
        if os.path.exists(clips_dir):
            # Find the most recently created video file
            video_files = [os.path.join(clips_dir, f) for f in os.listdir(clips_dir) if f.endswith('.mp4')]
            if video_files:
                video_files.sort(key=os.path.getmtime, reverse=True)
                latest_video = video_files[0]
                
                # Move the video to our output path
                shutil.move(latest_video, output_path)
                print(f"Moved video from {latest_video} to {output_path}")
                return True
            else:
                print("No video files found in clips directory")
                return False
        else:
            print(f"Clips directory {clips_dir} not found")
            return False
    
    except Exception as e:
        print(f"Exception during video generation: {str(e)}")
        # Create a placeholder file in case of error
        with open(output_path, 'w') as f:
            f.write(f"Error generating video for prompt: {prompt}")
        return False
    
    finally:
        # Clean up temporary prompt file
        if os.path.exists(temp_prompt_file):
            os.remove(temp_prompt_file)

def main():
    parser = argparse.ArgumentParser(description="Generate videos from prompts for HDMY 5 Movie")
    parser.add_argument("--prompts", default="/home/tdeshane/movie_maker/hdmy5movie_prompts.txt", 
                        help="Path to the prompts file")
    parser.add_argument("--output", default="/home/tdeshane/movie_maker/hdmy5movie_videos", 
                        help="Output directory for generated videos")
    parser.add_argument("--model", default=None, 
                        help="Path to the video generation model")
    parser.add_argument("--start", type=int, default=0, 
                        help="Start from this prompt index (0-based)")
    parser.add_argument("--end", type=int, default=None, 
                        help="End at this prompt index (exclusive)")
    parser.add_argument("--seed", type=int, default=None, 
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Read prompts from file
    prompts = read_prompts(args.prompts)
    total_prompts = len(prompts)
    
    print(f"Found {total_prompts} prompts in {args.prompts}")
    
    # Create section directories and progress file
    progress_file = create_output_dirs(args.output)
    
    # Assign prompts to sections
    sections = assign_prompts_to_sections(prompts)
    
    # Determine start and end indices
    start_idx = args.start
    end_idx = args.end if args.end is not None else total_prompts
    
    # Track which section we're in
    current_section = None
    section_prompt_count = 0
    
    # Generate videos for each prompt
    for i, prompt in enumerate(prompts[start_idx:end_idx], start=start_idx):
        # Determine which section this prompt belongs to
        for section, section_prompts in sections.items():
            if prompt in section_prompts:
                current_section = section
                section_prompt_count = section_prompts.index(prompt) + 1
                break
        
        # Create output filename
        output_filename = f"{i+1:03d}_{section_prompt_count:03d}.mp4"
        output_path = os.path.join(args.output, current_section, output_filename)
        
        # Generate the video
        success = generate_video(prompt, output_path, args.model, args.seed)
        
        if success:
            # Update progress
            update_progress(progress_file, total_prompts, i+1, current_section, prompt)
            print(f"Progress: {i+1}/{total_prompts} ({((i+1)/total_prompts)*100:.2f}%)")
        else:
            print(f"Failed to generate video for prompt {i+1}: {prompt[:50]}...")
    
    print("Video generation complete!")

if __name__ == "__main__":
    main() 