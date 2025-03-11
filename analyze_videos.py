#!/usr/bin/env python3
"""
Video Analysis Script

This script provides a user-friendly interface to analyze videos using the vision_analysis_poc.py
functionality. It allows for analyzing multiple videos, reviewing results, and generating improved prompts.
"""

import os
import sys
import argparse
import json
import glob
from pathlib import Path
import logging
import subprocess
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("analyze_videos.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("analyze_videos")

# Define paths
CLIPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clips")

def list_available_videos():
    """List all available videos in the clips directory."""
    videos = glob.glob(os.path.join(CLIPS_DIR, "*.mp4"))
    videos.sort(key=os.path.getmtime, reverse=True)
    
    print("\n=== Available Videos ===")
    for i, video in enumerate(videos, 1):
        basename = os.path.basename(video)
        size_mb = os.path.getsize(video) / (1024 * 1024)
        print(f"{i}. {basename} ({size_mb:.2f} MB)")
    
    return videos

def analyze_video(video_path, max_iterations=3):
    """Run the vision analysis on a specific video."""
    print(f"\n=== Analyzing Video: {os.path.basename(video_path)} ===")
    
    # Check if the video exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return False
    
    # Run the vision analysis script
    cmd = [
        "python", "vision_analysis_poc.py",
        "--video", video_path,
        "--max-iterations", str(max_iterations)
    ]
    
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Stream the output in real-time
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        
        process.wait()
        return process.returncode == 0
    except Exception as e:
        print(f"Error running analysis: {e}")
        return False

def display_analysis_results(video_path):
    """Display the analysis results for a video."""
    base_path = video_path.rsplit('.', 1)[0]
    
    # Check for analysis file
    analysis_file = f"{base_path}_analysis.txt"
    if os.path.exists(analysis_file):
        print("\n=== Frame-by-Frame Analysis ===")
        with open(analysis_file, 'r') as f:
            print(f.read())
    else:
        print(f"No analysis file found for {os.path.basename(video_path)}")
    
    # Check for iteration files
    iteration_files = sorted(glob.glob(f"{base_path}_iteration_*.json"))
    if iteration_files:
        print("\n=== Improvement Iterations ===")
        for i, iter_file in enumerate(iteration_files, 1):
            try:
                with open(iter_file, 'r') as f:
                    data = json.load(f)
                    print(f"\nIteration {i}:")
                    print(f"Target Description: {data.get('target_description', 'N/A')}")
                    print(f"Final Prompt: {data.get('final_prompt', 'N/A')}")
            except Exception as e:
                print(f"Error reading iteration file {iter_file}: {e}")
    else:
        print(f"No iteration files found for {os.path.basename(video_path)}")
    
    # Check for final analysis
    final_analysis_file = f"{base_path}_final_analysis.txt"
    if os.path.exists(final_analysis_file):
        print("\n=== Final Analysis ===")
        with open(final_analysis_file, 'r') as f:
            print(f.read())

def main():
    parser = argparse.ArgumentParser(description="Analyze videos using vision models")
    parser.add_argument("--list", action="store_true", help="List available videos")
    parser.add_argument("--analyze", type=str, help="Analyze a specific video (provide path or number from list)")
    parser.add_argument("--results", type=str, help="Display results for a specific video (provide path or number from list)")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum number of improvement iterations")
    parser.add_argument("--latest", action="store_true", help="Analyze the most recent video")
    
    args = parser.parse_args()
    
    # List all videos if requested or if no arguments provided
    videos = list_available_videos()
    if not videos:
        print("No videos found in the clips directory.")
        return
    
    if args.list or (not args.analyze and not args.results and not args.latest):
        return
    
    # Handle the latest video option
    if args.latest:
        video_path = videos[0]
        print(f"Selected latest video: {os.path.basename(video_path)}")
        success = analyze_video(video_path, args.max_iterations)
        if success:
            display_analysis_results(video_path)
        return
    
    # Handle analyze option
    if args.analyze:
        # Check if the input is a number
        try:
            video_index = int(args.analyze) - 1
            if 0 <= video_index < len(videos):
                video_path = videos[video_index]
            else:
                print(f"Invalid video number. Please choose between 1 and {len(videos)}.")
                return
        except ValueError:
            # Assume it's a path
            video_path = args.analyze
            if not os.path.exists(video_path):
                # Try prepending the clips directory
                alt_path = os.path.join(CLIPS_DIR, os.path.basename(video_path))
                if os.path.exists(alt_path):
                    video_path = alt_path
                else:
                    print(f"Video not found: {video_path}")
                    return
        
        success = analyze_video(video_path, args.max_iterations)
        if success:
            display_analysis_results(video_path)
    
    # Handle results option
    if args.results:
        # Check if the input is a number
        try:
            video_index = int(args.results) - 1
            if 0 <= video_index < len(videos):
                video_path = videos[video_index]
            else:
                print(f"Invalid video number. Please choose between 1 and {len(videos)}.")
                return
        except ValueError:
            # Assume it's a path
            video_path = args.results
            if not os.path.exists(video_path):
                # Try prepending the clips directory
                alt_path = os.path.join(CLIPS_DIR, os.path.basename(video_path))
                if os.path.exists(alt_path):
                    video_path = alt_path
                else:
                    print(f"Video not found: {video_path}")
                    return
        
        display_analysis_results(video_path)

if __name__ == "__main__":
    main() 