#!/usr/bin/env python3
"""
View Analysis Results

This script provides a visual interface to view the analysis results for a video,
showing the original video frames alongside the analysis.
"""

import os
import sys
import argparse
import json
import glob
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import textwrap

# Define paths
CLIPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clips")

def list_analyzed_videos():
    """List all videos that have been analyzed."""
    analysis_files = glob.glob(os.path.join(CLIPS_DIR, "*_analysis.txt"))
    videos = []
    
    for analysis_file in analysis_files:
        video_path = analysis_file.rsplit('_analysis.txt', 1)[0] + '.mp4'
        if os.path.exists(video_path):
            videos.append(video_path)
    
    videos.sort(key=os.path.getmtime, reverse=True)
    
    print("\n=== Analyzed Videos ===")
    for i, video in enumerate(videos, 1):
        basename = os.path.basename(video)
        print(f"{i}. {basename}")
    
    return videos

def extract_frames(video_path, num_frames=5):
    """Extract frames from a video for display."""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps
    
    # Calculate frame indices to extract
    indices = [int(i * frame_count / num_frames) for i in range(num_frames)]
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert from BGR to RGB for matplotlib
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    return frames

def load_analysis_data(video_path):
    """Load analysis data for a video."""
    base_path = video_path.rsplit('.', 1)[0]
    
    # Load frame-by-frame analysis
    analysis_file = f"{base_path}_analysis.txt"
    frame_descriptions = []
    overall_analysis = ""
    
    if os.path.exists(analysis_file):
        with open(analysis_file, 'r') as f:
            content = f.read()
            
            # Extract frame descriptions
            if "=== Frame-by-Frame Analysis ===" in content:
                frame_section = content.split("=== Frame-by-Frame Analysis ===")[1]
                if "=== Overall Analysis ===" in frame_section:
                    frame_section = frame_section.split("=== Overall Analysis ===")[0]
                
                for line in frame_section.strip().split('\n'):
                    if line.startswith("Frame "):
                        parts = line.split(": ", 1)
                        if len(parts) > 1:
                            frame_descriptions.append(parts[1])
            
            # Extract overall analysis
            if "=== Overall Analysis ===" in content:
                overall_analysis = content.split("=== Overall Analysis ===")[1].strip()
    
    # Load iteration data
    iteration_files = sorted(glob.glob(f"{base_path}_iteration_*.json"))
    iterations = []
    
    for iter_file in iteration_files:
        try:
            with open(iter_file, 'r') as f:
                data = json.load(f)
                iterations.append(data)
        except Exception as e:
            print(f"Error reading iteration file {iter_file}: {e}")
    
    # Load final analysis if available
    final_analysis_file = f"{base_path}_final_analysis.txt"
    final_analysis = ""
    
    if os.path.exists(final_analysis_file):
        with open(final_analysis_file, 'r') as f:
            final_analysis = f.read()
    
    return {
        "frame_descriptions": frame_descriptions,
        "overall_analysis": overall_analysis,
        "iterations": iterations,
        "final_analysis": final_analysis
    }

def display_results(video_path):
    """Display the analysis results for a video."""
    # Extract frames from the video
    frames = extract_frames(video_path)
    
    # Load analysis data
    analysis_data = load_analysis_data(video_path)
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(len(frames), 2, figure=fig)
    
    # Display frames and descriptions
    for i, frame in enumerate(frames):
        # Frame
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.imshow(frame)
        ax1.set_title(f"Frame {i+1}")
        ax1.axis('off')
        
        # Description
        ax2 = fig.add_subplot(gs[i, 1])
        if i < len(analysis_data["frame_descriptions"]):
            description = analysis_data["frame_descriptions"][i]
            wrapped_text = '\n'.join(textwrap.wrap(description, width=70))
            ax2.text(0.05, 0.5, wrapped_text, fontsize=10, va='center')
        ax2.axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"Analysis Results for {os.path.basename(video_path)}", fontsize=16)
    plt.subplots_adjust(top=0.95)
    
    # Show the plot
    plt.show()
    
    # Display overall analysis and improvement iterations in the console
    print("\n=== Overall Analysis ===")
    print(analysis_data["overall_analysis"])
    
    if analysis_data["iterations"]:
        print("\n=== Improvement Iterations ===")
        for i, iteration in enumerate(analysis_data["iterations"], 1):
            print(f"\nIteration {i}:")
            print(f"Target Description: {iteration.get('target_description', 'N/A')}")
            print(f"Final Prompt: {iteration.get('final_prompt', 'N/A')}")
    
    if analysis_data["final_analysis"]:
        print("\n=== Final Analysis ===")
        print(analysis_data["final_analysis"])

def main():
    parser = argparse.ArgumentParser(description="View analysis results for videos")
    parser.add_argument("--list", action="store_true", help="List analyzed videos")
    parser.add_argument("--view", type=str, help="View results for a specific video (provide path or number from list)")
    parser.add_argument("--latest", action="store_true", help="View results for the most recently analyzed video")
    
    args = parser.parse_args()
    
    # List all analyzed videos if requested or if no arguments provided
    videos = list_analyzed_videos()
    if not videos:
        print("No analyzed videos found.")
        return
    
    if args.list or (not args.view and not args.latest):
        return
    
    # Handle the latest video option
    if args.latest:
        video_path = videos[0]
        print(f"Selected latest analyzed video: {os.path.basename(video_path)}")
        display_results(video_path)
        return
    
    # Handle view option
    if args.view:
        # Check if the input is a number
        try:
            video_index = int(args.view) - 1
            if 0 <= video_index < len(videos):
                video_path = videos[video_index]
            else:
                print(f"Invalid video number. Please choose between 1 and {len(videos)}.")
                return
        except ValueError:
            # Assume it's a path
            video_path = args.view
            if not os.path.exists(video_path):
                # Try prepending the clips directory
                alt_path = os.path.join(CLIPS_DIR, os.path.basename(video_path))
                if os.path.exists(alt_path):
                    video_path = alt_path
                else:
                    print(f"Video not found: {video_path}")
                    return
        
        display_results(video_path)

if __name__ == "__main__":
    main() 