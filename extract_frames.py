#!/usr/bin/env python3
"""
Extract frames from videos to use as poster images.

This script extracts a frame from each video file in the hdmy5movie_videos directory
and saves it as a poster image in the hdmy5movie_posters directory.

It can be run in two modes:
1. One-time mode: Process all videos and exit
2. Monitor mode: Run continuously and process new videos as they appear

Usage:
    # One-time mode
    python3 extract_frames.py
    
    # Monitor mode
    python3 extract_frames.py --monitor --interval 30
    
    # Specify a custom log file
    python3 extract_frames.py --log-file /path/to/logfile.log

The script will:
1. Scan all video directories for MP4 files
2. Extract a frame from each video at the 1-second mark
3. Save the frame as a JPEG image in the corresponding poster directory
4. Skip videos that already have poster images
5. Log all activities to the console and a log file (if specified)

The log file is automatically rotated when it reaches 5MB in size,
and up to 3 backup files are kept.
"""

import os
import subprocess
import sys
import time
import argparse
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import datetime

# Set up logging
def setup_logging(log_file=None):
    """
    Set up logging with rotation capabilities.
    
    Args:
        log_file (str, optional): Path to the log file. If None, logging is only to console.
        
    Returns:
        logging.Logger: Configured logger object.
    """
    logger = logging.getLogger('hdmy5movie_poster')
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if log file is specified
    if log_file:
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Create rotating file handler (max 5MB per file, keep 3 backup files)
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def log_message(logger, message):
    """
    Log a message using the provided logger.
    
    Args:
        logger (logging.Logger): Logger object to use for logging.
        message (str): Message to log.
    """
    logger.info(message)

def extract_frame(logger, video_path, output_path):
    """
    Extract a frame from a video file and save it as a poster image.
    
    Args:
        logger (logging.Logger): Logger object to use for logging.
        video_path (str): Path to the video file.
        output_path (str): Path where the extracted frame will be saved.
        
    Returns:
        bool: True if the frame was extracted successfully, False otherwise.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Build the ffmpeg command
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-ss', '00:00:01',  # Extract frame at 1-second mark
        '-frames:v', '1',   # Extract only one frame
        '-q:v', '2',        # High quality (lower values = higher quality)
        '-f', 'image2',     # Output format
        output_path
    ]
    
    # Run the command
    log_message(logger, f"Extracting frame from {video_path} to {output_path}")
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if process.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        log_message(logger, f"Frame extracted successfully to {output_path}")
        return True
    else:
        log_message(logger, f"Frame extraction failed: {process.stderr.decode()}")
        return False

def get_video_files(videos_dir, sections):
    """
    Get all video files in all sections.
    
    Args:
        videos_dir (str): Base directory containing video sections.
        sections (list): List of section names to scan.
        
    Returns:
        dict: Dictionary mapping video paths to information about the video.
    """
    video_files = {}
    
    for section in sections:
        section_dir = os.path.join(videos_dir, section)
        if not os.path.exists(section_dir):
            continue
        
        # Get all MP4 files in the section directory
        mp4_files = [f for f in os.listdir(section_dir) if f.endswith('.mp4')]
        
        for mp4_file in mp4_files:
            video_path = os.path.join(section_dir, mp4_file)
            video_files[video_path] = {
                'section': section,
                'filename': mp4_file
            }
    
    return video_files

def process_videos(logger, base_dir, videos_dir, posters_dir, sections, processed_videos=None):
    """
    Process videos and extract frames.
    
    Args:
        logger (logging.Logger): Logger object to use for logging.
        base_dir (str): Base directory of the project.
        videos_dir (str): Directory containing video files.
        posters_dir (str): Directory where poster images will be saved.
        sections (list): List of section names to process.
        processed_videos (set, optional): Set of video paths that have already been processed.
        
    Returns:
        set: Updated set of processed video paths.
    """
    if processed_videos is None:
        processed_videos = set()
    
    # Get all video files
    log_message(logger, "Scanning for video files...")
    video_files = get_video_files(videos_dir, sections)
    log_message(logger, f"Found {len(video_files)} total video files")
    
    # Process each video file
    new_videos = 0
    successful_extractions = 0
    
    for video_path, video_info in video_files.items():
        # Skip if already processed
        if video_path in processed_videos:
            continue
        
        section = video_info['section']
        mp4_file = video_info['filename']
        
        # Create the poster path
        poster_path = os.path.join(posters_dir, section, mp4_file.replace('.mp4', '.jpg'))
        
        # Check if poster already exists
        if os.path.exists(poster_path) and os.path.getsize(poster_path) > 0:
            log_message(logger, f"Poster already exists for {video_path}, skipping")
            processed_videos.add(video_path)
            continue
        
        # Extract a frame from the video
        if extract_frame(logger, video_path, poster_path):
            successful_extractions += 1
        
        # Mark as processed
        processed_videos.add(video_path)
        new_videos += 1
    
    # Print a summary
    if new_videos > 0:
        log_message(logger, f"Extracted frames from {successful_extractions} of {new_videos} new videos.")
    else:
        log_message(logger, "No new videos found to process.")
    
    return processed_videos

def monitor_mode(logger, base_dir, videos_dir, posters_dir, sections, interval=60):
    """
    Run in monitor mode, continuously checking for new videos.
    
    Args:
        logger (logging.Logger): Logger object to use for logging.
        base_dir (str): Base directory of the project.
        videos_dir (str): Directory containing video files.
        posters_dir (str): Directory where poster images will be saved.
        sections (list): List of section names to process.
        interval (int, optional): Interval in seconds between checks. Defaults to 60.
    """
    log_message(logger, f"Starting monitor mode. Checking for new videos every {interval} seconds.")
    log_message(logger, "Press Ctrl+C to stop.")
    
    # Keep track of processed videos
    processed_videos = set()
    
    try:
        while True:
            # Process videos
            log_message(logger, f"Running check at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            processed_videos = process_videos(logger, base_dir, videos_dir, posters_dir, sections, processed_videos)
            
            # Wait for the next check
            log_message(logger, f"Sleeping for {interval} seconds until next check...")
            time.sleep(interval)
    except KeyboardInterrupt:
        log_message(logger, "Monitor mode stopped.")

def main():
    """
    Main entry point for the script.
    
    Parses command-line arguments, sets up logging, and runs the script
    in either one-time mode or monitor mode.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract frames from videos to use as poster images.")
    parser.add_argument("--monitor", action="store_true", help="Run in monitor mode, continuously checking for new videos.")
    parser.add_argument("--interval", type=int, default=60, help="Interval in seconds between checks in monitor mode (default: 60).")
    parser.add_argument("--log-file", type=str, default="logs/poster_service.log", help="Path to log file (default: logs/poster_service.log).")
    args = parser.parse_args()
    
    # Get the base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    videos_dir = os.path.join(base_dir, 'hdmy5movie_videos')
    posters_dir = os.path.join(base_dir, 'hdmy5movie_posters')
    
    # Set up logging
    log_file = os.path.join(base_dir, args.log_file)
    logger = setup_logging(log_file)
    
    # Create the posters directory if it doesn't exist
    os.makedirs(posters_dir, exist_ok=True)
    
    # Get all sections
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
    
    log_message(logger, f"Script started with arguments: monitor={args.monitor}, interval={args.interval}")
    log_message(logger, f"Base directory: {base_dir}")
    log_message(logger, f"Videos directory: {videos_dir}")
    log_message(logger, f"Posters directory: {posters_dir}")
    log_message(logger, f"Log file: {log_file}")
    
    if args.monitor:
        # Run in monitor mode
        monitor_mode(logger, base_dir, videos_dir, posters_dir, sections, args.interval)
    else:
        # Run in one-time mode
        process_videos(logger, base_dir, videos_dir, posters_dir, sections)
        log_message(logger, f"Poster images are saved in {posters_dir}")

if __name__ == "__main__":
    main() 