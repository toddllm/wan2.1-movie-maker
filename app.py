#!/usr/bin/env python3
"""
Movie Maker - A web interface for generating and combining videos.
"""

import os
import sys
import time
import uuid
import logging
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("movie_maker.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("movie_maker")

# Create Flask app
app = Flask(__name__)

# Ensure directories exist
CLIP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clips")
MOVIE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "movies")
os.makedirs(CLIP_DIR, exist_ok=True)
os.makedirs(MOVIE_DIR, exist_ok=True)

# Set the WAN2.1 model directory
WAN_REPO_PATH = os.path.join(os.path.expanduser("~"), "development", "wan-video", "wan2.1", "wan_repo")
MODEL_DIR = os.path.join(os.path.expanduser("~"), "development", "wan-video", "wan2.1", "models", "Wan2.1-T2V-1.3B")
VENV_PATH = os.path.join(os.path.expanduser("~"), "development", "wan-video", "wan2.1", "venv")

def activate_venv():
    """Activate the virtual environment for the current process."""
    venv_bin = os.path.join(VENV_PATH, "bin")
    venv_activate = os.path.join(venv_bin, "activate_this.py")
    
    if not os.path.exists(venv_activate):
        # Create activate_this.py if it doesn't exist (older venvs might not have it)
        with open(venv_activate, "w") as f:
            f.write('''
# This file is used to activate a Python virtual environment from within a Python script
import os
import sys
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["VIRTUAL_ENV"] = base
prev_sys_path = list(sys.path)
site_packages = os.path.join(base, "lib", "python{}.{}".format(sys.version_info[0], sys.version_info[1]), "site-packages")
sys.path.insert(0, site_packages)
sys.prefix = base
# Add the virtual environments libraries to the host python import mechanism
prev_length = len(sys.path)
sys.real_prefix = sys.prefix
sys.prefix = base
# Move the added items to the front of the path
new_sys_path = []
for item in sys.path[prev_length:]:
    new_sys_path.append(item)
sys.path[prev_length:] = new_sys_path
''')
    
    # Activate the environment
    with open(venv_activate) as f:
        exec(f.read(), {'__file__': venv_activate})
    
    # Add venv bin to PATH for subprocess calls
    os.environ["PATH"] = venv_bin + os.pathsep + os.environ["PATH"]
    
    logger.info(f"Activated virtual environment: {VENV_PATH}")

def generate_video(prompt, output_path, frame_count=160, size="832*480"):
    """Generate a video using the WAN2.1 model by calling the script directly."""
    try:
        # Construct the command to run the generate.py script
        wan_dir = os.path.join(os.path.expanduser("~"), "development", "wan-video", "Wan2.1")
        model_dir = os.path.join(os.path.expanduser("~"), "development", "wan-video", "wan2.1", "models", "Wan2.1-T2V-1.3B")
        venv_python = os.path.join(os.path.expanduser("~"), "development", "wan-video", "wan2.1", "venv", "bin", "python")
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Construct the command using the venv Python
        cmd = [
            venv_python, "generate.py",
            "--task", "t2v-1.3B",
            "--size", size,
            "--ckpt_dir", model_dir,
            "--offload_model=True",
            "--frame_num", str(frame_count),
            "--prompt", prompt
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Set PYTHONPATH to include the wan_repo directory
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(os.path.expanduser("~"), "development", "wan-video", "wan2.1", "wan_repo") + ":" + env.get("PYTHONPATH", "")
        
        # Run the command in the Wan2.1 directory
        process = subprocess.Popen(
            cmd,
            cwd=wan_dir,
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
        for filename in os.listdir(wan_dir):
            if filename.startswith("t2v-1.3B") and filename.endswith(".mp4") and timestamp in filename:
                generated_files.append(filename)
        
        # Sort by modification time (newest first)
        generated_files.sort(key=lambda x: os.path.getmtime(os.path.join(wan_dir, x)), reverse=True)
        
        if not generated_files:
            raise Exception("No video file was generated")
        
        # Get the latest generated file
        latest_file = os.path.join(wan_dir, generated_files[0])
        logger.info(f"Found generated video: {latest_file}")
        
        # Copy the file to the desired output location
        import shutil
        shutil.copy2(latest_file, output_path)
        logger.info(f"Copied video to: {output_path}")
        
        return True, output_path
    except Exception as e:
        logger.error(f"Error generating video: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, str(e)

def combine_videos(clip_paths, output_path):
    """Combine multiple video clips into one movie using FFmpeg."""
    try:
        # Create a temporary file listing all input clips
        list_file = "temp_file_list.txt"
        with open(list_file, "w") as f:
            for clip_path in clip_paths:
                f.write(f"file '{clip_path}'\n")
        
        # Use FFmpeg to concatenate videos
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
            "-i", list_file, "-c", "copy", output_path
        ]
        
        logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        # Remove temporary file
        os.remove(list_file)
        
        return True, output_path
    except Exception as e:
        logger.error(f"Error combining videos: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, str(e)

@app.route('/')
def index():
    """Homepage with interface for generating and combining videos."""
    # Get list of all clips
    clips = []
    for filename in os.listdir(CLIP_DIR):
        if filename.endswith(".mp4"):
            file_path = os.path.join(CLIP_DIR, filename)
            creation_time = os.path.getctime(file_path)
            creation_datetime = datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')
            
            # Extract prompt from filename if available
            parts = filename.split("_", 1)
            prompt = parts[1].replace(".mp4", "").replace("_", " ") if len(parts) > 1 else "Unknown"
            
            clips.append({
                "filename": filename,
                "path": file_path,
                "created": creation_datetime,
                "prompt": prompt
            })
    
    # Get list of all movies
    movies = []
    for filename in os.listdir(MOVIE_DIR):
        if filename.endswith(".mp4"):
            file_path = os.path.join(MOVIE_DIR, filename)
            creation_time = os.path.getctime(file_path)
            creation_datetime = datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')
            
            movies.append({
                "filename": filename,
                "path": file_path,
                "created": creation_datetime
            })
    
    # Sort by creation time (newest first)
    clips.sort(key=lambda x: x["created"], reverse=True)
    movies.sort(key=lambda x: x["created"], reverse=True)
    
    return render_template('index.html', clips=clips, movies=movies)

@app.route('/generate', methods=['POST'])
def generate():
    """Generate a new video clip."""
    prompt = request.form.get('prompt')
    
    if not prompt:
        return jsonify({"status": "error", "message": "Prompt is required"})
    
    # Create a sanitized filename from the prompt
    safe_prompt = prompt.replace(" ", "_").replace("/", "_").replace("\\", "_")[:50]
    clip_filename = f"{int(time.time())}_{safe_prompt}.mp4"
    clip_path = os.path.join(CLIP_DIR, clip_filename)
    
    # Generate the video
    success, message = generate_video(prompt, clip_path)
    
    if success:
        return jsonify({
            "status": "success", 
            "message": "Video generated successfully", 
            "clip": clip_filename
        })
    else:
        return jsonify({"status": "error", "message": f"Error generating video: {message}"})

@app.route('/combine', methods=['POST'])
def combine():
    """Combine multiple video clips into one movie."""
    # Get selected clips
    selected_clips = request.form.getlist('clips')
    movie_title = request.form.get('movie_title', 'Untitled')
    
    if not selected_clips:
        return jsonify({"status": "error", "message": "No clips selected"})
    
    # Create a sanitized filename for the movie
    safe_title = movie_title.replace(" ", "_").replace("/", "_").replace("\\", "_")[:50]
    movie_filename = f"{int(time.time())}_{safe_title}.mp4"
    movie_path = os.path.join(MOVIE_DIR, movie_filename)
    
    # Get full paths for selected clips
    clip_paths = [os.path.join(CLIP_DIR, clip) for clip in selected_clips]
    
    # Combine the videos
    success, message = combine_videos(clip_paths, movie_path)
    
    if success:
        return jsonify({
            "status": "success", 
            "message": "Movie created successfully", 
            "movie": movie_filename
        })
    else:
        return jsonify({"status": "error", "message": f"Error creating movie: {message}"})

@app.route('/clips/<filename>')
def serve_clip(filename):
    """Serve a video clip."""
    return send_from_directory(CLIP_DIR, filename)

@app.route('/movies/<filename>')
def serve_movie(filename):
    """Serve a movie."""
    return send_from_directory(MOVIE_DIR, filename)

@app.route('/delete_clip/<filename>', methods=['POST'])
def delete_clip(filename):
    """Delete a video clip."""
    try:
        clip_path = os.path.join(CLIP_DIR, filename)
        if os.path.exists(clip_path):
            os.remove(clip_path)
            return jsonify({"status": "success", "message": f"Clip {filename} deleted"})
        else:
            return jsonify({"status": "error", "message": f"Clip {filename} not found"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error deleting clip: {str(e)}"})

@app.route('/delete_movie/<filename>', methods=['POST'])
def delete_movie(filename):
    """Delete a movie."""
    try:
        movie_path = os.path.join(MOVIE_DIR, filename)
        if os.path.exists(movie_path):
            os.remove(movie_path)
            return jsonify({"status": "success", "message": f"Movie {filename} deleted"})
        else:
            return jsonify({"status": "error", "message": f"Movie {filename} not found"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error deleting movie: {str(e)}"})

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Movie Maker Web Interface')
    parser.add_argument('--host', default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()
    
    print(f"Starting Movie Maker on http://{args.host}:{args.port}...")
    app.run(host=args.host, port=args.port, debug=args.debug) 