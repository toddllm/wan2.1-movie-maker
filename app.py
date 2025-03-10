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
import random
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for

# Import the TemplateEnhancer from enhance_prompts.py
from enhance_prompts import TemplateEnhancer

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

# Initialize the prompt enhancer
prompt_enhancer = TemplateEnhancer()

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
    """Render the main page."""
    # Get all clips
    clips = []
    if os.path.exists(CLIP_DIR):
        for filename in os.listdir(CLIP_DIR):
            if filename.endswith('.mp4'):
                # Extract the prompt from the filename
                parts = filename.split('_', 1)
                if len(parts) > 1:
                    timestamp = parts[0]
                    prompt = parts[1].replace('.mp4', '').replace('_', ' ')
                else:
                    timestamp = ''
                    prompt = filename
                
                # Get file creation time
                file_path = os.path.join(CLIP_DIR, filename)
                created = datetime.fromtimestamp(os.path.getctime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
                
                clips.append({
                    'filename': filename,
                    'prompt': prompt,
                    'created': created
                })
    
    # Sort clips by creation time (newest first)
    clips.sort(key=lambda x: x['created'], reverse=True)
    
    # Get all movies
    movies = []
    if os.path.exists(MOVIE_DIR):
        for filename in os.listdir(MOVIE_DIR):
            if filename.endswith('.mp4'):
                # Extract the title from the filename
                parts = filename.split('_', 1)
                if len(parts) > 1:
                    timestamp = parts[0]
                    title = parts[1].replace('.mp4', '').replace('_', ' ')
                else:
                    timestamp = ''
                    title = filename
                
                # Get file creation time
                file_path = os.path.join(MOVIE_DIR, filename)
                created = datetime.fromtimestamp(os.path.getctime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
                
                movies.append({
                    'filename': filename,
                    'title': title,
                    'created': created
                })
    
    # Sort movies by creation time (newest first)
    movies.sort(key=lambda x: x['created'], reverse=True)
    
    # Get sample enhanced prompts for the help section
    sample_prompts = [
        "A red ball bouncing",
        "A butterfly in a garden",
        "A car driving on a mountain road"
    ]
    sample_enhanced = [enhance_prompt_text(p) for p in sample_prompts]
    
    return render_template('index.html', clips=clips, movies=movies, 
                          sample_prompts=sample_prompts, sample_enhanced=sample_enhanced)

@app.route('/generate', methods=['POST'])
def generate():
    """Generate a video from a text prompt."""
    prompt = request.form.get('prompt')
    use_enhanced = request.form.get('use_enhanced', 'false')
    enhanced_prompt = request.form.get('enhanced_prompt')
    
    if not prompt:
        return jsonify({"status": "error", "message": "Prompt is required"})
    
    # Use the enhanced prompt if specified
    final_prompt = enhanced_prompt if use_enhanced == 'true' and enhanced_prompt else prompt
    
    try:
        # Generate a unique filename based on the prompt
        timestamp = int(time.time())
        safe_prompt = ''.join(c if c.isalnum() or c in ' _-' else '_' for c in prompt[:50])
        filename = f"{timestamp}_{safe_prompt}.mp4"
        output_path = os.path.join(CLIP_DIR, filename)
        
        # Generate the video
        generate_video(final_prompt, output_path)
        
        return jsonify({
            "status": "success",
            "message": "Video generated successfully",
            "clip": filename,
            "prompt": final_prompt
        })
    except Exception as e:
        logger.error(f"Error generating video: {e}")
        return jsonify({"status": "error", "message": f"Error generating video: {str(e)}"})

@app.route('/combine', methods=['POST'])
def combine():
    """Combine multiple video clips into a single movie."""
    movie_title = request.form.get('movie_title')
    clip_filenames = request.form.getlist('clips')
    
    if not movie_title:
        return jsonify({"status": "error", "message": "Movie title is required"})
    
    if not clip_filenames:
        return jsonify({"status": "error", "message": "Please select at least one clip to combine"})
    
    try:
        # Create a sanitized filename from the title
        safe_title = ''.join(c if c.isalnum() or c in ' _-' else '_' for c in movie_title[:50])
        timestamp = int(time.time())
        movie_filename = f"{timestamp}_{safe_title}.mp4"
        movie_path = os.path.join(MOVIE_DIR, movie_filename)
        
        # Get the full paths of the clips
        clip_paths = [os.path.join(CLIP_DIR, filename) for filename in clip_filenames]
        
        # Check if all clips exist
        missing_clips = [path for path in clip_paths if not os.path.exists(path)]
        if missing_clips:
            return jsonify({
                "status": "error", 
                "message": f"Some clips were not found: {', '.join(os.path.basename(p) for p in missing_clips)}"
            })
        
        # Calculate total duration
        total_duration = len(clip_paths) * 10  # Each clip is 10 seconds
        
        # Combine the clips
        combine_videos(clip_paths, movie_path)
        
        return jsonify({
            "status": "success",
            "message": f"Movie created successfully! Combined {len(clip_paths)} clips into a {total_duration}-second movie.",
            "movie": movie_filename,
            "title": movie_title,
            "clips_used": len(clip_paths),
            "duration": total_duration
        })
    except Exception as e:
        logger.error(f"Error combining videos: {e}")
        return jsonify({"status": "error", "message": f"Error combining videos: {str(e)}"})

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

def enhance_prompt_text(prompt):
    """Enhance a prompt using the TemplateEnhancer."""
    try:
        enhanced_prompt = prompt_enhancer.enhance_prompt(prompt)
        logger.info(f"Enhanced prompt: {enhanced_prompt}")
        return enhanced_prompt
    except Exception as e:
        logger.error(f"Error enhancing prompt: {e}")
        return prompt  # Return original prompt if enhancement fails

@app.route('/enhance_prompt', methods=['POST'])
def enhance_prompt():
    """Enhance a prompt using the TemplateEnhancer."""
    prompt = request.form.get('prompt')
    
    if not prompt:
        return jsonify({"status": "error", "message": "Prompt is required"})
    
    enhanced_prompt = enhance_prompt_text(prompt)
    
    return jsonify({
        "status": "success",
        "message": "Prompt enhanced successfully",
        "enhanced_prompt": enhanced_prompt
    })

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Movie Maker Web Interface')
    parser.add_argument('--host', default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()
    
    print(f"Starting Movie Maker on http://{args.host}:{args.port}...")
    app.run(host=args.host, port=args.port, debug=args.debug) 