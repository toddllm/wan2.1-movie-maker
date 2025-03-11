#!/usr/bin/env python3
"""
Movie Maker Beta - A web interface for generating and analyzing videos with scoring.
"""

import os
import sys
import time
import uuid
import logging
import argparse
import subprocess
import random
import json
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, session

# Import the TemplateEnhancer from enhance_prompts.py
from enhance_prompts import TemplateEnhancer

# Import the ScoringSystem
from scoring_system import ScoringSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("movie_maker_beta.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("movie_maker_beta")

# Create Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Ensure directories exist
CLIP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clips")
MOVIE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "movies")
CHELSEA_CLIP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chelsea_clips")
os.makedirs(CLIP_DIR, exist_ok=True)
os.makedirs(MOVIE_DIR, exist_ok=True)
os.makedirs(CHELSEA_CLIP_DIR, exist_ok=True)

# Set the WAN2.1 model directory
WAN_REPO_PATH = os.path.join(os.path.expanduser("~"), "development", "wan-video", "wan2.1", "wan_repo")
MODEL_DIR = os.path.join(os.path.expanduser("~"), "development", "wan-video", "wan2.1", "models", "Wan2.1-T2V-1.3B")
VENV_PATH = os.path.join(os.path.expanduser("~"), "development", "wan-video", "wan2.1", "venv")

# Initialize the prompt enhancer and scoring system
prompt_enhancer = TemplateEnhancer()
scoring_system = ScoringSystem()

# Default user preferences
DEFAULT_PREFERENCES = {
    'min_score': 0.6,
    'focus_areas': ['composition', 'lighting', 'camera angles', 'mood'],
    'weights': {
        'prompt_relevance': 0.25,
        'frame_consistency': 0.20,
        'detail_level': 0.20,
        'technical_accuracy': 0.15,
        'creative_elements': 0.10,
        'grammar_quality': 0.10
    }
}

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
            venv_python,
            os.path.join(wan_dir, "generate.py"),
            "--model_path", model_dir,
            "--prompt", prompt,
            "--output_path", output_path,
            "--frame_count", str(frame_count),
            "--size", size
        ]
        
        # Log the command
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run the command
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for the process to complete
        stdout, stderr = process.communicate()
        
        # Check if the process was successful
        if process.returncode == 0:
            logger.info(f"Video generated successfully: {output_path}")
            return True, output_path
        else:
            logger.error(f"Error generating video: {stderr}")
            return False, stderr
    except Exception as e:
        logger.error(f"Exception generating video: {e}")
        return False, str(e)

def analyze_video(video_path, target_description=None, user_preferences=None):
    """Analyze a video using the vision_analysis_poc.py script."""
    try:
        # Construct the command to run the vision_analysis_poc.py script
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "vision_analysis_poc.py"),
            "--video", video_path
        ]
        
        # Add target description if provided
        if target_description:
            cmd.extend(["--target", target_description])
        
        # Add user preferences if provided
        if user_preferences:
            if 'min_score' in user_preferences:
                cmd.extend(["--min-score", str(user_preferences['min_score'])])
            
            if 'focus_areas' in user_preferences:
                cmd.extend(["--focus"] + user_preferences['focus_areas'])
            
            if 'weights' in user_preferences:
                weight_args = [f"{k}={v}" for k, v in user_preferences['weights'].items()]
                cmd.extend(["--weights"] + weight_args)
        
        # Log the command
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run the command
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for the process to complete
        stdout, stderr = process.communicate()
        
        # Check if the process was successful
        if process.returncode == 0:
            logger.info(f"Video analyzed successfully: {video_path}")
            
            # Get the analysis file path
            analysis_file = os.path.splitext(video_path)[0] + "_final_analysis.txt"
            scores_file = os.path.splitext(video_path)[0] + "_scores.json"
            
            # Check if the analysis file exists
            if os.path.exists(analysis_file):
                with open(analysis_file, 'r') as f:
                    analysis_content = f.read()
            else:
                analysis_content = "Analysis file not found."
            
            # Check if the scores file exists
            scores_data = None
            if os.path.exists(scores_file):
                with open(scores_file, 'r') as f:
                    scores_data = json.load(f)
            
            return True, {
                'analysis': analysis_content,
                'scores': scores_data
            }
        else:
            logger.error(f"Error analyzing video: {stderr}")
            return False, stderr
    except Exception as e:
        logger.error(f"Exception analyzing video: {e}")
        return False, str(e)

def combine_videos(clip_paths, output_path):
    """Combine multiple video clips into a single video using FFmpeg."""
    try:
        # Create a temporary file with the list of input files
        temp_list_file = os.path.join(os.path.dirname(output_path), "temp_file_list.txt")
        with open(temp_list_file, 'w') as f:
            for clip_path in clip_paths:
                f.write(f"file '{os.path.abspath(clip_path)}'\n")
        
        # Construct the FFmpeg command
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", temp_list_file,
            "-c", "copy",
            output_path
        ]
        
        # Run the command
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Remove the temporary file
        os.remove(temp_list_file)
        
        logger.info(f"Videos combined successfully: {output_path}")
        return True, output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Error combining videos: {e.stderr}")
        return False, e.stderr
    except Exception as e:
        logger.error(f"Exception combining videos: {e}")
        return False, str(e)

def get_user_preferences():
    """Get user preferences from session or use defaults."""
    if 'preferences' not in session:
        session['preferences'] = DEFAULT_PREFERENCES.copy()
    return session['preferences']

@app.route('/')
def index():
    """Render the main page."""
    # Get all clips
    clips = []
    for filename in sorted(os.listdir(CLIP_DIR), key=lambda x: os.path.getmtime(os.path.join(CLIP_DIR, x)), reverse=True):
        if filename.endswith('.mp4'):
            clip_path = os.path.join(CLIP_DIR, filename)
            
            # Check if this clip has been analyzed
            analysis_path = os.path.splitext(clip_path)[0] + "_final_analysis.txt"
            scores_path = os.path.splitext(clip_path)[0] + "_scores.json"
            
            has_analysis = os.path.exists(analysis_path)
            has_scores = os.path.exists(scores_path)
            
            # Get scores if available
            scores = None
            if has_scores:
                try:
                    with open(scores_path, 'r') as f:
                        scores = json.load(f)
                except Exception as e:
                    logger.error(f"Error reading scores file {scores_path}: {e}")
            
            # Extract prompt from filename
            prompt = filename.split('_', 1)[1].rsplit('.', 1)[0] if '_' in filename else filename.rsplit('.', 1)[0]
            
            clips.append({
                'filename': filename,
                'path': clip_path,
                'prompt': prompt,
                'has_analysis': has_analysis,
                'has_scores': has_scores,
                'scores': scores,
                'created': datetime.fromtimestamp(os.path.getmtime(clip_path)).strftime('%Y-%m-%d %H:%M:%S'),
                'is_chelsea': False
            })
    
    # Also get Chelsea's clips
    if os.path.exists(CHELSEA_CLIP_DIR):
        for filename in sorted(os.listdir(CHELSEA_CLIP_DIR), key=lambda x: os.path.getmtime(os.path.join(CHELSEA_CLIP_DIR, x)), reverse=True):
            if filename.endswith('.mp4'):
                clip_path = os.path.join(CHELSEA_CLIP_DIR, filename)
                
                # Check if this clip has been analyzed
                analysis_path = os.path.splitext(clip_path)[0] + "_final_analysis.txt"
                scores_path = os.path.splitext(clip_path)[0] + "_scores.json"
                
                has_analysis = os.path.exists(analysis_path)
                has_scores = os.path.exists(scores_path)
                
                # Get scores if available
                scores = None
                if has_scores:
                    try:
                        with open(scores_path, 'r') as f:
                            scores = json.load(f)
                    except Exception as e:
                        logger.error(f"Error reading scores file {scores_path}: {e}")
                
                # Extract prompt from filename
                prompt = filename.split('_', 1)[1].rsplit('.', 1)[0] if '_' in filename else filename.rsplit('.', 1)[0]
                
                clips.append({
                    'filename': filename,
                    'path': clip_path,
                    'prompt': prompt,
                    'has_analysis': has_analysis,
                    'has_scores': has_scores,
                    'scores': scores,
                    'created': datetime.fromtimestamp(os.path.getmtime(clip_path)).strftime('%Y-%m-%d %H:%M:%S'),
                    'is_chelsea': True
                })
    
    # Get all movies
    movies = []
    for filename in sorted(os.listdir(MOVIE_DIR), key=lambda x: os.path.getmtime(os.path.join(MOVIE_DIR, x)), reverse=True):
        if filename.endswith('.mp4'):
            movie_path = os.path.join(MOVIE_DIR, filename)
            
            # Extract title from filename
            parts = filename.split('_', 1)
            if len(parts) > 1:
                timestamp = parts[0]
                title = parts[1].replace('.mp4', '').replace('_', ' ')
            else:
                timestamp = ''
                title = filename
            
            # Check if it's a Chelsea movie
            is_chelsea = 'Chelsea' in title
            
            movies.append({
                'filename': filename,
                'path': movie_path,
                'title': title,
                'created': datetime.fromtimestamp(os.path.getmtime(movie_path)).strftime('%Y-%m-%d %H:%M:%S'),
                'is_chelsea': is_chelsea
            })
    
    # Get user preferences
    preferences = get_user_preferences()
    
    # Get sample enhanced prompts for the help section
    sample_prompts = [
        "A red ball bouncing",
        "A butterfly in a garden",
        "A car driving on a mountain road"
    ]
    sample_enhanced = [enhance_prompt_text(p) for p in sample_prompts]
    
    return render_template('index.html', 
                          clips=clips, 
                          movies=movies, 
                          page_title="Movie Maker Beta", 
                          preferences=preferences,
                          sample_prompts=sample_prompts,
                          sample_enhanced=sample_enhanced,
                          is_chelsea=False)

@app.route('/generate', methods=['POST'])
def generate():
    """Generate a video from a prompt."""
    prompt = request.form.get('prompt', '')
    use_enhanced = request.form.get('use_enhanced', 'false') == 'true'
    enhanced_prompt = request.form.get('enhanced_prompt', '')
    
    if not prompt:
        return jsonify({'status': 'error', 'message': 'No prompt provided'})
    
    # Use the enhanced prompt if requested
    final_prompt = enhanced_prompt if use_enhanced and enhanced_prompt else prompt
    
    # Generate a unique filename
    timestamp = int(time.time())
    filename = f"{timestamp}_{final_prompt[:50]}.mp4"
    output_path = os.path.join(CLIP_DIR, filename)
    
    # Generate the video
    success, result = generate_video(final_prompt, output_path)
    
    if success:
        return jsonify({
            'status': 'success',
            'message': 'Video generated successfully',
            'clip': filename,
            'prompt': final_prompt
        })
    else:
        return jsonify({
            'status': 'error',
            'message': f'Error generating video: {result}'
        })

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze a video."""
    filename = request.form.get('filename', '')
    target_description = request.form.get('target_description', '')
    
    if not filename:
        return jsonify({'status': 'error', 'message': 'No filename provided'})
    
    # Get the video path
    video_path = os.path.join(CLIP_DIR, filename)
    
    if not os.path.exists(video_path):
        return jsonify({'status': 'error', 'message': 'Video file not found'})
    
    # Get user preferences
    preferences = get_user_preferences()
    
    # Analyze the video
    success, result = analyze_video(video_path, target_description, preferences)
    
    if success:
        return jsonify({
            'status': 'success',
            'message': 'Video analyzed successfully',
            'analysis': result['analysis'],
            'scores': result['scores']
        })
    else:
        return jsonify({
            'status': 'error',
            'message': f'Error analyzing video: {result}'
        })

@app.route('/combine', methods=['POST'])
def combine():
    """Combine multiple video clips into a single video."""
    selected_clips = request.form.getlist('selected_clips[]')
    output_name = request.form.get('output_name', '')
    
    if not selected_clips:
        return jsonify({'status': 'error', 'message': 'No clips selected'})
    
    if not output_name:
        output_name = f"combined_{int(time.time())}.mp4"
    elif not output_name.endswith('.mp4'):
        output_name += '.mp4'
    
    # Get the full paths of the selected clips
    clip_paths = [os.path.join(CLIP_DIR, clip) for clip in selected_clips]
    
    # Check if all clips exist
    for clip_path in clip_paths:
        if not os.path.exists(clip_path):
            return jsonify({
                'status': 'error',
                'message': f'Clip not found: {os.path.basename(clip_path)}'
            })
    
    # Combine the clips
    output_path = os.path.join(MOVIE_DIR, output_name)
    success, result = combine_videos(clip_paths, output_path)
    
    if success:
        return jsonify({
            'status': 'success',
            'message': 'Videos combined successfully',
            'movie': output_name
        })
    else:
        return jsonify({
            'status': 'error',
            'message': f'Error combining videos: {result}'
        })

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
    clip_path = os.path.join(CLIP_DIR, filename)
    
    if os.path.exists(clip_path):
        try:
            os.remove(clip_path)
            
            # Also delete associated analysis files
            base_path = os.path.splitext(clip_path)[0]
            for ext in ['_analysis.txt', '_final_analysis.txt', '_scores.json']:
                analysis_path = base_path + ext
                if os.path.exists(analysis_path):
                    os.remove(analysis_path)
            
            # Delete iteration files
            for iteration_file in Path(CLIP_DIR).glob(f"{os.path.splitext(filename)[0]}_iteration_*.json"):
                os.remove(iteration_file)
            
            return jsonify({'status': 'success', 'message': 'Clip deleted successfully'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Error deleting clip: {e}'})
    else:
        return jsonify({'status': 'error', 'message': 'Clip not found'})

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
        logger.error(f"Error deleting movie: {e}")
        return jsonify({"status": "error", "message": f"Error deleting movie: {str(e)}"})

@app.route('/view_analysis/<filename>')
def view_analysis(filename):
    """View the analysis for a video clip."""
    clip_path = os.path.join(CLIP_DIR, filename)
    base_path = os.path.splitext(clip_path)[0]
    
    # Check for analysis files
    analysis_path = base_path + "_final_analysis.txt"
    scores_path = base_path + "_scores.json"
    
    analysis_content = None
    scores_data = None
    
    if os.path.exists(analysis_path):
        with open(analysis_path, 'r') as f:
            analysis_content = f.read()
    
    if os.path.exists(scores_path):
        with open(scores_path, 'r') as f:
            scores_data = json.load(f)
    
    # Get iteration files
    iterations = []
    for iteration_file in sorted(Path(CLIP_DIR).glob(f"{os.path.splitext(filename)[0]}_iteration_*.json")):
        with open(iteration_file, 'r') as f:
            iterations.append(json.load(f))
    
    # Extract prompt from filename
    prompt = filename.split('_', 1)[1].rsplit('.', 1)[0] if '_' in filename else filename.rsplit('.', 1)[0]
    
    return render_template(
        'analysis.html',
        filename=filename,
        prompt=prompt,
        analysis=analysis_content,
        scores=scores_data,
        iterations=iterations,
        page_title=f"Analysis for {filename}"
    )

def enhance_prompt_text(prompt):
    """Enhance a prompt using the TemplateEnhancer."""
    try:
        enhanced = prompt_enhancer.enhance_prompt(prompt)
        return enhanced
    except Exception as e:
        logger.error(f"Error enhancing prompt: {e}")
        return prompt

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

@app.route('/chelsea/enhance_prompt', methods=['POST'])
def chelsea_enhance_prompt():
    """Enhance a prompt for Chelsea using the TemplateEnhancer."""
    prompt = request.form.get('prompt')
    
    if not prompt:
        return jsonify({"status": "error", "message": "Prompt is required"})
    
    enhanced_prompt = enhance_prompt_text(prompt)
    
    return jsonify({
        "status": "success",
        "message": "Prompt enhanced successfully for Chelsea",
        "enhanced_prompt": enhanced_prompt
    })

@app.route('/preferences', methods=['GET', 'POST'])
def preferences():
    """View and update user preferences."""
    if request.method == 'POST':
        # Update preferences
        preferences = {
            'min_score': float(request.form.get('min_score', 0.6)),
            'focus_areas': request.form.getlist('focus_areas[]'),
            'weights': {
                'prompt_relevance': float(request.form.get('weight_prompt_relevance', 0.25)),
                'frame_consistency': float(request.form.get('weight_frame_consistency', 0.20)),
                'detail_level': float(request.form.get('weight_detail_level', 0.20)),
                'technical_accuracy': float(request.form.get('weight_technical_accuracy', 0.15)),
                'creative_elements': float(request.form.get('weight_creative_elements', 0.10)),
                'grammar_quality': float(request.form.get('weight_grammar_quality', 0.10))
            }
        }
        
        # Normalize weights to sum to 1.0
        weight_sum = sum(preferences['weights'].values())
        if weight_sum != 1.0:
            for key in preferences['weights']:
                preferences['weights'][key] /= weight_sum
        
        # Save preferences to session
        session['preferences'] = preferences
        
        return redirect(url_for('index'))
    else:
        # Get current preferences
        preferences = get_user_preferences()
        
        return render_template(
            'preferences.html',
            preferences=preferences,
            page_title="User Preferences"
        )

@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    """Analyze multiple videos in batch mode."""
    selected_clips = request.form.getlist('selected_clips[]')
    
    if not selected_clips:
        return jsonify({'status': 'error', 'message': 'No clips selected'})
    
    # Get user preferences
    preferences = get_user_preferences()
    
    # Start a background process to analyze the videos
    results = []
    for filename in selected_clips:
        video_path = os.path.join(CLIP_DIR, filename)
        
        if os.path.exists(video_path):
            # Extract target description from filename
            target_description = filename.split('_', 1)[1].rsplit('.', 1)[0] if '_' in filename else filename.rsplit('.', 1)[0]
            
            # Analyze the video
            success, result = analyze_video(video_path, target_description, preferences)
            
            results.append({
                'filename': filename,
                'success': success,
                'result': result if success else str(result)
            })
    
    return jsonify({
        'status': 'success',
        'message': f'Analyzed {len(results)} videos',
        'results': results
    })

@app.route('/score_video', methods=['POST'])
def score_video():
    """Score a video using the scoring system."""
    filename = request.form.get('filename', '')
    
    if not filename:
        return jsonify({'status': 'error', 'message': 'No filename provided'})
    
    # Get the video path
    video_path = os.path.join(CLIP_DIR, filename)
    
    if not os.path.exists(video_path):
        return jsonify({'status': 'error', 'message': 'Video file not found'})
    
    # Check if analysis files exist
    base_path = os.path.splitext(video_path)[0]
    analysis_path = base_path + "_analysis.txt"
    
    if not os.path.exists(analysis_path):
        return jsonify({'status': 'error', 'message': 'Analysis file not found'})
    
    # Extract frame descriptions from analysis file
    frame_descriptions = []
    with open(analysis_path, 'r') as f:
        content = f.read()
        for line in content.split('\n'):
            if line.startswith("Frame "):
                parts = line.split(": ", 1)
                if len(parts) > 1:
                    frame_descriptions.append(parts[1])
    
    # Extract prompt from filename
    prompt = filename.split('_', 1)[1].rsplit('.', 1)[0] if '_' in filename else filename.rsplit('.', 1)[0]
    
    # Get user preferences
    preferences = get_user_preferences()
    
    # Create a scoring system with custom weights if provided
    if 'weights' in preferences:
        custom_scoring_system = ScoringSystem(weights=preferences['weights'])
        overall_score, scores, feedback = custom_scoring_system.evaluate_analysis(prompt, frame_descriptions)
    else:
        overall_score, scores, feedback = scoring_system.evaluate_analysis(prompt, frame_descriptions)
    
    # Save the scores to a file
    scores_file = base_path + "_scores.json"
    with open(scores_file, 'w') as f:
        json.dump({
            'overall_score': overall_score,
            'scores': scores,
            'feedback': feedback
        }, f, indent=2)
    
    return jsonify({
        'status': 'success',
        'message': 'Video scored successfully',
        'overall_score': overall_score,
        'scores': scores,
        'feedback': feedback
    })

@app.route('/chelsea')
def chelsea_index():
    """Render the Chelsea page."""
    # Get all Chelsea's clips
    clips = []
    if os.path.exists(CHELSEA_CLIP_DIR):
        for filename in os.listdir(CHELSEA_CLIP_DIR):
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
                file_path = os.path.join(CHELSEA_CLIP_DIR, filename)
                created = datetime.fromtimestamp(os.path.getctime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
                
                # Check if this clip has been analyzed
                analysis_path = os.path.splitext(file_path)[0] + "_final_analysis.txt"
                scores_path = os.path.splitext(file_path)[0] + "_scores.json"
                
                has_analysis = os.path.exists(analysis_path)
                has_scores = os.path.exists(scores_path)
                
                # Get scores if available
                scores = None
                if has_scores:
                    try:
                        with open(scores_path, 'r') as f:
                            scores = json.load(f)
                    except Exception as e:
                        logger.error(f"Error reading scores file {scores_path}: {e}")
                
                clips.append({
                    'filename': filename,
                    'prompt': prompt,
                    'created': created,
                    'has_analysis': has_analysis,
                    'has_scores': has_scores,
                    'scores': scores
                })
    
    # Sort clips by creation time (newest first)
    clips.sort(key=lambda x: x['created'], reverse=True)
    
    # Get all movies (same as main page)
    movies = []
    for filename in sorted(os.listdir(MOVIE_DIR), key=lambda x: os.path.getmtime(os.path.join(MOVIE_DIR, x)), reverse=True):
        if filename.endswith('.mp4'):
            movie_path = os.path.join(MOVIE_DIR, filename)
            movies.append({
                'filename': filename,
                'path': movie_path,
                'created': datetime.fromtimestamp(os.path.getmtime(movie_path)).strftime('%Y-%m-%d %H:%M:%S')
            })
    
    # Get user preferences
    preferences = get_user_preferences()
    
    # Get sample enhanced prompts for the help section
    sample_prompts = [
        "A red ball bouncing",
        "A butterfly in a garden",
        "A car driving on a mountain road"
    ]
    sample_enhanced = [enhance_prompt_text(p) for p in sample_prompts]
    
    return render_template('index.html', 
                          clips=clips, 
                          movies=movies, 
                          page_title="Chelsea's Movie Maker Beta", 
                          preferences=preferences,
                          sample_prompts=sample_prompts,
                          sample_enhanced=sample_enhanced,
                          is_chelsea=True)

@app.route('/chelsea/generate', methods=['POST'])
def chelsea_generate():
    """Generate a video from a text prompt for Chelsea."""
    prompt = request.form.get('prompt', '')
    use_enhanced = request.form.get('use_enhanced', 'false') == 'true'
    enhanced_prompt = request.form.get('enhanced_prompt', '')
    
    if not prompt:
        return jsonify({'status': 'error', 'message': 'No prompt provided'})
    
    # Use the enhanced prompt if requested
    final_prompt = enhanced_prompt if use_enhanced and enhanced_prompt else prompt
    
    # Generate a unique filename
    timestamp = int(time.time())
    filename = f"{timestamp}_{final_prompt[:50]}.mp4"
    output_path = os.path.join(CHELSEA_CLIP_DIR, filename)
    
    # Generate the video
    success, result = generate_video(final_prompt, output_path)
    
    if success:
        return jsonify({
            'status': 'success',
            'message': 'Video generated successfully for Chelsea',
            'clip': filename,
            'prompt': final_prompt
        })
    else:
        logger.error(f"Error generating video for Chelsea: {result}")
        return jsonify({'status': 'error', 'message': f'Error generating video: {result}'})

@app.route('/chelsea/clips/<filename>')
def serve_chelsea_clip(filename):
    """Serve a Chelsea video clip."""
    return send_from_directory(CHELSEA_CLIP_DIR, filename)

@app.route('/chelsea/delete_clip/<filename>', methods=['POST'])
def delete_chelsea_clip(filename):
    """Delete a Chelsea video clip."""
    try:
        clip_path = os.path.join(CHELSEA_CLIP_DIR, filename)
        if os.path.exists(clip_path):
            os.remove(clip_path)
            
            # Also delete analysis files if they exist
            analysis_path = os.path.splitext(clip_path)[0] + "_analysis.txt"
            if os.path.exists(analysis_path):
                os.remove(analysis_path)
                
            final_analysis_path = os.path.splitext(clip_path)[0] + "_final_analysis.txt"
            if os.path.exists(final_analysis_path):
                os.remove(final_analysis_path)
                
            iteration_json_path = os.path.splitext(clip_path)[0] + "_iteration_1.json"
            if os.path.exists(iteration_json_path):
                os.remove(iteration_json_path)
                
            scores_path = os.path.splitext(clip_path)[0] + "_scores.json"
            if os.path.exists(scores_path):
                os.remove(scores_path)
            
            return jsonify({"status": "success", "message": f"Clip {filename} deleted"})
        else:
            return jsonify({"status": "error", "message": f"Clip {filename} not found"})
    except Exception as e:
        logger.error(f"Error deleting clip: {e}")
        return jsonify({"status": "error", "message": f"Error deleting clip: {str(e)}"})

@app.route('/chelsea/combine', methods=['POST'])
def chelsea_combine():
    """Combine multiple clips into a movie for Chelsea."""
    movie_title = request.form.get('movie_title', '')
    clip_filenames = request.form.getlist('clips')
    
    if not movie_title:
        return jsonify({'status': 'error', 'message': 'Movie title is required'})
    
    if not clip_filenames:
        return jsonify({'status': 'error', 'message': 'No clips selected'})
    
    # Generate a unique filename for the movie
    timestamp = int(time.time())
    safe_title = ''.join(c if c.isalnum() or c in ' _-' else '_' for c in movie_title[:50])
    movie_filename = f"{timestamp}_{safe_title}.mp4"
    output_path = os.path.join(MOVIE_DIR, movie_filename)
    
    # Get the full paths of the selected clips
    clip_paths = [os.path.join(CHELSEA_CLIP_DIR, filename) for filename in clip_filenames]
    
    # Combine the clips
    success, result = combine_videos(clip_paths, output_path)
    
    if success:
        return jsonify({
            'status': 'success',
            'message': 'Movie created successfully',
            'movie': movie_filename,
            'title': movie_title
        })
    else:
        return jsonify({'status': 'error', 'message': f'Error creating movie: {result}'})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Movie Maker Beta web interface")
    parser.add_argument("--port", type=int, default=5002, help="Port to run the server on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    # Activate the virtual environment
    activate_venv()
    
    # Run the app
    app.run(host='0.0.0.0', port=args.port, debug=args.debug) 