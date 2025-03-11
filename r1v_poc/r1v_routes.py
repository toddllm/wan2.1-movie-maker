#!/usr/bin/env python3
"""
R1-V Routes - Flask Routes for R1-V POC Integration

This module implements Flask routes for the R1-V POC integration:
- Enhanced analysis dashboard
- R1-V and R1-Omni analysis endpoints
- Recommendation system endpoints

It integrates with the Movie Maker application to provide enhanced video analysis capabilities.
"""

import os
import sys
import logging
import json
from flask import Blueprint, render_template, request, jsonify, redirect, url_for, session
from pathlib import Path

# Import the R1-V POC modules
from r1v_poc.r1v_analyzer import R1VAnalyzer
from r1v_poc.emotion_detector import EmotionDetector
from r1v_poc.recommender import Recommender
from r1v_poc.model_utils import load_model_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("r1v_routes.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("r1v_routes")

# Create Blueprint for R1-V routes
r1v_blueprint = Blueprint('r1v', __name__, template_folder='templates')

# Define paths
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "r1v_config.json")

# Load configuration
config = load_model_config(CONFIG_PATH)

@r1v_blueprint.route('/r1v')
def r1v_index():
    """Render the R1-V dashboard index page."""
    # Get list of clips
    clip_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "clips")
    clips = []
    
    if os.path.exists(clip_dir):
        for filename in os.listdir(clip_dir):
            if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                # Check if this clip has been analyzed with R1-V
                r1v_analysis_path = os.path.join(clip_dir, os.path.splitext(filename)[0] + "_r1v_analysis.json")
                emotion_analysis_path = os.path.join(clip_dir, os.path.splitext(filename)[0] + "_emotion_analysis.json")
                
                clips.append({
                    'filename': filename,
                    'path': os.path.join("clips", filename),
                    'has_r1v_analysis': os.path.exists(r1v_analysis_path),
                    'has_emotion_analysis': os.path.exists(emotion_analysis_path)
                })
    
    # Get list of movies
    movie_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "movies")
    movies = []
    
    if os.path.exists(movie_dir):
        for filename in os.listdir(movie_dir):
            if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                # Check if this movie has been analyzed with R1-V
                r1v_analysis_path = os.path.join(movie_dir, os.path.splitext(filename)[0] + "_r1v_analysis.json")
                emotion_analysis_path = os.path.join(movie_dir, os.path.splitext(filename)[0] + "_emotion_analysis.json")
                
                movies.append({
                    'filename': filename,
                    'path': os.path.join("movies", filename),
                    'has_r1v_analysis': os.path.exists(r1v_analysis_path),
                    'has_emotion_analysis': os.path.exists(emotion_analysis_path)
                })
    
    return render_template(
        'r1v_dashboard.html',
        clips=clips,
        movies=movies,
        config=config
    )

@r1v_blueprint.route('/r1v/analyze', methods=['POST'])
def r1v_analyze():
    """Analyze a video using R1-V."""
    video_path = request.form.get('video_path')
    analysis_type = request.form.get('analysis_type', 'r1v')  # 'r1v', 'emotion', or 'both'
    num_frames = int(request.form.get('num_frames', 5))
    
    if not video_path:
        return jsonify({
            'success': False,
            'error': 'No video path provided'
        })
    
    # Get the full path to the video
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_video_path = os.path.join(base_dir, video_path)
    
    if not os.path.exists(full_video_path):
        return jsonify({
            'success': False,
            'error': f'Video file not found: {video_path}'
        })
    
    results = {}
    
    # Perform R1-V analysis if requested
    if analysis_type in ['r1v', 'both']:
        try:
            logger.info(f"Starting R1-V analysis for {video_path}")
            
            # Initialize the R1-V analyzer
            r1v_analyzer = R1VAnalyzer(model_name=config.get('r1v_model'))
            
            # Analyze the video
            r1v_results = r1v_analyzer.analyze_video(full_video_path, num_frames)
            
            if r1v_results.get('success', False):
                results['r1v'] = {
                    'success': True,
                    'summary': r1v_results.get('summary', ''),
                    'analysis_path': os.path.splitext(full_video_path)[0] + "_r1v_analysis.json"
                }
            else:
                results['r1v'] = {
                    'success': False,
                    'error': r1v_results.get('error', 'Unknown error during R1-V analysis')
                }
        
        except Exception as e:
            logger.error(f"Error during R1-V analysis: {e}")
            results['r1v'] = {
                'success': False,
                'error': str(e)
            }
    
    # Perform emotion analysis if requested
    if analysis_type in ['emotion', 'both']:
        try:
            logger.info(f"Starting emotion analysis for {video_path}")
            
            # Initialize the emotion detector
            emotion_detector = EmotionDetector(model_name=config.get('r1omni_model'))
            
            # Analyze the video
            emotion_results = emotion_detector.analyze_video(full_video_path, num_frames)
            
            if emotion_results.get('success', False):
                results['emotion'] = {
                    'success': True,
                    'summary': emotion_results.get('summary', ''),
                    'analysis_path': os.path.splitext(full_video_path)[0] + "_emotion_analysis.json"
                }
            else:
                results['emotion'] = {
                    'success': False,
                    'error': emotion_results.get('error', 'Unknown error during emotion analysis')
                }
        
        except Exception as e:
            logger.error(f"Error during emotion analysis: {e}")
            results['emotion'] = {
                'success': False,
                'error': str(e)
            }
    
    # Generate recommendations if both analyses were successful
    if analysis_type == 'both' and results.get('r1v', {}).get('success', False) and results.get('emotion', {}).get('success', False):
        try:
            logger.info(f"Generating recommendations for {video_path}")
            
            # Initialize the recommender
            recommender = Recommender()
            
            # Get the original prompt if available
            original_prompt = request.form.get('original_prompt', '')
            
            # Generate recommendations
            recommendation_results = recommender.generate_recommendations(
                results['r1v']['analysis_path'],
                results['emotion']['analysis_path'],
                original_prompt
            )
            
            if recommendation_results.get('success', False):
                results['recommendations'] = {
                    'success': True,
                    'summary': recommendation_results.get('summary', ''),
                    'recommendations_path': os.path.splitext(full_video_path)[0] + "_recommendations.json"
                }
            else:
                results['recommendations'] = {
                    'success': False,
                    'error': recommendation_results.get('error', 'Unknown error generating recommendations')
                }
        
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            results['recommendations'] = {
                'success': False,
                'error': str(e)
            }
    
    return jsonify({
        'success': True,
        'results': results
    })

@r1v_blueprint.route('/r1v/view_analysis/<path:video_path>')
def view_analysis(video_path):
    """View the R1-V and emotion analysis for a video."""
    # Get the full path to the video
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_video_path = os.path.join(base_dir, video_path)
    
    if not os.path.exists(full_video_path):
        return render_template(
            'r1v_analysis.html',
            error=f'Video file not found: {video_path}'
        )
    
    # Check for analysis files
    r1v_analysis_path = os.path.splitext(full_video_path)[0] + "_r1v_analysis.json"
    emotion_analysis_path = os.path.splitext(full_video_path)[0] + "_emotion_analysis.json"
    recommendations_path = os.path.splitext(full_video_path)[0] + "_recommendations.json"
    
    # Load analyses if they exist
    r1v_analysis = None
    emotion_analysis = None
    recommendations = None
    
    if os.path.exists(r1v_analysis_path):
        try:
            with open(r1v_analysis_path, 'r') as f:
                r1v_analysis = json.load(f)
        except Exception as e:
            logger.error(f"Error loading R1-V analysis: {e}")
    
    if os.path.exists(emotion_analysis_path):
        try:
            with open(emotion_analysis_path, 'r') as f:
                emotion_analysis = json.load(f)
        except Exception as e:
            logger.error(f"Error loading emotion analysis: {e}")
    
    if os.path.exists(recommendations_path):
        try:
            with open(recommendations_path, 'r') as f:
                recommendations = json.load(f)
        except Exception as e:
            logger.error(f"Error loading recommendations: {e}")
    
    return render_template(
        'r1v_analysis.html',
        video_path=video_path,
        r1v_analysis=r1v_analysis,
        emotion_analysis=emotion_analysis,
        recommendations=recommendations
    )

@r1v_blueprint.route('/r1v/recommend', methods=['POST'])
def generate_recommendations():
    """Generate recommendations based on existing analyses."""
    r1v_analysis_path = request.form.get('r1v_analysis_path')
    emotion_analysis_path = request.form.get('emotion_analysis_path')
    original_prompt = request.form.get('original_prompt', '')
    
    if not r1v_analysis_path:
        return jsonify({
            'success': False,
            'error': 'No R1-V analysis path provided'
        })
    
    # Get the full paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_r1v_path = os.path.join(base_dir, r1v_analysis_path) if not os.path.isabs(r1v_analysis_path) else r1v_analysis_path
    full_emotion_path = os.path.join(base_dir, emotion_analysis_path) if emotion_analysis_path and not os.path.isabs(emotion_analysis_path) else emotion_analysis_path
    
    if not os.path.exists(full_r1v_path):
        return jsonify({
            'success': False,
            'error': f'R1-V analysis file not found: {r1v_analysis_path}'
        })
    
    try:
        logger.info(f"Generating recommendations based on {r1v_analysis_path} and {emotion_analysis_path}")
        
        # Initialize the recommender
        recommender = Recommender()
        
        # Generate recommendations
        results = recommender.generate_recommendations(
            full_r1v_path,
            full_emotion_path,
            original_prompt
        )
        
        if results.get('success', False):
            return jsonify({
                'success': True,
                'summary': results.get('summary', ''),
                'recommendations_path': os.path.splitext(full_r1v_path)[0].replace('_r1v_analysis', '') + "_recommendations.json"
            })
        else:
            return jsonify({
                'success': False,
                'error': results.get('error', 'Unknown error generating recommendations')
            })
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@r1v_blueprint.route('/r1v/config', methods=['GET', 'POST'])
def r1v_config():
    """View or update the R1-V configuration."""
    if request.method == 'POST':
        # Update configuration
        r1v_model = request.form.get('r1v_model')
        r1omni_model = request.form.get('r1omni_model')
        
        # Update the config
        config['r1v_model'] = r1v_model if r1v_model else config.get('r1v_model')
        config['r1omni_model'] = r1omni_model if r1omni_model else config.get('r1omni_model')
        
        # Save the config
        try:
            with open(CONFIG_PATH, 'w') as f:
                json.dump(config, f, indent=2)
            
            return jsonify({
                'success': True,
                'message': 'Configuration updated successfully'
            })
        
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            })
    
    else:
        # Return the current configuration
        return jsonify({
            'success': True,
            'config': config
        })

def register_r1v_routes(app):
    """Register the R1-V routes with the Flask app."""
    app.register_blueprint(r1v_blueprint)
    logger.info("R1-V routes registered") 