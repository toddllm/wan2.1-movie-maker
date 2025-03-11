#!/usr/bin/env python3
"""
R1-V Integration Script

This script demonstrates how to integrate the R1-V POC with the Movie Maker application.
It can be used as a standalone script or imported into the main application.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("r1v_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("r1v_integration")

def integrate_with_flask_app(app):
    """
    Integrate the R1-V POC with a Flask application.
    
    Args:
        app: The Flask application instance.
    """
    try:
        # Import the register_r1v_routes function
        from r1v_poc import register_r1v_routes
        
        # Register the R1-V routes with the Flask app
        register_r1v_routes(app)
        
        logger.info("R1-V POC integrated successfully with Flask app")
        return True
    
    except Exception as e:
        logger.error(f"Error integrating R1-V POC with Flask app: {e}")
        return False

def integrate_with_analyze_video(analyze_video_func):
    """
    Integrate the R1-V POC with the analyze_video function.
    
    Args:
        analyze_video_func: The original analyze_video function.
        
    Returns:
        function: The enhanced analyze_video function.
    """
    try:
        # Import the R1-V analyzer and emotion detector
        from r1v_poc import R1VAnalyzer, EmotionDetector
        
        # Define the enhanced analyze_video function
        def enhanced_analyze_video(video_path, num_frames=5, analysis_type='both', **kwargs):
            """
            Enhanced video analysis function that uses both the original analysis
            and the R1-V/R1-Omni analysis.
            
            Args:
                video_path (str): Path to the video file.
                num_frames (int): Number of frames to analyze.
                analysis_type (str): Type of analysis to perform ('r1v', 'emotion', 'both', or 'original').
                **kwargs: Additional arguments for the original analyze_video function.
                
            Returns:
                dict: Combined analysis results.
            """
            # Call the original analyze_video function
            original_results = analyze_video_func(video_path, **kwargs)
            
            # If only original analysis is requested, return it
            if analysis_type == 'original':
                return original_results
            
            # Initialize combined results
            combined_results = {
                'original_analysis': original_results,
                'success': True
            }
            
            # Perform R1-V analysis if requested
            if analysis_type in ['r1v', 'both']:
                try:
                    logger.info(f"Performing R1-V analysis on {video_path}")
                    r1v_analyzer = R1VAnalyzer()
                    r1v_results = r1v_analyzer.analyze_video(video_path, num_frames)
                    combined_results['r1v_analysis'] = r1v_results
                except Exception as e:
                    logger.error(f"Error during R1-V analysis: {e}")
                    combined_results['r1v_analysis'] = {'success': False, 'error': str(e)}
            
            # Perform emotion analysis if requested
            if analysis_type in ['emotion', 'both']:
                try:
                    logger.info(f"Performing emotion analysis on {video_path}")
                    emotion_detector = EmotionDetector()
                    emotion_results = emotion_detector.analyze_video(video_path, num_frames)
                    combined_results['emotion_analysis'] = emotion_results
                except Exception as e:
                    logger.error(f"Error during emotion analysis: {e}")
                    combined_results['emotion_analysis'] = {'success': False, 'error': str(e)}
            
            # Generate recommendations if both analyses were performed
            if analysis_type == 'both' and 'r1v_analysis' in combined_results and 'emotion_analysis' in combined_results:
                try:
                    from r1v_poc import Recommender
                    
                    logger.info(f"Generating recommendations for {video_path}")
                    recommender = Recommender()
                    
                    # Get paths to the analysis files
                    r1v_analysis_path = os.path.splitext(video_path)[0] + "_r1v_analysis.json"
                    emotion_analysis_path = os.path.splitext(video_path)[0] + "_emotion_analysis.json"
                    
                    # Generate recommendations
                    recommendation_results = recommender.generate_recommendations(
                        r1v_analysis_path,
                        emotion_analysis_path
                    )
                    
                    combined_results['recommendations'] = recommendation_results
                except Exception as e:
                    logger.error(f"Error generating recommendations: {e}")
                    combined_results['recommendations'] = {'success': False, 'error': str(e)}
            
            return combined_results
        
        logger.info("R1-V POC integrated successfully with analyze_video function")
        return enhanced_analyze_video
    
    except Exception as e:
        logger.error(f"Error integrating R1-V POC with analyze_video function: {e}")
        return analyze_video_func  # Return the original function if integration fails

def main():
    """Main function for testing the integration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="R1-V Integration Script")
    parser.add_argument("--video", help="Path to a video file for testing")
    parser.add_argument("--frames", type=int, default=5, help="Number of frames to analyze")
    parser.add_argument("--type", choices=['r1v', 'emotion', 'both', 'original'], default='both',
                        help="Type of analysis to perform")
    args = parser.parse_args()
    
    if args.video:
        # Import a dummy analyze_video function for testing
        def dummy_analyze_video(video_path, **kwargs):
            return {
                'video_path': video_path,
                'dummy_analysis': True,
                'message': 'This is a dummy analysis for testing purposes'
            }
        
        # Integrate with the dummy analyze_video function
        enhanced_analyze_video = integrate_with_analyze_video(dummy_analyze_video)
        
        # Test the enhanced analyze_video function
        results = enhanced_analyze_video(args.video, args.frames, args.type)
        
        # Print the results
        import json
        print(json.dumps(results, indent=2))
    else:
        print("No video specified. Use --video to test the integration.")

if __name__ == "__main__":
    main() 