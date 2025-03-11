#!/usr/bin/env python3
"""
R1-V Analyzer - Enhanced Visual Analysis Module

This module implements advanced frame analysis capabilities using the R1-V model:
- Object counting and identification
- Scene composition analysis
- Visual quality assessment

It integrates with the existing analyze_video function in the Movie Maker application.
"""

import os
import sys
import logging
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import json
from transformers import AutoProcessor, AutoModelForCausalLM
from difflib import SequenceMatcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("r1v_analyzer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("r1v_analyzer")

class R1VAnalyzer:
    """
    Enhanced visual analysis using R1-V model capabilities.
    """
    
    def __init__(self, model_name="Qwen/Qwen2-VL-7B", device=None):
        """
        Initialize the R1-V analyzer with the specified model.
        
        Args:
            model_name (str): The name or path of the R1-V model to use.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing R1VAnalyzer with model: {model_name} on {self.device}")
        
        # Load model and processor
        self.model = None
        self.processor = None
        self.load_model()
    
    def load_model(self):
        """Load the R1-V model and processor."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map=self.device,
                trust_remote_code=True
            )
            logger.info(f"Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def extract_frames(self, video_path, num_frames=5):
        """
        Extract frames from a video file.
        
        Args:
            video_path (str): Path to the video file.
            num_frames (int): Number of frames to extract.
            
        Returns:
            list: List of extracted frames as PIL Images.
        """
        logger.info(f"Extracting {num_frames} frames from {video_path}")
        
        try:
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            
            # Check if the video file was opened successfully
            if not cap.isOpened():
                logger.error(f"Error opening video file: {video_path}")
                return []
            
            # Get video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0
            
            logger.info(f"Video properties: {frame_count} frames, {fps} fps, {duration:.2f} seconds")
            
            # Calculate frame indices to extract (evenly distributed)
            if frame_count <= num_frames:
                # If there are fewer frames than requested, use all frames
                frame_indices = list(range(frame_count))
            else:
                # Otherwise, select evenly spaced frames
                frame_indices = [int(i * frame_count / num_frames) for i in range(num_frames)]
            
            # Extract the frames
            frames = []
            for idx in frame_indices:
                # Set the frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                
                # Read the frame
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert to PIL Image
                    pil_image = Image.fromarray(frame_rgb)
                    
                    frames.append(pil_image)
                else:
                    logger.warning(f"Failed to read frame at index {idx}")
            
            # Release the video file
            cap.release()
            
            logger.info(f"Extracted {len(frames)} frames")
            return frames
        
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            return []
    
    def analyze_frame(self, image, prompt_template=None):
        """
        Analyze a single frame using the R1-V model.
        
        Args:
            image (PIL.Image): The image to analyze.
            prompt_template (str): Optional custom prompt template.
            
        Returns:
            str: The analysis result.
        """
        if prompt_template is None:
            prompt_template = (
                "Analyze this image in detail. Identify and count all objects, "
                "describe the scene composition, and assess the visual quality. "
                "Format your response as follows:\n"
                "1. Objects: List and count all visible objects\n"
                "2. Scene Composition: Describe the layout and arrangement\n"
                "3. Visual Quality: Assess clarity, lighting, and overall quality\n"
                "4. Summary: Provide a concise summary of the scene"
            )
        
        try:
            # Prepare inputs
            inputs = self.processor(
                text=prompt_template,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False
                )
            
            # Decode the response
            response = self.processor.decode(output[0], skip_special_tokens=True)
            
            # Extract the model's response (remove the prompt)
            response = response.split(prompt_template)[-1].strip()
            
            return response
        
        except Exception as e:
            logger.error(f"Error analyzing frame: {e}")
            return f"Error analyzing frame: {str(e)}"
    
    def count_objects(self, image):
        """
        Count objects in an image using R1-V's counting capabilities.
        
        Args:
            image (PIL.Image): The image to analyze.
            
        Returns:
            dict: Dictionary with object counts.
        """
        prompt = (
            "Count all distinct objects in this image. "
            "List each object type and its count in the format: "
            "Object: Count"
        )
        
        response = self.analyze_frame(image, prompt)
        
        # Parse the response to extract object counts
        object_counts = {}
        for line in response.split('\n'):
            if ':' in line:
                parts = line.split(':')
                if len(parts) == 2:
                    obj = parts[0].strip()
                    try:
                        count = int(parts[1].strip())
                        object_counts[obj] = count
                    except ValueError:
                        # If count is not a number, skip this line
                        continue
        
        return object_counts
    
    def analyze_scene_composition(self, image):
        """
        Analyze the scene composition of an image.
        
        Args:
            image (PIL.Image): The image to analyze.
            
        Returns:
            dict: Dictionary with scene composition analysis.
        """
        prompt = (
            "Analyze the scene composition of this image. "
            "Describe the following aspects:\n"
            "1. Foreground elements\n"
            "2. Background elements\n"
            "3. Spatial arrangement\n"
            "4. Visual balance\n"
            "5. Color composition"
        )
        
        response = self.analyze_frame(image, prompt)
        
        # Return the full analysis as a dictionary
        composition_analysis = {
            "full_analysis": response,
            "aspects": {}
        }
        
        # Extract specific aspects if they're mentioned
        aspects = ["Foreground", "Background", "Spatial", "Balance", "Color"]
        for aspect in aspects:
            for line in response.split('\n'):
                if aspect.lower() in line.lower():
                    composition_analysis["aspects"][aspect] = line.strip()
        
        return composition_analysis
    
    def assess_visual_quality(self, image):
        """
        Assess the visual quality of an image.
        
        Args:
            image (PIL.Image): The image to assess.
            
        Returns:
            dict: Dictionary with quality assessment scores.
        """
        prompt = (
            "Assess the visual quality of this image on a scale of 1-10 for each of these aspects:\n"
            "1. Clarity/Sharpness\n"
            "2. Lighting/Exposure\n"
            "3. Color balance\n"
            "4. Composition\n"
            "5. Overall quality\n"
            "Format your response as 'Aspect: Score/10 - Brief explanation'"
        )
        
        response = self.analyze_frame(image, prompt)
        
        # Parse the response to extract quality scores
        quality_scores = {
            "full_assessment": response,
            "scores": {}
        }
        
        for line in response.split('\n'):
            if ':' in line and '/10' in line:
                parts = line.split(':')
                if len(parts) >= 2:
                    aspect = parts[0].strip()
                    score_part = parts[1].strip()
                    
                    # Extract the numerical score
                    try:
                        score = float(score_part.split('/')[0].strip())
                        quality_scores["scores"][aspect] = score
                    except (ValueError, IndexError):
                        # If score cannot be parsed, skip this line
                        continue
        
        # Calculate average score if we have at least one valid score
        if quality_scores["scores"]:
            quality_scores["average_score"] = sum(quality_scores["scores"].values()) / len(quality_scores["scores"])
        else:
            quality_scores["average_score"] = 0.0
        
        return quality_scores
    
    def analyze_video(self, video_path, num_frames=5):
        """
        Perform comprehensive analysis on a video using R1-V capabilities.
        
        Args:
            video_path (str): Path to the video file.
            num_frames (int): Number of frames to analyze.
            
        Returns:
            dict: Dictionary with comprehensive analysis results.
        """
        logger.info(f"Starting comprehensive analysis of {video_path}")
        
        # Extract frames from the video
        frames = self.extract_frames(video_path, num_frames)
        
        if not frames:
            logger.error(f"No frames extracted from {video_path}")
            return {
                "error": "Failed to extract frames from video",
                "success": False
            }
        
        # Analyze each frame
        frame_analyses = []
        object_counts_all = {}
        quality_scores_all = []
        
        for i, frame in enumerate(frames):
            logger.info(f"Analyzing frame {i+1}/{len(frames)}")
            
            # Save frame for debugging
            frame_path = f"frame_{i+1}.jpg"
            frame.save(frame_path)
            logger.info(f"Saved frame to {frame_path}")
            
            # Comprehensive analysis
            analysis = self.analyze_frame(frame)
            
            # Object counting
            object_counts = self.count_objects(frame)
            
            # Update overall object counts
            for obj, count in object_counts.items():
                if obj in object_counts_all:
                    object_counts_all[obj] = max(object_counts_all[obj], count)
                else:
                    object_counts_all[obj] = count
            
            # Scene composition
            composition = self.analyze_scene_composition(frame)
            
            # Visual quality assessment
            quality = self.assess_visual_quality(frame)
            quality_scores_all.append(quality["average_score"])
            
            # Add to frame analyses
            frame_analyses.append({
                "frame_index": i,
                "general_analysis": analysis,
                "object_counts": object_counts,
                "composition": composition,
                "quality": quality
            })
        
        # Calculate average quality score
        avg_quality_score = sum(quality_scores_all) / len(quality_scores_all) if quality_scores_all else 0.0
        
        # Generate overall video analysis
        overall_analysis = {
            "video_path": video_path,
            "frames_analyzed": len(frames),
            "frame_analyses": frame_analyses,
            "overall_object_counts": object_counts_all,
            "average_quality_score": avg_quality_score,
            "success": True
        }
        
        # Generate a summary
        summary = self._generate_summary(overall_analysis)
        overall_analysis["summary"] = summary
        
        # Save the analysis results
        output_path = os.path.splitext(video_path)[0] + "_r1v_analysis.json"
        with open(output_path, 'w') as f:
            json.dump(overall_analysis, f, indent=2)
        
        logger.info(f"Analysis complete. Results saved to {output_path}")
        
        return overall_analysis
    
    def _generate_summary(self, analysis):
        """
        Generate a summary of the video analysis.
        
        Args:
            analysis (dict): The complete analysis results.
            
        Returns:
            str: A summary of the analysis.
        """
        # Extract key information
        num_frames = analysis["frames_analyzed"]
        object_counts = analysis["overall_object_counts"]
        avg_quality = analysis["average_quality_score"]
        
        # Format object counts for the summary
        objects_str = ", ".join([f"{count} {obj}" for obj, count in object_counts.items()])
        
        # Create the summary
        summary = (
            f"Video Analysis Summary:\n"
            f"- Analyzed {num_frames} frames\n"
            f"- Detected objects: {objects_str}\n"
            f"- Average visual quality score: {avg_quality:.1f}/10\n\n"
        )
        
        # Add frame-specific highlights if there are frames
        if analysis["frame_analyses"]:
            # Get the highest quality frame
            best_frame_idx = max(
                range(len(analysis["frame_analyses"])),
                key=lambda i: analysis["frame_analyses"][i]["quality"].get("average_score", 0)
            )
            best_frame = analysis["frame_analyses"][best_frame_idx]
            
            # Add best frame highlight
            summary += (
                f"Highlight (Frame {best_frame_idx + 1}):\n"
                f"{best_frame['general_analysis'][:200]}...\n"
            )
        
        return summary

# Main function for standalone testing
def main():
    """Main function for testing the R1VAnalyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="R1-V Video Analyzer")
    parser.add_argument("--video", required=True, help="Path to the video file")
    parser.add_argument("--frames", type=int, default=5, help="Number of frames to analyze")
    parser.add_argument("--model", default="Qwen/Qwen2-VL-7B", help="Model name or path")
    args = parser.parse_args()
    
    # Initialize the analyzer
    analyzer = R1VAnalyzer(model_name=args.model)
    
    # Analyze the video
    results = analyzer.analyze_video(args.video, args.frames)
    
    # Print the summary
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    print(results["summary"])
    print("="*50)

if __name__ == "__main__":
    main() 