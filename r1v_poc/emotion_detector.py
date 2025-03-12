#!/usr/bin/env python3
"""
Emotion Detector - R1-Omni Integration

This module implements emotion recognition capabilities using the R1-Omni model:
- Frame-by-frame emotion detection
- Overall emotional arc analysis
- Emotional impact scoring

It integrates with the Movie Maker application to provide emotional analysis of videos.
"""

import os
import sys
import logging
import torch
import numpy as np
import cv2
from PIL import Image
import json
import tempfile
from pathlib import Path
from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForVision2Seq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("emotion_detector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("emotion_detector")

class EmotionDetector:
    """
    Emotion recognition using R1-Omni model capabilities.
    """
    
    def __init__(self, model_name="StarJiaxing/R1-Omni-0.5B", device=None):
        """
        Initialize the emotion detector with the specified model.
        
        Args:
            model_name (str): The name or path of the R1-Omni model to use.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing EmotionDetector with model: {model_name} on {self.device}")
        
        # Define emotion categories
        self.emotion_categories = [
            "happy", "sad", "angry", "surprised", "fearful", "disgusted", 
            "neutral", "anxious", "excited", "bored", "confused", "disappointed"
        ]
        
        # Load model and processor
        self.model = None
        self.processor = None
        self.load_model()
    
    def load_model(self):
        """Load the R1-Omni model and processor."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            
            # Try different model classes
            try:
                # First try with Vision2Seq model class which is more appropriate for VL models
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                    device_map=self.device,
                    trust_remote_code=True
                )
                logger.info("Loaded model using AutoModelForVision2Seq")
            except Exception as e:
                logger.warning(f"Failed to load with AutoModelForVision2Seq: {e}")
                # Fall back to CausalLM
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                    device_map=self.device,
                    trust_remote_code=True
                )
                logger.info("Loaded model using AutoModelForCausalLM")
            
            logger.info(f"Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def extract_audio(self, video_path, output_path=None):
        """
        Extract audio from a video file.
        
        Args:
            video_path (str): Path to the video file.
            output_path (str): Path to save the extracted audio. If None, a temporary file is created.
            
        Returns:
            str: Path to the extracted audio file.
        """
        logger.info(f"Extracting audio from {video_path}")
        
        try:
            # Create a temporary file if output_path is not provided
            if output_path is None:
                temp_dir = tempfile.gettempdir()
                output_path = os.path.join(temp_dir, f"audio_{os.path.basename(video_path)}.wav")
            
            # Use ffmpeg to extract audio
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-q:a", "0",
                "-map", "a",
                "-y",  # Overwrite output file if it exists
                output_path
            ]
            
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
                logger.info(f"Audio extracted successfully: {output_path}")
                return output_path
            else:
                logger.error(f"Error extracting audio: {stderr}")
                return None
        
        except Exception as e:
            logger.error(f"Exception extracting audio: {e}")
            return None
    
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
    
    def detect_emotion_in_frame(self, image):
        """
        Detect emotion in a single frame using the R1-Omni model.
        
        Args:
            image (PIL.Image): The image to analyze.
            
        Returns:
            dict: Dictionary with emotion detection results.
        """
        prompt = (
            "As an emotional recognition expert; what emotion is conveyed in this image? "
            "Output the thinking process in <think> </think> and final emotion in <answer> </answer> tags."
        )
        
        try:
            # Prepare inputs
            inputs = self.processor(
                text=prompt,
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
            response = response.split(prompt)[-1].strip()
            
            # Extract thinking process and answer
            thinking = ""
            emotion = ""
            
            if "<think>" in response and "</think>" in response:
                thinking = response.split("<think>")[1].split("</think>")[0].strip()
            
            if "<answer>" in response and "</answer>" in response:
                emotion = response.split("<answer>")[1].split("</answer>")[0].strip().lower()
            
            # If no structured answer, try to extract the emotion from the response
            if not emotion:
                for category in self.emotion_categories:
                    if category in response.lower():
                        emotion = category
                        break
            
            return {
                "emotion": emotion,
                "thinking": thinking,
                "full_response": response
            }
        
        except Exception as e:
            logger.error(f"Error detecting emotion in frame: {e}")
            return {
                "emotion": "unknown",
                "thinking": f"Error: {str(e)}",
                "full_response": f"Error: {str(e)}"
            }
    
    def detect_emotion_in_audio(self, audio_path):
        """
        Detect emotion in audio using the R1-Omni model.
        
        Args:
            audio_path (str): Path to the audio file.
            
        Returns:
            dict: Dictionary with emotion detection results.
        """
        prompt = (
            "<audio>\n"
            "As an emotional recognition expert; what emotion is conveyed in this audio? "
            "Output the thinking process in <think> </think> and final emotion in <answer> </answer> tags."
        )
        
        try:
            # Prepare inputs with audio
            inputs = self.processor(
                text=prompt,
                audio=audio_path,
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
            response = response.split(prompt)[-1].strip()
            
            # Extract thinking process and answer
            thinking = ""
            emotion = ""
            
            if "<think>" in response and "</think>" in response:
                thinking = response.split("<think>")[1].split("</think>")[0].strip()
            
            if "<answer>" in response and "</answer>" in response:
                emotion = response.split("<answer>")[1].split("</answer>")[0].strip().lower()
            
            # If no structured answer, try to extract the emotion from the response
            if not emotion:
                for category in self.emotion_categories:
                    if category in response.lower():
                        emotion = category
                        break
            
            return {
                "emotion": emotion,
                "thinking": thinking,
                "full_response": response
            }
        
        except Exception as e:
            logger.error(f"Error detecting emotion in audio: {e}")
            return {
                "emotion": "unknown",
                "thinking": f"Error: {str(e)}",
                "full_response": f"Error: {str(e)}"
            }
    
    def analyze_emotional_arc(self, frame_emotions):
        """
        Analyze the emotional arc based on frame-by-frame emotions.
        
        Args:
            frame_emotions (list): List of emotions detected in frames.
            
        Returns:
            dict: Dictionary with emotional arc analysis.
        """
        # Count emotion occurrences
        emotion_counts = {}
        for frame_data in frame_emotions:
            emotion = frame_data["emotion"]
            if emotion in emotion_counts:
                emotion_counts[emotion] += 1
            else:
                emotion_counts[emotion] = 1
        
        # Determine dominant emotion
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "unknown"
        
        # Analyze emotional transitions
        transitions = []
        for i in range(1, len(frame_emotions)):
            prev_emotion = frame_emotions[i-1]["emotion"]
            curr_emotion = frame_emotions[i]["emotion"]
            if prev_emotion != curr_emotion:
                transitions.append({
                    "from": prev_emotion,
                    "to": curr_emotion,
                    "frame_index": i
                })
        
        # Determine emotional complexity
        complexity = len(set([frame_data["emotion"] for frame_data in frame_emotions]))
        
        return {
            "dominant_emotion": dominant_emotion,
            "emotion_counts": emotion_counts,
            "transitions": transitions,
            "complexity": complexity
        }
    
    def calculate_emotional_impact(self, emotional_arc, audio_emotion):
        """
        Calculate the emotional impact score based on the emotional arc and audio emotion.
        
        Args:
            emotional_arc (dict): Emotional arc analysis.
            audio_emotion (dict): Audio emotion detection results.
            
        Returns:
            dict: Dictionary with emotional impact scores.
        """
        # Define impact factors
        impact_factors = {
            "complexity": 0.3,  # Higher complexity can mean more engaging
            "consistency": 0.3,  # Consistency between audio and visual emotions
            "intensity": 0.4    # Intensity of emotions (some emotions have higher impact)
        }
        
        # Calculate complexity score (normalized)
        max_complexity = 12  # Maximum number of emotion categories
        complexity_score = min(emotional_arc["complexity"] / max_complexity, 1.0)
        
        # Calculate consistency score
        consistency_score = 0.0
        if audio_emotion["emotion"] == emotional_arc["dominant_emotion"]:
            consistency_score = 1.0
        elif audio_emotion["emotion"] in emotional_arc["emotion_counts"]:
            # Partial consistency if audio emotion appears in frames
            consistency_score = 0.5
        
        # Calculate intensity score
        high_intensity_emotions = ["angry", "surprised", "fearful", "excited"]
        medium_intensity_emotions = ["happy", "sad", "disgusted", "disappointed", "anxious"]
        low_intensity_emotions = ["neutral", "bored", "confused"]
        
        intensity_score = 0.0
        dominant_emotion = emotional_arc["dominant_emotion"]
        
        if dominant_emotion in high_intensity_emotions:
            intensity_score = 1.0
        elif dominant_emotion in medium_intensity_emotions:
            intensity_score = 0.7
        elif dominant_emotion in low_intensity_emotions:
            intensity_score = 0.4
        
        # Calculate overall impact score
        impact_score = (
            impact_factors["complexity"] * complexity_score +
            impact_factors["consistency"] * consistency_score +
            impact_factors["intensity"] * intensity_score
        )
        
        return {
            "overall_impact": impact_score,
            "complexity_score": complexity_score,
            "consistency_score": consistency_score,
            "intensity_score": intensity_score,
            "factors": impact_factors
        }
    
    def analyze_video(self, video_path, num_frames=5):
        """
        Perform comprehensive emotional analysis on a video using R1-Omni capabilities.
        
        Args:
            video_path (str): Path to the video file.
            num_frames (int): Number of frames to analyze.
            
        Returns:
            dict: Dictionary with comprehensive emotional analysis results.
        """
        logger.info(f"Starting comprehensive emotional analysis of {video_path}")
        
        # Extract frames from the video
        frames = self.extract_frames(video_path, num_frames)
        
        if not frames:
            logger.error(f"No frames extracted from {video_path}")
            return {
                "error": "Failed to extract frames from video",
                "success": False
            }
        
        # Extract audio from the video
        audio_path = self.extract_audio(video_path)
        
        if not audio_path:
            logger.warning(f"Failed to extract audio from {video_path}, proceeding with visual analysis only")
        
        # Analyze each frame for emotions
        frame_emotions = []
        for i, frame in enumerate(frames):
            logger.info(f"Analyzing emotions in frame {i+1}/{len(frames)}")
            
            # Save frame for debugging
            frame_path = f"emotion_frame_{i+1}.jpg"
            frame.save(frame_path)
            logger.info(f"Saved frame to {frame_path}")
            
            # Detect emotion in frame
            emotion_result = self.detect_emotion_in_frame(frame)
            emotion_result["frame_index"] = i
            frame_emotions.append(emotion_result)
        
        # Analyze audio for emotions
        audio_emotion = None
        if audio_path:
            logger.info(f"Analyzing emotions in audio")
            audio_emotion = self.detect_emotion_in_audio(audio_path)
        else:
            audio_emotion = {
                "emotion": "unknown",
                "thinking": "Audio extraction failed",
                "full_response": "Audio extraction failed"
            }
        
        # Analyze emotional arc
        emotional_arc = self.analyze_emotional_arc(frame_emotions)
        
        # Calculate emotional impact
        emotional_impact = self.calculate_emotional_impact(emotional_arc, audio_emotion)
        
        # Generate overall analysis
        overall_analysis = {
            "video_path": video_path,
            "frames_analyzed": len(frames),
            "frame_emotions": frame_emotions,
            "audio_emotion": audio_emotion,
            "emotional_arc": emotional_arc,
            "emotional_impact": emotional_impact,
            "success": True
        }
        
        # Generate a summary
        summary = self._generate_summary(overall_analysis)
        overall_analysis["summary"] = summary
        
        # Save the analysis results
        output_path = os.path.splitext(video_path)[0] + "_emotion_analysis.json"
        with open(output_path, 'w') as f:
            json.dump(overall_analysis, f, indent=2)
        
        logger.info(f"Emotional analysis complete. Results saved to {output_path}")
        
        return overall_analysis
    
    def _generate_summary(self, analysis):
        """
        Generate a summary of the emotional analysis.
        
        Args:
            analysis (dict): The complete analysis results.
            
        Returns:
            str: A summary of the analysis.
        """
        # Extract key information
        num_frames = analysis["frames_analyzed"]
        dominant_emotion = analysis["emotional_arc"]["dominant_emotion"]
        audio_emotion = analysis["audio_emotion"]["emotion"]
        impact_score = analysis["emotional_impact"]["overall_impact"]
        complexity = analysis["emotional_arc"]["complexity"]
        
        # Create the summary
        summary = (
            f"Emotional Analysis Summary:\n"
            f"- Analyzed {num_frames} frames\n"
            f"- Dominant visual emotion: {dominant_emotion}\n"
            f"- Audio emotion: {audio_emotion}\n"
            f"- Emotional complexity: {complexity} different emotions\n"
            f"- Overall emotional impact score: {impact_score:.2f}/1.0\n\n"
        )
        
        # Add emotional arc description
        transitions = analysis["emotional_arc"]["transitions"]
        if transitions:
            summary += "Emotional Arc:\n"
            for i, transition in enumerate(transitions):
                summary += f"- Frame {transition['frame_index']}: {transition['from']} → {transition['to']}\n"
        else:
            summary += f"Emotional Arc: Consistent {dominant_emotion} throughout the video\n"
        
        # Add recommendations based on emotional analysis
        summary += "\nRecommendations:\n"
        
        if impact_score < 0.4:
            summary += "- Consider adding more emotional variety to increase viewer engagement\n"
        elif impact_score > 0.7:
            summary += "- Strong emotional impact detected, suitable for engaging content\n"
        
        if audio_emotion != dominant_emotion and audio_emotion != "unknown":
            summary += f"- Audio emotion ({audio_emotion}) differs from visual emotion ({dominant_emotion}), which may create an interesting contrast or confusion\n"
        
        return summary

# Import subprocess for audio extraction
import subprocess

# Main function for standalone testing
def main():
    """Main function for testing the EmotionDetector."""
    import argparse
    
    parser = argparse.ArgumentParser(description="R1-Omni Emotion Detector")
    parser.add_argument("--video", required=True, help="Path to the video file")
    parser.add_argument("--frames", type=int, default=5, help="Number of frames to analyze")
    parser.add_argument("--model", default="StarJiaxing/R1-Omni-0.5B", help="Model name or path")
    args = parser.parse_args()
    
    # Initialize the detector
    detector = EmotionDetector(model_name=args.model)
    
    # Analyze the video
    results = detector.analyze_video(args.video, args.frames)
    
    # Print the summary
    print("\n" + "="*50)
    print("EMOTIONAL ANALYSIS SUMMARY")
    print("="*50)
    print(results["summary"])
    print("="*50)

if __name__ == "__main__":
    main() 