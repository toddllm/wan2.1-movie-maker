#!/usr/bin/env python3
"""
Recommender System - R1-V and R1-Omni Integration

This module implements a recommendation system based on the analysis from R1-V and R1-Omni:
- Generate improvement suggestions
- Recommend optimal clip combinations
- Suggest emotional enhancements

It integrates with the Movie Maker application to provide recommendations for video creation.
"""

import os
import sys
import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("recommender.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("recommender")

class Recommender:
    """
    Recommendation system based on R1-V and R1-Omni analysis.
    """
    
    def __init__(self):
        """Initialize the recommender system."""
        logger.info("Initializing Recommender")
        
        # Define recommendation categories
        self.recommendation_categories = [
            "visual_quality",
            "emotional_impact",
            "scene_composition",
            "clip_combination",
            "prompt_enhancement"
        ]
    
    def load_analysis(self, r1v_analysis_path, emotion_analysis_path=None):
        """
        Load analysis results from R1-V and R1-Omni.
        
        Args:
            r1v_analysis_path (str): Path to the R1-V analysis JSON file.
            emotion_analysis_path (str): Path to the R1-Omni emotion analysis JSON file.
            
        Returns:
            tuple: (r1v_analysis, emotion_analysis) dictionaries.
        """
        r1v_analysis = None
        emotion_analysis = None
        
        # Load R1-V analysis
        if r1v_analysis_path and os.path.exists(r1v_analysis_path):
            try:
                with open(r1v_analysis_path, 'r') as f:
                    r1v_analysis = json.load(f)
                logger.info(f"Loaded R1-V analysis from {r1v_analysis_path}")
            except Exception as e:
                logger.error(f"Error loading R1-V analysis: {e}")
        
        # Load emotion analysis if provided
        if emotion_analysis_path and os.path.exists(emotion_analysis_path):
            try:
                with open(emotion_analysis_path, 'r') as f:
                    emotion_analysis = json.load(f)
                logger.info(f"Loaded emotion analysis from {emotion_analysis_path}")
            except Exception as e:
                logger.error(f"Error loading emotion analysis: {e}")
        
        return r1v_analysis, emotion_analysis
    
    def generate_visual_quality_recommendations(self, r1v_analysis):
        """
        Generate recommendations for improving visual quality.
        
        Args:
            r1v_analysis (dict): R1-V analysis results.
            
        Returns:
            list: List of visual quality recommendations.
        """
        recommendations = []
        
        if not r1v_analysis or "average_quality_score" not in r1v_analysis:
            return ["No visual quality data available for recommendations."]
        
        avg_quality = r1v_analysis["average_quality_score"]
        
        # General quality recommendations
        if avg_quality < 5.0:
            recommendations.append("The overall visual quality is low. Consider adjusting lighting and composition.")
        elif avg_quality < 7.0:
            recommendations.append("The visual quality is moderate. Consider fine-tuning clarity and color balance.")
        else:
            recommendations.append("The visual quality is good. Minor adjustments may still enhance specific aspects.")
        
        # Frame-specific recommendations
        if "frame_analyses" in r1v_analysis:
            for frame_analysis in r1v_analysis["frame_analyses"]:
                if "quality" in frame_analysis and "scores" in frame_analysis["quality"]:
                    scores = frame_analysis["quality"]["scores"]
                    
                    # Check for low scores in specific aspects
                    for aspect, score in scores.items():
                        if score < 5.0:
                            recommendations.append(f"Improve {aspect} (currently rated {score}/10) in frame {frame_analysis['frame_index'] + 1}.")
        
        return recommendations
    
    def generate_emotional_impact_recommendations(self, emotion_analysis):
        """
        Generate recommendations for improving emotional impact.
        
        Args:
            emotion_analysis (dict): R1-Omni emotion analysis results.
            
        Returns:
            list: List of emotional impact recommendations.
        """
        recommendations = []
        
        if not emotion_analysis or "emotional_impact" not in emotion_analysis:
            return ["No emotional analysis data available for recommendations."]
        
        impact = emotion_analysis["emotional_impact"]
        arc = emotion_analysis["emotional_arc"]
        
        # Overall impact recommendations
        overall_impact = impact.get("overall_impact", 0)
        if overall_impact < 0.4:
            recommendations.append("The emotional impact is low. Consider adding more emotional variety or intensity.")
        elif overall_impact < 0.7:
            recommendations.append("The emotional impact is moderate. Consider enhancing the dominant emotion or adding contrast.")
        else:
            recommendations.append("The emotional impact is strong. Maintain this level of emotional engagement in future clips.")
        
        # Complexity recommendations
        complexity_score = impact.get("complexity_score", 0)
        if complexity_score < 0.3:
            recommendations.append("The emotional complexity is low. Consider introducing more emotional variety.")
        
        # Consistency recommendations
        consistency_score = impact.get("consistency_score", 0)
        if consistency_score < 0.5:
            recommendations.append("There's a mismatch between audio and visual emotions. Consider aligning them for stronger impact.")
        
        # Transition recommendations
        transitions = arc.get("transitions", [])
        if not transitions:
            recommendations.append("The emotional arc is flat. Consider adding emotional transitions for more engaging content.")
        elif len(transitions) > 3:
            recommendations.append("The emotional arc has many transitions. Consider simplifying for a more coherent experience.")
        
        return recommendations
    
    def generate_scene_composition_recommendations(self, r1v_analysis):
        """
        Generate recommendations for improving scene composition.
        
        Args:
            r1v_analysis (dict): R1-V analysis results.
            
        Returns:
            list: List of scene composition recommendations.
        """
        recommendations = []
        
        if not r1v_analysis or "frame_analyses" not in r1v_analysis:
            return ["No scene composition data available for recommendations."]
        
        # Analyze object counts
        if "overall_object_counts" in r1v_analysis:
            object_counts = r1v_analysis["overall_object_counts"]
            
            # Check for overcrowded scenes
            total_objects = sum(object_counts.values())
            if total_objects > 10:
                recommendations.append(f"The scene contains many objects ({total_objects}). Consider simplifying for clearer focus.")
            elif total_objects < 3:
                recommendations.append("The scene has few objects. Consider adding more elements for visual interest.")
        
        # Analyze composition aspects
        for frame_analysis in r1v_analysis["frame_analyses"]:
            if "composition" in frame_analysis and "aspects" in frame_analysis["composition"]:
                aspects = frame_analysis["composition"]["aspects"]
                
                # Check for specific composition issues
                if "Balance" in aspects and "imbalanced" in aspects["Balance"].lower():
                    recommendations.append(f"Frame {frame_analysis['frame_index'] + 1} has imbalanced composition. Consider reframing.")
                
                if "Foreground" in aspects and "empty" in aspects["Foreground"].lower():
                    recommendations.append(f"Frame {frame_analysis['frame_index'] + 1} has an empty foreground. Consider adding a focal point.")
        
        return recommendations
    
    def generate_clip_combination_recommendations(self, r1v_analysis, emotion_analysis):
        """
        Generate recommendations for optimal clip combinations.
        
        Args:
            r1v_analysis (dict): R1-V analysis results.
            emotion_analysis (dict): R1-Omni emotion analysis results.
            
        Returns:
            list: List of clip combination recommendations.
        """
        recommendations = []
        
        # Check if we have both analyses
        if not r1v_analysis or not emotion_analysis:
            return ["Insufficient data for clip combination recommendations."]
        
        # Get emotional information
        dominant_emotion = emotion_analysis.get("emotional_arc", {}).get("dominant_emotion", "unknown")
        
        # Generate recommendations based on emotion and visual quality
        if dominant_emotion != "unknown":
            # Recommendations for different emotions
            if dominant_emotion in ["happy", "excited"]:
                recommendations.append("For happy/excited content, combine with clips that have bright colors and dynamic movement.")
            elif dominant_emotion in ["sad", "disappointed"]:
                recommendations.append("For sad content, combine with clips that have muted colors and slower pacing.")
            elif dominant_emotion in ["angry", "fearful"]:
                recommendations.append("For intense emotions, combine with clips that have strong contrast and dramatic composition.")
            elif dominant_emotion in ["neutral", "bored"]:
                recommendations.append("For neutral content, combine with clips that have more emotional variety to increase engagement.")
        
        # Visual quality based recommendations
        avg_quality = r1v_analysis.get("average_quality_score", 0)
        if avg_quality < 6.0:
            recommendations.append("Consider combining with higher quality clips to maintain viewer engagement.")
        
        # General combination recommendations
        recommendations.append("For smoother transitions, combine clips with similar color palettes or thematic elements.")
        recommendations.append("Consider emotional pacing when combining clips - build toward emotional peaks for maximum impact.")
        
        return recommendations
    
    def generate_prompt_enhancement_suggestions(self, r1v_analysis, emotion_analysis, original_prompt=None):
        """
        Generate suggestions for enhancing prompts based on analysis.
        
        Args:
            r1v_analysis (dict): R1-V analysis results.
            emotion_analysis (dict): R1-Omni emotion analysis results.
            original_prompt (str): The original prompt used to generate the video.
            
        Returns:
            dict: Enhanced prompt suggestions.
        """
        suggestions = {
            "visual_elements": [],
            "emotional_elements": [],
            "composition_elements": [],
            "enhanced_prompt": ""
        }
        
        # Extract key information from analyses
        visual_elements = []
        if r1v_analysis and "overall_object_counts" in r1v_analysis:
            # Get the most common objects
            objects = list(r1v_analysis["overall_object_counts"].items())
            objects.sort(key=lambda x: x[1], reverse=True)
            visual_elements = [obj[0] for obj in objects[:3]]
        
        emotional_elements = []
        if emotion_analysis and "emotional_arc" in emotion_analysis:
            dominant_emotion = emotion_analysis["emotional_arc"].get("dominant_emotion", "")
            if dominant_emotion:
                emotional_elements.append(dominant_emotion)
        
        # Generate suggestions for visual elements
        if visual_elements:
            suggestions["visual_elements"] = [
                f"Include {', '.join(visual_elements)} for visual consistency",
                "Specify lighting conditions for better control over mood",
                "Add details about camera angle and distance"
            ]
        
        # Generate suggestions for emotional elements
        if emotional_elements:
            emotion = emotional_elements[0]
            suggestions["emotional_elements"] = [
                f"Emphasize {emotion} emotion through character expressions or actions",
                "Specify the emotional tone or atmosphere",
                "Include emotional transitions for dynamic content"
            ]
        
        # Generate suggestions for composition
        suggestions["composition_elements"] = [
            "Specify foreground and background elements separately",
            "Include details about spatial arrangement",
            "Mention color palette for consistent visual style"
        ]
        
        # Create an enhanced prompt if original prompt is provided
        if original_prompt:
            # Start with the original prompt
            enhanced_prompt = original_prompt
            
            # Add visual elements if not already present
            for element in visual_elements:
                if element.lower() not in original_prompt.lower():
                    enhanced_prompt += f", with {element}"
            
            # Add emotional elements if not already present
            if emotional_elements and emotional_elements[0].lower() not in original_prompt.lower():
                enhanced_prompt += f", conveying a {emotional_elements[0]} mood"
            
            # Add composition elements
            if "foreground" not in original_prompt.lower() and "background" not in original_prompt.lower():
                enhanced_prompt += ", with clear foreground and background separation"
            
            suggestions["enhanced_prompt"] = enhanced_prompt
        
        return suggestions
    
    def generate_recommendations(self, r1v_analysis_path, emotion_analysis_path=None, original_prompt=None):
        """
        Generate comprehensive recommendations based on R1-V and R1-Omni analyses.
        
        Args:
            r1v_analysis_path (str): Path to the R1-V analysis JSON file.
            emotion_analysis_path (str): Path to the R1-Omni emotion analysis JSON file.
            original_prompt (str): The original prompt used to generate the video.
            
        Returns:
            dict: Comprehensive recommendations.
        """
        logger.info(f"Generating recommendations based on analyses")
        
        # Load analyses
        r1v_analysis, emotion_analysis = self.load_analysis(r1v_analysis_path, emotion_analysis_path)
        
        # Check if we have valid analyses
        if not r1v_analysis:
            return {
                "error": "No valid R1-V analysis available",
                "success": False
            }
        
        # Generate recommendations for each category
        visual_quality_recs = self.generate_visual_quality_recommendations(r1v_analysis)
        
        emotional_impact_recs = []
        if emotion_analysis:
            emotional_impact_recs = self.generate_emotional_impact_recommendations(emotion_analysis)
        
        scene_composition_recs = self.generate_scene_composition_recommendations(r1v_analysis)
        
        clip_combination_recs = []
        if emotion_analysis:
            clip_combination_recs = self.generate_clip_combination_recommendations(r1v_analysis, emotion_analysis)
        
        prompt_enhancement = self.generate_prompt_enhancement_suggestions(r1v_analysis, emotion_analysis, original_prompt)
        
        # Compile all recommendations
        recommendations = {
            "visual_quality": visual_quality_recs,
            "emotional_impact": emotional_impact_recs,
            "scene_composition": scene_composition_recs,
            "clip_combination": clip_combination_recs,
            "prompt_enhancement": prompt_enhancement,
            "success": True
        }
        
        # Generate a summary
        summary = self._generate_summary(recommendations)
        recommendations["summary"] = summary
        
        # Save the recommendations
        video_path = r1v_analysis.get("video_path", "unknown_video")
        output_path = os.path.splitext(video_path)[0] + "_recommendations.json"
        with open(output_path, 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        logger.info(f"Recommendations generated and saved to {output_path}")
        
        return recommendations
    
    def _generate_summary(self, recommendations):
        """
        Generate a summary of the recommendations.
        
        Args:
            recommendations (dict): The complete recommendations.
            
        Returns:
            str: A summary of the recommendations.
        """
        # Create the summary
        summary = "Recommendation Summary:\n\n"
        
        # Add visual quality recommendations
        summary += "Visual Quality:\n"
        for rec in recommendations["visual_quality"][:2]:  # Limit to top 2
            summary += f"- {rec}\n"
        
        # Add emotional impact recommendations if available
        if recommendations["emotional_impact"]:
            summary += "\nEmotional Impact:\n"
            for rec in recommendations["emotional_impact"][:2]:  # Limit to top 2
                summary += f"- {rec}\n"
        
        # Add scene composition recommendations
        summary += "\nScene Composition:\n"
        for rec in recommendations["scene_composition"][:2]:  # Limit to top 2
            summary += f"- {rec}\n"
        
        # Add clip combination recommendations if available
        if recommendations["clip_combination"]:
            summary += "\nClip Combination:\n"
            for rec in recommendations["clip_combination"][:2]:  # Limit to top 2
                summary += f"- {rec}\n"
        
        # Add prompt enhancement suggestions
        summary += "\nPrompt Enhancement:\n"
        for category, suggestions in recommendations["prompt_enhancement"].items():
            if category == "enhanced_prompt" and suggestions:
                summary += f"- Enhanced prompt: {suggestions}\n"
                break
        
        return summary

# Main function for standalone testing
def main():
    """Main function for testing the Recommender."""
    import argparse
    
    parser = argparse.ArgumentParser(description="R1-V and R1-Omni Recommender")
    parser.add_argument("--r1v", required=True, help="Path to the R1-V analysis JSON file")
    parser.add_argument("--emotion", help="Path to the R1-Omni emotion analysis JSON file")
    parser.add_argument("--prompt", help="Original prompt used to generate the video")
    args = parser.parse_args()
    
    # Initialize the recommender
    recommender = Recommender()
    
    # Generate recommendations
    results = recommender.generate_recommendations(args.r1v, args.emotion, args.prompt)
    
    # Print the summary
    print("\n" + "="*50)
    print("RECOMMENDATION SUMMARY")
    print("="*50)
    print(results["summary"])
    print("="*50)

if __name__ == "__main__":
    main() 