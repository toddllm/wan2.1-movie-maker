#!/usr/bin/env python3
"""
Scoring System for Video Analysis

This module provides a scoring system to evaluate the quality of video analysis
and prompt generation based on various metrics.
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from difflib import SequenceMatcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scoring_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("scoring_system")

class ScoringSystem:
    """
    A system for scoring video analysis quality based on various metrics.
    """
    
    def __init__(self, weights=None):
        """
        Initialize the scoring system with customizable weights for different metrics.
        
        Args:
            weights (dict): Custom weights for different metrics. If None, default weights are used.
        """
        # Default weights for different metrics (sum to 1.0)
        self.default_weights = {
            "prompt_relevance": 0.25,  # How well the prompt matches the video content
            "frame_consistency": 0.20,  # Consistency between frame descriptions
            "detail_level": 0.20,      # Level of detail in descriptions
            "technical_accuracy": 0.15, # Accuracy of technical elements described
            "creative_elements": 0.10,  # Creative elements in the prompt
            "grammar_quality": 0.10,    # Grammar and language quality
        }
        
        # Use custom weights if provided, otherwise use defaults
        self.weights = weights if weights is not None else self.default_weights
        
        # Normalize weights to ensure they sum to 1.0
        weight_sum = sum(self.weights.values())
        if weight_sum != 1.0:
            for key in self.weights:
                self.weights[key] /= weight_sum
    
    def calculate_text_similarity(self, text1, text2):
        """
        Calculate similarity between two text strings using SequenceMatcher.
        
        Args:
            text1 (str): First text string
            text2 (str): Second text string
            
        Returns:
            float: Similarity score between 0.0 and 1.0
        """
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def score_prompt_relevance(self, prompt, frame_descriptions):
        """
        Score how relevant the prompt is to the frame descriptions.
        
        Args:
            prompt (str): The prompt used or generated
            frame_descriptions (list): List of frame descriptions
            
        Returns:
            float: Score between 0.0 and 1.0
        """
        # Calculate similarity between prompt and each frame description
        similarities = [self.calculate_text_similarity(prompt, desc) for desc in frame_descriptions]
        
        # Return average similarity
        return np.mean(similarities) if similarities else 0.0
    
    def score_frame_consistency(self, frame_descriptions):
        """
        Score the consistency between frame descriptions.
        
        Args:
            frame_descriptions (list): List of frame descriptions
            
        Returns:
            float: Score between 0.0 and 1.0
        """
        if len(frame_descriptions) <= 1:
            return 1.0  # Perfect consistency if only one frame
        
        # Calculate pairwise similarities between consecutive frames
        similarities = []
        for i in range(len(frame_descriptions) - 1):
            similarity = self.calculate_text_similarity(
                frame_descriptions[i], frame_descriptions[i+1]
            )
            similarities.append(similarity)
        
        # Return average similarity
        return np.mean(similarities)
    
    def score_detail_level(self, frame_descriptions):
        """
        Score the level of detail in frame descriptions.
        
        Args:
            frame_descriptions (list): List of frame descriptions
            
        Returns:
            float: Score between 0.0 and 1.0
        """
        # Calculate average word count per description
        word_counts = [len(desc.split()) for desc in frame_descriptions]
        avg_word_count = np.mean(word_counts) if word_counts else 0
        
        # Normalize to a score between 0 and 1
        # Assuming a good description has at least 50 words
        # and anything above 150 words is excellent
        return min(1.0, avg_word_count / 100)
    
    def score_technical_accuracy(self, frame_descriptions, technical_terms=None):
        """
        Score the technical accuracy of frame descriptions.
        
        Args:
            frame_descriptions (list): List of frame descriptions
            technical_terms (list): List of technical terms to look for
            
        Returns:
            float: Score between 0.0 and 1.0
        """
        if not technical_terms:
            # Default technical terms related to video/film
            technical_terms = [
                "angle", "shot", "lighting", "composition", "focus", 
                "depth", "field", "frame", "perspective", "zoom",
                "pan", "tilt", "tracking", "dolly", "crane",
                "wide", "close-up", "medium", "long", "establishing"
            ]
        
        # Count occurrences of technical terms in descriptions
        term_counts = []
        for desc in frame_descriptions:
            desc_lower = desc.lower()
            count = sum(1 for term in technical_terms if term.lower() in desc_lower)
            term_counts.append(count)
        
        # Calculate average technical term count
        avg_term_count = np.mean(term_counts) if term_counts else 0
        
        # Normalize to a score between 0 and 1
        # Assuming a good technical description has at least 3 technical terms
        return min(1.0, avg_term_count / 5)
    
    def score_creative_elements(self, prompt):
        """
        Score the creative elements in the prompt.
        
        Args:
            prompt (str): The prompt used or generated
            
        Returns:
            float: Score between 0.0 and 1.0
        """
        # Creative elements to look for
        creative_elements = [
            "color", "mood", "atmosphere", "emotion", "style",
            "theme", "aesthetic", "visual", "artistic", "dramatic",
            "cinematic", "dynamic", "vibrant", "contrast", "texture"
        ]
        
        # Count occurrences of creative elements in prompt
        prompt_lower = prompt.lower()
        count = sum(1 for element in creative_elements if element.lower() in prompt_lower)
        
        # Normalize to a score between 0 and 1
        # Assuming a good creative prompt has at least 5 creative elements
        return min(1.0, count / 8)
    
    def score_grammar_quality(self, text):
        """
        Score the grammar and language quality of text.
        
        Args:
            text (str): The text to evaluate
            
        Returns:
            float: Score between 0.0 and 1.0
        """
        # This is a simplified approach - in a real system, you might use
        # a more sophisticated grammar checker or language model
        
        # Check for common grammar issues
        grammar_issues = [
            "  ",  # Double spaces
            " ,",  # Space before comma
            " .",  # Space before period
            "  ",  # Double spaces
            "...",  # Ellipsis (not necessarily an issue, but can be overused)
            "!!"   # Multiple exclamation marks
        ]
        
        # Count occurrences of grammar issues
        issue_count = sum(text.count(issue) for issue in grammar_issues)
        
        # Calculate words per sentence (very long sentences might indicate poor grammar)
        sentences = text.split('.')
        if len(sentences) > 0:
            words_per_sentence = np.mean([len(s.split()) for s in sentences if s.strip()])
            sentence_penalty = max(0, (words_per_sentence - 25) / 15) if words_per_sentence > 25 else 0
        else:
            sentence_penalty = 0
        
        # Calculate total penalty (issues per 100 words + sentence length penalty)
        word_count = len(text.split())
        if word_count > 0:
            issue_density = (issue_count / word_count) * 100
        else:
            issue_density = 0
        
        total_penalty = min(1.0, (issue_density / 10) + sentence_penalty)
        
        # Return score (1.0 - penalty)
        return max(0.0, 1.0 - total_penalty)
    
    def calculate_overall_score(self, prompt, frame_descriptions, technical_terms=None):
        """
        Calculate an overall score based on all metrics.
        
        Args:
            prompt (str): The prompt used or generated
            frame_descriptions (list): List of frame descriptions
            technical_terms (list): List of technical terms to look for
            
        Returns:
            dict: Dictionary containing overall score and individual metric scores
        """
        # Calculate individual metric scores
        prompt_relevance = self.score_prompt_relevance(prompt, frame_descriptions)
        frame_consistency = self.score_frame_consistency(frame_descriptions)
        detail_level = self.score_detail_level(frame_descriptions)
        technical_accuracy = self.score_technical_accuracy(frame_descriptions, technical_terms)
        creative_elements = self.score_creative_elements(prompt)
        
        # Combine all frame descriptions for grammar scoring
        all_text = " ".join(frame_descriptions) + " " + prompt
        grammar_quality = self.score_grammar_quality(all_text)
        
        # Calculate weighted overall score
        overall_score = (
            self.weights["prompt_relevance"] * prompt_relevance +
            self.weights["frame_consistency"] * frame_consistency +
            self.weights["detail_level"] * detail_level +
            self.weights["technical_accuracy"] * technical_accuracy +
            self.weights["creative_elements"] * creative_elements +
            self.weights["grammar_quality"] * grammar_quality
        )
        
        # Return all scores
        return {
            "overall_score": overall_score,
            "prompt_relevance": prompt_relevance,
            "frame_consistency": frame_consistency,
            "detail_level": detail_level,
            "technical_accuracy": technical_accuracy,
            "creative_elements": creative_elements,
            "grammar_quality": grammar_quality
        }
    
    def get_feedback(self, scores):
        """
        Generate feedback based on scores.
        
        Args:
            scores (dict): Dictionary of scores from calculate_overall_score
            
        Returns:
            dict: Dictionary containing feedback for each metric and overall feedback
        """
        feedback = {}
        
        # Overall feedback
        overall_score = scores["overall_score"]
        if overall_score >= 0.8:
            feedback["overall"] = "Excellent analysis! The prompt and descriptions are detailed, consistent, and technically accurate."
        elif overall_score >= 0.6:
            feedback["overall"] = "Good analysis. There's room for improvement in some areas, but overall the quality is solid."
        elif overall_score >= 0.4:
            feedback["overall"] = "Average analysis. Several areas need improvement to better capture the video content."
        else:
            feedback["overall"] = "Poor analysis. The descriptions lack detail, consistency, and technical accuracy."
        
        # Prompt relevance feedback
        prompt_relevance = scores["prompt_relevance"]
        if prompt_relevance >= 0.8:
            feedback["prompt_relevance"] = "The prompt is highly relevant to the video content."
        elif prompt_relevance >= 0.6:
            feedback["prompt_relevance"] = "The prompt is mostly relevant but could better match the video content."
        elif prompt_relevance >= 0.4:
            feedback["prompt_relevance"] = "The prompt is somewhat relevant but misses key elements of the video."
        else:
            feedback["prompt_relevance"] = "The prompt has little relevance to the video content."
        
        # Frame consistency feedback
        frame_consistency = scores["frame_consistency"]
        if frame_consistency >= 0.8:
            feedback["frame_consistency"] = "The frame descriptions are highly consistent with each other."
        elif frame_consistency >= 0.6:
            feedback["frame_consistency"] = "The frame descriptions are mostly consistent but have some discrepancies."
        elif frame_consistency >= 0.4:
            feedback["frame_consistency"] = "The frame descriptions show significant inconsistencies."
        else:
            feedback["frame_consistency"] = "The frame descriptions are highly inconsistent with each other."
        
        # Detail level feedback
        detail_level = scores["detail_level"]
        if detail_level >= 0.8:
            feedback["detail_level"] = "The descriptions are very detailed and comprehensive."
        elif detail_level >= 0.6:
            feedback["detail_level"] = "The descriptions have good detail but could be more comprehensive."
        elif detail_level >= 0.4:
            feedback["detail_level"] = "The descriptions lack sufficient detail."
        else:
            feedback["detail_level"] = "The descriptions are extremely sparse and lack necessary detail."
        
        # Technical accuracy feedback
        technical_accuracy = scores["technical_accuracy"]
        if technical_accuracy >= 0.8:
            feedback["technical_accuracy"] = "The descriptions use appropriate technical terminology."
        elif technical_accuracy >= 0.6:
            feedback["technical_accuracy"] = "The descriptions use some technical terms but could be more precise."
        elif technical_accuracy >= 0.4:
            feedback["technical_accuracy"] = "The descriptions use few technical terms and lack precision."
        else:
            feedback["technical_accuracy"] = "The descriptions lack technical terminology entirely."
        
        # Creative elements feedback
        creative_elements = scores["creative_elements"]
        if creative_elements >= 0.8:
            feedback["creative_elements"] = "The prompt includes excellent creative elements and artistic direction."
        elif creative_elements >= 0.6:
            feedback["creative_elements"] = "The prompt includes good creative elements but could be more artistic."
        elif creative_elements >= 0.4:
            feedback["creative_elements"] = "The prompt includes few creative elements and lacks artistic direction."
        else:
            feedback["creative_elements"] = "The prompt lacks creative elements and artistic direction."
        
        # Grammar quality feedback
        grammar_quality = scores["grammar_quality"]
        if grammar_quality >= 0.8:
            feedback["grammar_quality"] = "The text has excellent grammar and language quality."
        elif grammar_quality >= 0.6:
            feedback["grammar_quality"] = "The text has good grammar with minor issues."
        elif grammar_quality >= 0.4:
            feedback["grammar_quality"] = "The text has several grammar and language issues."
        else:
            feedback["grammar_quality"] = "The text has poor grammar and language quality."
        
        return feedback
    
    def evaluate_analysis(self, prompt, frame_descriptions, technical_terms=None):
        """
        Evaluate the quality of video analysis.
        
        Args:
            prompt (str): The prompt used or generated
            frame_descriptions (list): List of frame descriptions
            technical_terms (list): List of technical terms to look for
            
        Returns:
            tuple: (overall_score, scores_dict, feedback_dict)
        """
        # Calculate scores
        scores = self.calculate_overall_score(prompt, frame_descriptions, technical_terms)
        
        # Generate feedback
        feedback = self.get_feedback(scores)
        
        return scores["overall_score"], scores, feedback
    
    def save_evaluation(self, video_path, prompt, scores, feedback):
        """
        Save evaluation results to a JSON file.
        
        Args:
            video_path (str): Path to the video file
            prompt (str): The prompt used or generated
            scores (dict): Dictionary of scores
            feedback (dict): Dictionary of feedback
            
        Returns:
            str: Path to the saved evaluation file
        """
        # Create evaluation data
        evaluation = {
            "video_path": video_path,
            "prompt": prompt,
            "scores": scores,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        }
        
        # Create output path
        output_path = os.path.splitext(video_path)[0] + "_evaluation.json"
        
        # Save to file
        with open(output_path, "w") as f:
            json.dump(evaluation, f, indent=2)
        
        logger.info(f"Saved evaluation to {output_path}")
        return output_path

# Add a main function for testing
if __name__ == "__main__":
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description="Test the scoring system")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to evaluate")
    parser.add_argument("--descriptions", type=str, nargs="+", required=True, help="Frame descriptions")
    
    args = parser.parse_args()
    
    # Create scoring system
    scoring_system = ScoringSystem()
    
    # Evaluate
    overall_score, scores, feedback = scoring_system.evaluate_analysis(
        args.prompt, args.descriptions
    )
    
    # Print results
    print(f"Overall Score: {overall_score:.2f}")
    print("\nIndividual Scores:")
    for metric, score in scores.items():
        if metric != "overall_score":
            print(f"  {metric}: {score:.2f}")
    
    print("\nFeedback:")
    for metric, fb in feedback.items():
        print(f"  {metric}: {fb}") 