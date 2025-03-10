#!/usr/bin/env python3
"""
Prompt Enhancement Script for Movie Maker

This script reads prompts from a file, enhances them using templates,
and saves the enhanced prompts to a new file for later use in video generation.
"""

import os
import sys
import argparse
import random
from datetime import datetime

class TemplateEnhancer:
    """Class to enhance prompts using predefined templates and details."""
    
    def __init__(self):
        """Initialize the template enhancer with predefined details."""
        # Visual attributes
        self.colors = ["vibrant", "rich", "deep", "bright", "saturated", "vivid", "glowing"]
        self.textures = ["smooth", "rough", "glossy", "matte", "textured", "polished", "detailed"]
        self.materials = ["metallic", "wooden", "glass", "plastic", "fabric", "stone", "crystal"]
        
        # Lighting conditions
        self.lighting = [
            "soft natural lighting", "dramatic side lighting", "warm golden hour light",
            "cool blue lighting", "high contrast lighting", "diffused lighting", "backlit scene"
        ]
        
        # Camera angles and movements
        self.camera = [
            "close-up shot", "wide angle view", "tracking shot", "slow motion capture",
            "dynamic camera movement", "steady cam shot", "low angle perspective"
        ]
        
        # Temporal progression
        self.movement = [
            "gradually changing", "smoothly transitioning", "rhythmically moving",
            "continuously evolving", "dynamically shifting", "steadily progressing"
        ]
        
        # Background details
        self.background = [
            "blurred background", "minimalist setting", "detailed environment",
            "abstract backdrop", "natural scenery", "studio setting", "gradient background"
        ]
        
        # Cinematic qualities
        self.cinematic = [
            "cinematic quality", "professional production", "high-definition detail",
            "film-like aesthetic", "movie-quality visuals", "professional cinematography"
        ]
    
    def enhance_prompt(self, prompt):
        """Enhance a prompt to make it more detailed and descriptive."""
        # Select random elements from each category
        color = random.choice(self.colors)
        texture = random.choice(self.textures)
        material = random.choice(self.materials)
        light = random.choice(self.lighting)
        cam = random.choice(self.camera)
        move = random.choice(self.movement)
        bg = random.choice(self.background)
        cinema = random.choice(self.cinematic)
        
        # Create enhanced prompt with selected details
        enhanced_prompt = (
            f"{prompt}, with {color} colors and {texture} textures. "
            f"The scene features {light} and is captured in a {cam}. "
            f"The movement is {move} throughout the sequence, set against a {bg}. "
            f"The overall look has a {cinema}."
        )
        
        print(f"Original prompt: {prompt}")
        print(f"Enhanced prompt: {enhanced_prompt}")
        
        return enhanced_prompt

def enhance_prompts_from_file(input_file, output_file=None):
    """Enhance all prompts in the given file and save to a new file."""
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return False
    
    # Read prompts from file
    with open(input_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    if not prompts:
        print("Error: No prompts found in the file.")
        return False
    
    print(f"Found {len(prompts)} prompts to enhance.")
    
    # Initialize template enhancer
    enhancer = TemplateEnhancer()
    
    # Create output file if not specified
    if output_file is None:
        os.makedirs("enhanced_prompts", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"enhanced_prompts/enhanced_{timestamp}.txt"
    
    # Process each prompt
    enhanced_prompts = []
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\nEnhancing prompt {i}/{len(prompts)}:")
        
        try:
            enhanced_prompt = enhancer.enhance_prompt(prompt)
            enhanced_prompts.append(enhanced_prompt)
            
            # Save to file as we go
            with open(output_file, "a") as f:
                f.write(f"Original: {prompt}\nEnhanced: {enhanced_prompt}\n\n")
                
        except Exception as e:
            print(f"Error enhancing prompt: {e}")
            enhanced_prompts.append(prompt)  # Use original if enhancement fails
            
            # Save the original to the file
            with open(output_file, "a") as f:
                f.write(f"Original: {prompt}\nEnhanced: {prompt} (enhancement failed)\n\n")
    
    print(f"\nPrompt enhancement complete!")
    print(f"Enhanced prompts saved to: {output_file}")
    
    # Also create a file with just the enhanced prompts for easy use
    enhanced_only_file = output_file.replace(".txt", "_only.txt")
    with open(enhanced_only_file, "w") as f:
        for prompt in enhanced_prompts:
            f.write(f"{prompt}\n")
    
    print(f"Enhanced prompts only (for video generation) saved to: {enhanced_only_file}")
    
    return enhanced_only_file

def main():
    parser = argparse.ArgumentParser(description="Enhance prompts for video generation")
    parser.add_argument("input_file", help="File containing prompts (one per line)")
    parser.add_argument("--output", help="Output file for enhanced prompts")
    
    args = parser.parse_args()
    
    # Enhance prompts
    enhance_prompts_from_file(
        args.input_file,
        args.output
    )

if __name__ == "__main__":
    main() 