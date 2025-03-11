#!/usr/bin/env python3
"""
Re-enhance Prompts

This script finds the 8 most recent enhanced prompts and re-enhances them
using a different enhancement technique.
"""

import os
import sys
import json
import argparse
import logging
import glob
import re
from datetime import datetime
from pathlib import Path
from enhance_prompts import TemplateEnhancer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("re_enhance_prompts.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("re_enhance_prompts")

# Paths
ENHANCED_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "enhanced_prompts")
os.makedirs(ENHANCED_PROMPTS_DIR, exist_ok=True)

# Initialize the prompt enhancer
prompt_enhancer = TemplateEnhancer()

def find_recent_enhanced_prompts(count=8):
    """
    Find the most recent enhanced prompt files
    
    Args:
        count: Number of recent files to find
    
    Returns:
        List of paths to enhanced prompt files
    """
    # Find all enhanced prompt files
    enhanced_files = glob.glob(os.path.join(ENHANCED_PROMPTS_DIR, "enhanced_*.txt"))
    
    # Sort by modification time (newest first)
    enhanced_files.sort(key=os.path.getmtime, reverse=True)
    
    # Return the specified number of files
    return enhanced_files[:count]

def extract_prompts_from_file(file_path):
    """
    Extract prompts from an enhanced prompts file
    
    Args:
        file_path: Path to the enhanced prompts file
    
    Returns:
        List of prompts
    """
    with open(file_path, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    return prompts

def re_enhance_prompt(prompt):
    """
    Re-enhance a prompt using a different technique
    
    Args:
        prompt: The original enhanced prompt
    
    Returns:
        re_enhanced_prompt: The re-enhanced prompt
    """
    # Extract the core prompt by removing enhancement patterns
    # This is a simple approach - we're assuming the original prompt is at the beginning
    # and the enhancements are added after
    core_prompt = re.sub(r', with.*$', '', prompt).strip()
    
    # If we couldn't extract a core prompt, use the original
    if not core_prompt or len(core_prompt) < 5:
        core_prompt = prompt
    
    # Add different enhancement patterns
    re_enhanced_prompt = f"{core_prompt}, featuring dynamic motion and fluid transitions. "
    re_enhanced_prompt += "The scene has dramatic lighting with high contrast between light and shadow. "
    re_enhanced_prompt += "The camera moves smoothly, following the action with precision. "
    re_enhanced_prompt += "The composition emphasizes depth and perspective, creating a sense of immersion. "
    re_enhanced_prompt += "The overall aesthetic is cinematic with rich colors and detailed textures."
    
    logger.info(f"Original prompt: {prompt}")
    logger.info(f"Core prompt: {core_prompt}")
    logger.info(f"Re-enhanced prompt: {re_enhanced_prompt}")
    
    return re_enhanced_prompt

def save_prompts_to_file(prompts, output_file):
    """
    Save prompts to a file
    
    Args:
        prompts: List of prompts
        output_file: Path to save prompts
    """
    with open(output_file, 'w') as f:
        for prompt in prompts:
            f.write(prompt + '\n')
    
    logger.info(f"Saved {len(prompts)} prompts to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Re-enhance recent prompts")
    parser.add_argument("--count", "-c", type=int, default=8, help="Number of recent prompts to re-enhance")
    parser.add_argument("--output", "-o", help="Output file for re-enhanced prompts")
    
    args = parser.parse_args()
    
    # Generate default output filename if not provided
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(ENHANCED_PROMPTS_DIR, f"re_enhanced_{timestamp}.txt")
    
    # Find recent enhanced prompt files
    enhanced_files = find_recent_enhanced_prompts(args.count)
    logger.info(f"Found {len(enhanced_files)} recent enhanced prompt files")
    
    # Extract and re-enhance prompts
    re_enhanced_prompts = []
    for file_path in enhanced_files:
        prompts = extract_prompts_from_file(file_path)
        logger.info(f"Found {len(prompts)} prompts in {file_path}")
        
        for prompt in prompts:
            re_enhanced_prompt = re_enhance_prompt(prompt)
            re_enhanced_prompts.append(re_enhanced_prompt)
            
            # Limit to the specified count
            if len(re_enhanced_prompts) >= args.count:
                break
        
        # Limit to the specified count
        if len(re_enhanced_prompts) >= args.count:
            break
    
    # Save re-enhanced prompts to file
    save_prompts_to_file(re_enhanced_prompts[:args.count], args.output)
    
    logger.info(f"Re-enhanced {len(re_enhanced_prompts[:args.count])} prompts")
    logger.info(f"Output file: {args.output}")
    
    return args.output

if __name__ == "__main__":
    main() 