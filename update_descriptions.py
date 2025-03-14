#!/usr/bin/env python3
"""
Script to update voice sample descriptions based on feedback.
This script compares feedback with sample descriptions and updates
the descriptions when they don't match.
"""

import os
import json
import re
import time
import logging

# Configuration
FEEDBACK_FILE = os.path.expanduser("~/movie_maker/voice_feedback_db.json")
SAMPLES_FILE = os.path.expanduser("~/movie_maker/voice_samples.js")
LOG_FILE = os.path.expanduser("~/movie_maker/description_updates.log")

# Set up logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_feedback_db():
    """Load the feedback database."""
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logging.error(f"Error reading {FEEDBACK_FILE}")
            return None
    else:
        logging.error(f"Feedback file {FEEDBACK_FILE} does not exist")
        return None

def load_samples():
    """Load the voice samples data."""
    if not os.path.exists(SAMPLES_FILE):
        logging.error(f"Samples file {SAMPLES_FILE} does not exist")
        return None
    
    try:
        with open(SAMPLES_FILE, 'r') as f:
            content = f.read()
            # Extract the JSON part from the JavaScript variable assignment
            match = re.search(r'const\s+voiceExplorationSamples\s*=\s*({.*});', content, re.DOTALL)
            if not match:
                logging.error("Could not find voice samples data in the file")
                return None
            
            json_text = match.group(1)
            return json.loads(json_text)
    except Exception as e:
        logging.error(f"Error reading samples file: {e}")
        return None

def save_samples(samples_data):
    """Save the updated voice samples data."""
    if not os.path.exists(SAMPLES_FILE):
        logging.error(f"Samples file {SAMPLES_FILE} does not exist")
        return False
    
    try:
        # Read the original file to preserve the structure
        with open(SAMPLES_FILE, 'r') as f:
            content = f.read()
        
        # Create the updated JSON string
        json_str = json.dumps(samples_data, indent=2)
        
        # Replace the JSON part in the JavaScript file
        updated_content = re.sub(
            r'const\s+voiceExplorationSamples\s*=\s*({.*});', 
            f'const voiceExplorationSamples = {json_str};', 
            content, 
            flags=re.DOTALL
        )
        
        # Write the updated content back to the file
        with open(SAMPLES_FILE, 'w') as f:
            f.write(updated_content)
        
        logging.info(f"Updated samples file: {SAMPLES_FILE}")
        return True
    except Exception as e:
        logging.error(f"Error updating samples file: {e}")
        return False

def update_descriptions(last_update_time=None):
    """
    Update sample descriptions based on feedback.
    
    Args:
        last_update_time: If provided, only process feedback entries newer than this time
    """
    feedback_data = load_feedback_db()
    samples_data = load_samples()
    
    if not feedback_data or not samples_data:
        return
    
    updates_made = 0
    
    # Create a mapping of sample ID to sample object
    sample_map = {sample['id']: sample for sample in samples_data['samples']}
    
    # Process each feedback entry
    for entry in feedback_data['feedback']:
        sample_id = entry['sampleId']
        feedback = entry['feedback']
        
        # Skip entries older than last_update_time if specified
        if last_update_time and 'timestamp' in entry:
            try:
                entry_time = time.strptime(entry['timestamp'], '%Y-%m-%dT%H:%M:%S.%fZ')
                if entry_time <= last_update_time:
                    continue
            except (ValueError, TypeError):
                # If we can't parse the timestamp, process the entry anyway
                pass
        
        # Convert sample_id to int if it's a string
        if isinstance(sample_id, str) and sample_id.isdigit():
            sample_id = int(sample_id)
        
        if sample_id not in sample_map:
            logging.warning(f"Sample {sample_id} not found in samples data")
            # Try to find the sample by string comparison
            for sid in sample_map:
                if str(sid) == str(sample_id):
                    sample_id = sid
                    break
            else:
                continue
        
        sample = sample_map[sample_id]
        description = sample['description']
        updated = False
        
        # Process user notes
        if 'notes' in feedback and feedback['notes']:
            if 'user_notes' not in sample:
                sample['user_notes'] = []
            
            # Add note if it's not already in the list
            if feedback['notes'] not in sample['user_notes']:
                sample['user_notes'].append(feedback['notes'])
                updated = True
        
        # Process attributes
        if 'attributes' in feedback and feedback['attributes']:
            if 'attributes' not in sample:
                sample['attributes'] = {}
            
            # Update attributes
            for attr, value in feedback['attributes'].items():
                if attr not in sample['attributes'] or sample['attributes'][attr] != value:
                    sample['attributes'][attr] = value
                    updated = True
        
        if updated:
            updates_made += 1
    
    # Save updated samples if any changes were made
    if updates_made > 0:
        if save_samples(samples_data):
            logging.info(f"Made {updates_made} description updates")
    else:
        logging.info("No description updates needed")
    
    return updates_made

if __name__ == "__main__":
    update_descriptions() 