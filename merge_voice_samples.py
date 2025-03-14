#!/usr/bin/env python3
"""
Merge voice samples from backup and current files.
"""

import json
import os
import sys

# Paths to the voice sample files
BACKUP_FILE = "voice_samples_backup.js"
CURRENT_FILE = "voice_samples.js"
OUTPUT_FILE = "voice_samples_merged.js"

def read_js_file(file_path):
    """Read a JavaScript file that contains a JSON object."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Remove the JavaScript variable declaration
    if content.startswith("const voiceExplorationSamples = "):
        content = content[len("const voiceExplorationSamples = "):]
    
    # Remove trailing semicolon if present
    if content.endswith(";"):
        content = content[:-1]
    
    return json.loads(content)

def write_js_file(data, file_path):
    """Write a JSON object to a JavaScript file."""
    with open(file_path, 'w') as f:
        f.write("const voiceExplorationSamples = ")
        json.dump(data, f, indent=2)
    
    print(f"Wrote merged data to {file_path}")

def merge_voice_samples():
    """Merge voice samples from backup and current files."""
    try:
        # Read the backup file
        backup_data = read_js_file(BACKUP_FILE)
        backup_samples = backup_data.get("samples", [])
        print(f"Read {len(backup_samples)} samples from backup file")
        
        # Read the current file
        current_data = read_js_file(CURRENT_FILE)
        current_samples = current_data.get("samples", [])
        print(f"Read {len(current_samples)} samples from current file")
        
        # Create a set of existing IDs to avoid duplicates
        existing_ids = {sample["id"] for sample in backup_samples}
        
        # Add new samples that don't exist in the backup
        new_samples = []
        for sample in current_samples:
            if sample["id"] not in existing_ids:
                new_samples.append(sample)
                existing_ids.add(sample["id"])
        
        # Merge the samples
        merged_samples = backup_samples + new_samples
        print(f"Added {len(new_samples)} new samples")
        print(f"Total merged samples: {len(merged_samples)}")
        
        # Create the merged data
        merged_data = backup_data.copy()
        merged_data["samples"] = merged_samples
        merged_data["date_generated"] = current_data.get("date_generated", backup_data.get("date_generated"))
        
        # Write the merged data to the output file
        write_js_file(merged_data, OUTPUT_FILE)
        
        # Optionally, replace the current file with the merged file
        if len(sys.argv) > 1 and sys.argv[1] == "--replace":
            import shutil
            shutil.copy(OUTPUT_FILE, CURRENT_FILE)
            print(f"Replaced {CURRENT_FILE} with merged data")
        
        return True
    except Exception as e:
        print(f"Error merging voice samples: {e}")
        return False

if __name__ == "__main__":
    merge_voice_samples() 