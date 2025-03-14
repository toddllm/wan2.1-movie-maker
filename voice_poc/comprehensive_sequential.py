#!/usr/bin/env python3
"""
Comprehensive Sequential CSM Voice Explorer

This script generates a comprehensive set of voice samples sequentially (one at a time),
incorporating insights from the CSM-1B implementation for more diverse and interesting samples.
"""

import os
import sys
import torch
import torchaudio
import time
import json
import argparse
import random
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# Try to import CSM model
try:
    from csm_link import load_csm_1b, Generator, Segment
except ImportError:
    print("CSM link not found. Make sure setup_csm.py has been run.")
    sys.exit(1)

# Base directory for output
BASE_OUTPUT_DIR = Path(os.path.expanduser("~/movie_maker/hdmy5movie_voices"))
EXPLORE_DIR = BASE_OUTPUT_DIR / "explore"
SAMPLES_DIR = BASE_OUTPUT_DIR / "samples"
SCENES_DIR = BASE_OUTPUT_DIR / "scenes"

# Default HuggingFace model path 
DEFAULT_MODEL_PATH = os.path.expanduser("~/.cache/huggingface/hub/models--sesame--csm-1b")
DEFAULT_MODEL_ID = "sesame/csm-1b"

# Ensure output directories exist
for dir_path in [EXPLORE_DIR, SAMPLES_DIR, SCENES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Sample texts with varying styles and lengths
SAMPLE_TEXTS = {
    "short": [
        # Short samples (narrative descriptions, 5-15 words)
        "Epic orchestral music builds as stars appear in darkness.",
        "A pixelated spaceship drifts through an asteroid field.",
        "Ancient ruins glow with mysterious blue energy.",
        "Lava bubbles around an obsidian fortress.",
        "Dawn breaks over a cyberpunk cityscape.",
    ],
    "medium": [
        # Medium samples (20-30 words)
        "The camera pans slowly across a desolate landscape, revealing the remnants of an ancient civilization buried beneath crimson sand.",
        "Electronic beeps and ambient synth create tension as the player navigates through a dimly lit abandoned research facility.",
        "Rhythmic tribal drums accompany the hero's journey through dense jungle foliage, occasionally interrupted by exotic wildlife calls.",
        "The spacecraft engines roar to life, pushing against the planet's gravity as stars slowly become visible through the clearing atmosphere.",
    ],
    "technical": [
        # Technical/specialized
        "The quantum entanglement device initializes its calibration sequence, emitting a soft blue glow throughout the laboratory.",
        "Players can combine lithium deposits with thermal extractors to create advanced power cells for their exosuits.",
        "The procedurally generated ecosystem adapts to player actions, creating emergent gameplay scenarios unique to each playthrough.",
    ],
    "emotional": [
        # Emotional/dramatic with explicit style markers
        "[excited] We've finally discovered the artifact after years of searching!",
        "[ominous] Something watches from the shadows, tracking your every move.",
        "[suspenseful] The countdown timer flashes red as oxygen levels drop to critical levels.",
        "[melancholic] The melody echoes memories of a world long forgotten.",
    ],
    "conversation": [
        # Dialogue/conversational context
        "Hey there! Have you tried the new game yet?",
        "I'm not sure I understand. Could you explain that again?",
        "That's incredible news! When did you find out?",
        "Let me think about it. I'll get back to you tomorrow.",
    ],
    "game_instructions": [
        # Game instructions/tutorials
        "Press X to jump and hold Circle to charge your special attack.",
        "Collect the blue crystals to upgrade your character's abilities in the skill tree.",
        "Mission objective: Infiltrate the enemy base and retrieve the stolen data.",
        "New quest available: 'The Lost Artifact' - Speak to the village elder to begin.",
    ],
    "accent_variation": [
        # Different accents/styles
        "[british] I say, this adventure is turning out to be quite splendid!",
        "[southern] Y'all better watch out for them critters in the swamp.",
        "[robotic] Initiating primary protocol. Scanning environment for threats.",
        "[whisper] We need to be very quiet. The guards are right around the corner.",
    ]
}

# Conversation context examples (pairs of dialogue)
CONVERSATION_CONTEXTS = [
    [
        ("Hello! How are you doing today?", 1),
        ("I'm doing great! Just exploring this amazing world.", 0)
    ],
    [
        ("Did you hear about the ancient temple they discovered?", 0),
        ("Yes! I can't believe they found it after all these years.", 1)
    ],
    [
        ("We need to be careful. The enemythis approaching.", 0),
        ("I'll watch your back. Let's stick to the shadows.", 1)
    ],
    [
        ("The mission is simple: get in, grab the artifact, get out.", 0),
        ("Nothing is ever that simple. What about the security systems?", 1)
    ]
]

# Speaker voice styles
SPEAKER_DESCRIPTIONS = {
    0: "Medium-pitched male voice with clear articulation",
    1: "Deep male voice with dramatic tone",
    2: "Female voice with medium pitch",
    3: "Young male voice with higher pitch",
    4: "Authoritative male voice with gravitas",
    5: "Smooth female voice with warm tone",
    6: "Energetic male voice with medium-high pitch",
    7: "Deep female voice with mature tone"
}

# Various parameter combinations to explore
TEMPERATURE_DESCRIPTIONS = {
    0.5: "low randomness and consistent delivery",
    0.75: "moderate expressivity and natural flow",
    0.9: "balanced expressivity and consistency",
    1.1: "high expressivity and natural variation",
    1.3: "maximum expressivity and creativity"
}

TOPK_DESCRIPTIONS = {
    20: "focused and consistent word choices",
    50: "balanced vocabulary diversity",
    80: "diverse and potentially creative phrasing"
}

# Additional configurations from CSM-1B model examination
LENGTHS_MS = {
    "short": 10000,     # 10 seconds 
    "medium": 20000,    # 20 seconds
    "long": 30000       # 30 seconds
}

def get_description(speaker_id, temperature, topk, style=None, context=None):
    """Generate a description of the voice sample based on parameters."""
    speaker_desc = SPEAKER_DESCRIPTIONS.get(speaker_id, f"Speaker {speaker_id}")
    temp_desc = TEMPERATURE_DESCRIPTIONS.get(temperature, f"temperature {temperature}")
    topk_desc = TOPK_DESCRIPTIONS.get(topk, f"topk {topk}")
    
    description = f"{speaker_desc} with {temp_desc} and {topk_desc}"
    
    if style:
        description += f" using {style} style"
    
    if context:
        description += " with conversation context"
    
    return description

def load_csm_model(model_path=None, model_id=None, device="cpu"):
    """Load the CSM model from either a local path or Hugging Face model ID."""
    if model_id:
        print(f"Loading CSM model from Hugging Face: {model_id} on {device}")
        return load_csm_1b(model_id=model_id, device=device)
    else:
        print(f"Loading CSM model from {model_path} on {device}")
        return load_csm_1b(model_path, device)

def preload_audio(audio_file, device="cpu"):
    """Load a pre-existing audio file for context."""
    waveform, sample_rate = torchaudio.load(audio_file)
    if sample_rate != 24000:  # CSM model sample rate
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=24000)
    return waveform.squeeze(0).to(device)

def generate_voice_sample(
    generator,
    text,
    speaker_id,
    temperature,
    topk,
    output_dir,
    sample_id,
    style=None,
    context_segments=None,
    max_audio_length_ms=20000
):
    """Generate a single voice sample with the given parameters."""
    # Create a unique filename based on parameters and a unique identifier
    base_name = f"speaker_{speaker_id}_temp_{temperature}_topk_{topk}"
    
    if style:
        base_name += f"_{style}"
    
    if context_segments:
        base_name += "_with_context"
    
    # Add unique identifier to prevent overwriting when using same parameters but different text
    filename = f"{base_name}_{sample_id}.wav"
    output_path = output_dir / filename
    
    # Skip if file already exists
    if output_path.exists():
        print(f"Skipping existing file: {filename}")
        return {
            "id": sample_id,
            "filename": filename,
            "exists": True,
            "speaker_id": speaker_id,
            "temperature": temperature,
            "topk": topk,
            "style": style,
            "has_context": bool(context_segments),
            "description": get_description(speaker_id, temperature, topk, style, context_segments),
            "generation_time": "N/A (skipped)",
            "text": text
        }
    
    # Prepare context if provided
    context = []
    if context_segments:
        for text_ctx, speaker_ctx in context_segments:
            # In a real implementation, we'd have the audio for these segments
            # Here we're just creating empty context structures
            context.append(Segment(speaker=speaker_ctx, text=text_ctx, audio=torch.zeros(1000)))
    
    # Generate the sample
    start_time = time.time()
    try:
        audio = generator.generate(
            text=text,
            speaker=speaker_id,
            context=context,
            temperature=temperature,
            topk=topk,
            max_audio_length_ms=max_audio_length_ms
        )
        
        # Save the audio
        torchaudio.save(
            output_path,
            audio.unsqueeze(0).cpu(),
            generator.sample_rate
        )
        
        generation_time = time.time() - start_time
        exists = True
        
    except Exception as e:
        print(f"Error generating sample: {e}")
        generation_time = time.time() - start_time
        exists = False
    
    # Return metadata
    return {
        "id": sample_id,
        "filename": filename,
        "exists": exists,
        "speaker_id": speaker_id,
        "temperature": temperature,
        "topk": topk,
        "style": style,
        "has_context": bool(context_segments),
        "description": get_description(speaker_id, temperature, topk, style, context_segments),
        "generation_time": f"{generation_time:.2f}s",
        "text": text
    }

def create_combinations_comprehensive(args):
    """Create comprehensive test combinations based on args."""
    combinations = []
    sample_id = 0
    
    # Basic parameter exploration
    for speaker in args.speakers:
        for temp in args.temperatures:
            for topk in args.topks:
                # Standard samples - one sample for each text type with default params
                for text_type in SAMPLE_TEXTS.keys():
                    # Only use each text type for a subset of combinations to reduce total count
                    # Special text types only for specific parameter combinations
                    if (text_type in ["emotional", "accent_variation"] and temp < 0.9) or \
                       (text_type in ["technical"] and topk < 50):
                        continue
                        
                    sample_id += 1
                    text = random.choice(SAMPLE_TEXTS[text_type])
                    combinations.append({
                        "id": sample_id,
                        "text": text, 
                        "speaker": speaker,
                        "temperature": temp,
                        "topk": topk,
                        "max_audio_length_ms": LENGTHS_MS["medium"],
                        "style": text_type,
                        "context": None
                    })
    
    # Conversation samples with context (only for a few key combinations)
    if args.with_context:
        key_speakers = args.speakers[:min(4, len(args.speakers))]
        for speaker in key_speakers:
            # Use middle temperature and topk values for context examples
            context_temp = 0.9
            context_topk = 50
            
            # Create conversation samples
            for i, (text, alt_speaker) in enumerate(zip(SAMPLE_TEXTS["conversation"], 
                                                     [s for s in key_speakers if s != speaker])):
                # Limit to just a few examples
                if i >= 2:
                    break
                    
                sample_id += 1
                # Create a random context exchange
                context = random.choice(CONVERSATION_CONTEXTS)
                combinations.append({
                    "id": sample_id,
                    "text": text,
                    "speaker": speaker,
                    "temperature": context_temp,
                    "topk": context_topk,
                    "max_audio_length_ms": LENGTHS_MS["medium"],
                    "style": "conversation",
                    "context": context
                })
    
    # If requested, randomize the order
    if args.random:
        random.shuffle(combinations)
    
    # Limit to max_samples if specified
    if args.max_samples and args.max_samples > 0:
        combinations = combinations[:args.max_samples]
    
    return combinations

def generate_samples_sequentially(args):
    """Generate voice samples sequentially, one at a time."""
    # Load the model
    generator = load_csm_model(args.model_path, args.model_id, args.device)
    
    # Create parameter combinations
    combinations = create_combinations_comprehensive(args)
    
    # Print summary
    print(f"Will generate {len(combinations)} samples sequentially")
    print(f"Speakers: {args.speakers}")
    print(f"Temperatures: {args.temperatures}")
    print(f"TopK values: {args.topks}")
    
    # Metadata for all samples
    metadata = {
        "sample_text": "Various texts (see individual samples)",
        "base_path": "/voices/explore",  # Web-relative path
        "absolute_path": str(EXPLORE_DIR),  # Keep absolute path for reference
        "date_generated": datetime.now().isoformat(),
        "samples": []
    }
    
    # Generate each sample sequentially
    for idx, combo in enumerate(tqdm(combinations, desc="Generating voice samples")):
        try:
            # Generate the sample
            sample_metadata = generate_voice_sample(
                generator,
                combo["text"],
                combo["speaker"],
                combo["temperature"],
                combo["topk"],
                EXPLORE_DIR,
                combo["id"],
                style=combo["style"],
                context_segments=combo["context"],
                max_audio_length_ms=combo["max_audio_length_ms"]
            )
            
            # Add metadata
            metadata["samples"].append(sample_metadata)
            
            # Save metadata after each sample to track progress
            with open(EXPLORE_DIR / "voice_samples.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Update HTML after each sample if requested
            if args.update_html:
                update_html_file(metadata)
            
        except Exception as e:
            print(f"Error generating sample {idx}: {e}")
    
    print(f"Generated {len(metadata['samples'])} samples")
    return metadata

def update_html_file(metadata):
    """Update the voice_samples.js file with the latest metadata."""
    js_file_path = Path(os.path.expanduser("~/movie_maker/voice_samples.js"))
    
    # Check if the file exists and read existing data
    if js_file_path.exists():
        try:
            with open(js_file_path, "r") as f:
                content = f.read()
                
            # Extract the JSON part
            if content.startswith("const voiceExplorationSamples = "):
                content = content[len("const voiceExplorationSamples = "):]
            
            # Remove trailing semicolon if present
            if content.endswith(";"):
                content = content[:-1]
                
            existing_data = json.loads(content)
            existing_samples = existing_data.get("samples", [])
            
            # Create a set of existing IDs to avoid duplicates
            existing_ids = {sample["id"] for sample in existing_samples}
            
            # Add new samples that don't exist in the existing data
            for sample in metadata["samples"]:
                if sample["id"] not in existing_ids:
                    existing_samples.append(sample)
                    existing_ids.add(sample["id"])
            
            # Update the metadata with the merged samples
            merged_data = existing_data.copy()
            merged_data["samples"] = existing_samples
            merged_data["date_generated"] = metadata["date_generated"]
            
            # Write the merged data back to the file
            js_content = f"const voiceExplorationSamples = {json.dumps(merged_data, indent=2)};"
        except Exception as e:
            print(f"Error reading existing voice_samples.js: {e}")
            # Fall back to overwriting with new data
            js_content = f"const voiceExplorationSamples = {json.dumps(metadata, indent=2)};"
    else:
        # File doesn't exist, create it with the new data
        js_content = f"const voiceExplorationSamples = {json.dumps(metadata, indent=2)};"
    
    # Save the JavaScript file
    with open(js_file_path, "w") as f:
        f.write(js_content)
    
    print("Updated voice_samples.js with latest data")

def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive voice samples sequentially")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, 
                        help=f"Path to the CSM model checkpoint (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--model-id", default=None,
                        help=f"Hugging Face model ID (default: {DEFAULT_MODEL_ID} if no model path is specified)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Device to run the model on")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to generate (default: all combinations)")
    parser.add_argument("--random", action="store_true", default=True,
                        help="Randomize the order of sample generation")
    parser.add_argument("--sequential", action="store_false", dest="random",
                        help="Generate samples in sequential order (not random)")
    parser.add_argument("--speakers", type=int, nargs="+", default=list(range(8)),
                        help="Speaker IDs to use (default: 0-7)")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.5, 0.75, 0.9, 1.1, 1.3],
                        help="Temperature values to use")
    parser.add_argument("--topks", type=int, nargs="+", default=[20, 50, 80],
                        help="TopK values to use")
    parser.add_argument("--with-context", action="store_true", default=True,
                        help="Include samples with conversation context")
    parser.add_argument("--no-context", action="store_false", dest="with_context",
                        help="Skip samples with conversation context")
    parser.add_argument("--no-update-html", action="store_false", dest="update_html",
                        help="Don't update the HTML file after each sample")
    
    args = parser.parse_args()
    
    # If no model path is specified, use the default Hugging Face model ID
    if args.model_path == DEFAULT_MODEL_PATH and not os.path.exists(args.model_path) and args.model_id is None:
        args.model_id = DEFAULT_MODEL_ID
        args.model_path = None
    
    # Print configuration
    print("Configuration:")
    if args.model_id:
        print(f"  Model ID: {args.model_id}")
    else:
        print(f"  Model Path: {args.model_path}")
    print(f"  Device: {args.device}")
    print(f"  Max Samples: {args.max_samples or 'All combinations'}")
    print(f"  Randomize: {args.random}")
    print(f"  Include Context Examples: {args.with_context}")
    print(f"  Speakers: {args.speakers}")
    print(f"  Temperatures: {args.temperatures}")
    print(f"  TopK values: {args.topks}")
    print(f"  Update HTML: {args.update_html}")
    
    # Generate samples
    metadata = generate_samples_sequentially(args)
    
    print("Done!")
    print(f"Metadata saved to {EXPLORE_DIR / 'voice_samples.json'}")
    print(f"To view the samples, start the web server and go to http://localhost:8000/voice_explorer.html")

if __name__ == "__main__":
    main() 