# Wan2.1 Movie Maker

A web interface for generating and combining 10-second videos using the Wan2.1 text-to-video model.

## Features

- Generate videos from text prompts
- Enhance prompts with detailed descriptions for better video generation
- Combine multiple clips into longer movies
- Batch process multiple prompts
- GPU usage monitoring to prevent conflicts
- Dark mode UI for reduced eye strain
- Automatic poster image generation for video thumbnails
- Restart capability for interrupted generation processes
- Voice sample feedback system for collecting and processing user feedback
- Voice exploration and generation tools for creating and managing voice samples
- Voice status monitoring for tracking generation progress
- IP address monitoring for servers with dynamic IPs
- Enhanced voice generation with female-dominant voice sets

## Requirements

- Python 3.8+
- FFmpeg
- Wan2.1 model
- Flask
- CUDA-compatible GPU

## Installation

1. Clone this repository
2. Ensure FFmpeg is installed
3. Set up the Wan2.1 model
4. Install Python dependencies

## Usage

### Starting the Web Interface

```bash
./run.sh --port=5001
```

### Running a Test Server

For testing new features without affecting the main service:

```bash
./test_server.sh [port]
```

Example:
```bash
./test_server.sh 5002
```

This will start a test instance on port 5002 (or any specified port).

### Enhancing Prompts

```bash
./enhance_only.sh --input=prompts.txt
```

### Generating Videos with Enhanced Prompts

```bash
./run_enhanced_generation.sh --input=enhanced_prompts/enhanced_TIMESTAMP_only.txt --seconds=1
```

### Batch Processing

```bash
./batch_generate.py prompts.txt --seconds=1
```

### Poster Image Generation

The system includes an automatic poster image generation service that extracts frames from videos to use as thumbnails.

```bash
# Run manually (one-time)
python3 extract_frames.py

# Run in monitor mode
python3 extract_frames.py --monitor --interval 30
```

For more details, see [README_POSTER_SERVICE.md](README_POSTER_SERVICE.md).

### Restarting Generation Process

If the video generation process stops unexpectedly, you can restart it from where it left off:

```bash
./restart_hdmy5movie_generation.sh
```

This script will automatically determine the next prompt to process and create any necessary directories.

For more details, see [README_RESTART_GENERATION.md](README_RESTART_GENERATION.md).

### Voice Sample Feedback System

The system includes a feedback collection and processing system for voice samples:

```bash
# Start the feedback server
python3 feedback_server.py

# Update voice sample descriptions based on feedback
python3 update_descriptions.py
```

For more details, see [FEEDBACK_SYSTEM.md](FEEDBACK_SYSTEM.md).

### Voice Exploration and Generation

The system includes tools for exploring and generating voice samples:

```bash
# Start the feedback server which also serves the voice explorer interface
python3 feedback_server.py
# Then navigate to http://localhost:8000/voice_explorer.html

# Generate voice samples with different parameters
./explore_voices.sh --quick

# Monitor voice generation status
./start_status_server.sh
```

For more details, see [VOICE_SYSTEM.md](VOICE_SYSTEM.md).

### IP Address Monitoring

The system includes scripts to monitor the server's public IP address and send notifications when it changes:

```bash
# Check for IP changes manually
./check_ip_change.sh

# Run the complete monitoring process
./monitor_ip_changes.sh
```

A cron job is set up to run this check automatically every hour.

For more details, see [IP_MONITOR_README.md](IP_MONITOR_README.md).

## Voice Generation

### Generating New Voice Sets

To generate a new set of 100 voices with 80% female voices:

```bash
./generate_more_female_voices.sh
```

To generate additional voice sets:

```bash
./generate_100_more_voices.sh
```

These scripts generate:
- 80 female voices with higher expressivity
- 20 male voices with varied characteristics

### Preserving Voice Samples

To ensure all generated voices are preserved:

```bash
python3 merge_voice_samples.py --replace
```

For automatic preservation during generation:

```bash
./auto_merge_voices.sh
```

For more details, see [VOICE_GENERATION.md](VOICE_GENERATION.md).

### Exploring Generated Voices

Start the feedback server and navigate to the Voice Explorer:

```bash
python3 feedback_server.py
```

Then open http://localhost:8000/voice_explorer.html in your browser.

## Voice Analysis System

This system provides tools for analyzing voice samples, organizing them by character, and browsing the resulting catalog.

### Components

1. **Voice Gender Analysis**: Analyzes voice samples to determine gender and other characteristics.
2. **Voice Sample Organization**: Organizes voice samples into character profiles.
3. **Voice Browser**: Web interface for browsing voice characters.

### Scripts

- `run_voice_gender_analysis.sh`: Runs gender analysis on voice samples.
- `organize_voice_samples.py`: Organizes samples into character profiles.
- `start_voice_browser.sh`: Starts the web interface for browsing voice characters.
- `test_audio.sh`: Tests audio analysis on a single sample.
- `reset_phi4_model.sh`: Resets the Phi-4 model directory if needed.
- `phi4_audio_test.py`: Uses the Phi-4 multimodal model for advanced audio analysis.
- `run_phi4_audio_test.sh`: Wrapper script for running the Phi-4 multimodal audio test.

### Usage

#### Testing Audio Analysis

To test audio analysis on a single sample:

```bash
cd movie_maker
./test_audio.sh [path/to/audio.wav]
```

If no audio file is specified, the script will find a sample file automatically.

#### Using Phi-4 Multimodal for Audio Analysis

For more advanced audio analysis using the Phi-4 multimodal model:

```bash
cd movie_maker
./run_phi4_audio_test.sh [path/to/audio.wav]
```

This will:
1. Load the Phi-4 multimodal model
2. Process the audio file
3. Analyze the voice to determine gender and other characteristics
4. Provide a detailed analysis with confidence scores

#### Running Full Analysis

To run analysis on all voice samples:

```bash
cd movie_maker
./run_voice_gender_analysis.sh
```

This will test 5 samples and prompt you to continue with the full analysis.

#### Organizing Voice Samples

To organize voice samples into character profiles:

```bash
cd movie_maker
./organize_voice_samples.py
```

This creates character profiles and a master catalog.

#### Browsing Voice Characters

To browse voice characters through the web interface:

```bash
cd movie_maker
./start_voice_browser.sh
```

Then access the provided URL in your browser.

### Troubleshooting

If you encounter issues with the Phi-4 model:

1. Run `./reset_phi4_model.sh` to reset the model directory.
2. Ensure audio files are in WAV format with a sample rate of 16000 Hz.
3. Check that the required Python packages are installed.
4. Look for error messages in the console output.
5. For GPU acceleration, ensure CUDA is properly installed.

### Requirements

- Python 3.8+
- PyTorch
- Transformers library
- SoundFile
- Flask (for the voice browser)
- CUDA-compatible GPU (recommended for faster processing)

## Setup

1. Make sure you have Python 3.8+ installed
2. Install required packages:
   ```
   pip install torch torchaudio transformers
   ```
3. Run the setup script to download the Phi-4 model:
   ```
   ./setup_phi4_model.py
   ```
4. Fix the audio configuration (required for proper audio analysis):
   ```
   ./fix_phi4_audio.py
   ```

## Usage

### Testing Audio Analysis

To test if the system can analyze audio files correctly:

```bash
./test_audio.sh [optional_audio_file.wav]
```

This will:
1. Check if the Phi-4 model is installed
2. Fix the audio configuration if needed
3. Analyze a sample audio file (or the one you specify)
4. Display the analysis results

### Running Voice Gender Analysis

To analyze voice samples for gender identification:

```bash
./run_voice_gender_analysis.sh
```

This will:
1. Test the analysis with 5 samples first
2. Ask if you want to run the full analysis
3. Save results to `hdmy5movie_voices/explore/gender_analysis.json`

You can also run the analysis script directly with options:

```bash
./analyze_voice_gender.py --voice-dir PATH --metadata PATH --output PATH --limit NUM
```

### Organizing Voice Samples

To organize voice samples into character profiles:

```bash
./organize_voice_samples.py
```

This will:
1. Load the gender analysis results
2. Group samples by speaker and gender
3. Create character profiles with descriptions
4. Organize files into a character-based structure
5. Generate a catalog of all characters

### Browsing Voice Characters

To browse the organized voice characters in a web interface:

```bash
./start_voice_browser.sh
```

This will:
1. Start a simple HTTP server
2. Open a browser to view the voice characters
3. Allow you to listen to samples and view character information

## Troubleshooting

If you encounter issues with audio analysis:

1. Make sure you've run `fix_phi4_audio.py` to update the model configuration
2. Check that your audio files are WAV format with 16000 Hz sample rate
3. Verify that the paths to your voice samples are correct
4. Check the log files for detailed error messages

### Audio Format Requirements

The Phi-4 model requires audio files to be:
- WAV format
- 16000 Hz sample rate
- Mono (single channel)

You can check and convert your audio files using the provided script:
```bash
./check_audio_format.py your_audio_file.wav
```

This script will:
- Check if the audio file meets the requirements
- Automatically resample to 16000 Hz if needed
- Convert to mono if needed
- Save the converted file with a suffix (_resampled.wav or _mono.wav)

The `test_audio.sh` script automatically runs this check and uses the converted file if available.

### Common Errors and Solutions

#### SequenceFeatureExtractor Error

If you see an error like `SequenceFeatureExtractor.__init__() got multiple values for argument 'feature_size'`, this indicates a configuration issue with the audio parameters. Try these steps:

1. Run the fix script again:
   ```bash
   ./fix_phi4_audio.py
   ```

2. If that doesn't work, reset the model configuration:
   ```bash
   ./reset_phi4_model.sh --restore
   ```
   Then run the fix script again.

3. If all else fails, you can completely remove and reinstall the model:
   ```bash
   ./reset_phi4_model.sh --remove
   ./setup_phi4_model.py
   ./fix_phi4_audio.py
   ```

#### Trust Remote Code Prompt

If you're repeatedly asked to trust remote code, you can set the environment variable:
```bash
export TRANSFORMERS_TRUST_REMOTE_CODE=1
```

Or use the updated scripts which automatically set this variable.

## Files

- `setup_phi4_model.py` - Downloads and sets up the Phi-4 model
- `fix_phi4_audio.py`