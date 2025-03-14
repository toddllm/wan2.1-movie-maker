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
# Start the voice explorer interface
python3 -m http.server 8000
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

Start the web server and navigate to the Voice Explorer:

```bash
./start_server.sh
```

Then open http://localhost:8000/voice_explorer.html in your browser.

## Project Structure

- `app.py`: Main Flask application
- `run.sh`: Script to start the web interface
- `test_server.sh`: Script to start a test server on a specified port
- `enhance_prompts.py`: Script to enhance prompts
- `batch_generate.py`: Script for batch processing
- `extract_frames.py`: Script to extract frames from videos for poster images
- `hdmy5movie_poster_service.service`: Systemd service file for poster generation
- `restart_hdmy5movie_generation.sh`: Script to restart the generation process
- `feedback_server.py`: Server for collecting voice sample feedback
- `update_descriptions.py`: Script for updating voice sample descriptions based on feedback
- `voice_feedback_db.json`: Database of user feedback on voice samples
- `voice_samples.js`: Voice sample data with descriptions
- `voice_explorer.html`: Web interface for exploring voice samples
- `listen.html`: Simplified interface for playing voice samples
- `voice_status.html`: Interface for monitoring voice generation status
- `start_feedback_server.sh`: Script to start the feedback server
- `start_status_server.sh`: Script to start the status server
- `explore_voices.sh`: Script to generate voice samples with different parameters
- `check_files.py`: Script to check and monitor voice generation files
- `check_ip_change.sh`: Script to check for IP address changes
- `send_ip_notification.sh`: Script to send IP change notifications
- `monitor_ip_changes.sh`: Combined script for IP monitoring
- `templates/`: HTML templates for the web interface
- `static/`: Static files (CSS, JS)
- `clips/`: Generated video clips
- `movies/`: Combined movies
- `hdmy5movie_videos/`: Video files for the HDMY 5 Movie project
- `hdmy5movie_posters/`: Poster images for the HDMY 5 Movie project
- `hdmy5movie_voices/`: Voice samples for the HDMY 5 Movie project
- `voices/`: Directory containing general voice samples
- `voice_poc/`: Directory containing voice generation proof of concept implementations
- `logs/`: Log files for various services

## License

MIT 