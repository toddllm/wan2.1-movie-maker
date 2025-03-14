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
- `templates/`: HTML templates for the web interface
- `static/`: Static files (CSS, JS)
- `clips/`: Generated video clips
- `movies/`: Combined movies
- `hdmy5movie_videos/`: Video files for the HDMY 5 Movie project
- `hdmy5movie_posters/`: Poster images for the HDMY 5 Movie project
- `logs/`: Log files for various services

## License

MIT 