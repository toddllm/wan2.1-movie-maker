# Wan2.1 Movie Maker

A web interface for generating and combining 10-second videos using the Wan2.1 text-to-video model.

## Features

- Generate videos from text prompts
- Enhance prompts with detailed descriptions for better video generation
- Combine multiple clips into longer movies
- Batch process multiple prompts
- GPU usage monitoring to prevent conflicts

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

## Project Structure

- `app.py`: Main Flask application
- `run.sh`: Script to start the web interface
- `enhance_prompts.py`: Script to enhance prompts
- `batch_generate.py`: Script for batch processing
- `templates/`: HTML templates for the web interface
- `static/`: Static files (CSS, JS)
- `clips/`: Generated video clips
- `movies/`: Combined movies

## License

MIT 