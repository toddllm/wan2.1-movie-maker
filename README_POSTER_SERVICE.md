# HDMY 5 Movie Poster Generation Service

This service automatically extracts frames from video files to use as poster images for the HDMY 5 Movie website.

## Overview

The poster generation service monitors the video directories for new videos and automatically extracts a frame from each video to use as a poster image. This ensures that all videos on the website have appropriate thumbnail images without requiring manual intervention.

## Features

- **Automatic Frame Extraction**: Extracts a frame from each video file at the 1-second mark
- **Monitoring Mode**: Continuously checks for new videos and processes them automatically
- **Log Rotation**: Maintains logs with automatic rotation to prevent disk space issues
- **Systemd Integration**: Runs as a system service that starts automatically on boot

## Files

- `extract_frames.py`: The main script that extracts frames from videos
- `hdmy5movie_poster_service.service`: The systemd service file that runs the script
- `logs/poster_service.log`: The log file for the service (rotated automatically)

## Usage

### Manual Execution

You can run the script manually with the following commands:

```bash
# One-time mode: Process all videos and exit
python3 extract_frames.py

# Monitor mode: Continuously check for new videos
python3 extract_frames.py --monitor --interval 30
```

### Command-line Options

- `--monitor`: Run in monitor mode, continuously checking for new videos
- `--interval SECONDS`: Interval in seconds between checks in monitor mode (default: 60)
- `--log-file PATH`: Path to log file (default: logs/poster_service.log)

### Service Management

The service is managed using systemd:

```bash
# Start the service
sudo systemctl start hdmy5movie_poster_service.service

# Stop the service
sudo systemctl stop hdmy5movie_poster_service.service

# Check the status of the service
sudo systemctl status hdmy5movie_poster_service.service

# Enable the service to start on boot
sudo systemctl enable hdmy5movie_poster_service.service

# View the logs
sudo journalctl -u hdmy5movie_poster_service.service
```

## Log Rotation

The service uses Python's `RotatingFileHandler` to manage log files:

- Maximum log file size: 5MB
- Number of backup files: 3
- Rotation happens automatically when the log file reaches the size limit

## Directory Structure

```
movie_maker/
├── extract_frames.py                # Main script
├── hdmy5movie_poster_service.service # Systemd service file
├── hdmy5movie_videos/               # Video files
│   ├── 01_opening_credits/
│   ├── 02_prologue/
│   └── ...
├── hdmy5movie_posters/              # Generated poster images
│   ├── 01_opening_credits/
│   ├── 02_prologue/
│   └── ...
└── logs/
    └── poster_service.log           # Log file
```

## Troubleshooting

If the service fails to start, check the following:

1. Ensure the log directory exists and has the correct permissions:
   ```bash
   sudo mkdir -p /home/tdeshane/movie_maker/logs
   sudo chown -R tdeshane:tdeshane /home/tdeshane/movie_maker/logs
   ```

2. Check the service logs:
   ```bash
   sudo journalctl -u hdmy5movie_poster_service.service
   ```

3. Try running the script manually to see if there are any errors:
   ```bash
   cd /home/tdeshane/movie_maker
   python3 extract_frames.py
   ``` 