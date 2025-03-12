# HDMY 5 Movie Generation Restart Guide

This document explains how to restart the HDMY 5 Movie generation process if it stops unexpectedly.

## Overview

The video generation process for HDMY 5 Movie can sometimes stop due to various reasons, such as:
- System restarts
- Network issues
- GPU memory errors
- Missing directories for new sections

The `restart_hdmy5movie_generation.sh` script provides a way to restart the generation process from where it left off, without regenerating videos that have already been created.

## Features

- **Resume from Last Position**: Automatically determines the next prompt to process based on the specified index
- **Directory Creation**: Creates necessary directories for new sections to prevent errors
- **Logging**: Logs all activities to the generation log file with timestamps
- **Process Tracking**: Saves the process ID for easy monitoring and management

## Usage

```bash
# Run the restart script
./restart_hdmy5movie_generation.sh
```

## How It Works

1. The script counts the number of existing video files to determine where to start
2. It creates any missing directories for upcoming sections
3. It logs the restart information to the generation log file
4. It starts the generation process with the correct starting index
5. It saves the process ID for future reference

## Troubleshooting

### Determining the Correct Start Index

If you're unsure about the correct start index, you can:

1. Count the number of existing video files:
   ```bash
   find /home/tdeshane/movie_maker/hdmy5movie_videos -name "*.mp4" | wc -l
   ```

2. Check the last generated video:
   ```bash
   ls -la /home/tdeshane/movie_maker/hdmy5movie_videos/*/*.mp4 | sort | tail -n 1
   ```

3. Check the prompt for the last generated video:
   ```bash
   cat /home/tdeshane/movie_maker/hdmy5movie_videos/*/temp_prompt.txt
   ```

### Stopping the Generation Process

To stop the generation process:
```bash
kill $(cat /home/tdeshane/movie_maker/logs/generator.pid)
```

### Monitoring the Generation Process

To monitor the generation process:
```bash
tail -f /home/tdeshane/movie_maker/logs/generation.log
```

## Common Issues

### Missing Directories

If the generation process stops with an error like:
```
FileNotFoundError: [Errno 2] No such file or directory: '/path/to/section/temp_prompt.txt'
```

This indicates that a directory for a new section is missing. The restart script automatically creates these directories.

### Duplicate Videos

If you notice that the same video is being generated twice, you may need to adjust the start index. Check the last generated video and the current prompt to ensure they don't match.

## Maintenance

The restart script should be updated if:
- New sections are added to the movie
- The directory structure changes
- The prompt file is modified 