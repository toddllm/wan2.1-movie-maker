# Video Analysis Tools

This directory contains tools for analyzing and improving AI-generated videos using vision models.

## Overview

The analysis process uses a multimodal vision model (Phi-4-multimodal-instruct) to:

1. Analyze frames from a video
2. Generate descriptions of each frame
3. Create an overall analysis of the video
4. Compare the video content to the target description
5. Generate improved prompts for better video generation

## Scripts

### 1. analyze_videos.py

This script provides a user-friendly interface to analyze videos and review results.

```bash
# List all available videos
python analyze_videos.py --list

# Analyze a specific video (by number from list)
python analyze_videos.py --analyze 1

# Analyze a specific video (by path)
python analyze_videos.py --analyze clips/my_video.mp4

# Analyze the most recent video
python analyze_videos.py --latest

# Display results for a specific video
python analyze_videos.py --results 1

# Set maximum number of improvement iterations
python analyze_videos.py --analyze 1 --max-iterations 5
```

### 2. view_results.py

This script provides a visual interface to view the analysis results, showing the original video frames alongside the analysis.

```bash
# List all analyzed videos
python view_results.py --list

# View results for a specific video (by number from list)
python view_results.py --view 1

# View results for a specific video (by path)
python view_results.py --view clips/my_video.mp4

# View results for the most recently analyzed video
python view_results.py --latest
```

### 3. vision_analysis_poc.py

This is the core script that performs the analysis. It can be run directly, but it's recommended to use `analyze_videos.py` instead.

```bash
# Analyze a specific video
python vision_analysis_poc.py --video clips/my_video.mp4

# Set maximum number of improvement iterations
python vision_analysis_poc.py --video clips/my_video.mp4 --max-iterations 5

# Analyze the most recent video
python vision_analysis_poc.py --latest
```

## Output Files

For each analyzed video, the following files are generated:

- `<video_name>_analysis.txt`: Frame-by-frame analysis and overall analysis
- `<video_name>_iteration_<N>.json`: Improvement suggestions for iteration N
- `<video_name>_final_analysis.txt`: Final analysis after all iterations

## Workflow

1. Generate a video using the Wan2.1 text-to-video model
2. Analyze the video using `analyze_videos.py`
3. Review the analysis results using `view_results.py`
4. Use the improved prompt to generate a better video
5. Repeat the process until satisfied with the results

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- OpenCV
- Matplotlib
- Phi-4-multimodal-instruct model (downloaded automatically)

## Tips

- The analysis process can be resource-intensive, especially for high-resolution videos
- For best results, use videos with clear subjects and good lighting
- The improved prompts are designed to be more specific and detailed than the original prompts
- The analysis can help identify issues with the video generation process 