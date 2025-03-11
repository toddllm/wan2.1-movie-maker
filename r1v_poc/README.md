# R1-V and R1-Omni Integration for Movie Maker

This module implements a proof of concept (POC) integration of the R1-V and R1-Omni models into the Movie Maker application, enhancing its video analysis capabilities.

## Features

- **Enhanced Visual Analysis**: Using R1-V for detailed scene analysis, object counting, and visual quality assessment.
- **Emotion Recognition**: Using R1-Omni for detecting emotions in video frames and audio.
- **Recommendation System**: Generating improvement suggestions based on analyses.
- **Web Dashboard**: Interactive interface for analyzing videos and viewing results.

## Components

- `r1v_analyzer.py`: Implements visual analysis using R1-V.
- `emotion_detector.py`: Implements emotion recognition using R1-Omni.
- `recommender.py`: Generates recommendations based on analyses.
- `model_utils.py`: Utilities for loading and using models.
- `r1v_routes.py`: Flask routes for the web dashboard.

## Usage

### Command Line

Each module can be run as a standalone script:

```bash
# Visual analysis
python r1v_analyzer.py --video path/to/video.mp4 --frames 5 --model "Qwen/Qwen2-VL-7B"

# Emotion analysis
python emotion_detector.py --video path/to/video.mp4 --frames 5 --model "HumanMLLM/R1-Omni-0.5B"

# Generate recommendations
python recommender.py --r1v path/to/r1v_analysis.json --emotion path/to/emotion_analysis.json --prompt "Original prompt"
```

### Web Dashboard

The web dashboard is integrated with the Movie Maker application. To access it:

1. Start the Movie Maker application
2. Navigate to `/r1v` in your browser
3. Use the dashboard to analyze videos and view results

## Configuration

Model configuration is stored in `r1v_config.json`. You can modify this file directly or use the web dashboard to update the configuration.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- OpenCV 4.5+
- Flask 2.0+

## Integration with Movie Maker

To integrate this POC with the Movie Maker application, add the following to your app.py:

```python
from r1v_poc import register_r1v_routes

# Initialize your Flask app
app = Flask(__name__)

# Register R1-V routes
register_r1v_routes(app) 