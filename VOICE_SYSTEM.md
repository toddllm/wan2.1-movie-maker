# Voice System

## Overview
The Voice System is a comprehensive set of tools for exploring, generating, and managing voice samples for the Wan2.1 Movie Maker project. It consists of several components that work together to provide a complete voice management solution.

## Components

### 1. Voice Explorer (`voice_explorer.html`)
A web-based interface for exploring and providing feedback on voice samples:
- Browse through available voice samples
- Play and listen to samples
- Filter samples by various attributes
- Provide feedback on voice characteristics
- Add notes to samples
- Submit feedback to the server

### 2. Voice Player (`listen.html`)
A simplified web interface for playing voice samples:
- Play individual voice samples
- Select different scenes or contexts
- Focus on listening without the exploration features

### 3. Voice Status Monitor (`voice_status.html`)
A web interface for monitoring the status of voice generation:
- Track progress of voice generation tasks
- View statistics on completed and pending generations
- Monitor system performance during generation

### 4. Feedback System
A system for collecting and processing user feedback on voice samples:
- Feedback Server (`feedback_server.py`): Collects and stores user feedback
- Update Script (`update_descriptions.py`): Processes feedback to update sample descriptions
- Feedback Database (`voice_feedback_db.json`): Stores all user feedback

### 5. Voice Generation Tools
Scripts for generating and exploring different voice parameters:
- `explore_voices.sh`: Script to generate voice samples with different parameter sets
- Voice PoC directory (`voice_poc/`): Proof of concept implementations for voice generation

## Usage

### Exploring Voice Samples
1. Open the Voice Explorer interface:
   ```
   python3 -m http.server 8000
   ```
   Then navigate to `http://localhost:8000/voice_explorer.html`

2. Use the interface to browse, filter, and listen to voice samples
3. Provide feedback on samples using the feedback form

### Starting the Feedback Server
```bash
./start_feedback_server.sh
```
This will start the feedback server on port 8000, allowing the Voice Explorer to submit feedback.

### Monitoring Voice Generation Status
```bash
./start_status_server.sh
```
This will start the status server and open the status page at `http://localhost:8000/voice_status.html`.

### Generating Voice Samples with Different Parameters
```bash
./explore_voices.sh [options]
```

Options:
- `--quick`: Generate a small set of samples quickly
- `--speakers`: Generate samples with different speaker IDs
- `--temperature`: Generate samples with different temperature values
- `--topk`: Generate samples with different topk values
- `--comprehensive`: Generate a comprehensive set of samples
- `--device [device]`: Specify the device to use (cpu or cuda)

### Updating Voice Sample Descriptions
```bash
python3 update_descriptions.py
```
This will process the feedback database and update the voice sample descriptions accordingly.

## Files
- `voice_explorer.html`: Web interface for exploring voice samples
- `listen.html`: Simplified interface for playing voice samples
- `voice_status.html`: Interface for monitoring voice generation status
- `feedback_server.py`: Server for collecting feedback
- `update_descriptions.py`: Script for updating descriptions based on feedback
- `voice_feedback_db.json`: Database of user feedback
- `voice_samples.js`: Voice sample data with descriptions
- `start_feedback_server.sh`: Script to start the feedback server
- `start_status_server.sh`: Script to start the status server
- `explore_voices.sh`: Script to generate voice samples with different parameters
- `check_files.py`: Script to check and monitor voice generation files
- `voice_poc/`: Directory containing voice generation proof of concept implementations
- `hdmy5movie_voices/`: Directory containing voice samples for the HDMY 5 Movie project
- `voices/`: Directory containing general voice samples 