# Voice Sample Feedback System

## Overview
The Voice Sample Feedback System allows users to provide feedback on voice samples, which is then stored server-side and used to update the sample descriptions. This system consists of several components:

1. **Feedback Collection**: Client-side interface for users to submit feedback on voice samples
2. **Feedback Server**: Server-side component that stores feedback in a database
3. **Update Script**: Process that applies feedback to update voice sample descriptions

## Components

### 1. Feedback Database (`voice_feedback_db.json`)
Stores all user feedback in JSON format with the following structure:
```json
{
  "feedback": [
    {
      "sampleId": "123",
      "feedback": {
        "notes": "This is a user note",
        "attributes": {
          "gender": "Male",
          "other_attribute": "value"
        }
      },
      "timestamp": "2023-05-01T12:34:56.789Z"
    }
  ]
}
```

### 2. Feedback Server (`feedback_server.py`)
- Runs on port 8000
- Receives feedback submissions from users
- Stores feedback in the database
- Handles API endpoints for feedback submission

### 3. Update Script (`update_descriptions.py`)
- Processes feedback entries from the database
- Updates voice sample descriptions in `voice_samples.js`
- Adds user notes and attributes to samples
- Logs updates to `description_updates.log`

## Usage

### Running the System
1. Start the feedback server:
   ```
   python3 feedback_server.py
   ```

2. To update descriptions based on feedback:
   ```
   python3 update_descriptions.py
   ```

### Feedback Processing Logic
- User notes are added to samples if they don't already exist
- Attributes (like gender) are updated based on user feedback
- Updates are logged to track changes
- The system can process only new feedback since the last update

## Files
- `voice_feedback_db.json`: Feedback database
- `voice_samples.js`: Voice sample data with descriptions
- `feedback_server.py`: Server for collecting feedback
- `update_descriptions.py`: Script for updating descriptions
- `description_updates.log`: Log of description updates 