# Voice Classification and Description Update Process

This document outlines the process for analyzing voice samples, classifying their gender, and updating their descriptions in the voice explorer.

## Prerequisites

1. Python environment with required packages:
   ```bash
   source voice_poc/venv/bin/activate
   pip install torch torchaudio transformers
   ```

2. Required files:
   - `ecapa_gender_analysis.py`: Main script for gender classification
   - `model.py`: ECAPA-TDNN model implementation
   - `voice_samples.js`: Voice explorer metadata
   - `update_voice_descriptions.py`: Script to update descriptions

## Process Steps

### 1. Running Gender Classification

The ECAPA-TDNN model is used to classify voice samples by gender:

```bash
python ecapa_gender_analysis.py --dir hdmy5movie_voices/explore --device cuda
```

This will:
- Load the ECAPA-TDNN model
- Process all audio files in the specified directory
- Generate `ecapa_gender_results.json` with:
  - Gender classification (male/female)
  - Confidence scores
  - Audio characteristics (volume, speech rate)

### 2. Updating Voice Descriptions

The `update_voice_descriptions.py` script updates the voice explorer metadata:

```bash
python update_voice_descriptions.py
```

This will:
- Load gender analysis results
- Create a backup of `voice_samples.js`
- Update gender descriptions while preserving other characteristics
- Save changes back to `voice_samples.js`

### 3. Deploying Changes

1. Ensure the voice explorer server is running:
   ```bash
   python -m http.server 8000
   ```

2. Access the voice explorer at:
   ```
   http://localhost:8000/voice_explorer.html
   ```

3. Refresh the page to see updated descriptions

## File Structure

```
movie_maker/
├── voice_poc/
│   └── venv/              # Python virtual environment
├── hdmy5movie_voices/
│   └── explore/           # Voice sample audio files
├── ecapa_gender_analysis.py
├── model.py
├── voice_samples.js
├── update_voice_descriptions.py
├── ecapa_gender_results.json
└── voice_samples.js.bak   # Backup of original descriptions
```

## Troubleshooting

1. If gender classification fails:
   - Check CUDA availability
   - Verify model file exists
   - Check audio file format and paths

2. If description updates fail:
   - Verify `ecapa_gender_results.json` exists
   - Check file permissions
   - Restore from backup if needed

3. If voice explorer doesn't update:
   - Clear browser cache
   - Verify server is running
   - Check file paths in `voice_samples.js`

## Future Improvements

1. Add confidence scores to descriptions
2. Implement batch processing for large datasets
3. Add more audio characteristics analysis
4. Create a web interface for manual verification
5. Add automated testing for the classification process

## Notes

- The ECAPA-TDNN model provides gender classification with confidence scores
- Descriptions are updated while preserving other characteristics (pitch, expressivity, style)
- Original descriptions are backed up before updates
- The process can be repeated for new voice samples 