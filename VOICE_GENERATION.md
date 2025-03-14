# Voice Generation System

## Overview
The Voice Generation System allows for the creation of diverse voice samples using the CSM-1B model. This document explains how to generate new voices, particularly focusing on creating female-dominant voice sets.

## Voice Generation Scripts

### `generate_more_female_voices.sh`
This script generates 100 voice samples with 80% female voices and 20% male voices.

```bash
./generate_more_female_voices.sh
```

The script:
1. Generates 80 female voices using speakers 2, 5, and 7 with higher expressivity (temperatures 1.1 and 1.3)
2. Generates 20 male voices using speakers 0, 1, 3, 4, and 6 with varied expressivity (temperatures 0.9, 1.1, and 1.3)
3. Updates the voice explorer interface automatically

### `generate_100_more_voices.sh`
This script generates an additional 100 voice samples with the same 80/20 female/male ratio.

```bash
./generate_100_more_voices.sh
```

The script follows the same pattern as `generate_more_female_voices.sh` but can be run multiple times to generate more voices.

### `merge_voice_samples.py`
This utility script ensures that newly generated voices are added to the existing collection without overwriting previous samples.

```bash
python3 merge_voice_samples.py --replace
```

The script:
1. Reads the backup and current voice sample files
2. Merges the samples, avoiding duplicates
3. Updates the voice_samples.js file with the combined data

### `auto_merge_voices.sh`
This script automatically runs the merge operation at regular intervals to preserve all voice samples during generation.

```bash
./auto_merge_voices.sh
```

The script:
1. Runs the merge_voice_samples.py script every 5 minutes
2. Logs the number of samples in the voice_samples.js file
3. Ensures no samples are lost if the generation process overwrites the file

## Voice Characteristics

### Female Voices
- **Speaker 2**: Female voice with medium pitch
- **Speaker 5**: Smooth female voice with warm tone
- **Speaker 7**: Deep female voice with mature tone
- **Temperature 1.1-1.3**: Highly varied/expressive delivery for more distinctive voices

### Male Voices
- **Speaker 0**: Medium-pitched male voice with clear articulation
- **Speaker 1**: Deep male voice with dramatic tone
- **Speaker 3**: Young male voice with higher pitch
- **Speaker 4**: Authoritative male voice with gravitas
- **Speaker 6**: Energetic male voice with medium-high pitch
- **Temperature 0.9-1.3**: Range from natural to highly expressive delivery

## Voice Explorer
Generated voices can be explored through the Voice Explorer interface:

1. Start the web server:
   ```bash
   cd /home/tdeshane/movie_maker && ./start_server.sh
   ```

2. Navigate to http://localhost:8000/voice_explorer.html in your browser

3. Use the interface to:
   - Browse all generated voices
   - Filter by characteristics (gender, speaker, temperature)
   - Play voice samples
   - Mark favorites
   - Provide feedback on voice quality

## Voice Feedback System
The feedback system allows for marking voices with specific characteristics:

1. Gender classification (Male/Female/Androgynous)
2. Adding notes about voice quality
3. Rating voice characteristics

Feedback is stored in `voice_feedback_db.json` and can be used to update voice descriptions using the `update_descriptions.py` script.

## Generation Process
Voice generation is a CPU-intensive process. Each voice sample takes approximately 2-3 minutes to generate on a standard CPU. The full set of 100 voices may take several hours to complete.

The generation process:
1. Loads the CSM-1B model from Hugging Face
2. Generates each voice sequentially
3. Saves audio files to `/home/tdeshane/movie_maker/hdmy5movie_voices/explore`
4. Updates metadata in `voice_samples.js`
5. Makes new voices immediately available in the Voice Explorer

## Tips for Creating Distinctive Voices
- Higher temperature values (1.1-1.3) create more expressive and varied voices
- Speaker 2 and 5 produce the most natural-sounding female voices
- Speaker 7 produces deeper female voices with a mature tone
- Using different text samples can highlight different aspects of voice quality 