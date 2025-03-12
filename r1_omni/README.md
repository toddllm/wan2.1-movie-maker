# R1-Omni Setup and Usage Guide

This guide provides instructions on how to set up and use the R1-Omni model for emotion recognition in videos.

## Setup

1. Clone the repositories:
   ```bash
   git clone git@github.com:HumanMLLM/R1-Omni.git
   git clone git@github.com:HumanMLLM/HumanOmni.git
   ```

2. Add HumanOmni to your Python path:
   ```bash
   echo 'export PYTHONPATH=$PYTHONPATH:/home/tdeshane/HumanOmni' >> ~/.bashrc
   source ~/.bashrc
   ```

3. Install dependencies:
   ```bash
   cd HumanOmni
   pip install -r requirements.txt
   ```

4. Download the required models:
   ```bash
   mkdir -p models
   cd models
   git lfs install
   git clone https://huggingface.co/StarJiaxing/R1-Omni-0.5B
   git clone https://huggingface.co/google/siglip-base-patch16-224
   git clone https://huggingface.co/openai/whisper-large-v3
   ```

5. Update the config.json file in the R1-Omni-0.5B model to point to the local models:
   ```json
   "mm_audio_tower": "/home/tdeshane/models/whisper-large-v3",
   "mm_vision_tower": "/home/tdeshane/models/siglip-base-patch16-224",
   ```

## Usage

You can use the provided test script to run inference on a video:

```bash
python test_r1_omni.py --video_path /path/to/your/video.mp4
```

Or you can use the original inference script from the R1-Omni repository:

```bash
python inference.py --modal video_audio \
  --model_path /home/tdeshane/models/R1-Omni-0.5B \
  --video_path /path/to/your/video.mp4 \
  --instruct "As an emotional recognition expert; throughout the video, which emotion conveyed by the characters is the most obvious to you? Output the thinking process in <think> </think> and final emotion in <answer> </answer> tags."
```

## Troubleshooting

If you encounter any issues, check the following:

1. Make sure the HumanOmni package is in your Python path.
2. Make sure the paths in the config.json file are correct.
3. Make sure you have all the required dependencies installed.
4. Make sure you have a GPU with enough memory to run the model.

## References

- [R1-Omni GitHub Repository](https://github.com/HumanMLLM/R1-Omni)
- [HumanOmni GitHub Repository](https://github.com/HumanMLLM/HumanOmni)
- [R1-Omni Paper](https://arxiv.org/abs/2503.05379) 