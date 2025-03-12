import os
import argparse
from humanomni import model_init, mm_infer
from humanomni.utils import disable_torch_init
from transformers import BertTokenizer

# Set environment variables
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    parser = argparse.ArgumentParser(description="R1-Omni Inference Script")
    parser.add_argument('--modal', type=str, default='video_audio', help='Modal type (video or video_audio)')
    parser.add_argument('--model_path', type=str, default='/home/tdeshane/models/R1-Omni-0.5B', help='Path to the model')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the video file')
    parser.add_argument('--instruct', type=str, default="As an emotional recognition expert; throughout the video, which emotion conveyed by the characters is the most obvious to you? Output the thinking process in <think> </think> and final emotion in <answer> </answer> tags.", help='Instruction for the model')

    args = parser.parse_args()

    # Initialize BERT tokenizer
    bert_model = "bert-base-uncased"
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model)

    # Disable Torch initialization
    disable_torch_init()

    # Initialize model, processor, and tokenizer
    print(f"Initializing model from {args.model_path}...")
    model, processor, tokenizer = model_init(args.model_path)
    print("Model initialized successfully!")

    # Process video input
    print(f"Processing video from {args.video_path}...")
    video_tensor = processor['video'](args.video_path)
    print("Video processed successfully!")
    
    # Process audio based on modal type
    if args.modal == 'video_audio' or args.modal == 'audio':
        print("Processing audio...")
        audio = processor['audio'](args.video_path)[0]
        print("Audio processed successfully!")
    else:
        audio = None

    # Execute inference
    print("Running inference...")
    output = mm_infer(video_tensor, args.instruct, model=model, tokenizer=tokenizer, modal=args.modal, question=args.instruct, bert_tokeni=bert_tokenizer, do_sample=False, audio=audio)
    print("\nModel output:")
    print(output)

if __name__ == "__main__":
    main() 