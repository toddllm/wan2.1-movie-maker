#!/bin/bash

# Script to run direct video generation with re-enhancement
# This script handles re-enhancing prompts and generating videos

set -e

# Default values
NUM_PROMPTS=5
VIDEO_DURATION=10

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --count|-c)
            if [[ -n "$2" ]] && [[ "$2" =~ ^[0-9]+$ ]]; then
                NUM_PROMPTS="$2"
                shift 2
            else
                echo "Error: --count requires a numeric argument"
                exit 1
            fi
            ;;
        --duration|-d)
            if [[ -n "$2" ]] && [[ "$2" =~ ^[0-9]+$ ]]; then
                VIDEO_DURATION="$2"
                shift 2
            else
                echo "Error: --duration requires a numeric argument"
                exit 1
            fi
            ;;
        --help|-h)
            echo "Usage: $0 [--count|-c NUM_PROMPTS] [--duration|-d VIDEO_DURATION]"
            echo "  --count, -c NUM_PROMPTS    Number of prompts to re-enhance (default: 5)"
            echo "  --duration, -d DURATION    Duration of videos in seconds (default: 10)"
            echo "  --help, -h                 Show this help message"
            exit 0
            ;;
        *)
            if [[ "$1" =~ ^[0-9]+$ ]] && [[ -z "$FIRST_NUM" ]]; then
                NUM_PROMPTS="$1"
                FIRST_NUM=1
                shift
            elif [[ "$1" =~ ^[0-9]+$ ]] && [[ -n "$FIRST_NUM" ]]; then
                VIDEO_DURATION="$1"
                shift
            else
                echo "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
            fi
            ;;
    esac
done

echo "Starting direct re-enhancement and video generation at $(date)"
echo "Number of prompts to re-enhance: $NUM_PROMPTS"
echo "Video duration (seconds): $VIDEO_DURATION"

# Re-enhance prompts
echo "Re-enhancing prompts..."
python3 re_enhance_prompts.py --count "$NUM_PROMPTS"

# Get the most recent re-enhanced prompts file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="enhanced_prompts/re_enhanced_${TIMESTAMP}.txt"

# Generate videos
echo "Generating videos from re-enhanced prompts using direct generation..."
python3 direct_generate.py --input "$OUTPUT_FILE" --duration "$VIDEO_DURATION"

echo "Direct re-enhancement and video generation completed at $(date)"
echo "Re-enhanced prompts saved to: $OUTPUT_FILE"
echo "Check the clips directory for generated videos" 