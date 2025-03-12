#!/bin/bash
# R1-V and R1-Omni Model Tester Script
# This script runs the model tester with common options.

# Set the directory to the script's location
cd "$(dirname "$0")"

# Default values
R1V_MODEL="Qwen/Qwen-VL-Chat"
R1OMNI_MODEL="StarJiaxing/R1-Omni-0.5B"
TEST_IMAGE=""
TEST_VIDEO=""
DEVICE=""
SKIP_R1V=false
SKIP_R1OMNI=false
SKIP_VIDEO=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --r1v)
      R1V_MODEL="$2"
      shift 2
      ;;
    --r1omni)
      R1OMNI_MODEL="$2"
      shift 2
      ;;
    --image)
      TEST_IMAGE="$2"
      shift 2
      ;;
    --video)
      TEST_VIDEO="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --skip-r1v)
      SKIP_R1V=true
      shift
      ;;
    --skip-r1omni)
      SKIP_R1OMNI=true
      shift
      ;;
    --skip-video)
      SKIP_VIDEO=true
      shift
      ;;
    --cpu)
      DEVICE="cpu"
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --r1v MODEL       Specify the R1-V model name or path (default: $R1V_MODEL)"
      echo "  --r1omni MODEL    Specify the R1-Omni model name or path (default: $R1OMNI_MODEL)"
      echo "  --image PATH      Specify a test image path"
      echo "  --video PATH      Specify a test video path"
      echo "  --device DEVICE   Specify the device to use (cuda or cpu)"
      echo "  --cpu             Use CPU instead of GPU"
      echo "  --skip-r1v        Skip testing the R1-V model"
      echo "  --skip-r1omni     Skip testing the R1-Omni model"
      echo "  --skip-video      Skip testing video frame extraction"
      echo "  --help            Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help to see available options"
      exit 1
      ;;
  esac
done

# Build the command
CMD="python test_models.py"

if [ -n "$R1V_MODEL" ]; then
  CMD="$CMD --r1v \"$R1V_MODEL\""
fi

if [ -n "$R1OMNI_MODEL" ]; then
  CMD="$CMD --r1omni \"$R1OMNI_MODEL\""
fi

if [ -n "$TEST_IMAGE" ]; then
  CMD="$CMD --image \"$TEST_IMAGE\""
fi

if [ -n "$TEST_VIDEO" ]; then
  CMD="$CMD --video \"$TEST_VIDEO\""
fi

if [ -n "$DEVICE" ]; then
  CMD="$CMD --device \"$DEVICE\""
fi

if [ "$SKIP_R1V" = true ]; then
  CMD="$CMD --skip-r1v"
fi

if [ "$SKIP_R1OMNI" = true ]; then
  CMD="$CMD --skip-r1omni"
fi

if [ "$SKIP_VIDEO" = true ]; then
  CMD="$CMD --skip-video"
fi

# Print the command
echo "Running: $CMD"

# Execute the command
eval $CMD

# Check the exit code
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
  echo "All tests passed successfully!"
else
  echo "Some tests failed. Check the logs for details."
fi

exit $EXIT_CODE 