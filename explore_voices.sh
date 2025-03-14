#!/bin/bash
# Script to run the CSM Voice Explorer with different parameter sets

POC_DIR="$HOME/movie_maker/voice_poc"
OUTPUT_DIR="$HOME/movie_maker/hdmy5movie_voices/explore"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print usage
function print_usage {
    echo -e "${BLUE}CSM Voice Explorer${NC}"
    echo
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  --quick              Generate a small set of samples quickly"
    echo "  --speakers           Generate samples with different speaker IDs"
    echo "  --temperature        Generate samples with different temperature values"
    echo "  --topk               Generate samples with different topk values"
    echo "  --comprehensive      Generate a comprehensive set of samples (many combinations)"
    echo "  --device [device]    Specify the device to use (cpu or cuda, default: cpu)"
    echo "  --help               Show this help message"
    echo
    echo "Examples:"
    echo "  $0 --quick                     # Generate a small set of samples quickly"
    echo "  $0 --speakers --device cpu     # Test different speaker IDs"
    echo "  $0 --temperature --device cpu  # Test different temperature values"
    echo "  $0 --comprehensive             # Generate a comprehensive set"
}

# Check if no arguments provided
if [ $# -eq 0 ]; then
    print_usage
    exit 0
fi

# Default values
DEVICE="cpu"
MODE="quick"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            MODE="quick"
            shift
            ;;
        --speakers)
            MODE="speakers"
            shift
            ;;
        --temperature)
            MODE="temperature"
            shift
            ;;
        --topk)
            MODE="topk"
            shift
            ;;
        --comprehensive)
            MODE="comprehensive"
            shift
            ;;
        --device)
            DEVICE=$2
            shift 2
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

echo -e "${GREEN}Running CSM Voice Explorer in ${MODE} mode...${NC}"

# Ensure the virtual environment is activated
cd "$POC_DIR" || { echo -e "${RED}Failed to change to POC directory${NC}"; exit 1; }

if [ -f "./venv/bin/activate" ]; then
    source "./venv/bin/activate"
else
    echo -e "${RED}Virtual environment not found. Please run setup first.${NC}"
    exit 1
fi

# Run the voice explorer script
python explore_voices.py --mode "$MODE" --device "$DEVICE" --update-html

# Provide instructions on how to view the results
echo -e "${GREEN}Voice exploration complete!${NC}"
echo -e "To view the results, run the web server:"
echo -e "${YELLOW}~/movie_maker/start_server.sh${NC}"
echo -e "Then open a browser and go to:"
echo -e "${YELLOW}http://localhost:8000/voice_explorer.html${NC}" 