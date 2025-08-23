#!/bin/bash

# Auto-sync with file watching
RUNPOD_USER="szmfys6qmnlc3s-64411015"
RUNPOD_HOST="ssh.runpod.io"
SSH_KEY="~/.ssh/runpod_ed25519"
REMOTE_DIR="~/simple-vision"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if fswatch is installed
if ! command -v fswatch &> /dev/null; then
    echo -e "${RED}fswatch not found!${NC}"
    echo "Install with:"
    echo "  macOS:  brew install fswatch"
    echo "  Ubuntu: sudo apt-get install fswatch"
    echo "  Arch:   sudo pacman -S fswatch"
    exit 1
fi

echo -e "${GREEN}👁️  Watching for changes...${NC}"
echo "Press Ctrl+C to stop"
echo ""

# Initial sync
./sync.sh

# Watch for changes and sync
fswatch -o . -e "\.git" -e "\.venv" -e "__pycache__" | while read f; do
    echo -e "${YELLOW}Change detected, syncing...${NC}"
    ./sync.sh
    echo -e "${GREEN}Watching for changes...${NC}"
done