#!/bin/bash

# RunPod sync script
RUNPOD_USER="szmfys6qmnlc3s-64411015"
RUNPOD_HOST="ssh.runpod.io"
SSH_KEY="~/.ssh/runpod_ed25519"
REMOTE_DIR="~/simple-vision" # Change this to your preferred remote directory

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Syncing to RunPod GPU instance...${NC}"

# Create remote directory if it doesn't exist
ssh -i "$SSH_KEY" "$RUNPOD_USER@$RUNPOD_HOST" "mkdir -p $REMOTE_DIR"

# Sync files using rsync
rsync -avz --progress \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    --exclude 'blip2_description.txt' \
    -e "ssh -i $SSH_KEY" \
    ./ "$RUNPOD_USER@$RUNPOD_HOST:$REMOTE_DIR/"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Sync completed successfully!${NC}"
    echo -e "Remote path: $REMOTE_DIR"
else
    echo -e "${RED}✗ Sync failed!${NC}"
    exit 1
fi
