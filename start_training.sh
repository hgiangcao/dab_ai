#!/bin/bash

# Start the AlphaZero training process
echo "Initializing AlphaZero training for Dots and Boxes..."

# Ensure we have the correct Python environment
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate pyenv312 || echo "Conda env pyenv312 not found, using default env."
fi

# Launch tensorboard in the background (kill existing ones first)
echo "Starting TensorBoard on http://localhost:6006..."
pkill -f "tensorboard" || true
sleep 1
tensorboard --logdir=temp --port 6006 &
TB_PID=$!

# Run the training script
python main.py

# When training ends (or is interrupted), kill TensorBoard
kill $TB_PID
echo "Training finished and TensorBoard stopped."
