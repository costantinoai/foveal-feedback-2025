#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define repository name and paths
REPO_NAME="foveal-feedback-2025"
REPO_URL="https://github.com/costantinoai/$REPO_NAME.git"
TDANN_REPO="https://github.com/neuroailab/TDANN.git"
TDANN_DIR="external/TDANN-main"

# Step 1: Clone the repository (skip if already in the directory)
if [ ! -d "$REPO_NAME" ]; then
    echo "Cloning repository..."
    git clone $REPO_URL
    cd $REPO_NAME
else
    cd $REPO_NAME
fi

# Step 2: Create and activate conda environment
echo "Setting up conda environment..."
conda create -n foveal-feedback python=3.8 -y
conda activate foveal-feedback

# Step 3: Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Step 4: Clone and install TDANN
if [ ! -d "$TDANN_DIR" ]; then
    echo "Cloning TDANN repository..."
    mkdir -p external && cd external
    git clone $TDANN_REPO TDANN-main
    cd TDANN-main

    echo "Installing TDANN dependencies..."
    pip install -r requirements.txt

    echo "Installing spacetorch..."
    pip install -e .

    cd ../../
else
    echo "TDANN already installed, skipping..."
fi

# Step 5: Set up Python paths
echo "Adding modules to Python path..."
export PYTHONPATH=$PWD/modules:$PWD/$TDANN_DIR:$PYTHONPATH

# Step 6: Verification
echo "Verifying installation..."
python -c "import torch; print('Torch version:', torch.__version__)"
python -c "import spacetorch; print('Spacetorch installation successful')"

echo "Setup complete! Activate your environment with 'conda activate foveal-feedback'."
