#! /bin/bash

echo "Setting up your Mac for Pytorch"
echo "==============================="

echo "Downloading MiniForge"
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
sh Miniforge3-MacOSX-arm64.sh


echo "Installing PyTorch"
conda create --name="metal" python
conda activate metal
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
pip install wandb tqdm