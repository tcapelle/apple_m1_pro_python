#! /bin/bash

echo "Setting up your Mac for Pytorch"
echo "==============================="

echo "Downloading MiniForge"
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
sh Miniforge3-MacOSX-arm64.sh


echo "Installing PyTorch"
conda create --name="pt" "python<3.11"
conda activate pt
conda install pytorch torchvision torchaudio -c pytorch
pip install wandb tqdm

echo "Installing Huggingface Stack"
pip install transformers datasets
