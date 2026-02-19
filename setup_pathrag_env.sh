#!/bin/bash
#SBATCH -J setup-pathrag
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o logs/setup_%j.out
#SBATCH -e logs/setup_%j.err

set -euo pipefail

module load python3
module load cuda/12.4
module load gcc

mkdir -p logs

conda create -n path-rag python=3.10 -y
source activate path-rag

pip install --upgrade pip
pip install numpy pandas scikit-learn
pip install opencv-python pillow
pip install openslide-python
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

cd "$SCRATCH"
if [ ! -d histocartography ]; then
    git clone https://github.com/BiomedSciAI/histocartography
fi
cd histocartography
pip install -e .
mkdir -p checkpoints

echo "Download required HistoCartography checkpoints into $SCRATCH/histocartography/checkpoints before running extraction."

echo "Environment setup complete"
