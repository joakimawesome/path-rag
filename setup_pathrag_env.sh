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
pip install numpy==1.26.4 pandas==2.2.3 scikit-learn==1.5.2
pip install opencv-python==4.10.0.84 pillow==11.1.0
pip install openslide-python==1.4.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

cd "$SCRATCH"
if [ ! -d histocartography ]; then
    git clone https://github.com/BiomedSciAI/histocartography
fi
cd histocartography
pip install -e .
mkdir -p checkpoints

echo "Download required HistoCartography checkpoints into $SCRATCH/histocartography/checkpoints before running extraction."

echo "Environment setup complete"
