#!/bin/bash
#SBATCH -J test-nuclei
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -o logs/test_%j.out
#SBATCH -e logs/test_%j.err
#SBATCH -A YOUR_PROJECT_ID

set -euo pipefail

TEST_INPUT="$WORK/wsi_input"
TEST_OUTPUT="$SCRATCH/test_output"

mkdir -p "$TEST_OUTPUT" logs

module load python3
module load cuda/12.4
module load gcc

source activate path-rag

cd "$SCRATCH/path-rag"

echo "Testing nuclei extraction"
python process_wsi_batch.py \
    --input_dir "$TEST_INPUT" \
    --output_dir "$TEST_OUTPUT" \
    --temp_dir /tmp

echo "Test complete. Output: $TEST_OUTPUT"
ls -lh "$TEST_OUTPUT"
