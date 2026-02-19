#!/bin/bash
#SBATCH -J test-nuclei
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -o logs/test_%j.out
#SBATCH -e logs/test_%j.err
#SBATCH -A ASC25123

set -euo pipefail

DATA_ROOT="$SCRATCH/Pediatric-Brain-Tumor/data"
TEST_INPUT="$DATA_ROOT/wsi_raw"
TEST_OUTPUT="$DATA_ROOT/wsi_processed/test_${SLURM_JOB_ID}"
TEST_TEMP_DIR="/tmp/nuclei_extract_${SLURM_JOB_ID}"

mkdir -p "$TEST_OUTPUT" "$TEST_TEMP_DIR" logs

module load python3
module load cuda/12.4
module load gcc

source activate path-rag

cd "$SCRATCH/path-rag"

echo "Testing nuclei extraction"
python process_wsi_batch.py \
    --input_dir "$TEST_INPUT" \
    --output_dir "$TEST_OUTPUT" \
    --temp_dir "$TEST_TEMP_DIR"

echo "Test complete. Output: $TEST_OUTPUT"
ls -lh "$TEST_OUTPUT"
