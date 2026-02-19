#!/bin/bash
#SBATCH -J nuclei-extract
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -o logs/nuclei_%j.out
#SBATCH -e logs/nuclei_%j.err
#SBATCH -A ASC25123

set -euo pipefail

DATA_ROOT="$SCRATCH/Pediatric-Brain-Tumor/data"
INPUT_DIR="$DATA_ROOT/wsi_raw"
OUTPUT_DIR="$DATA_ROOT/wsi_processed/nuclei_batches/${SLURM_JOB_ID}"
RESULT_DIR="$DATA_ROOT/wsi_processed/nuclei_results"
TEMP_DIR="/tmp/nuclei_extract_${SLURM_JOB_ID}"

mkdir -p "$OUTPUT_DIR" "$RESULT_DIR" "$TEMP_DIR" logs

module load python3
module load cuda/12.4
module load gcc

source activate path-rag

cd "$SCRATCH/path-rag"

echo "Starting nuclei extraction at $(date)"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Hostname: $(hostname)"

python process_wsi_batch.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --temp_dir "$TEMP_DIR"

cp -r "$OUTPUT_DIR"/* "$RESULT_DIR"/
rm -rf "$TEMP_DIR"

echo "Job completed at $(date)"
