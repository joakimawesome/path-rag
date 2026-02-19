#!/bin/bash
#SBATCH -J nuclei-smoke
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:20:00
#SBATCH -o logs/nuclei_smoke_%j.out
#SBATCH -e logs/nuclei_smoke_%j.err
#SBATCH -A ASC25123

set -euo pipefail

DATA_ROOT="$SCRATCH/Pediatric-Brain-Tumor/data"
INPUT_DIR="$DATA_ROOT/wsi_raw"
SMOKE_OUTPUT="$DATA_ROOT/wsi_processed/smoke_${SLURM_JOB_ID}"
SMOKE_LIST="$SCRATCH/path-rag/smoke_first_wsi_${SLURM_JOB_ID}.txt"
TEMP_DIR="/tmp/nuclei_smoke_${SLURM_JOB_ID}"

mkdir -p "$SMOKE_OUTPUT" "$TEMP_DIR" logs

module load python3
module load cuda/12.4
module load gcc

source activate path-rag

cd "$SCRATCH/path-rag"

first_wsi=$(find "$INPUT_DIR" -maxdepth 1 -type f -name "*.svs" | sort | head -n 1)
if [ -z "$first_wsi" ]; then
    echo "No .svs files found under $INPUT_DIR"
    exit 1
fi
printf "%s\n" "$first_wsi" > "$SMOKE_LIST"

python process_wsi_batch.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$SMOKE_OUTPUT" \
    --temp_dir "$TEMP_DIR" \
    --file_list "$SMOKE_LIST"

echo "Smoke test complete. Output: $SMOKE_OUTPUT"
