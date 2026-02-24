#!/bin/bash
#SBATCH -J wsi-graphs
#SBATCH -o logs/wsi_graphs_%j.out
#SBATCH -e logs/wsi_graphs_%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 8:00:00
#SBATCH -A ASC25123

set -euo pipefail

# Optional overrides when submitting:
# sbatch --export=ALL,PATHRAG_INPUT_DIR=/path/to/wsi_raw,PATHRAG_OUTPUT_ROOT=/path/to/wsi_processed run_wsi_graphs.slurm

DATA_ROOT="${PATHRAG_DATA_ROOT:-$SCRATCH/Pediatric-Brain-Tumor/data}"
INPUT_DIR="${PATHRAG_INPUT_DIR:-$SCRATCH/tcga_gbm_lgg_pedi}"
OUTPUT_ROOT="${PATHRAG_OUTPUT_ROOT:-$DATA_ROOT/wsi_processed/tcga_wsi_graphs}"
OUTPUT_DIR="${PATHRAG_OUTPUT_DIR:-$OUTPUT_ROOT/${SLURM_JOB_ID}}"
RESULT_DIR="${PATHRAG_RESULT_DIR:-$DATA_ROOT/wsi_processed/wsi_graph_results}"
TEMP_DIR="${PATHRAG_TEMP_DIR:-/tmp/wsi_graphs_${SLURM_JOB_ID}}"
REPO_DIR="${PATHRAG_REPO_DIR:-$SCRATCH/path-rag}"

# Graph extraction parameters (override via --export)
THUMBNAIL_SIZE="${PATHRAG_THUMBNAIL_SIZE:-4096}"
KNN_K="${PATHRAG_KNN_K:-5}"
KNN_THRESH="${PATHRAG_KNN_THRESH:-50}"
PATCH_SIZE="${PATHRAG_PATCH_SIZE:-224}"
RESIZE_SIZE="${PATHRAG_RESIZE_SIZE:-224}"

mkdir -p "$OUTPUT_DIR" "$RESULT_DIR" "$TEMP_DIR" logs

module load gcc cuda
module load python3

source activate path-rag

cd "$REPO_DIR"

if [ ! -d "$INPUT_DIR" ]; then
    echo "Input directory does not exist: $INPUT_DIR"
    exit 1
fi

svs_count=$(find "$INPUT_DIR" -maxdepth 1 -type f -name "*.svs" | wc -l)
if [ "$svs_count" -eq 0 ]; then
    echo "No .svs files found under $INPUT_DIR"
    exit 1
fi

echo "Starting WSI graph extraction at $(date)"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Result directory: $RESULT_DIR"
echo "SVS file count: $svs_count"

python process_wsi_graphs.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" --temp_dir "$TEMP_DIR" --thumbnail_size "$THUMBNAIL_SIZE" --k "$KNN_K" --thresh "$KNN_THRESH" --patch_size "$PATCH_SIZE" --resize_size "$RESIZE_SIZE"

cp -r "$OUTPUT_DIR"/* "$RESULT_DIR"/
rm -rf "$TEMP_DIR"

echo "Completed at $(date)"
