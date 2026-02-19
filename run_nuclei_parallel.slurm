#!/bin/bash
#SBATCH -J nuclei-parallel
#SBATCH -p gh
#SBATCH -N 4
#SBATCH -n 4
#SBATCH -t 24:00:00
#SBATCH -o logs/nuclei_parallel_%j.out
#SBATCH -e logs/nuclei_parallel_%j.err
#SBATCH -A YOUR_PROJECT_ID

set -euo pipefail

INPUT_DIR="$WORK/wsi_input"
OUTPUT_DIR="$SCRATCH/wsi_output"
RESULT_DIR="$WORK/nuclei_results"
REPO_DIR="$SCRATCH/path-rag"

mkdir -p "$OUTPUT_DIR" "$RESULT_DIR" logs

module load python3
module load cuda/12.4
module load gcc
module load pylauncher

source activate path-rag

cd "$REPO_DIR"

ls "$INPUT_DIR"/*.svs > all_files.txt
split -n l/"$SLURM_NTASKS" --numeric-suffixes=0 -a 2 all_files.txt wsi_batch_

: > commands.txt
for i in $(seq 0 $((SLURM_NTASKS - 1))); do
    batch_file=$(printf "wsi_batch_%02d" "$i")
    if [ -f "$batch_file" ]; then
        echo "python process_wsi_batch.py --input_dir $INPUT_DIR --output_dir ${OUTPUT_DIR}/batch_${i} --file_list ${REPO_DIR}/${batch_file}" >> commands.txt
    fi
done

python3 "$TACC_PYLAUNCHER_DIR/pylauncher.py" commands.txt

mkdir -p "$OUTPUT_DIR/final"
find "$OUTPUT_DIR"/batch_* -mindepth 1 -maxdepth 1 -type d -exec cp -r {} "$OUTPUT_DIR/final"/ \;
cp -r "$OUTPUT_DIR/final"/* "$RESULT_DIR"/

echo "Parallel processing complete"
