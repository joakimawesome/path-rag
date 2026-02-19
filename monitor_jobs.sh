#!/bin/bash
set -euo pipefail

echo "=== Current Jobs ==="
squeue -u "$USER"

echo
echo "=== Disk Usage ==="
echo "HOME: $(du -sh "$HOME" 2>/dev/null | cut -f1)"
if [ -d "$SCRATCH/Pediatric-Brain-Tumor/data/wsi_processed" ]; then
    echo "SCRATCH (wsi_processed): $(du -sh "$SCRATCH/Pediatric-Brain-Tumor/data/wsi_processed" 2>/dev/null | cut -f1)"
else
    echo "SCRATCH (wsi_processed): N/A"
fi

echo
echo "=== Recent Log Files ==="
ls -lt logs/*.out 2>/dev/null | head -5 || true

echo
echo "=== Output Files ==="
echo "$(find "$SCRATCH/wsi_output" -name "*.npz" 2>/dev/null | wc -l) nuclei result files found"
