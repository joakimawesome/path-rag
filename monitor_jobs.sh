#!/bin/bash
set -euo pipefail

echo "=== Current Jobs ==="
squeue -u "$USER"

echo
echo "=== Disk Usage ==="
echo "HOME: $(du -sh "$HOME" 2>/dev/null | cut -f1)"
echo "WORK: $(du -sh "$WORK" 2>/dev/null | cut -f1)"
echo "SCRATCH: $(du -sh "$SCRATCH" 2>/dev/null | cut -f1)"

echo
echo "=== Recent Log Files ==="
ls -lt logs/*.out 2>/dev/null | head -5 || true

echo
echo "=== Output Files ==="
echo "$(find "$SCRATCH/wsi_output" -name "*.npz" 2>/dev/null | wc -l) nuclei result files found"
