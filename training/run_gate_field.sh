#!/bin/bash
# Certification of the trained g_geo and g_eff surrogates (gate_field.py):
# one Slurm array task per (field, dimension), 10 tasks. Task I: field I div 5
# (0 = g_geo, 1 = g_eff), D = 2 + I mod 5. Reads the packed fields in
# fields/ and writes one JSON per field and dimension to results/.
#
#   sbatch --array=0-9 run_gate_field.sh
#SBATCH --job-name=field_gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/field_gate_%A_%a.out
#SBATCH --error=logs/field_gate_%A_%a.err
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
I="${SLURM_ARRAY_TASK_ID:-0}"
FIELDS=(v2 geff); FIELD="${FIELDS[$(( I / 5 ))]}"; D=$(( 2 + I % 5 ))

# module load ...   # your Python environment with torch, numpy, scipy
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
mkdir -p "$REPO/logs"
cd "$REPO/training"
python -u gate_field.py --field "$FIELD" --D "$D"
