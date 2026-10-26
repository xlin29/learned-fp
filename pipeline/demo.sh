#!/usr/bin/env bash
# pipeline/demo.sh — 2-3 minute CPU smoke test on the bundled lab_data/.
#
# Extracts 5 lab devices spanning the 3 paper Table 4 categories
# (browser-default / browser-defense / extension), trains 9 keys × 2
# epochs with the fixed-config trainer, runs vote, and prints Top-k.
#
# Goal: prove the train + eval code path runs end-to-end on a clean
# checkout. NOT a paper-scale reproduction (paper used 1,022–3,303
# devices, 9 keys, 30 epochs / Bayesian search). At this scale all
# Top-k numbers stay at the 5-class base rate (top1=0.2, top5+=1.0)
# — that is expected; uplift requires the paper-scale corpus.
#
# Usage:
#   ./pipeline/demo.sh
# Or via Docker (see pipeline/Dockerfile).
#
# Outputs land under demo_out/ (gitignored).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LAB="$REPO_ROOT/lab_data"
EXTRACT="$REPO_ROOT/lab_data/.demo_extract"
OUT="$REPO_ROOT/demo_out"
CFG="$REPO_ROOT/pipeline/configs/demo_lab.yaml"

# Paper Table 4 categories — five slugs spanning all three categories.
# These pattern-match a single physical device + its browser config in
# lab_data/manifest.json. Five classes is plenty for a smoke-test demo.
PATTERNS=(
  "*chrome-standard*.tar.gz"
  "*firefox-standard*.tar.gz"
  "*brave*.tar.gz"
  "*samsung-sat-strict*.tar.gz"
  "*canvas-fp-defender-chrome*.tar.gz"
)

cyan()  { printf '\033[36m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
red()   { printf '\033[31m%s\033[0m\n' "$*" >&2; }

cyan "[demo] checking prerequisites…"
command -v python3 >/dev/null || { red "python3 not on PATH"; exit 1; }
python3 -c "import torch, numpy, yaml" 2>/dev/null \
  || { red "Missing deps. Install: pip install -r pipeline/requirements.txt"; exit 1; }

cyan "[demo] picking 5 representative devices from lab_data/devices/"
rm -rf "$EXTRACT"
mkdir -p "$EXTRACT"
PICKED=()
for pat in "${PATTERNS[@]}"; do
    # shellcheck disable=SC2086
    file=$(ls $LAB/devices/$pat 2>/dev/null | head -1 || true)
    if [ -z "${file:-}" ]; then
        red "no device matching pattern: $pat"; exit 1
    fi
    PICKED+=("$file")
    tar -xzf "$file" -C "$EXTRACT"
    printf '  %s\n' "$(basename "$file")"
done

cyan "[demo] extracted $(ls -d "$EXTRACT"/dev_* | wc -l | tr -d ' ') device dirs"

# Run training (loops over the 9 keys in demo_lab.yaml internally).
rm -rf "$OUT"
mkdir -p "$OUT"
cyan "[demo] training (9 keys × 2 epochs, batch=64)…"
python3 "$REPO_ROOT/pipeline/cli.py" train \
    --config "$CFG" \
    --data "$EXTRACT" \
    --out "$OUT"

cyan "[demo] evaluating (mode=vote on the same split)…"
python3 "$REPO_ROOT/pipeline/cli.py" eval \
    --config "$CFG" \
    --data "$EXTRACT" \
    --models "$OUT" \
    --mode vote

green "[demo] PASS — train + eval pipeline works end-to-end."
echo
echo "Output:        $OUT/"
echo "Trained models: $(ls "$OUT/models" 2>/dev/null | wc -l | tr -d ' ') .pt"
echo "Cleanup:       rm -rf $EXTRACT $OUT"
