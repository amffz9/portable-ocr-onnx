#!/usr/bin/env bash
# Download the Nemotron OCR v2 checkpoints from Hugging Face into weights/.
#
# The weights are licensed by NVIDIA Corporation under the NVIDIA Open Model
# License and are never committed to this repository.
#
# Usage: scripts/fetch_weights.sh [english|multilingual|all]   (default: all)
#
# Layout produced:
#   weights/v2_english/{detector,recognizer,relational}.pth, charset.txt, model_config.json
#   weights/v2_multilingual/...
#   weights/example.py, weights/ocr-example-input-1.png   (upstream sample)

set -euo pipefail

REPO_ID="nvidia/nemotron-ocr-v2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${NEMOTRON_OCR_WEIGHTS:-$ROOT/weights}"
WHICH="${1:-all}"

case "$WHICH" in
  english)      PATTERNS=("v2_english/*") ;;
  multilingual) PATTERNS=("v2_multilingual/*") ;;
  all)          PATTERNS=("v2_*/*") ;;
  *) echo "usage: $0 [english|multilingual|all]" >&2; exit 2 ;;
esac
PATTERNS+=("example.py" "ocr-example-input-1.png")

mkdir -p "$DEST"

if command -v hf >/dev/null 2>&1; then
  args=()
  for p in "${PATTERNS[@]}"; do args+=(--include "$p"); done
  hf download "$REPO_ID" "${args[@]}" --local-dir "$DEST"
elif command -v git >/dev/null 2>&1 && git lfs version >/dev/null 2>&1; then
  echo "hf CLI not found; falling back to git lfs" >&2
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 "https://huggingface.co/$REPO_ID" "$tmp/repo"
  inc="$(IFS=,; echo "${PATTERNS[*]}")"
  git -C "$tmp/repo" lfs pull --include "$inc"
  for p in "${PATTERNS[@]}"; do
    # shellcheck disable=SC2086
    (cd "$tmp/repo" && cp -r --parents $p "$DEST/")
  done
else
  echo "error: need either the 'hf' CLI (pip install huggingface_hub) or git-lfs" >&2
  exit 1
fi

for d in "$DEST"/v2_*; do
  for f in detector.pth recognizer.pth relational.pth charset.txt; do
    [[ -s "$d/$f" ]] || { echo "error: missing $d/$f" >&2; exit 1; }
  done
done
echo "Weights ready in $DEST"
