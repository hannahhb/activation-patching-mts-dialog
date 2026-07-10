#!/usr/bin/env bash
# Runs redeep_sentence.py (ECS/PKS word-level activation caching) across all
# 6 dataset splits, capped to the first N generations per sample.
#
# Each split gets its OWN --out directory (redeep_out/{config}_{split}/) --
# do not point multiple splits at the same --out, since every split numbers
# its samples from 0 and a shared activations/ dir would collide.
#
# Usage:
#   ./run_redeep_all_splits.sh [N]
#   N defaults to 5 if not given.
#
# Must be run from sae_experiments/ (same directory as redeep_sentence.py).
set -euo pipefail

N="${1:-5}"
SPLITS=(aci/test1 aci/test2 aci/test3 virtscribe/test1 virtscribe/test2 virtscribe/test3)

for CONFIG_SPLIT in "${SPLITS[@]}"; do
  TAG="${CONFIG_SPLIT//\//_}"   # e.g. aci/test1 -> aci_test1
  OUT="redeep_out/${TAG}"
  echo "=== ${CONFIG_SPLIT} -> ${OUT} (first ${N} generations/sample) ==="
  mkdir -p "${OUT}"

  # Reuse the already-computed (model-only, data-independent) copying-head
  # scores instead of re-running the OV eigendecomposition for every split.
  if [ -f copying_head_scores.npy ]; then
    cp copying_head_scores.npy "${OUT}/"
  fi

  python redeep_sentence.py \
    --generations "luq_out/llama/generations/${CONFIG_SPLIT}" \
    --sentences   "luq_out/llama/generations/${CONFIG_SPLIT}/sentences" \
    --out         "${OUT}" \
    --notes "${N}"

  echo
done

echo "Done. Activation caches written to redeep_out/{tag}/activations/ for each split."
