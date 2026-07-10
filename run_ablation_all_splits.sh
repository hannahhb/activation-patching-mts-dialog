#!/usr/bin/env bash
# Runs causal_intervention.py (Design 1, ablation/necessity test) across all
# 6 dataset splits, reading the activation caches produced by
# run_redeep_all_splits.sh.
#
# Uses the single-split form of --span-dir (luq_out/llama_judge/{config}/{split},
# not the shared root) -- this is required for correctness: every split
# numbers its samples from 0, so the shared-root form silently matches the
# wrong split's judge labels via first-glob-wins.
#
# Usage:
#   ./run_ablation_all_splits.sh [N]
#   N defaults to 5 if not given -- should match (or be <=) the N used for
#   run_redeep_all_splits.sh, since --max-gen-idx only filters generations
#   that were actually cached in that stage.
#
# Must be run from sae_experiments/ (same directory as causal_intervention.py),
# AFTER run_redeep_all_splits.sh has completed for these splits.
set -euo pipefail

N="${1:-5}"
SPLITS=(aci/test1 aci/test2 aci/test3 virtscribe/test1 virtscribe/test2 virtscribe/test3)

for CONFIG_SPLIT in "${SPLITS[@]}"; do
  TAG="${CONFIG_SPLIT//\//_}"
  ACT_DIR="redeep_out/${TAG}/activations"

  if [ ! -d "${ACT_DIR}" ]; then
    echo "=== ${CONFIG_SPLIT}: skipping, no ${ACT_DIR} (run run_redeep_all_splits.sh first) ==="
    continue
  fi

  echo "=== ${CONFIG_SPLIT} -> causal_out/${TAG} (ablation, first ${N} generations/sample) ==="

  python causal_intervention.py \
    --mode ablation \
    --act-dir   "${ACT_DIR}" \
    --span-dir  "luq_out/llama_judge/${CONFIG_SPLIT}" \
    --gen-dir   "luq_out/llama/generations/${CONFIG_SPLIT}" \
    --copy-mask "redeep_out/${TAG}/copying_head_mask.npy" \
    --out       "causal_out/${TAG}" \
    --max-gen-idx "${N}"

  echo
done

echo "Done. Ablation results written to causal_out/{tag}/ablation_results.csv for each split."
echo
echo "To merge all splits into one combined result:"
cat <<'EOF'
  python3 -c "
import pandas as pd
from pathlib import Path
dfs = [pd.read_csv(f).assign(split=f.parent.name) for f in Path('causal_out').glob('*/ablation_results.csv')]
combined = pd.concat(dfs, ignore_index=True)
combined.to_csv('causal_out/combined_ablation_results.csv', index=False)
print(combined.groupby('split').size())
"
EOF
