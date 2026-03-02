#!/bin/bash
set -e
# Compare init_state=random vs init_state=buffer across 3 seeds

# small models (without halting)
for seed in 1 2 3; do
  for mode in random buffer; do
    echo "=== init_state=$mode seed=$seed ==="
    uv run main.py --batch_size 128 --h_dim 256 --N_supervision 8 --halt_loss_weight 0.0 --halt_prob_thresh 2.0 \
      --steps 15_000 --skip_test \
      --init_state "$mode" \
      --checkpoint "models/small_init_${mode}_seed${seed}.pt"
  done
done
