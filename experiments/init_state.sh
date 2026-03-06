#!/bin/bash
# Compare init_state=random vs init_state=buffer across 3 seeds with small model (without halting)

for seed in 1 2 3; do
  for mode in random buffer; do

    # For now, hardcode k=3 for random init
    k_passes=$( [ "$mode" = "random" ] && echo 3 || echo 1 )
    checkpoint="models/small_init=${mode}_k=${k_passes}_seed=${seed}.pt"

    if [ -f "$checkpoint" ]; then
      echo "Checkpoint $checkpoint exists; skipping."
      continue
    fi
    echo "Starting training for $checkpoint"

    uv run main.py --batch_size 128 --h_dim 256 --N_supervision 8 --halt_loss_weight 0.0 --halt_prob_thresh 2.0 \
      --steps 10_000 --test_size 8192 --N_supervision_test 256 --k_passes "$k_passes" \
      --init_state "$mode" \
      --checkpoint "$checkpoint" \
      --seed "$seed"
  done
done